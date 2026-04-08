//! Anthropic Claude Messages API client with streaming support.
//!
//! Implements the `LlmClient` trait for Anthropic's Messages API, providing
//! full support for streaming responses, tool calling, extended thinking,
//! prompt caching, and vision.
//!
//! # API Endpoints
//!
//! ## Direct Anthropic API
//!
//! - Non-streaming: POST `https://api.anthropic.com/v1/messages`
//! - Streaming: POST `https://api.anthropic.com/v1/messages` with `stream: true`
//!
//! ## Azure Anthropic
//!
//! - Non-streaming: POST `{base_url}/v1/messages`
//! - Streaming: POST `{base_url}/v1/messages` with `stream: true`
//!
//! ## AWS Bedrock
//!
//! - Non-streaming: POST `https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke`
//! - Streaming: POST `https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke-with-response-stream`
//!
//! # Authentication
//!
//! - **Direct Anthropic**: `x-api-key` header with API key from console.anthropic.com
//! - **Azure Anthropic**: `x-api-key` or `Authorization: Bearer ...`
//! - **AWS Bedrock**: `Authorization: Bearer {token}` header with Bedrock API key

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{sign, SignableBody, SignableRequest, SigningSettings};
use aws_sigv4::sign::v4;
use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use futures::StreamExt;
use once_cell::sync::Lazy;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, HOST};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};
use tracing::{debug, error, info, trace, warn};

use super::config::{AnthropicConfig, AzureAnthropicAuthMethod, BedrockAuthMethod};
use super::convert::{from_unified_messages, from_unified_tools};
use super::rate_limiter::RateLimiter;
use super::streaming::{AccumulatedBlock, ErrorData, StreamEvent};
use super::types::{
    CacheControl, ContentBlock, ErrorResponse, Message, SystemBlock, SystemPrompt, Tool,
};
use crate::llm::provider::{LlmClient, ProviderFailureCapture};
use crate::llm::unified::{UnifiedMessage, UnifiedTool, UnifiedToolCall};

/// Custom error type for retryable stream errors.
///
/// Used to signal that a streaming error should trigger a retry of the entire request.
/// Covers both API-level errors (rate limits, overload) and network-level errors
/// (connection drops, EOF during chunk reading).
#[derive(Debug)]
enum RetryableStreamError {
    /// API error from structured error event in stream
    ApiError { error_data: ErrorData },
    /// Network error during chunk reading (EOF, connection reset, etc.)
    NetworkError { message: String },
}

impl std::fmt::Display for RetryableStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiError { error_data } => write!(
                f,
                "Retryable API error ({}): {}",
                error_data.error_type, error_data.message
            ),
            Self::NetworkError { message } => {
                write!(f, "Retryable network error during streaming: {}", message)
            }
        }
    }
}

impl std::error::Error for RetryableStreamError {}

/// Global rate limiter instance shared across all Anthropic clients.
///
/// Initialized on first use based on the first client's configuration.
/// Coordinates token usage across all parallel workers to prevent org-wide rate limit violations.
static GLOBAL_RATE_LIMITER: Lazy<Mutex<Option<RateLimiter>>> = Lazy::new(|| Mutex::new(None));

/// Anthropic Claude Messages API client.
///
/// Handles authentication, request construction, SSE parsing, and response
/// conversion for the Anthropic Messages API.
///
/// # Features
///
/// - Streaming responses via Server-Sent Events
/// - Tool calling with parallel execution support
/// - Extended thinking with token budgets
/// - Prompt caching with 5m/1h TTL
/// - Vision (images) and documents (PDFs)
/// - Server tools (web search, bash, code execution)
///
/// # Examples
///
/// ```ignore
/// use appam::llm::anthropic::{AnthropicClient, AnthropicConfig};
/// use appam::llm::unified::UnifiedMessage;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let config = AnthropicConfig::default();
///     let client = AnthropicClient::new(config)?;
///
///     let messages = vec![UnifiedMessage::user("Hello!")];
///
///     client.chat_with_tools_streaming(
///         &messages,
///         &[],
///         |chunk| { print!("{}", chunk); Ok(()) },
///         |_tools| Ok(()),
///         |_reasoning| Ok(()),
///         |_partial| Ok(()),
///     ).await?;
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    http: reqwest::Client,
    config: AnthropicConfig,
    /// Last failed provider exchange captured for diagnostics.
    last_failed_exchange: std::sync::Arc<Mutex<Option<ProviderFailureCapture>>>,
}

impl AnthropicClient {
    /// Create a new Anthropic Messages API client.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - HTTP client construction fails
    /// - Configuration validation fails
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        let mut config = config;
        config.validate()?;

        if let Some(ref mut azure) = config.azure {
            azure.base_url = azure.normalized_base_url()?;
        }

        // For Bedrock, use the Bedrock endpoint; otherwise use the standard base URL
        let base_url = if let Some(ref bedrock) = config.bedrock {
            format!("https://bedrock-runtime.{}.amazonaws.com", bedrock.region)
        } else if let Some(ref azure) = config.azure {
            azure.base_url.clone()
        } else {
            config.base_url.clone()
        };

        let http = crate::http::client_pool::get_or_init_client(&base_url, |ctx| {
            // Configure HTTP client with optimizations:
            // - connect_timeout: 30s to prevent hanging on connection establishment
            // - pool_idle_timeout: 120s to clean up stale connections
            // - pool_max_idle_per_host: Limit to 10 idle connections per host
            // - tcp_keepalive: 60s to detect dead connections
            // - tcp_nodelay: Disable Nagle's algorithm for lower latency
            // - gzip: Enable compression to reduce bandwidth
            // - resolve_to_addrs: Cache DNS lookups to prevent flaky DNS errors
            // - NO read timeout: Allow infinite active connection time for streaming responses
            //
            // Note: We let reqwest negotiate HTTP/1.1 or HTTP/2 via ALPN instead of
            // forcing http2_prior_knowledge(), as some servers don't support raw HTTP/2.
            let mut builder = reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(30))
                .pool_idle_timeout(std::time::Duration::from_secs(120))
                .pool_max_idle_per_host(10)
                .tcp_keepalive(std::time::Duration::from_secs(60))
                .tcp_nodelay(true)
                .gzip(true)
                .user_agent("appam/0.1.1");

            if let Some(addrs) = ctx.resolved_addrs() {
                builder = builder.resolve_to_addrs(ctx.host(), addrs);
            }

            builder.build().context("Failed to create HTTP client")
        })?;

        Ok(Self {
            http,
            config,
            last_failed_exchange: std::sync::Arc::new(Mutex::new(None)),
        })
    }

    /// Clear any stale failed-exchange diagnostics before a new request starts.
    fn clear_last_failed_exchange(&self) {
        *self
            .last_failed_exchange
            .lock()
            .expect("last_failed_exchange mutex poisoned") = None;
    }

    /// Persist failed provider exchange diagnostics for later retrieval.
    fn record_failed_exchange(
        &self,
        http_status: Option<reqwest::StatusCode>,
        request_payload: &str,
        response_payload: impl Into<String>,
    ) {
        let (provider, model) = if let Some(ref bedrock) = self.config.bedrock {
            ("bedrock", bedrock.model_id.clone())
        } else if self.is_azure() {
            ("azure-anthropic", self.config.model.clone())
        } else {
            ("anthropic", self.config.model.clone())
        };

        let capture = ProviderFailureCapture {
            provider: provider.to_string(),
            model,
            http_status: http_status.map(|status| status.as_u16()),
            request_payload: request_payload.to_string(),
            response_payload: response_payload.into(),
            provider_response_id: None,
        };

        *self
            .last_failed_exchange
            .lock()
            .expect("last_failed_exchange mutex poisoned") = Some(capture);
    }

    /// Retrieve and clear the most recent failed provider exchange.
    pub fn take_last_failed_exchange(&self) -> Option<ProviderFailureCapture> {
        self.last_failed_exchange
            .lock()
            .expect("last_failed_exchange mutex poisoned")
            .take()
    }

    /// Check if this client is configured for AWS Bedrock.
    ///
    /// Returns true if `bedrock` configuration is present.
    fn is_bedrock(&self) -> bool {
        self.config.bedrock.is_some()
    }

    /// Check if this client is configured for Azure Anthropic.
    ///
    /// Returns true when Azure transport settings are present. This mode keeps
    /// Anthropic's Messages wire format but switches endpoint and authentication
    /// handling to Azure-hosted semantics.
    fn is_azure(&self) -> bool {
        self.config.azure.is_some()
    }

    /// Check if streaming is supported for the current configuration.
    ///
    /// AWS Bedrock with Bearer token authentication only supports the non-streaming
    /// `/invoke` endpoint. The streaming `/invoke-with-response-stream` endpoint
    /// requires SigV4 authentication.
    ///
    /// Returns:
    /// - `true` for direct Anthropic API (always supports streaming)
    /// - `true` for Bedrock with SigV4 auth (supports streaming)
    /// - `false` for Bedrock with Bearer token auth (non-streaming only)
    #[allow(dead_code)]
    fn supports_streaming(&self) -> bool {
        if let Some(ref bedrock) = self.config.bedrock {
            // SigV4 supports streaming, Bearer token does not
            bedrock.auth_method == BedrockAuthMethod::SigV4
        } else {
            // Direct Anthropic API always supports streaming
            true
        }
    }

    /// Build the API endpoint URL based on configuration.
    ///
    /// # Direct Anthropic API
    ///
    /// Uses the base URL from config: `{base_url}/v1/messages`
    ///
    /// # Azure Anthropic
    ///
    /// Uses the normalized Azure base URL from config: `{azure.base_url}/v1/messages`
    ///
    /// # AWS Bedrock
    ///
    /// - SigV4 auth: Uses streaming endpoint when `config.stream` is true
    /// - Bearer token: Always uses non-streaming endpoint (streaming not supported)
    fn build_endpoint_url(&self) -> String {
        if let Some(ref bedrock) = self.config.bedrock {
            match bedrock.auth_method {
                BedrockAuthMethod::SigV4 => {
                    // SigV4 supports streaming
                    if self.config.stream {
                        bedrock.streaming_endpoint()
                    } else {
                        bedrock.invoke_endpoint()
                    }
                }
                BedrockAuthMethod::BearerToken => {
                    // Bearer token only supports non-streaming
                    bedrock.invoke_endpoint()
                }
            }
        } else if let Some(ref azure) = self.config.azure {
            format!("{}/v1/messages", azure.base_url)
        } else {
            format!("{}/v1/messages", self.config.base_url)
        }
    }

    /// Build HTTP headers for API requests.
    ///
    /// # Direct Anthropic API Headers
    ///
    /// - `x-api-key`: Authentication (from config or `ANTHROPIC_API_KEY` env)
    /// - `anthropic-version`: API version (2023-06-01)
    /// - `anthropic-beta`: Beta feature headers (if any)
    /// - `content-type`: application/json
    /// - `accept`: text/event-stream (for streaming)
    ///
    /// # Azure Anthropic Headers
    ///
    /// - `anthropic-version`: API version (2023-06-01)
    /// - `anthropic-beta`: Beta feature headers (if any)
    /// - `x-api-key`: API-key auth when `AzureAnthropicAuthMethod::XApiKey`
    /// - `Authorization: Bearer ...`: bearer auth when `AzureAnthropicAuthMethod::BearerToken`
    /// - `content-type`: application/json
    /// - `accept`: text/event-stream
    ///
    /// # AWS Bedrock Headers (Bearer Token)
    ///
    /// - `Authorization`: Bearer token (from config or `AWS_BEARER_TOKEN_BEDROCK` env)
    /// - `content-type`: application/json
    /// - `accept`: application/json
    ///
    /// # AWS Bedrock Headers (SigV4)
    ///
    /// For SigV4 authentication, this method returns base headers only.
    /// The actual auth headers are added by `sign_request_sigv4()` after
    /// the request body is known.
    ///
    /// Note: For Bedrock, `anthropic_version` is sent in the request body, not as a header.
    ///
    /// # Errors
    ///
    /// Returns an error if API key/token is missing (for non-SigV4) or headers cannot be constructed.
    ///
    /// # Security
    ///
    /// Never logs the API key/token header value.
    fn build_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();

        // Content type (same for all)
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(ref bedrock) = self.config.bedrock {
            match bedrock.auth_method {
                BedrockAuthMethod::SigV4 => {
                    // SigV4: Base headers only - auth headers added by sign_request_sigv4()
                    // Accept header depends on whether we're streaming
                    if self.config.stream {
                        headers.insert(
                            ACCEPT,
                            HeaderValue::from_static("application/vnd.amazon.eventstream"),
                        );
                    } else {
                        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
                    }

                    // Add host header (required for SigV4)
                    let host = format!("bedrock-runtime.{}.amazonaws.com", bedrock.region);
                    headers.insert(
                        HOST,
                        HeaderValue::from_str(&host).context("Invalid host header")?,
                    );
                }
                BedrockAuthMethod::BearerToken => {
                    // Bearer token authentication
                    let bearer_token = self
                        .config
                        .api_key
                        .clone()
                        .or_else(|| std::env::var("AWS_BEARER_TOKEN_BEDROCK").ok())
                        .ok_or_else(|| {
                            anyhow!(
                                "Missing AWS Bedrock bearer token. Set AWS_BEARER_TOKEN_BEDROCK env var or config.api_key"
                            )
                        })?;

                    headers.insert(
                        AUTHORIZATION,
                        HeaderValue::from_str(&format!("Bearer {}", bearer_token))
                            .context("Invalid bearer token header format")?,
                    );

                    headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
                }
            }
        } else if let Some(ref azure) = self.config.azure {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

            let direct_anthropic_env_key = std::env::var("ANTHROPIC_API_KEY").ok();
            let explicit_config_credential = self.config.api_key.as_ref().filter(|key| {
                direct_anthropic_env_key
                    .as_ref()
                    .map(|env_key| env_key != *key)
                    .unwrap_or(true)
            });

            let credential = if let Some(key) = explicit_config_credential {
                key.clone()
            } else {
                match azure.auth_method {
                    AzureAnthropicAuthMethod::XApiKey => std::env::var("AZURE_ANTHROPIC_API_KEY")
                        .or_else(|_| std::env::var("AZURE_API_KEY"))
                        .context(
                            "Missing Azure Anthropic API key. Set AZURE_ANTHROPIC_API_KEY or AZURE_API_KEY env var",
                        )?,
                    AzureAnthropicAuthMethod::BearerToken => std::env::var(
                        "AZURE_ANTHROPIC_AUTH_TOKEN",
                    )
                    .or_else(|_| std::env::var("AZURE_API_KEY"))
                    .context(
                        "Missing Azure Anthropic bearer token. Set AZURE_ANTHROPIC_AUTH_TOKEN or AZURE_API_KEY env var",
                    )?,
                }
            };

            match azure.auth_method {
                AzureAnthropicAuthMethod::XApiKey => {
                    headers.insert(
                        "x-api-key",
                        HeaderValue::from_str(&credential)
                            .context("Invalid Azure Anthropic API key header format")?,
                    );
                }
                AzureAnthropicAuthMethod::BearerToken => {
                    headers.insert(
                        AUTHORIZATION,
                        HeaderValue::from_str(&format!("Bearer {}", credential))
                            .context("Invalid Azure Anthropic bearer header format")?,
                    );
                }
            }
        } else {
            // Direct Anthropic API authentication
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

            let api_key = self
                .config
                .api_key
                .clone()
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                .ok_or_else(|| {
                    anyhow!(
                        "Missing Anthropic API key. Set ANTHROPIC_API_KEY env var or config.api_key"
                    )
                })?;

            headers.insert(
                "x-api-key",
                HeaderValue::from_str(&api_key).context("Invalid API key header format")?,
            );

            // Beta features (only for direct Anthropic API)
            if self.config.beta_features.has_any() {
                let beta_values = self.config.beta_features.to_header_values();
                let beta_header = beta_values.join(",");
                headers.insert(
                    "anthropic-beta",
                    HeaderValue::from_str(&beta_header).context("Invalid beta header format")?,
                );
                debug!(beta_features = %beta_header, "Beta features enabled");
            }
        }

        Ok(headers)
    }

    /// Load AWS credentials from environment variables.
    ///
    /// Reads credentials from standard AWS environment variables:
    /// - `AWS_ACCESS_KEY_ID` - Required
    /// - `AWS_SECRET_ACCESS_KEY` - Required
    /// - `AWS_SESSION_TOKEN` - Optional (for temporary credentials)
    ///
    /// # Errors
    ///
    /// Returns an error if required credentials are missing.
    fn load_aws_credentials(&self) -> Result<Credentials> {
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID")
            .context("Missing AWS_ACCESS_KEY_ID environment variable")?;
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .context("Missing AWS_SECRET_ACCESS_KEY environment variable")?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();

        Ok(Credentials::new(
            access_key_id,
            secret_access_key,
            session_token,
            None, // Expiry time - we don't track this
            "appam-bedrock-sigv4",
        ))
    }

    /// Sign a request using AWS Signature Version 4.
    ///
    /// Creates SigV4 authentication headers for AWS Bedrock requests.
    /// This is required for the streaming endpoint (`invoke-with-response-stream`).
    ///
    /// # Arguments
    ///
    /// * `url` - The full request URL
    /// * `body` - The JSON request body as bytes
    /// * `headers` - Base headers to include in signature calculation
    ///
    /// # Returns
    ///
    /// HeaderMap containing the SigV4 authentication headers to add to the request:
    /// - `Authorization` - The SigV4 signature
    /// - `x-amz-date` - Timestamp of the request
    /// - `x-amz-security-token` - Session token (if using temporary credentials)
    /// - `x-amz-content-sha256` - SHA256 hash of the request body
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - AWS credentials are missing
    /// - Signing fails
    /// - Bedrock config is not present
    fn sign_request_sigv4(&self, url: &str, body: &[u8], headers: &HeaderMap) -> Result<HeaderMap> {
        let bedrock = self
            .config
            .bedrock
            .as_ref()
            .ok_or_else(|| anyhow!("SigV4 signing requires Bedrock configuration"))?;

        // Load AWS credentials and convert to Identity
        let credentials = self.load_aws_credentials()?;
        let identity = aws_smithy_runtime_api::client::identity::Identity::new(credentials, None);

        // Calculate body hash
        let body_hash = hex::encode(Sha256::digest(body));

        // Build signing settings
        let mut signing_settings = SigningSettings::default();
        signing_settings.signature_location = aws_sigv4::http_request::SignatureLocation::Headers;

        // Build signing params
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&bedrock.region)
            .name("bedrock")
            .time(SystemTime::now())
            .settings(signing_settings)
            .build()
            .context("Failed to build SigV4 signing params")?;

        // Convert reqwest headers to http::HeaderMap for signing
        let mut http_headers = http::HeaderMap::new();
        for (name, value) in headers.iter() {
            if let Ok(name) = http::HeaderName::from_bytes(name.as_str().as_bytes()) {
                if let Ok(value) = http::HeaderValue::from_bytes(value.as_bytes()) {
                    http_headers.insert(name, value);
                }
            }
        }

        // Add content-sha256 header (required for signing)
        http_headers.insert(
            http::HeaderName::from_static("x-amz-content-sha256"),
            http::HeaderValue::from_str(&body_hash).context("Invalid body hash")?,
        );

        // Parse URL
        let parsed_url = url::Url::parse(url).context("Invalid URL for signing")?;
        let path_and_query = if let Some(query) = parsed_url.query() {
            format!("{}?{}", parsed_url.path(), query)
        } else {
            parsed_url.path().to_string()
        };

        // Create signable request
        let signable_request = SignableRequest::new(
            "POST",
            &path_and_query,
            http_headers
                .iter()
                .map(|(k, v)| (k.as_str(), v.to_str().unwrap_or(""))),
            SignableBody::Bytes(body),
        )
        .context("Failed to create signable request")?;

        // Sign the request
        let signing_params_ref: aws_sigv4::http_request::SigningParams = signing_params.into();
        let (signing_instructions, _signature) = sign(signable_request, &signing_params_ref)
            .context("Failed to sign request with SigV4")?
            .into_parts();

        // Build result headers from signing instructions
        let mut auth_headers = HeaderMap::new();

        // Apply signing instructions to build auth headers
        for (name, value) in signing_instructions.headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                .context("Invalid signed header name")?;
            let header_value =
                HeaderValue::from_str(value).context("Invalid signed header value")?;
            auth_headers.insert(header_name, header_value);
        }

        // Add the content SHA256 header
        auth_headers.insert(
            reqwest::header::HeaderName::from_static("x-amz-content-sha256"),
            HeaderValue::from_str(&body_hash).context("Invalid body hash header")?,
        );

        trace!(
            region = %bedrock.region,
            "Request signed with SigV4 for Bedrock"
        );

        Ok(auth_headers)
    }

    /// Check if this client uses SigV4 authentication for Bedrock.
    fn uses_sigv4(&self) -> bool {
        self.config
            .bedrock
            .as_ref()
            .map(|b| b.auth_method == BedrockAuthMethod::SigV4)
            .unwrap_or(false)
    }

    /// Build the Messages API request body.
    ///
    /// Converts unified messages and tools to Anthropic's format.
    ///
    /// # Important Details
    ///
    /// - System messages are extracted and placed in `system` field
    /// - Remaining messages are converted to Anthropic format
    /// - Tools are converted to `input_schema` format
    /// - Thinking config becomes `thinking` parameter
    /// - Prompt caching uses Anthropic's top-level `cache_control` helper on
    ///   direct Anthropic and Azure Anthropic requests
    /// - Bedrock prompt caching uses block-level `cache_control` checkpoints
    ///   inside supported Anthropic fields (`system`, `messages`, `tools`)
    ///
    /// # Bedrock Differences
    ///
    /// - `model` field is NOT included (it's in the endpoint URL)
    /// - `anthropic_version` is added to the body (not a header)
    /// - `stream` field is NOT included (determined by endpoint choice)
    fn build_request_body(
        &self,
        messages: &[UnifiedMessage],
        tools: &[UnifiedTool],
    ) -> Result<serde_json::Value> {
        // Extract system prompt
        let (mut system_prompt, mut conversation_messages) =
            from_unified_messages(messages, &self.config)?;
        let mut anthropic_tools = if !tools.is_empty() {
            Some(from_unified_tools(tools, &self.config)?)
        } else {
            None
        };

        if self.is_bedrock() {
            if let Some(cache_control) = self
                .config
                .caching
                .as_ref()
                .and_then(|caching| caching.top_level_cache_control())
            {
                Self::apply_bedrock_prompt_caching(
                    &mut system_prompt,
                    &mut conversation_messages,
                    anthropic_tools.as_deref_mut(),
                    cache_control,
                );
            }
        }

        // Build base request - Bedrock doesn't include model (it's in URL)
        let mut body = if self.is_bedrock() {
            let bedrock = self.config.bedrock.as_ref().unwrap();
            json!({
                "anthropic_version": bedrock.anthropic_version,
                "max_tokens": self.config.max_tokens,
                "messages": conversation_messages,
            })
        } else {
            json!({
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": conversation_messages,
            })
        };

        // Add system prompt if present
        if let Some(system) = system_prompt {
            body["system"] = serde_json::to_value(system)?;
        }

        // Add Anthropic's top-level prompt cache helper. Anthropic applies the
        // breakpoint to the last cacheable block in the request, which avoids
        // brittle client-side guesses about whether the final cacheable prefix
        // ends in `system`, `tools`, or `messages`.
        if !self.is_bedrock() {
            if let Some(cache_control) = self
                .config
                .caching
                .as_ref()
                .and_then(|caching| caching.top_level_cache_control())
            {
                body["cache_control"] = serde_json::to_value(cache_control)?;
            }
        }

        // Add tools if provided
        if let Some(anthropic_tools) = anthropic_tools {
            body["tools"] = serde_json::to_value(anthropic_tools)?;
        }

        // Add tool choice if configured
        if let Some(ref tool_choice) = self.config.tool_choice {
            body["tool_choice"] = super::convert::tool_choice_to_json(tool_choice)?;
        }

        // Add streaming - only for direct Anthropic API (Bedrock uses endpoint for streaming)
        if self.config.stream && !self.is_bedrock() {
            body["stream"] = json!(true);
        }

        // Add optional parameters
        if let Some(temp) = self.config.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(top_p) = self.config.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(top_k) = self.config.top_k {
            body["top_k"] = json!(top_k);
        }
        if !self.config.stop_sequences.is_empty() {
            body["stop_sequences"] = json!(self.config.stop_sequences);
        }

        // Add extended thinking (adaptive for Opus 4.6+, legacy for older models)
        if let Some(ref thinking) = self.config.thinking {
            if thinking.adaptive {
                // Adaptive thinking: Claude dynamically decides when/how much to think.
                // Recommended for Opus 4.6+. Automatically enables interleaved thinking.
                body["thinking"] = json!({"type": "adaptive"});
            } else if thinking.enabled {
                // Legacy fixed-budget thinking for pre-Opus 4.6 models.
                body["thinking"] = json!({
                    "type": "enabled",
                    "budget_tokens": thinking.budget_tokens
                });
            }
        }

        // Add effort level via output_config (Opus 4.5+, GA on Opus 4.6)
        if let Some(ref effort) = self.config.effort {
            body["output_config"] = json!({"effort": effort.as_str()});
        }

        // Add Bedrock beta features in the request body (not HTTP headers)
        if self.is_bedrock() && self.config.beta_features.has_any() {
            let beta_values = self.config.beta_features.to_header_values();
            body["anthropic_beta"] = json!(beta_values);
        }

        // Add metadata
        if let Some(ref metadata) = self.config.metadata {
            body["metadata"] = json!(metadata);
        }

        Ok(body)
    }

    /// Apply Bedrock prompt caching checkpoints to the supported Anthropic
    /// request fields present in the request.
    ///
    /// Bedrock's Anthropic InvokeModel integration expects explicit
    /// `cache_control` objects inside supported `system`, `messages`, and
    /// `tools` fields rather than Anthropic's request-level cache helper. Appam
    /// therefore injects up to one checkpoint at the end of each supported
    /// section when automatic caching is enabled.
    ///
    /// This keeps the public `CachingConfig` transport-agnostic while still
    /// preserving message-level caching for workloads where the repeated prefix
    /// lives in user content instead of in `system` or `tools`.
    fn apply_bedrock_prompt_caching(
        system_prompt: &mut Option<SystemPrompt>,
        conversation_messages: &mut [Message],
        tools: Option<&mut [Tool]>,
        cache_control: CacheControl,
    ) {
        Self::apply_cache_control_to_system_prompt(system_prompt, cache_control.clone());
        Self::apply_cache_control_to_messages(conversation_messages, cache_control.clone());

        if let Some(tools) = tools {
            if let Some(last_tool) = tools.last_mut() {
                last_tool.cache_control = Some(cache_control);
            }
        }
    }

    /// Apply a cache checkpoint to the end of the system prompt section.
    fn apply_cache_control_to_system_prompt(
        system_prompt: &mut Option<SystemPrompt>,
        cache_control: CacheControl,
    ) {
        match system_prompt {
            Some(SystemPrompt::String(text)) => {
                let text = text.clone();
                *system_prompt = Some(SystemPrompt::Blocks(vec![SystemBlock {
                    block_type: "text".to_string(),
                    text,
                    cache_control: Some(cache_control),
                }]));
            }
            Some(SystemPrompt::Blocks(blocks)) => {
                if let Some(last_block) = blocks.last_mut() {
                    last_block.cache_control = Some(cache_control);
                }
            }
            None => {}
        }
    }

    /// Apply a cache checkpoint to the last cacheable message content block.
    fn apply_cache_control_to_messages(
        conversation_messages: &mut [Message],
        cache_control: CacheControl,
    ) {
        for message in conversation_messages.iter_mut().rev() {
            for block in message.content.iter_mut().rev() {
                if Self::set_cache_control_on_content_block(block, cache_control.clone()) {
                    return;
                }
            }
        }
    }

    /// Attach cache control to a content block when the block type supports it.
    fn set_cache_control_on_content_block(
        block: &mut ContentBlock,
        cache_control: CacheControl,
    ) -> bool {
        match block {
            ContentBlock::Text {
                cache_control: slot,
                ..
            }
            | ContentBlock::Image {
                cache_control: slot,
                ..
            }
            | ContentBlock::Document {
                cache_control: slot,
                ..
            }
            | ContentBlock::ToolUse {
                cache_control: slot,
                ..
            }
            | ContentBlock::ToolResult {
                cache_control: slot,
                ..
            } => {
                *slot = Some(cache_control);
                true
            }
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => false,
        }
    }

    /// Initialize the global rate limiter if configured and not already initialized.
    ///
    /// The rate limiter is global (shared across all Anthropic clients) and
    /// initialized once on first use. If already initialized, this is a no-op.
    fn ensure_rate_limiter_initialized(&self) {
        if let Some(config) = &self.config.rate_limiter {
            if config.enabled {
                let mut limiter = GLOBAL_RATE_LIMITER.lock().unwrap();
                if limiter.is_none() {
                    info!(
                        tokens_per_minute = config.tokens_per_minute,
                        "Initializing global rate limiter for Anthropic API"
                    );
                    *limiter = Some(RateLimiter::new(config.tokens_per_minute));
                }
            }
        }
    }

    /// Acquire a request slot from the global rate limiter.
    ///
    /// If rate limiting is enabled, this checks the sliding window and blocks
    /// if we're at/over the threshold. Does NOT deduct tokens (that happens
    /// in `record_rate_limit_usage` after getting the actual response).
    async fn acquire_rate_limit_slot(&self) {
        if let Some(config) = &self.config.rate_limiter {
            if config.enabled {
                // Get a clone of the limiter in a limited scope to release the lock immediately
                let limiter_opt = {
                    let limiter_guard = GLOBAL_RATE_LIMITER.lock().unwrap();
                    limiter_guard.clone()
                    // MutexGuard is dropped here automatically
                };

                if let Some(limiter) = limiter_opt {
                    debug!("Acquiring rate limiter slot");

                    // This blocks if we're at threshold
                    // Lock is already released, so this is safe across await
                    limiter.acquire_slot().await;
                }
            }
        }
    }

    /// Record actual token usage in the global rate limiter.
    ///
    /// Called after receiving a response with actual token counts.
    /// This updates the sliding window with real consumption data.
    async fn record_rate_limit_usage(&self, input_tokens: u32, output_tokens: u32) {
        if let Some(config) = &self.config.rate_limiter {
            if config.enabled {
                let limiter_opt = {
                    let limiter_guard = GLOBAL_RATE_LIMITER.lock().unwrap();
                    limiter_guard.clone()
                };

                if let Some(limiter) = limiter_opt {
                    let total_tokens = input_tokens + output_tokens;
                    limiter.record_usage(total_tokens).await;
                }
            }
        }
    }

    /// Extract retry-after duration from response headers.
    ///
    /// Checks for the `retry-after` header and parses it as seconds.
    /// Anthropic returns this header in rate limit (429) responses.
    ///
    /// # Arguments
    ///
    /// * `response` - HTTP response to extract headers from
    ///
    /// # Returns
    ///
    /// Optional retry-after duration in seconds.
    fn extract_retry_after(response: &reqwest::Response) -> Option<u64> {
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
    }

    /// Calculate wait duration for retry with exponential backoff.
    ///
    /// Uses the retry configuration to calculate backoff, with special handling
    /// for retry-after headers that override the calculated backoff.
    ///
    /// # Arguments
    ///
    /// * `attempt` - Current retry attempt number (1-indexed)
    /// * `retry_after_secs` - Optional retry-after from API response
    ///
    /// # Returns
    ///
    /// Duration to wait before retrying, respecting max_backoff cap.
    fn calculate_retry_wait(&self, attempt: u32, retry_after_secs: Option<u64>) -> Duration {
        let retry_config = self.config.retry.as_ref().expect("Retry config required");

        let wait_ms = if let Some(retry_after) = retry_after_secs {
            // Use retry-after from header, but respect max_backoff
            let retry_after_ms = retry_after * 1000;
            debug!(
                retry_after_secs = retry_after,
                "Using retry-after from API response header"
            );
            retry_after_ms.min(retry_config.max_backoff_ms)
        } else {
            // Calculate exponential backoff with jitter
            retry_config.calculate_backoff(attempt)
        };

        Duration::from_millis(wait_ms)
    }

    /// Calculate retry wait duration for network errors.
    ///
    /// Uses exponential backoff with optional jitter for network-level failures
    /// (connection timeouts, DNS errors). Separate from API retry logic to allow
    /// faster failure detection while maintaining reliability.
    ///
    /// # Arguments
    ///
    /// * `attempt` - Current retry attempt number (1-indexed)
    ///
    /// # Returns
    ///
    /// Duration to wait before retrying network request.
    fn calculate_network_retry_wait(&self, attempt: u32) -> Duration {
        let network_retry_config = self
            .config
            .network_retry
            .as_ref()
            .expect("Network retry config required");

        let wait_ms = network_retry_config.calculate_backoff(attempt);
        Duration::from_millis(wait_ms)
    }

    /// Check if an error is a retryable network error.
    ///
    /// Returns true for transient network failures that may succeed on retry:
    /// - Connection errors (TCP connection failed, refused, etc.)
    /// - Timeout errors (connection establishment timed out)
    /// - DNS resolution failures
    ///
    /// # Arguments
    ///
    /// * `error` - The reqwest error to check
    ///
    /// # Returns
    ///
    /// True if the error is retryable, false otherwise.
    fn is_network_error_retryable(error: &reqwest::Error) -> bool {
        // Check standard reqwest error categories
        if error.is_connect() || error.is_timeout() {
            return true;
        }

        Self::is_network_error_message_retryable(&error.to_string())
    }

    /// Check whether a lower-level network error message indicates a retryable transport failure.
    fn is_network_error_message_retryable(error_message: &str) -> bool {
        let error_msg = error_message.to_ascii_lowercase();
        error_msg.contains("connection reset")
            || error_msg.contains("broken pipe")
            || error_msg.contains("connection closed")
            || error_msg.contains("sendrequest")
            || error_msg.contains("dns error")
            || error_msg.contains("failed to lookup address")
            || error_msg.contains("nodename nor servname provided")
    }

    /// Check if an HTTP status code indicates a retryable error.
    ///
    /// Returns true for transient infrastructure errors that should be retried:
    /// - 502 Bad Gateway
    /// - 503 Service Unavailable
    /// - 504 Gateway Timeout
    ///
    /// These errors typically indicate temporary infrastructure issues that
    /// may succeed on retry.
    fn is_status_code_retryable(status: reqwest::StatusCode) -> bool {
        matches!(status.as_u16(), 502..=504)
    }

    /// Check if a stream chunk reading error is retryable.
    ///
    /// Returns true for transient network errors during stream reading:
    /// - EOF errors (connection closed unexpectedly)
    /// - Connection resets
    /// - Incomplete chunk reads
    /// - DNS resolution failures
    ///
    /// These errors indicate the connection was interrupted during streaming,
    /// but a retry may succeed with a fresh connection.
    ///
    /// # Arguments
    ///
    /// * `error` - The error that occurred while reading a stream chunk
    ///
    /// # Returns
    ///
    /// True if the error should trigger a retry, false otherwise.
    fn is_chunk_error_retryable(error: &anyhow::Error) -> bool {
        // Convert to string for pattern matching on error messages
        let error_str = format!("{:#}", error);
        let error_str_lower = error_str.to_lowercase();

        // Check for known retryable patterns in reqwest/hyper errors
        error_str_lower.contains("unexpected eof")
            || error_str_lower.contains("connection reset")
            || error_str_lower.contains("broken pipe")
            || error_str_lower.contains("connection closed")
            || error_str_lower.contains("incomplete")
            || error_str_lower.contains("chunk size")
            || error_str_lower.contains("dns error")
            || error_str_lower.contains("failed to lookup address")
            || error_str_lower.contains("nodename nor servname provided")
    }

    /// Parse SSE stream from Anthropic API.
    ///
    /// Handles all event types, accumulates content blocks, and invokes
    /// callbacks as content arrives.
    ///
    /// # Event Processing
    ///
    /// - `message_start`: Initialize response
    /// - `content_block_start`: Begin tracking new block
    /// - `content_block_delta`: Accumulate deltas, emit callbacks
    /// - `content_block_stop`: Finalize block, emit tool calls if applicable
    /// - `message_delta`: Update stop reason
    /// - `message_stop`: Complete
    ///
    /// # Callbacks
    ///
    /// - `on_content`: Called for each text delta
    /// - `on_reasoning`: Called for each thinking delta (text only, for display)
    /// - `on_tool_calls_partial`: Called as tool arguments accumulate
    /// - `on_tool_calls`: Called when tool calls are finalized
    /// - `on_content_block_complete`: Called for complete blocks (preserves thinking signatures)
    ///
    /// # Returns
    ///
    /// Returns (input_tokens, output_tokens) for rate limiter tracking.
    #[allow(clippy::too_many_arguments)]
    async fn parse_sse_stream<FContent, FTool, FReason, FToolPartial, FContentBlock, FUsage>(
        &self,
        response: reqwest::Response,
        on_content: &mut FContent,
        on_tool_calls: &mut FTool,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
        on_content_block_complete: &mut FContentBlock,
        on_usage: &mut FUsage,
    ) -> Result<(u32, u32)>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(crate::llm::unified::UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        // Variables to track token usage across the stream
        let mut total_input_tokens = 0u32;
        let mut total_output_tokens = 0u32;

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut event_lines: Vec<String> = Vec::new();

        // Track content blocks by index
        let mut blocks: std::collections::HashMap<usize, AccumulatedBlock> =
            std::collections::HashMap::new();

        // Track final tool calls
        let mut finalized_tool_calls: Vec<UnifiedToolCall> = Vec::new();

        let debug_enabled = std::env::var("ANTHROPIC_DEBUG")
            .ok()
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        while let Some(chunk) = stream.next().await {
            // Handle chunk reading with retry detection
            let bytes = match chunk.context("Failed to read stream chunk") {
                Ok(b) => b,
                Err(e) => {
                    // Check if this is a retryable network error during chunk reading
                    if Self::is_chunk_error_retryable(&e) {
                        // Return retryable error so outer loop can retry the entire request
                        return Err(RetryableStreamError::NetworkError {
                            message: format!("{:#}", e),
                        }
                        .into());
                    }
                    // Non-retryable error - propagate immediately
                    return Err(e);
                }
            };
            let part = String::from_utf8_lossy(&bytes);
            buffer.push_str(&part);

            // Process complete lines
            while let Some(line_end) = buffer.find('\n') {
                let mut line = buffer[..line_end].to_string();
                buffer = buffer[line_end + 1..].to_string();

                // Trim \r if present
                if line.ends_with('\r') {
                    line.pop();
                }

                if debug_enabled {
                    eprintln!("[anthropic::sse] {}", line);
                }

                // Empty line = end of event
                if line.is_empty() {
                    if !event_lines.is_empty() {
                        let data_payload = event_lines.join("\n");
                        event_lines.clear();

                        // Parse and handle event
                        if let Err(e) = self.handle_event(
                            &data_payload,
                            &mut blocks,
                            &mut finalized_tool_calls,
                            &mut total_input_tokens,
                            &mut total_output_tokens,
                            on_content,
                            on_tool_calls,
                            on_reasoning,
                            on_tool_calls_partial,
                            on_content_block_complete,
                            on_usage,
                            debug_enabled,
                        ) {
                            if debug_enabled {
                                eprintln!("[anthropic::error] {}", e);
                            }
                            error!(error = %e, "Failed to handle event");
                        }
                    }
                } else if let Some(data) = line.strip_prefix("data: ") {
                    event_lines.push(data.to_string());
                } else if let Some(event_type) = line.strip_prefix("event: ") {
                    // Event type line (optional, type is in data JSON)
                    debug!(event = %event_type, "Event type");
                }
                // Ignore comment lines starting with ':'
            }
        }

        // Return token usage (extracted from MessageDelta events handled earlier)
        Ok((total_input_tokens, total_output_tokens))
    }

    /// Parse a Bedrock EventStream binary response into Anthropic events.
    ///
    /// AWS Bedrock's `invoke-with-response-stream` endpoint returns responses
    /// wrapped in Amazon EventStream binary framing, not raw SSE text. Each
    /// binary frame contains:
    ///
    /// - Prelude: total length + headers length + CRC32
    /// - Headers: `:event-type`, `:content-type`, `:message-type`
    /// - Payload: JSON `{"bytes": "<base64-encoded-anthropic-event-json>"}`
    /// - Message CRC32
    ///
    /// This method decodes the binary frames using `MessageFrameDecoder`,
    /// extracts and base64-decodes the Anthropic event JSON from each frame,
    /// and feeds it into the same `handle_event()` pipeline used by the
    /// direct Anthropic SSE parser.
    ///
    /// # Error Frames
    ///
    /// Frames with `:message-type: exception` indicate Bedrock-level errors
    /// (throttling, validation, internal errors). These are extracted and
    /// returned as errors for the retry logic to handle.
    ///
    /// # Arguments
    ///
    /// Same callback signature as `parse_sse_stream`.
    ///
    /// # Returns
    ///
    /// `(input_tokens, output_tokens)` from the streamed response.
    #[allow(clippy::too_many_arguments)]
    async fn parse_bedrock_eventstream<
        FContent,
        FTool,
        FReason,
        FToolPartial,
        FContentBlock,
        FUsage,
    >(
        &self,
        response: reqwest::Response,
        on_content: &mut FContent,
        on_tool_calls: &mut FTool,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
        on_content_block_complete: &mut FContentBlock,
        on_usage: &mut FUsage,
    ) -> Result<(u32, u32)>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(crate::llm::unified::UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        let mut total_input_tokens = 0u32;
        let mut total_output_tokens = 0u32;

        let mut stream = response.bytes_stream();
        let mut buffer = bytes::BytesMut::new();
        let mut decoder = MessageFrameDecoder::new();

        // Reuse the same block tracking as SSE parser
        let mut blocks: std::collections::HashMap<usize, AccumulatedBlock> =
            std::collections::HashMap::new();
        let mut finalized_tool_calls: Vec<UnifiedToolCall> = Vec::new();

        let debug_enabled = std::env::var("ANTHROPIC_DEBUG")
            .ok()
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        while let Some(chunk) = stream.next().await {
            let bytes = match chunk.context("Failed to read Bedrock EventStream chunk") {
                Ok(b) => b,
                Err(e) => {
                    if Self::is_chunk_error_retryable(&e) {
                        return Err(RetryableStreamError::NetworkError {
                            message: format!("{:#}", e),
                        }
                        .into());
                    }
                    return Err(e);
                }
            };

            buffer.extend_from_slice(&bytes);

            // Decode as many complete frames as possible from the buffer
            loop {
                match decoder.decode_frame(&mut buffer) {
                    Ok(DecodedFrame::Complete(message)) => {
                        // Extract message type from headers
                        let mut message_type = None;
                        let mut event_type = None;

                        for header in message.headers() {
                            let name = header.name().as_str();
                            if name == ":message-type" {
                                if let Ok(val) = header.value().as_string() {
                                    message_type = Some(val.as_str().to_string());
                                }
                            } else if name == ":event-type" {
                                if let Ok(val) = header.value().as_string() {
                                    event_type = Some(val.as_str().to_string());
                                }
                            }
                        }

                        if debug_enabled {
                            eprintln!(
                                "[bedrock::eventstream] message_type={:?}, event_type={:?}, payload_len={}",
                                message_type,
                                event_type,
                                message.payload().len()
                            );
                        }

                        // Handle exception frames (Bedrock-level errors)
                        if message_type.as_deref() == Some("exception") {
                            let error_payload =
                                String::from_utf8_lossy(message.payload()).to_string();
                            error!(
                                event_type = ?event_type,
                                payload = %error_payload,
                                "Bedrock EventStream exception"
                            );

                            // Check for retryable Bedrock exceptions
                            let is_retryable = matches!(
                                event_type.as_deref(),
                                Some("throttlingException")
                                    | Some("serviceUnavailableException")
                                    | Some("internalServerException")
                            );

                            if is_retryable {
                                return Err(RetryableStreamError::NetworkError {
                                    message: format!(
                                        "Bedrock exception ({}): {}",
                                        event_type.unwrap_or_default(),
                                        error_payload
                                    ),
                                }
                                .into());
                            }

                            return Err(anyhow!(
                                "Bedrock EventStream exception ({}): {}",
                                event_type.unwrap_or_default(),
                                error_payload
                            ));
                        }

                        // Only process "chunk" event frames
                        if event_type.as_deref() != Some("chunk") {
                            if debug_enabled {
                                eprintln!(
                                    "[bedrock::eventstream] Skipping non-chunk event: {:?}",
                                    event_type
                                );
                            }
                            continue;
                        }

                        // Parse payload JSON: {"bytes": "<base64-encoded-data>"}
                        let payload_json: serde_json::Value =
                            serde_json::from_slice(message.payload())
                                .context("Failed to parse Bedrock chunk payload as JSON")?;

                        let b64_data = payload_json["bytes"]
                            .as_str()
                            .ok_or_else(|| anyhow!("Bedrock chunk missing 'bytes' field"))?;

                        // Base64-decode to get the Anthropic event JSON
                        let decoded_bytes = base64::Engine::decode(
                            &base64::engine::general_purpose::STANDARD,
                            b64_data,
                        )
                        .context("Failed to base64-decode Bedrock chunk bytes")?;

                        let event_json = String::from_utf8(decoded_bytes)
                            .context("Bedrock decoded bytes are not valid UTF-8")?;

                        if debug_enabled {
                            eprintln!("[bedrock::eventstream] decoded event: {}", event_json);
                        }

                        // Feed into the same handle_event pipeline used by SSE parser
                        if let Err(e) = self.handle_event(
                            &event_json,
                            &mut blocks,
                            &mut finalized_tool_calls,
                            &mut total_input_tokens,
                            &mut total_output_tokens,
                            on_content,
                            on_tool_calls,
                            on_reasoning,
                            on_tool_calls_partial,
                            on_content_block_complete,
                            on_usage,
                            debug_enabled,
                        ) {
                            if debug_enabled {
                                eprintln!("[bedrock::error] {}", e);
                            }
                            error!(error = %e, "Failed to handle Bedrock event");
                        }
                    }
                    Ok(DecodedFrame::Incomplete) => {
                        // Need more data to decode the next frame
                        break;
                    }
                    Err(e) => {
                        error!(error = %e, "Failed to decode Bedrock EventStream frame");
                        return Err(anyhow!("Bedrock EventStream frame decode error: {}", e));
                    }
                }
            }
        }

        Ok((total_input_tokens, total_output_tokens))
    }

    /// Handle a single SSE event.
    #[allow(clippy::too_many_arguments)]
    fn handle_event<FContent, FTool, FReason, FToolPartial, FContentBlock, FUsage>(
        &self,
        data: &str,
        blocks: &mut std::collections::HashMap<usize, AccumulatedBlock>,
        finalized_tool_calls: &mut Vec<UnifiedToolCall>,
        total_input_tokens: &mut u32,
        total_output_tokens: &mut u32,
        on_content: &mut FContent,
        on_tool_calls: &mut FTool,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
        on_content_block_complete: &mut FContentBlock,
        on_usage: &mut FUsage,
        debug_enabled: bool,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()>,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()>,
        FReason: FnMut(&str) -> Result<()>,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()>,
        FContentBlock: FnMut(crate::llm::unified::UnifiedContentBlock) -> Result<()>,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()>,
    {
        // Try to parse as StreamEvent
        let event: StreamEvent = serde_json::from_str(data).context("Failed to parse SSE event")?;

        if debug_enabled {
            match &event {
                StreamEvent::MessageStart { .. } => eprintln!("[anthropic::event] message_start"),
                StreamEvent::ContentBlockStart {
                    index,
                    content_block,
                } => {
                    eprintln!(
                        "[anthropic::event] content_block_start: index={}, type={:?}",
                        index, content_block
                    );
                }
                StreamEvent::ContentBlockDelta { index, delta } => {
                    eprintln!(
                        "[anthropic::event] content_block_delta: index={}, delta={:?}",
                        index, delta
                    );
                }
                StreamEvent::ContentBlockStop { index } => {
                    eprintln!("[anthropic::event] content_block_stop: index={}", index);
                }
                StreamEvent::MessageDelta { .. } => eprintln!("[anthropic::event] message_delta"),
                StreamEvent::MessageStop => eprintln!("[anthropic::event] message_stop"),
                StreamEvent::Ping => eprintln!("[anthropic::event] ping"),
                StreamEvent::Error { .. } => eprintln!("[anthropic::event] error"),
            }
        }

        match event {
            StreamEvent::MessageStart { message } => {
                debug!(id = %message.id, model = %message.model, "Message started");

                // Initialize token tracking from message start
                *total_input_tokens = message.usage.input_tokens;
                *total_output_tokens = message.usage.output_tokens;
            }

            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let block = AccumulatedBlock::from_start(&content_block);
                blocks.insert(index, block);
            }

            StreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(block) = blocks.get_mut(&index) {
                    // Apply delta
                    block.apply_delta(&delta);

                    // Emit callbacks based on delta type
                    match &delta {
                        super::streaming::Delta::TextDelta { text } => {
                            on_content(text)?;
                        }
                        super::streaming::Delta::ThinkingDelta { thinking } => {
                            on_reasoning(thinking)?;
                        }
                        super::streaming::Delta::InputJsonDelta { partial_json: _ } => {
                            // Build partial tool call for progress updates
                            if let (Some(id), Some(name)) = (&block.tool_id, &block.tool_name) {
                                let parsed_input =
                                    serde_json::from_str::<serde_json::Value>(&block.partial_json)
                                        .unwrap_or(serde_json::Value::Null);
                                let partial_call = UnifiedToolCall {
                                    id: id.clone(),
                                    name: name.clone(),
                                    input: parsed_input,
                                    raw_input_json: Some(block.partial_json.clone()),
                                };
                                on_tool_calls_partial(&[partial_call])?;
                            }
                        }
                        super::streaming::Delta::SignatureDelta { .. } => {
                            // Signature captured, no callback needed
                        }
                    }
                }
            }

            StreamEvent::ContentBlockStop { index } => {
                // Finalize the block
                if let Some(block) = blocks.get(&index) {
                    match block.to_content_block() {
                        Ok(ContentBlock::ToolUse {
                            id, name, input, ..
                        }) => {
                            // Tool call complete - add to finalized list
                            let raw_input_json = Some(input.to_string());
                            finalized_tool_calls.push(UnifiedToolCall {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                                raw_input_json,
                            });
                            // Also emit as complete block for message reconstruction
                            let unified_block = crate::llm::unified::UnifiedContentBlock::ToolUse {
                                id,
                                name,
                                input,
                            };
                            on_content_block_complete(unified_block)?;
                        }
                        Ok(ContentBlock::Thinking {
                            thinking,
                            signature,
                        }) => {
                            // Thinking block complete - emit with signature preserved
                            let unified_block =
                                crate::llm::unified::UnifiedContentBlock::Thinking {
                                    thinking,
                                    signature: Some(signature),
                                    encrypted_content: None,
                                    redacted: false,
                                };
                            on_content_block_complete(unified_block)?;
                        }
                        Ok(ContentBlock::RedactedThinking { data }) => {
                            // Redacted thinking block - emit as redacted
                            let unified_block =
                                crate::llm::unified::UnifiedContentBlock::Thinking {
                                    thinking: data,
                                    signature: None,
                                    encrypted_content: None,
                                    redacted: true,
                                };
                            on_content_block_complete(unified_block)?;
                        }
                        Ok(ContentBlock::Text { text, .. }) => {
                            // Text block complete - emit for message reconstruction
                            let unified_block =
                                crate::llm::unified::UnifiedContentBlock::Text { text };
                            on_content_block_complete(unified_block)?;
                        }
                        Ok(_) => {
                            // Other block types (images, documents) - emit if needed
                            // For now, skip as they're less common
                        }
                        Err(e) => {
                            // Malformed JSON in tool call - skip and continue processing.
                            // This can happen when streaming is interrupted or API returns incomplete JSON.
                            // The tool call is not added to finalized_tool_calls, so it won't be executed.
                            // The model can observe this and retry or continue without the tool result.
                            if let Some(block) = blocks.get(&index) {
                                error!(
                                    error = %e,
                                    index = index,
                                    block_type = ?block.block_type,
                                    tool_name = ?block.tool_name,
                                    partial_json = %block.partial_json,
                                    "Failed to finalize content block - malformed JSON in tool call"
                                );
                            } else {
                                error!(error = %e, index = index, "Failed to finalize content block");
                            }
                        }
                    }
                }
            }

            StreamEvent::MessageDelta { delta, usage } => {
                debug!(
                    stop_reason = ?delta.stop_reason,
                    usage = ?usage,
                    "Message delta"
                );

                // Track actual token usage for rate limiter
                *total_input_tokens = usage.input_tokens;
                *total_output_tokens = usage.output_tokens;

                // Convert Anthropic usage to UnifiedUsage and call callback
                let unified_usage = crate::llm::unified::UnifiedUsage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cache_creation_input_tokens: usage.cache_creation_input_tokens,
                    cache_read_input_tokens: usage.cache_read_input_tokens,
                    reasoning_tokens: None, // Anthropic doesn't separate reasoning tokens
                };
                on_usage(unified_usage)?;
            }

            StreamEvent::MessageStop => {
                // Emit finalized tool calls
                if !finalized_tool_calls.is_empty() {
                    let calls = std::mem::take(finalized_tool_calls);
                    on_tool_calls(calls)?;
                }
                debug!("Message stopped");
            }

            StreamEvent::Ping => {
                // Keepalive, no action needed
            }

            StreamEvent::Error { error } => {
                error!(
                    error_type = %error.error_type,
                    message = %error.message,
                    retryable = error.is_retryable(),
                    "API error in stream"
                );

                // If this is a retryable error (rate_limit_error, overloaded_error),
                // return a custom error type that the outer retry loop can catch
                if error.is_retryable() {
                    return Err(RetryableStreamError::ApiError { error_data: error }.into());
                }

                // Non-retryable error - fail immediately
                return Err(anyhow!(
                    "Anthropic API error ({}): {}",
                    error.error_type,
                    error.message
                ));
            }
        }

        Ok(())
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn chat_with_tools_streaming<
        FContent,
        FTool,
        FReason,
        FToolPartial,
        FContentBlock,
        FUsage,
    >(
        &self,
        messages: &[UnifiedMessage],
        tools: &[UnifiedTool],
        mut on_content: FContent,
        mut on_tool_calls: FTool,
        mut on_reasoning: FReason,
        mut on_tool_calls_partial: FToolPartial,
        mut on_content_block_complete: FContentBlock,
        mut on_usage: FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(crate::llm::unified::UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        self.clear_last_failed_exchange();

        // Initialize global rate limiter if configured
        self.ensure_rate_limiter_initialized();

        // Build request body once (doesn't change across retries)
        let body = self.build_request_body(messages, tools)?;
        let request_payload = serde_json::to_string_pretty(&body)?;

        // Serialize body to bytes for SigV4 signing (needs to be done once, reused)
        let body_bytes = serde_json::to_vec(&body).context("Failed to serialize request body")?;

        // Log request info - use Bedrock model_id if configured, otherwise standard model
        let model_name = if let Some(ref bedrock) = self.config.bedrock {
            &bedrock.model_id
        } else {
            &self.config.model
        };
        let provider = if self.is_bedrock() {
            if self.uses_sigv4() {
                "AWS Bedrock (SigV4)"
            } else {
                "AWS Bedrock (Bearer)"
            }
        } else if let Some(ref azure) = self.config.azure {
            match azure.auth_method {
                AzureAnthropicAuthMethod::XApiKey => "Azure Anthropic (x-api-key)",
                AzureAnthropicAuthMethod::BearerToken => "Azure Anthropic (Bearer)",
            }
        } else {
            "Anthropic"
        };

        info!(
            model = %model_name,
            provider = provider,
            messages = messages.len(),
            tools = tools.len(),
            "Sending Claude Messages API request"
        );

        debug!(
            request = %serde_json::to_string_pretty(&body)?,
            "Request body (API key redacted)"
        );

        // Retry loop for API errors (rate limits, overload)
        let mut attempt = 0u32;
        let max_attempts = self
            .config
            .retry
            .as_ref()
            .map(|r| r.max_retries + 1)
            .unwrap_or(1);

        // Build endpoint URL once
        let endpoint_url = self.build_endpoint_url();

        loop {
            attempt += 1;

            // Acquire rate limiter slot before sending (blocks if at threshold)
            self.acquire_rate_limit_slot().await;

            // Network retry wrapper for connection/timeout errors
            let network_max_attempts = self
                .config
                .network_retry
                .as_ref()
                .map(|r| r.max_retries + 1)
                .unwrap_or(1);

            let response = {
                let mut network_attempt = 0u32;

                loop {
                    network_attempt += 1;

                    // Build base headers
                    let mut headers = self.build_headers()?;

                    // For SigV4, sign the request and add auth headers
                    // Note: SigV4 must be signed fresh each attempt due to timestamp
                    if self.uses_sigv4() {
                        let auth_headers =
                            self.sign_request_sigv4(&endpoint_url, &body_bytes, &headers)?;
                        for (name, value) in auth_headers.iter() {
                            headers.insert(name.clone(), value.clone());
                        }
                    }

                    // Attempt to send request
                    // For SigV4, we send raw bytes since the signature is computed over the exact bytes
                    let request_result = if self.uses_sigv4() {
                        self.http
                            .post(&endpoint_url)
                            .headers(headers)
                            .body(body_bytes.clone())
                            .send()
                            .await
                    } else {
                        self.http
                            .post(&endpoint_url)
                            .headers(headers)
                            .json(&body)
                            .send()
                            .await
                    };

                    match request_result {
                        Ok(resp) => {
                            // Success - break out of network retry loop
                            break resp;
                        }
                        Err(e) => {
                            // Check if error is a retryable network error
                            if Self::is_network_error_retryable(&e)
                                && self.config.network_retry.is_some()
                                && network_attempt < network_max_attempts
                            {
                                // Log network error and retry
                                let wait_duration =
                                    self.calculate_network_retry_wait(network_attempt);

                                warn!(
                                    provider = %provider,
                                    attempt = network_attempt,
                                    max_attempts = network_max_attempts,
                                    error = %e,
                                    wait_secs = wait_duration.as_secs_f64(),
                                    "Network error, retrying after backoff"
                                );

                                tokio::time::sleep(wait_duration).await;
                                continue;
                            } else {
                                // Non-retryable network error or max attempts exceeded
                                if network_attempt >= network_max_attempts
                                    && Self::is_network_error_retryable(&e)
                                {
                                    warn!(
                                        network_max_attempts = network_max_attempts,
                                        "Max network retry attempts exceeded"
                                    );
                                }

                                // Propagate error with context
                                self.record_failed_exchange(None, &request_payload, e.to_string());
                                return Err(e).context(format!(
                                    "Failed to send request to {} API",
                                    provider
                                ));
                            }
                        }
                    }
                }
            };

            // Check status
            let status = response.status();
            if !status.is_success() {
                // Extract retry-after header before consuming response body
                let retry_after = Self::extract_retry_after(&response);

                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());

                // Try to parse as structured error
                if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&error_text) {
                    error!(
                        error_type = %error_response.error.error_type,
                        message = %error_response.error.message,
                        attempt = attempt,
                        "Anthropic API error"
                    );

                    // Check if error is retryable and we have retries left
                    if error_response.error.is_retryable()
                        && self.config.retry.is_some()
                        && attempt < max_attempts
                    {
                        // Calculate wait duration
                        let wait_duration = self.calculate_retry_wait(
                            attempt,
                            retry_after.or(error_response.error.retry_after.map(|s| s as u64)),
                        );

                        warn!(
                            provider = %provider,
                            attempt = attempt,
                            max_attempts = max_attempts,
                            wait_secs = wait_duration.as_secs_f64(),
                            error_type = %error_response.error.error_type,
                            error_message = %error_response.error.message,
                            "Rate limit or overload error, retrying after backoff"
                        );

                        // Wait before retry
                        tokio::time::sleep(wait_duration).await;

                        // Retry
                        continue;
                    } else {
                        // Non-retryable error or max retries exceeded
                        if attempt >= max_attempts && error_response.error.is_retryable() {
                            warn!(
                                max_attempts = max_attempts,
                                "Max retry attempts exceeded, failing request"
                            );
                        }

                        self.record_failed_exchange(Some(status), &request_payload, &error_text);
                        return Err(anyhow!(
                            "Anthropic API error ({}): {}",
                            error_response.error.error_type,
                            error_response.error.message
                        ));
                    }
                } else {
                    // Failed to parse as structured error - check if HTTP status code is retryable
                    error!(
                        status = %status,
                        error_text = %error_text,
                        attempt = attempt,
                        "Anthropic API error (non-structured response)"
                    );

                    // Check if status code indicates a retryable error (502, 503, 504)
                    if Self::is_status_code_retryable(status)
                        && self.config.retry.is_some()
                        && attempt < max_attempts
                    {
                        // Calculate wait duration
                        let wait_duration = self.calculate_retry_wait(attempt, retry_after);

                        warn!(
                            provider = %provider,
                            attempt = attempt,
                            max_attempts = max_attempts,
                            wait_secs = wait_duration.as_secs_f64(),
                            status = %status,
                            "Transient HTTP error, retrying after backoff"
                        );

                        // Wait before retry
                        tokio::time::sleep(wait_duration).await;

                        // Retry
                        continue;
                    } else {
                        // Non-retryable status code or max retries exceeded
                        if attempt >= max_attempts && Self::is_status_code_retryable(status) {
                            warn!(
                                max_attempts = max_attempts,
                                "Max retry attempts exceeded for transient HTTP error"
                            );
                        }

                        self.record_failed_exchange(Some(status), &request_payload, &error_text);
                        return Err(anyhow!("Anthropic API error ({}): {}", status, error_text));
                    }
                }
            }

            // Parse the streaming response. Bedrock uses EventStream binary framing,
            // while the direct Anthropic API uses Server-Sent Events (SSE) text format.
            let stream_result = if self.is_bedrock() && self.config.stream {
                self.parse_bedrock_eventstream(
                    response,
                    &mut on_content,
                    &mut on_tool_calls,
                    &mut on_reasoning,
                    &mut on_tool_calls_partial,
                    &mut on_content_block_complete,
                    &mut on_usage,
                )
                .await
            } else {
                self.parse_sse_stream(
                    response,
                    &mut on_content,
                    &mut on_tool_calls,
                    &mut on_reasoning,
                    &mut on_tool_calls_partial,
                    &mut on_content_block_complete,
                    &mut on_usage,
                )
                .await
            };

            // Check if this is a retryable stream error
            match stream_result {
                Ok((input_tokens, output_tokens)) => {
                    // Record actual token usage in rate limiter (based on API response)
                    if input_tokens > 0 || output_tokens > 0 {
                        self.record_rate_limit_usage(input_tokens, output_tokens)
                            .await;

                        debug!(
                            input_tokens = input_tokens,
                            output_tokens = output_tokens,
                            total_tokens = input_tokens + output_tokens,
                            "Recorded actual token usage in rate limiter"
                        );
                    }

                    return Ok(());
                }
                Err(e) => {
                    // Check if this is a retryable stream error
                    if let Some(retryable_error) = e.downcast_ref::<RetryableStreamError>() {
                        // Check if we have retries configured and attempts left
                        if self.config.retry.is_some() && attempt < max_attempts {
                            // Calculate wait duration (stream errors don't provide retry_after)
                            let wait_duration = self.calculate_retry_wait(attempt, None);

                            // Log based on error type
                            match retryable_error {
                                RetryableStreamError::ApiError { error_data } => {
                                    warn!(
                                        provider = %provider,
                                        attempt = attempt,
                                        max_attempts = max_attempts,
                                        wait_secs = wait_duration.as_secs_f64(),
                                        error_type = %error_data.error_type,
                                        "Retryable API stream error (overload/rate limit), retrying after backoff"
                                    );
                                }
                                RetryableStreamError::NetworkError { message } => {
                                    warn!(
                                        provider = %provider,
                                        attempt = attempt,
                                        max_attempts = max_attempts,
                                        wait_secs = wait_duration.as_secs_f64(),
                                        error = %message,
                                        "Retryable network stream error (connection interrupted), retrying after backoff"
                                    );
                                }
                            }

                            // Wait before retry
                            tokio::time::sleep(wait_duration).await;

                            // Retry - continue will make a new HTTP request and parse the stream again
                            continue;
                        } else {
                            // No retries configured or max attempts exceeded
                            if attempt >= max_attempts {
                                match retryable_error {
                                    RetryableStreamError::ApiError { error_data } => {
                                        warn!(
                                            max_attempts = max_attempts,
                                            error_type = %error_data.error_type,
                                            "Max retry attempts exceeded for API stream error"
                                        );
                                    }
                                    RetryableStreamError::NetworkError { message } => {
                                        warn!(
                                            max_attempts = max_attempts,
                                            error = %message,
                                            "Max retry attempts exceeded for network stream error"
                                        );
                                    }
                                }
                            }

                            // Return appropriate error message based on type
                            self.record_failed_exchange(None, &request_payload, format!("{:#}", e));
                            return Err(match retryable_error {
                                RetryableStreamError::ApiError { error_data } => anyhow!(
                                    "Anthropic API error ({}): {}",
                                    error_data.error_type,
                                    error_data.message
                                ),
                                RetryableStreamError::NetworkError { message } => {
                                    anyhow!("Network error during streaming: {}", message)
                                }
                            });
                        }
                    }

                    // Not a retryable stream error - propagate immediately
                    self.record_failed_exchange(None, &request_payload, format!("{:#}", e));
                    return Err(e);
                }
            }
        }
    }

    fn provider_name(&self) -> &str {
        if self.is_bedrock() {
            "bedrock"
        } else if self.is_azure() {
            "azure-anthropic"
        } else {
            "anthropic"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::anthropic::{CacheTTL, CachingConfig};
    use once_cell::sync::Lazy;
    use std::collections::HashMap;
    use std::sync::Mutex;

    static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn restore_env(name: &str, previous: Option<String>) {
        if let Some(value) = previous {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
    }

    #[test]
    fn test_client_creation() {
        let config = AnthropicConfig::default();
        let client = AnthropicClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_client_validation_fails() {
        let config = AnthropicConfig {
            thinking: Some(super::super::config::ThinkingConfig {
                enabled: true,
                budget_tokens: 10000,
                adaptive: false,
            }),
            max_tokens: 5000, // < budget_tokens
            ..Default::default()
        };

        let client = AnthropicClient::new(config);
        assert!(client.is_err());
    }

    #[test]
    fn test_is_status_code_retryable() {
        // Retryable status codes
        assert!(AnthropicClient::is_status_code_retryable(
            reqwest::StatusCode::BAD_GATEWAY
        ));
        assert!(AnthropicClient::is_status_code_retryable(
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(AnthropicClient::is_status_code_retryable(
            reqwest::StatusCode::GATEWAY_TIMEOUT
        ));

        // Non-retryable status codes
        assert!(!AnthropicClient::is_status_code_retryable(
            reqwest::StatusCode::BAD_REQUEST
        ));
        assert!(!AnthropicClient::is_status_code_retryable(
            reqwest::StatusCode::UNAUTHORIZED
        ));
        assert!(!AnthropicClient::is_status_code_retryable(
            reqwest::StatusCode::FORBIDDEN
        ));
        assert!(!AnthropicClient::is_status_code_retryable(
            reqwest::StatusCode::NOT_FOUND
        ));
        assert!(!AnthropicClient::is_status_code_retryable(
            reqwest::StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(!AnthropicClient::is_status_code_retryable(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        ));
    }

    #[test]
    fn test_is_chunk_error_retryable() {
        // Retryable chunk reading errors (network issues during streaming)
        let eof_error = anyhow!("error decoding response body: error reading a body from connection: unexpected EOF during chunk size line");
        assert!(AnthropicClient::is_chunk_error_retryable(&eof_error));

        let connection_reset = anyhow!("connection reset by peer");
        assert!(AnthropicClient::is_chunk_error_retryable(&connection_reset));

        let broken_pipe = anyhow!("broken pipe");
        assert!(AnthropicClient::is_chunk_error_retryable(&broken_pipe));

        let connection_closed = anyhow!("Connection closed before message completed");
        assert!(AnthropicClient::is_chunk_error_retryable(
            &connection_closed
        ));

        let incomplete = anyhow!("incomplete message");
        assert!(AnthropicClient::is_chunk_error_retryable(&incomplete));

        // Non-retryable errors (parsing, validation, etc.)
        let parse_error = anyhow!("failed to parse JSON");
        assert!(!AnthropicClient::is_chunk_error_retryable(&parse_error));

        let validation_error = anyhow!("invalid event type");
        assert!(!AnthropicClient::is_chunk_error_retryable(
            &validation_error
        ));

        let auth_error = anyhow!("authentication failed");
        assert!(!AnthropicClient::is_chunk_error_retryable(&auth_error));
    }

    #[test]
    fn test_is_network_error_message_retryable() {
        assert!(AnthropicClient::is_network_error_message_retryable(
            "client error (SendRequest): connection error: connection reset"
        ));
        assert!(AnthropicClient::is_network_error_message_retryable(
            "broken pipe while sending request body"
        ));
        assert!(AnthropicClient::is_network_error_message_retryable(
            "dns error: failed to lookup address information"
        ));
        assert!(!AnthropicClient::is_network_error_message_retryable(
            "authentication failed"
        ));
    }

    #[test]
    fn test_handle_event_usage_conversion() {
        let config = AnthropicConfig::default();
        let client = AnthropicClient::new(config).expect("client should initialize");

        let mut blocks = HashMap::new();
        let mut finalized_tool_calls = Vec::new();
        let mut total_input_tokens = 0u32;
        let mut total_output_tokens = 0u32;
        let mut captured_usage = Vec::new();

        let event = serde_json::json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_sequence": null
            },
            "usage": {
                "input_tokens": 321,
                "output_tokens": 654,
                "cache_creation_input_tokens": 42,
                "cache_read_input_tokens": 7
            }
        });

        client
            .handle_event(
                &event.to_string(),
                &mut blocks,
                &mut finalized_tool_calls,
                &mut total_input_tokens,
                &mut total_output_tokens,
                &mut |_text| Ok(()),
                &mut |_calls| Ok(()),
                &mut |_reason| Ok(()),
                &mut |_partial| Ok(()),
                &mut |_block| Ok(()),
                &mut |usage| {
                    captured_usage.push(usage);
                    Ok(())
                },
                false,
            )
            .expect("message_delta should be handled");

        assert_eq!(total_input_tokens, 321);
        assert_eq!(total_output_tokens, 654);

        let usage = captured_usage.first().expect("usage callback should fire");
        assert_eq!(usage.input_tokens, 321);
        assert_eq!(usage.output_tokens, 654);
        assert_eq!(usage.cache_creation_input_tokens, Some(42));
        assert_eq!(usage.cache_read_input_tokens, Some(7));
        assert!(usage.reasoning_tokens.is_none());
    }

    #[test]
    fn test_azure_endpoint_url_uses_normalized_base() {
        let client = AnthropicClient::new(AnthropicConfig {
            azure: Some(super::super::config::AzureAnthropicConfig {
                base_url: "https://example-resource.services.ai.azure.com/anthropic/v1/messages/"
                    .to_string(),
                auth_method: AzureAnthropicAuthMethod::XApiKey,
            }),
            ..Default::default()
        })
        .expect("azure client should initialize");

        assert_eq!(
            client.build_endpoint_url(),
            "https://example-resource.services.ai.azure.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn test_azure_headers_use_x_api_key_and_env_fallback() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous_key = std::env::var("AZURE_ANTHROPIC_API_KEY").ok();
        let previous_fallback = std::env::var("AZURE_API_KEY").ok();
        std::env::remove_var("AZURE_ANTHROPIC_API_KEY");
        std::env::set_var("AZURE_API_KEY", "azure-fallback-key");

        let client = AnthropicClient::new(AnthropicConfig {
            azure: Some(super::super::config::AzureAnthropicConfig {
                base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
                auth_method: AzureAnthropicAuthMethod::XApiKey,
            }),
            ..Default::default()
        })
        .expect("azure client should initialize");

        let headers = client.build_headers().expect("headers should build");
        assert_eq!(
            headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok()),
            Some("azure-fallback-key")
        );
        assert!(headers.get(AUTHORIZATION).is_none());
        assert_eq!(
            headers
                .get("anthropic-version")
                .and_then(|value| value.to_str().ok()),
            Some("2023-06-01")
        );

        restore_env("AZURE_ANTHROPIC_API_KEY", previous_key);
        restore_env("AZURE_API_KEY", previous_fallback);
    }

    #[test]
    fn test_azure_headers_use_bearer_token_and_env_fallback() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous_token = std::env::var("AZURE_ANTHROPIC_AUTH_TOKEN").ok();
        let previous_fallback = std::env::var("AZURE_API_KEY").ok();
        std::env::remove_var("AZURE_ANTHROPIC_AUTH_TOKEN");
        std::env::set_var("AZURE_API_KEY", "azure-bearer-fallback");

        let client = AnthropicClient::new(AnthropicConfig {
            azure: Some(super::super::config::AzureAnthropicConfig {
                base_url: "https://example-resource.openai.azure.com/anthropic".to_string(),
                auth_method: AzureAnthropicAuthMethod::BearerToken,
            }),
            ..Default::default()
        })
        .expect("azure client should initialize");

        let headers = client.build_headers().expect("headers should build");
        assert_eq!(
            headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer azure-bearer-fallback")
        );
        assert!(headers.get("x-api-key").is_none());

        restore_env("AZURE_ANTHROPIC_AUTH_TOKEN", previous_token);
        restore_env("AZURE_API_KEY", previous_fallback);
    }

    #[test]
    fn test_provider_name_for_azure_anthropic() {
        let client = AnthropicClient::new(AnthropicConfig {
            azure: Some(super::super::config::AzureAnthropicConfig {
                base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
                auth_method: AzureAnthropicAuthMethod::XApiKey,
            }),
            api_key: Some("test-key".to_string()),
            ..Default::default()
        })
        .expect("azure client should initialize");

        assert_eq!(client.provider_name(), "azure-anthropic");
    }

    #[test]
    fn test_build_request_body_uses_top_level_cache_control_for_direct_anthropic() {
        let client = AnthropicClient::new(AnthropicConfig {
            caching: Some(CachingConfig {
                enabled: true,
                ttl: CacheTTL::OneHour,
            }),
            ..Default::default()
        })
        .expect("anthropic client should initialize");

        let body = client
            .build_request_body(&[UnifiedMessage::user("Cache this request prefix.")], &[])
            .expect("request body should build");

        assert_eq!(body["cache_control"]["type"], "ephemeral");
        assert_eq!(body["cache_control"]["ttl"], "1h");
        assert!(body["messages"][0]["content"][0]["cache_control"].is_null());
        assert!(body.get("system").is_none());
    }

    #[test]
    fn test_build_request_body_uses_top_level_cache_control_for_azure_anthropic() {
        let client = AnthropicClient::new(AnthropicConfig {
            azure: Some(super::super::config::AzureAnthropicConfig {
                base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
                auth_method: AzureAnthropicAuthMethod::XApiKey,
            }),
            caching: Some(CachingConfig::default()),
            ..Default::default()
        })
        .expect("azure client should initialize");

        let body = client
            .build_request_body(
                &[UnifiedMessage::user("Cache this Azure request prefix.")],
                &[],
            )
            .expect("request body should build");

        assert_eq!(body["cache_control"]["type"], "ephemeral");
        assert_eq!(body["cache_control"]["ttl"], "5m");
    }

    #[test]
    fn test_build_request_body_uses_block_cache_control_for_bedrock_messages() {
        let client = AnthropicClient::new(AnthropicConfig {
            bedrock: Some(super::super::config::BedrockConfig::default()),
            caching: Some(CachingConfig::default()),
            ..Default::default()
        })
        .expect("bedrock client should initialize");

        let body = client
            .build_request_body(
                &[UnifiedMessage::user("Cache this Bedrock request prefix.")],
                &[],
            )
            .expect("request body should build");

        assert!(body.get("cache_control").is_none());
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["ttl"],
            "5m"
        );
    }

    #[test]
    fn test_build_request_body_uses_block_cache_control_for_bedrock_system_and_tools() {
        let client = AnthropicClient::new(AnthropicConfig {
            bedrock: Some(super::super::config::BedrockConfig::default()),
            caching: Some(CachingConfig {
                enabled: true,
                ttl: CacheTTL::OneHour,
            }),
            ..Default::default()
        })
        .expect("bedrock client should initialize");

        let body = client
            .build_request_body(
                &[
                    UnifiedMessage::system("You are a Bedrock cached assistant."),
                    UnifiedMessage::user("Use the tools."),
                ],
                &[UnifiedTool {
                    name: "get_weather".to_string(),
                    description: "Get weather".to_string(),
                    parameters: json!({"type": "object"}),
                }],
            )
            .expect("request body should build");

        assert!(body.get("cache_control").is_none());
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["ttl"],
            "1h"
        );
        assert_eq!(body["tools"][0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn test_azure_ignores_direct_anthropic_env_key_when_resolving_azure_auth() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous_anthropic = std::env::var("ANTHROPIC_API_KEY").ok();
        let previous_azure = std::env::var("AZURE_API_KEY").ok();

        std::env::set_var("ANTHROPIC_API_KEY", "direct-anthropic-env-key");
        std::env::set_var("AZURE_API_KEY", "azure-env-key");

        let client = AnthropicClient::new(AnthropicConfig {
            api_key: Some("direct-anthropic-env-key".to_string()),
            azure: Some(super::super::config::AzureAnthropicConfig {
                base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
                auth_method: AzureAnthropicAuthMethod::BearerToken,
            }),
            ..Default::default()
        })
        .expect("azure client should initialize");

        let headers = client.build_headers().expect("headers should build");
        assert_eq!(
            headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer azure-env-key")
        );

        restore_env("ANTHROPIC_API_KEY", previous_anthropic);
        restore_env("AZURE_API_KEY", previous_azure);
    }
}
