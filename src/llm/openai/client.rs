//! OpenAI Responses API client with streaming support.
//!
//! Implements the `LlmClient` trait for OpenAI's Responses API, providing
//! full support for streaming responses, tool calling, reasoning, and
//! structured outputs.
//!
//! # API Endpoint
//!
//! - Endpoint: POST `https://api.openai.com/v1/responses`
//! - Streaming: SSE with `stream: true`
//!
//! # Authentication
//!
//! Requires `Authorization: Bearer {api_key}` header with API key from OpenAI.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER};
use reqwest::StatusCode;
use reqwest::Url;
use std::collections::{HashMap, HashSet};
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

use super::config::{
    model_supports_sampling_parameters, normalize_openai_model, resolve_reasoning_effort_for_model,
    OpenAIConfig, RetryConfig,
};
use super::convert::{
    extract_instructions, from_unified_messages, from_unified_tools, to_unified_content_blocks,
};
use super::streaming::{is_chunk_error_recoverable, StreamAccumulator, StreamEvent};
use super::types::*;
use crate::llm::provider::{LlmClient, ProviderFailureCapture};
use crate::llm::unified::{
    UnifiedContentBlock, UnifiedMessage, UnifiedRole, UnifiedTool, UnifiedToolCall,
};

/// OpenAI Responses API client.
///
/// Handles authentication, request construction, SSE parsing, and response
/// conversion for the OpenAI Responses API.
///
/// # Features
///
/// - Streaming responses via Server-Sent Events
/// - Tool calling with parallel execution support
/// - Reasoning for o-series models
/// - Structured outputs via JSON schema
/// - Prompt caching with cache keys
/// - Service tier selection for latency/cost control
/// - Automatic HTTP/1 fallback when HTTP/2 stream limits are exceeded
///
/// # HTTP/2 Stream Saturation and HTTP/1 Fallback
///
/// This client maintains two HTTP client instances:
/// - **Primary (HTTP/2)**: Used by default for multiplexing and efficiency
/// - **Fallback (HTTP/1)**: Activated automatically when HTTP/2 stream limits are hit
///
/// When high-concurrency workloads (e.g., 200+ parallel requests) exceed Cloudflare's
/// `SETTINGS_MAX_CONCURRENT_STREAMS` limit (~100 streams), the server resets streams
/// with `PROTOCOL_ERROR`. The client detects these errors and automatically switches
/// to HTTP/1 for all subsequent requests, opening additional TCP connections as needed.
///
/// The fallback is permanent for the lifetime of the client instance and provides
/// graceful degradation without dropping requests.
///
/// # Examples
///
/// ```ignore
/// use appam::llm::openai::{OpenAIClient, OpenAIConfig};
/// use appam::llm::unified::UnifiedMessage;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let config = OpenAIConfig::default();
///     let client = OpenAIClient::new(config)?;
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
///         |_block| Ok(()),
///         |_usage| Ok(()),
///     ).await?;
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    /// Primary HTTP client using HTTP/2 for multiplexing (default)
    http_primary: reqwest::Client,
    /// Fallback HTTP client using HTTP/1 only (activated on stream saturation)
    http_fallback: reqwest::Client,
    /// Atomic flag indicating whether HTTP/1 fallback mode is active
    use_http1_fallback: Arc<AtomicBool>,
    /// Client configuration (model, API key, reasoning, etc.)
    config: OpenAIConfig,
    /// Mutable continuation anchor used for `previous_response_id`.
    previous_response_id: Arc<Mutex<Option<String>>>,
    /// Most recent completed response ID observed during streaming.
    latest_response_id: Arc<Mutex<Option<String>>>,
    /// Last failed provider exchange captured for diagnostics.
    last_failed_exchange: Arc<Mutex<Option<ProviderFailureCapture>>>,
}

/// Classification of retryable streaming failures.
///
/// Used to provide structured logging when the SSE stream terminates unexpectedly
/// and we decide to retry the request.
#[derive(Debug)]
enum StreamRetryReason {
    /// Network-level interruption such as unexpected EOF or connection reset.
    Network { message: String },
    /// API-level retryable error such as internal server errors or timeouts.
    ApiError {
        #[allow(dead_code)]
        code: Option<String>,
        message: String,
    },
}

impl StreamRetryReason {
    fn label(&self) -> &'static str {
        match self {
            Self::Network { .. } => "network",
            Self::ApiError { .. } => "api_error",
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Network { message } => message,
            Self::ApiError { message, .. } => message,
        }
    }
}

/// Build an HTTP/1-only fallback client for stream saturation scenarios.
///
/// Creates a dedicated HTTP client configured to use HTTP/1.1 exclusively,
/// bypassing HTTP/2 multiplexing. This client is used as a safety valve when
/// the primary HTTP/2 transport exceeds the server's concurrent stream limit.
///
/// # Configuration
///
/// Matches the primary client's settings (timeouts, keepalive, DNS caching)
/// but forces HTTP/1 via `.http1_only()`, allowing reqwest to open additional
/// TCP connections instead of multiplexing on a saturated HTTP/2 connection.
///
/// # Use Case
///
/// When Cloudflare (fronting api.openai.com) rejects new streams with
/// `PROTOCOL_ERROR` because `SETTINGS_MAX_CONCURRENT_STREAMS` is exceeded,
/// the fallback client opens fresh HTTP/1.1 connections to continue operations.
///
/// # Arguments
///
/// * `config` - OpenAI configuration containing base URL and connection parameters
///
/// # Returns
///
/// A configured `reqwest::Client` restricted to HTTP/1.1
///
/// # Errors
///
/// Returns an error if:
/// - Base URL parsing fails
/// - Host extraction fails
/// - Client construction fails
fn build_http1_fallback_client(config: &OpenAIConfig) -> Result<reqwest::Client> {
    let base = Url::parse(&config.base_url)
        .context("Failed to parse OpenAI base URL when constructing HTTP/1 fallback client")?;
    let host = base
        .host_str()
        .ok_or_else(|| anyhow!("OpenAI base URL is missing host component"))?;
    let port = base.port_or_known_default().ok_or_else(|| {
        anyhow!(
            "Unable to determine port for OpenAI base URL (scheme: {}, host: {})",
            base.scheme(),
            host
        )
    })?;

    let mut builder = reqwest::Client::builder()
        .http1_only()
        .connect_timeout(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(120))
        .pool_max_idle_per_host(10)
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .gzip(true)
        .user_agent("appam/0.1.0");

    if let Some(addrs) = resolve_host_for_http1(host, port) {
        builder = builder.resolve_to_addrs(host, addrs.as_slice());
    }

    builder
        .build()
        .context("Failed to create HTTP/1 fallback client for OpenAI")
}

/// Resolve DNS for the HTTP/1 fallback client to avoid runtime lookup delays.
///
/// Pre-resolves the OpenAI hostname to socket addresses and caches them in the
/// HTTP client configuration via `resolve_to_addrs()`. This prevents DNS flakiness
/// from causing request failures when the fallback is activated.
///
/// # Implementation Notes
///
/// Uses `ToSocketAddrs` (OS resolver) rather than async DNS to keep construction
/// synchronous. Failures are logged but not fatal—the client falls back to runtime
/// resolution within hyper's connector if pre-resolution fails.
///
/// # Arguments
///
/// * `host` - Hostname to resolve (e.g., "api.openai.com")
/// * `port` - Port number for the socket address (e.g., 443)
///
/// # Returns
///
/// `Some(Vec<SocketAddr>)` if resolution succeeds, `None` if it fails
fn resolve_host_for_http1(host: &str, port: u16) -> Option<Vec<SocketAddr>> {
    let target = format!("{host}:{port}");
    match target.to_socket_addrs() {
        Ok(iter) => {
            let addrs: Vec<_> = iter.collect();
            if addrs.is_empty() {
                warn!(
                    host = host,
                    port = port,
                    "DNS lookup returned no addresses for HTTP/1 fallback client"
                );
                None
            } else {
                debug!(
                    host = host,
                    port = port,
                    addr_count = addrs.len(),
                    "Resolved HTTP/1 fallback client addresses"
                );
                Some(addrs)
            }
        }
        Err(err) => {
            warn!(
                host = host,
                port = port,
                error = %err,
                "Failed to resolve HTTP/1 fallback client host"
            );
            None
        }
    }
}

impl OpenAIClient {
    /// Create a new OpenAI Responses API client.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - HTTP client construction fails
    /// - Configuration validation fails
    pub fn new(config: OpenAIConfig) -> Result<Self> {
        config.validate()?;

        // Build primary HTTP/2 client (default transport)
        let http_primary = crate::http::client_pool::get_or_init_client(&config.base_url, |ctx| {
            // Configure HTTP client with optimizations:
            // - HTTP/2 by default for multiplexing (allows up to ~100 concurrent streams)
            // - connect_timeout: 30s to prevent hanging on connection establishment
            // - pool_idle_timeout: 120s to clean up stale connections
            // - pool_max_idle_per_host: Limit to 10 idle connections per host
            // - tcp_keepalive: 60s to detect dead connections
            // - tcp_nodelay: Disable Nagle's algorithm for lower latency
            // - gzip: Enable compression to reduce bandwidth
            // - resolve_to_addrs: Cache DNS lookups to prevent flaky DNS errors
            // - NO read timeout: Allow infinite active connection time for streaming responses
            let mut builder = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .pool_idle_timeout(Duration::from_secs(120))
                .pool_max_idle_per_host(10)
                .tcp_keepalive(Duration::from_secs(60))
                .tcp_nodelay(true)
                .gzip(true)
                .user_agent("appam/0.1.0");

            if let Some(addrs) = ctx.resolved_addrs() {
                builder = builder.resolve_to_addrs(ctx.host(), addrs);
            }

            builder.build().context("Failed to create HTTP client")
        })?;

        // Build HTTP/1 fallback client (activated on stream saturation)
        let http_fallback = build_http1_fallback_client(&config)?;
        let previous_response_id = config
            .conversation
            .as_ref()
            .and_then(|conversation| conversation.previous_response_id.clone());

        Ok(Self {
            http_primary,
            http_fallback,
            use_http1_fallback: Arc::new(AtomicBool::new(false)),
            config,
            previous_response_id: Arc::new(Mutex::new(previous_response_id)),
            latest_response_id: Arc::new(Mutex::new(None)),
            last_failed_exchange: Arc::new(Mutex::new(None)),
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
        http_status: Option<StatusCode>,
        request_payload: &str,
        response_payload: impl Into<String>,
    ) {
        let provider = if self.config.azure.is_some() {
            "azure-openai"
        } else {
            "openai"
        };
        let capture = ProviderFailureCapture {
            provider: provider.to_string(),
            model: normalize_openai_model(&self.config.model),
            http_status: http_status.map(|status| status.as_u16()),
            request_payload: request_payload.to_string(),
            response_payload: response_payload.into(),
            provider_response_id: self.latest_response_id(),
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

    /// Return the latest completed Responses API ID observed by this client.
    pub fn latest_response_id(&self) -> Option<String> {
        self.latest_response_id
            .lock()
            .expect("latest response id mutex poisoned")
            .clone()
    }

    /// Update the `previous_response_id` used for the next request.
    pub fn set_previous_response_id(&self, response_id: Option<String>) {
        *self
            .previous_response_id
            .lock()
            .expect("previous response id mutex poisoned") = response_id;
    }

    /// Build the API endpoint URL based on configuration.
    ///
    /// # Azure OpenAI
    ///
    /// When Azure configuration is present, constructs the Azure-specific URL:
    /// `https://{resource_name}.cognitiveservices.azure.com/openai/responses?api-version={api_version}`
    ///
    /// # Standard OpenAI
    ///
    /// Uses the base URL from config: `{base_url}/responses`
    fn build_endpoint_url(&self) -> String {
        if let Some(ref azure) = self.config.azure {
            format!(
                "https://{}.cognitiveservices.azure.com/openai/responses?api-version={}",
                azure.resource_name, azure.api_version
            )
        } else {
            format!("{}/responses", self.config.base_url)
        }
    }

    /// Check if this client is configured for Azure OpenAI.
    fn is_azure(&self) -> bool {
        self.config.azure.is_some()
    }

    /// Build HTTP headers for OpenAI API requests.
    ///
    /// # Standard OpenAI Headers
    ///
    /// - `Authorization`: Bearer token authentication
    /// - `Content-Type`: application/json
    /// - `Accept`: text/event-stream (for streaming)
    /// - `OpenAI-Organization`: Optional organization ID
    /// - `OpenAI-Project`: Optional project ID
    ///
    /// # Azure OpenAI Headers
    ///
    /// - `api-key`: API key authentication (instead of Bearer token)
    /// - `Content-Type`: application/json
    /// - `Accept`: text/event-stream (for streaming)
    ///
    /// # API Key Resolution
    ///
    /// - Standard OpenAI: `config.api_key` → `OPENAI_API_KEY` env var
    /// - Azure OpenAI: `config.api_key` → `AZURE_OPENAI_API_KEY` → `OPENAI_API_KEY` env var
    ///
    /// # Errors
    ///
    /// Returns an error if API key is missing or headers cannot be constructed.
    ///
    /// # Security
    ///
    /// Never logs the API key header value.
    fn build_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));

        // Resolve API key based on provider type
        let api_key = if let Some(ref key) = self.config.api_key {
            key.clone()
        } else if self.is_azure() {
            // Azure: Try AZURE_OPENAI_API_KEY first, then fallback to OPENAI_API_KEY
            std::env::var("AZURE_OPENAI_API_KEY")
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .context("Missing Azure OpenAI API key. Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY env var")?
        } else {
            // Standard OpenAI
            std::env::var("OPENAI_API_KEY")
                .context("Missing OpenAI API key. Set OPENAI_API_KEY env var or config.api_key")?
        };

        if self.is_azure() {
            // Azure uses api-key header instead of Bearer token
            headers.insert(
                "api-key",
                HeaderValue::from_str(&api_key).context("Invalid API key header format")?,
            );
        } else {
            // Standard OpenAI uses Bearer token
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .context("Invalid API key header format")?,
            );

            // Optional organization header (OpenAI only, not used for Azure)
            if let Some(ref org) = self.config.organization {
                headers.insert(
                    "OpenAI-Organization",
                    HeaderValue::from_str(org).context("Invalid organization header")?,
                );
            }

            // Optional project header (OpenAI only, not used for Azure)
            if let Some(ref project) = self.config.project {
                headers.insert(
                    "OpenAI-Project",
                    HeaderValue::from_str(project).context("Invalid project header")?,
                );
            }
        }

        Ok(headers)
    }

    /// Build the Responses API request body.
    ///
    /// Converts unified messages and tools to OpenAI's format and applies
    /// all configuration options.
    fn build_request_body(
        &self,
        messages: &[UnifiedMessage],
        tools: &[UnifiedTool],
    ) -> Result<ResponseCreateParams> {
        let previous_response_id = self
            .previous_response_id
            .lock()
            .expect("previous response id mutex poisoned")
            .clone();
        // Convert unified messages to OpenAI input format. System prompts are
        // lifted into top-level `instructions`, while `input` contains only the
        // conversation items that should participate in the item stream.
        let instructions = extract_instructions(messages);
        let input_messages: Vec<UnifiedMessage> = messages
            .iter()
            .filter(|message| message.role != UnifiedRole::System)
            .cloned()
            .collect();
        let input = from_unified_messages(&input_messages, previous_response_id.as_deref());
        let normalized_model = normalize_openai_model(&self.config.model);
        let requested_effort = self
            .config
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort);
        let resolved_effort = self
            .config
            .reasoning
            .as_ref()
            .map(|_| resolve_reasoning_effort_for_model(&normalized_model, requested_effort));
        let sampling_supported =
            model_supports_sampling_parameters(&normalized_model, requested_effort);

        let params = ResponseCreateParams {
            model: normalized_model,
            input: Some(input),
            stream: Some(self.config.stream),

            max_output_tokens: self.config.max_output_tokens,
            // GPT-5.4 only accepts sampling parameters in `reasoning.effort = "none"`.
            temperature: sampling_supported
                .then_some(self.config.temperature)
                .flatten(),
            top_p: sampling_supported.then_some(self.config.top_p).flatten(),

            tools: if tools.is_empty() {
                None
            } else {
                Some(from_unified_tools(tools))
            },

            tool_choice: if tools.is_empty() {
                None
            } else {
                Some(ToolChoice::String("auto".to_string()))
            },

            parallel_tool_calls: Some(false), // Match existing behavior
            max_tool_calls: None,

            reasoning: self.config.reasoning.as_ref().map(|r| Reasoning {
                effort: resolved_effort.as_ref().map(|e| {
                    match e {
                        super::config::ReasoningEffort::None => "none",
                        super::config::ReasoningEffort::Minimal => "minimal",
                        super::config::ReasoningEffort::Low => "low",
                        super::config::ReasoningEffort::Medium => "medium",
                        super::config::ReasoningEffort::High => "high",
                        super::config::ReasoningEffort::XHigh => "xhigh",
                    }
                    .to_string()
                }),
                summary: r.summary.as_ref().map(|s| {
                    match s {
                        super::config::ReasoningSummary::Auto => "auto",
                        super::config::ReasoningSummary::Concise => "concise",
                        super::config::ReasoningSummary::Detailed => "detailed",
                    }
                    .to_string()
                }),
            }),

            text: {
                // Build text configuration with both format and verbosity
                if self.config.text_format.is_some() || self.config.text_verbosity.is_some() {
                    Some(ResponseTextConfig {
                        format: self.config.text_format.as_ref().map(|fmt| match fmt {
                            super::config::TextFormatConfig::Text => ResponseTextFormat::Text,
                            super::config::TextFormatConfig::JsonObject => {
                                ResponseTextFormat::JsonObject
                            }
                            super::config::TextFormatConfig::JsonSchema {
                                name,
                                description,
                                schema,
                                strict,
                            } => ResponseTextFormat::JsonSchema {
                                name: name.clone(),
                                description: description.clone(),
                                schema: schema.clone(),
                                strict: *strict,
                            },
                        }),
                        verbosity: self.config.text_verbosity.map(|v| match v {
                            super::config::TextVerbosity::Low => TextVerbosity::Low,
                            super::config::TextVerbosity::Medium => TextVerbosity::Medium,
                            super::config::TextVerbosity::High => TextVerbosity::High,
                        }),
                    })
                } else {
                    None
                }
            },

            service_tier: self.config.service_tier.map(|st| {
                match st {
                    super::config::ServiceTier::Auto => "auto",
                    super::config::ServiceTier::Default => "default",
                    super::config::ServiceTier::Flex => "flex",
                    super::config::ServiceTier::Priority => "priority",
                    super::config::ServiceTier::Scale => "scale",
                }
                .to_string()
            }),

            conversation: self.config.conversation.as_ref().and_then(|conversation| {
                conversation
                    .id
                    .as_ref()
                    .map(|id| Conversation::Simple(id.clone()))
            }),

            previous_response_id,

            store: self.config.store,
            background: self.config.background,
            metadata: self.config.metadata.clone(),
            prompt_cache_key: self.config.prompt_cache_key.clone(),
            safety_identifier: self.config.safety_identifier.clone(),
            top_logprobs: sampling_supported
                .then_some(self.config.top_logprobs)
                .flatten(),

            instructions,
            stream_options: None,
            include: self
                .config
                .store
                .filter(|store| !store)
                .map(|_| vec!["reasoning.encrypted_content".to_string()]),
            truncation: None,
        };

        Ok(params)
    }

    /// Get the effective retry configuration, defaulting when not provided.
    fn retry_config(&self) -> RetryConfig {
        self.config.retry.clone().unwrap_or_default()
    }

    /// Determine whether the given HTTP status code is retryable.
    fn should_retry_status(status: StatusCode) -> bool {
        matches!(
            status,
            StatusCode::TOO_MANY_REQUESTS
                | StatusCode::INTERNAL_SERVER_ERROR
                | StatusCode::BAD_GATEWAY
                | StatusCode::SERVICE_UNAVAILABLE
                | StatusCode::GATEWAY_TIMEOUT
                | StatusCode::REQUEST_TIMEOUT
        )
    }

    /// Determine whether a reqwest error indicates a transient network failure.
    ///
    /// Retries on:
    /// - Timeouts
    /// - Connection errors (including DNS failures)
    /// - Request errors
    /// - Body errors
    fn should_retry_reqwest_error(error: &reqwest::Error) -> bool {
        // Check standard reqwest error categories
        if error.is_timeout() || error.is_connect() || error.is_request() || error.is_body() {
            return true;
        }

        // Check for DNS errors in the error chain
        let error_msg = error.to_string().to_ascii_lowercase();
        if error_msg.contains("dns error")
            || error_msg.contains("failed to lookup address")
            || error_msg.contains("nodename nor servname provided")
        {
            return true;
        }

        if Self::is_http2_protocol_error(&error_msg) {
            return true;
        }

        false
    }

    /// Extract retry-after header value (in seconds) if present.
    fn retry_after_from_headers(headers: &HeaderMap) -> Option<Duration> {
        headers
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|raw| raw.parse::<u64>().ok())
            .map(Duration::from_secs)
    }

    /// Calculate retry delay using exponential backoff and optional retry-after header.
    fn compute_retry_delay(
        retry_config: &RetryConfig,
        attempt: u32,
        retry_after: Option<Duration>,
    ) -> Duration {
        if let Some(delay) = retry_after {
            let max_backoff = Duration::from_millis(retry_config.max_backoff_ms);
            return std::cmp::min(delay, max_backoff);
        }

        let backoff_ms = retry_config.calculate_backoff(attempt);
        Duration::from_millis(backoff_ms)
    }

    /// Detect HTTP/2 protocol errors indicating stream saturation.
    ///
    /// Checks if an error message contains the signature of an HTTP/2 `PROTOCOL_ERROR`,
    /// which Cloudflare (and other CDNs) send when a client attempts to open more
    /// concurrent streams than the server's `SETTINGS_MAX_CONCURRENT_STREAMS` allows.
    ///
    /// # Arguments
    ///
    /// * `message` - Error message string (case-insensitive lowercase recommended)
    ///
    /// # Returns
    ///
    /// `true` if the message indicates HTTP/2 stream saturation, `false` otherwise
    fn is_http2_protocol_error(message: &str) -> bool {
        message.contains("http2 error: stream error received")
    }

    /// Select the appropriate HTTP client based on fallback state.
    ///
    /// Returns the HTTP/2 client by default, switching to the HTTP/1 fallback client
    /// once `enable_http1_fallback()` has been called. The selection is thread-safe
    /// via atomic flag and uses relaxed ordering for read performance.
    ///
    /// # Returns
    ///
    /// Reference to either `http_primary` (HTTP/2) or `http_fallback` (HTTP/1)
    fn select_http_client(&self) -> &reqwest::Client {
        if self.use_http1_fallback.load(Ordering::Relaxed) {
            &self.http_fallback
        } else {
            &self.http_primary
        }
    }

    /// Activate HTTP/1 fallback mode permanently for this client.
    ///
    /// Switches the client from HTTP/2 multiplexing to HTTP/1 connection pooling.
    /// This is a one-way transition triggered when HTTP/2 stream saturation is
    /// detected. Uses atomic compare-exchange to ensure only one thread logs the
    /// transition, even under high concurrency.
    ///
    /// # Behavior
    ///
    /// - **First call**: Sets the flag, logs the transition, returns `true`
    /// - **Subsequent calls**: No-op, returns `false`
    ///
    /// All threads immediately see the new state due to `SeqCst` ordering.
    ///
    /// # Returns
    ///
    /// `true` if this call activated the fallback, `false` if already active
    fn enable_http1_fallback(&self) -> bool {
        if self
            .use_http1_fallback
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            info!(
                "Switching OpenAI client to HTTP/1 fallback mode after HTTP/2 protocol error spike"
            );
            true
        } else {
            false
        }
    }

    /// Classify whether a streaming error should trigger a retry.
    fn classify_stream_error(error: &anyhow::Error) -> Option<StreamRetryReason> {
        // Check for reqwest errors
        for cause in error.chain() {
            if let Some(reqwest_error) = cause.downcast_ref::<reqwest::Error>() {
                if Self::should_retry_reqwest_error(reqwest_error) {
                    return Some(StreamRetryReason::Network {
                        message: reqwest_error.to_string(),
                    });
                }
            }
        }

        let message = error.to_string();
        let normalized = message.to_ascii_lowercase();

        // Check for API error patterns
        if normalized.contains("retryable api error") {
            return Some(StreamRetryReason::ApiError {
                code: None,
                message: message.clone(),
            });
        }

        // Check for network patterns
        const NETWORK_PATTERNS: &[&str] = &[
            "unexpected eof",
            "connection reset",
            "broken pipe",
            "connection closed",
            "connection aborted",
            "incomplete message",
            "error reading a body from connection",
            "dns error",
            "failed to lookup address",
            "nodename nor servname provided",
        ];

        if NETWORK_PATTERNS
            .iter()
            .any(|needle| normalized.contains(needle))
        {
            return Some(StreamRetryReason::Network { message });
        }

        None
    }

    /// Parse SSE stream and invoke callbacks.
    ///
    /// Processes Server-Sent Events from the streaming response and invokes
    /// the appropriate callbacks for content, tool calls, and reasoning. The
    /// callbacks are borrowed mutably so the caller can reuse the same
    /// closures across retry attempts.
    #[allow(clippy::too_many_arguments)]
    async fn parse_stream<FContent, FTool, FReason, FToolPartial, FContentBlock, FUsage>(
        &self,
        response: reqwest::Response,
        on_content: &mut FContent,
        on_tool_calls: &mut FTool,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
        on_content_block_complete: &mut FContentBlock,
        on_usage: &mut FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut pending_bytes = Vec::new();
        let mut accumulator = StreamAccumulator::new();
        let mut completed_tool_calls = Vec::new();
        let mut function_call_meta: HashMap<String, (String, String)> = HashMap::new();
        let mut streamed_reasoning_segments: HashSet<(i32, i32)> = HashSet::new();
        let mut streamed_summary_segments: HashSet<(i32, i32)> = HashSet::new();

        // Track bytes received for error reporting
        let mut total_bytes_received: usize = 0;
        let mut events_processed: usize = 0;

        while let Some(chunk) = stream.next().await {
            // Handle chunk reading with graceful error recovery
            let chunk = match chunk.context("Failed to read stream chunk") {
                Ok(b) => b,
                Err(e) => {
                    // Check if this is a recoverable network error during chunk reading
                    if is_chunk_error_recoverable(&e) {
                        warn!(
                            target: "openai",
                            bytes_received = total_bytes_received,
                            events_processed = events_processed,
                            error = %e,
                            "Stream interrupted by recoverable error, returning partial response"
                        );
                        // Return success with partial content - the caller already received
                        // content via callbacks before the interruption
                        break;
                    }
                    // Non-recoverable error - propagate immediately
                    return Err(e);
                }
            };
            total_bytes_received += chunk.len();
            pending_bytes.extend_from_slice(&chunk);

            match std::str::from_utf8(&pending_bytes) {
                Ok(valid_str) => {
                    buffer.push_str(valid_str);
                    pending_bytes.clear();
                }
                Err(e) => {
                    let valid_up_to = e.valid_up_to();

                    if valid_up_to > 0 {
                        let valid_str = std::str::from_utf8(&pending_bytes[..valid_up_to])
                            .expect("valid UTF-8 prefix");
                        buffer.push_str(valid_str);
                        pending_bytes.drain(..valid_up_to);
                    }

                    if e.error_len().is_some() {
                        anyhow::bail!("Invalid UTF-8 in stream: encountered invalid byte sequence",);
                    }
                }
            }

            // Process complete SSE events
            while let Some(event_end) = buffer.find("\n\n") {
                let event_data = buffer[..event_end].to_string();
                buffer = buffer[event_end + 2..].to_string();

                // Parse SSE event
                let mut data_payload = String::new();
                for line in event_data.lines() {
                    if let Some(rest) = line.strip_prefix("data: ") {
                        if !data_payload.is_empty() {
                            data_payload.push('\n');
                        }
                        data_payload.push_str(rest);
                    }
                }

                if data_payload.is_empty() {
                    continue;
                }

                if data_payload == "[DONE]" {
                    break;
                }

                match serde_json::from_str::<StreamEvent>(&data_payload) {
                    Ok(event) => {
                        self.handle_stream_event(
                            &event,
                            &mut accumulator,
                            &mut function_call_meta,
                            &mut streamed_reasoning_segments,
                            &mut streamed_summary_segments,
                            on_content,
                            on_tool_calls,
                            on_reasoning,
                            on_tool_calls_partial,
                            on_content_block_complete,
                            on_usage,
                            &mut completed_tool_calls,
                        )?;
                        events_processed += 1;
                    }
                    Err(e) => {
                        debug!(
                            "Failed to parse stream event: {} - Data: {}",
                            e, data_payload
                        );
                    }
                }
            }
        }

        if !pending_bytes.is_empty() {
            match std::str::from_utf8(&pending_bytes) {
                Ok(valid_str) => {
                    buffer.push_str(valid_str);
                }
                Err(e) => {
                    let valid_up_to = e.valid_up_to();
                    if valid_up_to > 0 {
                        let valid_str = std::str::from_utf8(&pending_bytes[..valid_up_to])
                            .expect("valid UTF-8 prefix");
                        buffer.push_str(valid_str);
                    }

                    anyhow::bail!(
                        "Invalid UTF-8 in stream: stream ended with incomplete UTF-8 sequence",
                    );
                }
            }
        }

        // Invoke final callbacks if needed
        if !completed_tool_calls.is_empty() {
            on_tool_calls(completed_tool_calls)?;
        }

        Ok(())
    }

    /// Handle a single stream event.
    #[allow(clippy::too_many_arguments)]
    fn handle_stream_event<FContent, FTool, FReason, FToolPartial, FContentBlock, FUsage>(
        &self,
        event: &StreamEvent,
        accumulator: &mut StreamAccumulator,
        function_call_meta: &mut HashMap<String, (String, String)>,
        streamed_reasoning_segments: &mut HashSet<(i32, i32)>,
        streamed_summary_segments: &mut HashSet<(i32, i32)>,
        on_content: &mut FContent,
        _on_tool_calls: &mut FTool,
        on_reasoning: &mut FReason,
        _on_tool_calls_partial: &mut FToolPartial,
        on_content_block_complete: &mut FContentBlock,
        on_usage: &mut FUsage,
        completed_tool_calls: &mut Vec<UnifiedToolCall>,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        match event {
            StreamEvent::ResponseTextDelta { delta, .. } => {
                on_content(delta)?;
            }
            StreamEvent::ResponseTextDone {
                text,
                output_index,
                content_index,
                ..
            } => {
                let already_streamed = accumulator
                    .get_text(*output_index as usize, *content_index as usize)
                    .map(|buf| !buf.is_empty())
                    .unwrap_or(false);

                if !already_streamed && !text.is_empty() {
                    on_content(text)?;
                }
            }
            StreamEvent::ResponseReasoningTextDelta {
                delta,
                output_index,
                content_index,
                ..
            } => {
                streamed_reasoning_segments.insert((*output_index, *content_index));
                on_reasoning(delta)?;
            }
            StreamEvent::ResponseReasoningTextDone {
                text,
                output_index,
                content_index,
                ..
            } => {
                if !text.is_empty()
                    && streamed_reasoning_segments.insert((*output_index, *content_index))
                {
                    on_reasoning(text)?;
                }
            }
            StreamEvent::ResponseOutputItemAdded {
                item:
                    OutputItem::FunctionCall {
                        id, call_id, name, ..
                    },
                ..
            } => {
                function_call_meta.insert(id.clone(), (call_id.clone(), name.clone()));
            }
            StreamEvent::ResponseOutputItemAdded { .. } => {
                // Ignore non-function-call output items
            }
            StreamEvent::ResponseFunctionCallArgumentsDone {
                call_id,
                name,
                arguments,
                item_id,
                ..
            } => {
                let input = serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
                let (meta_call_id, meta_name) = function_call_meta
                    .get(item_id)
                    .cloned()
                    .unwrap_or_else(|| (item_id.clone(), String::new()));
                let id = call_id.clone().unwrap_or(meta_call_id);
                let tool_name = name.clone().unwrap_or(meta_name);
                let tool_call = UnifiedToolCall {
                    id,
                    name: tool_name,
                    input,
                    raw_input_json: Some(arguments.clone()),
                };
                completed_tool_calls.push(tool_call);
            }
            StreamEvent::ResponseReasoningSummaryPartAdded { .. }
            | StreamEvent::ResponseReasoningSummaryPartDone { .. } => {
                // Summary structure events do not stream visible text directly.
                // `response.reasoning_summary_text.delta`/`done` are handled below.
            }
            StreamEvent::ResponseReasoningSummaryTextDelta {
                delta,
                output_index,
                summary_index,
                ..
            } => {
                streamed_summary_segments.insert((*output_index, *summary_index));
                on_reasoning(delta)?;
            }
            StreamEvent::ResponseReasoningSummaryTextDone {
                text,
                output_index,
                summary_index,
                ..
            } => {
                if !text.is_empty()
                    && streamed_summary_segments.insert((*output_index, *summary_index))
                {
                    on_reasoning(text)?;
                }
            }
            StreamEvent::ResponseCompleted { response, .. } => {
                *self
                    .latest_response_id
                    .lock()
                    .expect("latest response id mutex poisoned") = Some(response.id.clone());

                // Convert and emit usage data
                if let Some(usage) = &response.usage {
                    let input_tokens = usage.input_tokens.max(0) as u32;
                    let output_tokens = usage.output_tokens.max(0) as u32;
                    let cache_read_tokens = usage.input_tokens_details.cached_tokens.max(0) as u32;
                    let reasoning_tokens =
                        usage.output_tokens_details.reasoning_tokens.max(0) as u32;

                    let unified_usage = crate::llm::unified::UnifiedUsage {
                        input_tokens,
                        output_tokens,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: (cache_read_tokens > 0)
                            .then_some(cache_read_tokens),
                        reasoning_tokens: (reasoning_tokens > 0).then_some(reasoning_tokens),
                    };
                    on_usage(unified_usage)?;
                }

                // Extract final content blocks for completion callback
                let content_blocks = to_unified_content_blocks(&response.output);
                for block in content_blocks {
                    on_content_block_complete(block)?;
                }
            }
            StreamEvent::ResponseFailed { error, .. }
            | StreamEvent::ResponseError { error, .. } => {
                // Check if this is a retryable API error
                if error.is_retryable() {
                    return Err(anyhow!(
                        "Retryable API error: {} (code: {:?})",
                        error.message,
                        error.code
                    ));
                } else {
                    return Err(anyhow!(
                        "API error: {} (code: {:?})",
                        error.message,
                        error.code
                    ));
                }
            }
            _ => {}
        }

        accumulator.handle_event(event);
        Ok(())
    }
}

#[async_trait]
impl LlmClient for OpenAIClient {
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
        on_content: FContent,
        on_tool_calls: FTool,
        on_reasoning: FReason,
        on_tool_calls_partial: FToolPartial,
        on_content_block_complete: FContentBlock,
        mut on_usage: FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        *self
            .latest_response_id
            .lock()
            .expect("latest response id mutex poisoned") = None;
        self.clear_last_failed_exchange();

        let request_body = self.build_request_body(messages, tools)?;
        let request_payload = serde_json::to_string_pretty(&request_body)?;

        // Debug: Log request body to verify tool schemas
        debug!(
            "OpenAI Request body: {}",
            serde_json::to_string_pretty(&request_body)?
        );

        let headers = self.build_headers()?;
        let retry_config = self.retry_config();
        let max_attempts = retry_config.max_retries.saturating_add(1).max(1);
        let mut attempt: u32 = 0;

        let mut on_content = on_content;
        let mut on_tool_calls = on_tool_calls;
        let mut on_reasoning = on_reasoning;
        let mut on_tool_calls_partial = on_tool_calls_partial;
        let mut on_content_block_complete = on_content_block_complete;

        loop {
            attempt += 1;

            // Check which transport is active (HTTP/2 or HTTP/1 fallback)
            let using_http1 = self.use_http1_fallback.load(Ordering::Relaxed);

            debug!(
                attempt = attempt,
                max_attempts = max_attempts,
                transport = if using_http1 { "http1" } else { "http2" },
                "Sending OpenAI Responses API request"
            );

            // Select HTTP client based on fallback state
            let http_client = self.select_http_client();

            let response = match http_client
                .post(self.build_endpoint_url())
                .headers(headers.clone())
                .json(&request_body)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(err) => {
                    // Check if this is an HTTP/2 stream saturation error
                    let error_lower = err.to_string().to_ascii_lowercase();
                    let is_http2_protocol = Self::is_http2_protocol_error(&error_lower);

                    // Activate HTTP/1 fallback if we hit the stream limit
                    if is_http2_protocol {
                        self.enable_http1_fallback();
                    }

                    if attempt < max_attempts
                        && (Self::should_retry_reqwest_error(&err) || is_http2_protocol)
                    {
                        let wait = Self::compute_retry_delay(&retry_config, attempt, None);
                        warn!(
                            attempt = attempt,
                            max_attempts = max_attempts,
                            wait_secs = wait.as_secs_f64(),
                            error = %err,
                            "OpenAI request failed, retrying after backoff"
                        );
                        eprintln!(
                            "⚠️  OpenAI request failed (attempt {}/{}), retrying in {:.1}s...",
                            attempt,
                            max_attempts,
                            wait.as_secs_f64()
                        );
                        sleep(wait).await;
                        continue;
                    }

                    self.record_failed_exchange(None, &request_payload, err.to_string());
                    return Err(err).context("OpenAI API request failed");
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let response_headers = response.headers().clone();
                let body = response.text().await.unwrap_or_default();

                error!(status = %status, attempt = attempt, body = %body, "OpenAI error response");

                if attempt < max_attempts && Self::should_retry_status(status) {
                    let retry_after = Self::retry_after_from_headers(&response_headers);
                    let wait = Self::compute_retry_delay(&retry_config, attempt, retry_after);
                    info!(
                        attempt = attempt,
                        max_attempts = max_attempts,
                        wait_secs = wait.as_secs_f64(),
                        status = %status,
                        "Retrying OpenAI request after API error"
                    );
                    eprintln!(
                        "⚠️  OpenAI API error {} (attempt {}/{}), retrying in {:.1}s...",
                        status,
                        attempt,
                        max_attempts,
                        wait.as_secs_f64()
                    );
                    sleep(wait).await;
                    continue;
                }

                self.record_failed_exchange(Some(status), &request_payload, body.clone());
                return Err(anyhow!("OpenAI error ({}): {}", status, body));
            }

            match self
                .parse_stream(
                    response,
                    &mut on_content,
                    &mut on_tool_calls,
                    &mut on_reasoning,
                    &mut on_tool_calls_partial,
                    &mut on_content_block_complete,
                    &mut on_usage,
                )
                .await
            {
                Ok(()) => return Ok(()),
                Err(err) => {
                    if attempt < max_attempts {
                        if let Some(reason) = Self::classify_stream_error(&err) {
                            // Check if streaming failure was due to HTTP/2 saturation
                            let reason_message_lower = reason.message().to_ascii_lowercase();
                            if Self::is_http2_protocol_error(&reason_message_lower) {
                                self.enable_http1_fallback();
                            }

                            let wait = Self::compute_retry_delay(&retry_config, attempt, None);
                            warn!(
                                attempt = attempt,
                                max_attempts = max_attempts,
                                wait_secs = wait.as_secs_f64(),
                                error_kind = reason.label(),
                                error_message = reason.message(),
                                "OpenAI streaming error, retrying"
                            );
                            eprintln!(
                                "⚠️  OpenAI streaming error (attempt {}/{}), retrying in {:.1}s...",
                                attempt,
                                max_attempts,
                                wait.as_secs_f64()
                            );
                            sleep(wait).await;
                            continue;
                        }
                    }

                    self.record_failed_exchange(None, &request_payload, format!("{:#}", err));
                    return Err(err);
                }
            }
        }
    }

    fn provider_name(&self) -> &str {
        "openai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};
    use std::collections::{HashMap, HashSet};

    use crate::llm::openai::{ConversationConfig, ReasoningConfig};
    use crate::llm::unified::UnifiedMessage;

    fn build_test_client(config: OpenAIConfig) -> OpenAIClient {
        OpenAIClient::new(OpenAIConfig {
            api_key: Some("test-openai-key".to_string()),
            ..config
        })
        .expect("test client should construct")
    }

    #[test]
    fn test_build_request_body_normalizes_gpt54_and_preserves_sampling_none_mode() {
        let client = build_test_client(OpenAIConfig {
            model: "openai/gpt-5.4".to_string(),
            reasoning: Some(ReasoningConfig::no_reasoning()),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_logprobs: Some(4),
            conversation: Some(ConversationConfig {
                id: None,
                previous_response_id: Some("resp_prev".to_string()),
            }),
            ..Default::default()
        });

        let request = client
            .build_request_body(&[UnifiedMessage::user("hello")], &[])
            .expect("request body should build");

        assert_eq!(request.model, "gpt-5.4");
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.top_p, Some(0.9));
        assert_eq!(request.top_logprobs, Some(4));
        assert_eq!(request.previous_response_id.as_deref(), Some("resp_prev"));
        assert!(request.conversation.is_none());
        assert_eq!(
            request
                .reasoning
                .as_ref()
                .and_then(|reasoning| reasoning.effort.as_deref()),
            Some("none")
        );
    }

    #[test]
    fn test_set_previous_response_id_updates_follow_up_requests() {
        let client = build_test_client(OpenAIConfig::default());
        client.set_previous_response_id(Some("resp_follow_up".to_string()));

        let request = client
            .build_request_body(&[UnifiedMessage::user("hello again")], &[])
            .expect("request body should build");

        assert_eq!(
            request.previous_response_id.as_deref(),
            Some("resp_follow_up")
        );
    }

    #[test]
    fn test_build_request_body_trims_replayed_history_when_continuing() {
        let client = build_test_client(OpenAIConfig {
            conversation: Some(ConversationConfig {
                id: None,
                previous_response_id: Some("resp_prev".to_string()),
            }),
            ..Default::default()
        });

        let request = client
            .build_request_body(
                &[
                    UnifiedMessage::system("Stay terse."),
                    UnifiedMessage::assistant("Prior answer"),
                    UnifiedMessage::user("Tool result or follow-up"),
                ],
                &[],
            )
            .expect("request body should build");

        match request.input.expect("input must exist") {
            ResponseInput::Structured(items) => {
                assert_eq!(items.len(), 1);
                assert!(matches!(
                    &items[0],
                    InputItem::Message {
                        role: MessageRole::User,
                        ..
                    }
                ));
            }
            _ => panic!("expected structured input"),
        }

        assert_eq!(request.instructions.as_deref(), Some("Stay terse."));
    }

    #[test]
    fn test_build_request_body_keeps_tool_outputs_first_class_on_continuation() {
        let client = build_test_client(OpenAIConfig {
            conversation: Some(ConversationConfig {
                id: None,
                previous_response_id: Some("resp_prev".to_string()),
            }),
            ..Default::default()
        });

        let request = client
            .build_request_body(
                &[
                    UnifiedMessage::system("Use tools carefully."),
                    UnifiedMessage {
                        role: UnifiedRole::Assistant,
                        content: vec![crate::llm::unified::UnifiedContentBlock::ToolUse {
                            id: "call_123".to_string(),
                            name: "read_file".to_string(),
                            input: serde_json::json!({"path": "src/main.rs"}),
                        }],
                        id: Some("msg_1".to_string()),
                        timestamp: None,
                        reasoning: None,
                        reasoning_details: None,
                    },
                    UnifiedMessage {
                        role: UnifiedRole::User,
                        content: vec![crate::llm::unified::UnifiedContentBlock::ToolResult {
                            tool_use_id: "call_123".to_string(),
                            content: serde_json::json!({"ok": true}),
                            is_error: Some(false),
                        }],
                        id: None,
                        timestamp: None,
                        reasoning: None,
                        reasoning_details: None,
                    },
                ],
                &[],
            )
            .expect("request body should build");

        match request.input.expect("input must exist") {
            ResponseInput::Structured(items) => {
                assert_eq!(items.len(), 1);
                assert!(matches!(
                    &items[0],
                    InputItem::FunctionCallOutput { call_id, .. } if call_id == "call_123"
                ));
            }
            _ => panic!("expected structured input"),
        }

        assert_eq!(
            request.instructions.as_deref(),
            Some("Use tools carefully.")
        );
    }

    #[test]
    fn test_build_request_body_requests_encrypted_reasoning_for_stateless_turns() {
        let client = build_test_client(OpenAIConfig {
            store: Some(false),
            ..Default::default()
        });

        let request = client
            .build_request_body(&[UnifiedMessage::user("hello")], &[])
            .expect("request body should build");

        assert_eq!(
            request.include,
            Some(vec!["reasoning.encrypted_content".to_string()])
        );
    }

    #[test]
    fn test_build_request_body_serializes_minimal_reasoning_effort() {
        let client = build_test_client(OpenAIConfig {
            reasoning: Some(crate::llm::openai::ReasoningConfig::minimal()),
            ..Default::default()
        });

        let request = client
            .build_request_body(&[UnifiedMessage::user("hello")], &[])
            .expect("request body should build");

        assert_eq!(
            request
                .reasoning
                .as_ref()
                .and_then(|reasoning| reasoning.effort.as_deref()),
            Some("minimal")
        );
    }

    #[test]
    fn test_handle_stream_event_records_latest_response_id() {
        let client = build_test_client(OpenAIConfig::default());
        let response = Response {
            id: "resp_recorded".to_string(),
            created_at: 0.0,
            object: "response".to_string(),
            model: "gpt-5.4".to_string(),
            status: ResponseStatus::Completed,
            output: vec![],
            instructions: None,
            tools: vec![],
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: false,
            temperature: None,
            top_p: None,
            usage: None,
            error: None,
            incomplete_details: None,
            conversation: None,
            previous_response_id: None,
        };

        let event = StreamEvent::ResponseCompleted {
            response,
            sequence_number: 1,
        };

        client
            .handle_stream_event(
                &event,
                &mut StreamAccumulator::new(),
                &mut HashMap::new(),
                &mut HashSet::new(),
                &mut HashSet::new(),
                &mut |_| Ok(()),
                &mut |_| Ok(()),
                &mut |_| Ok(()),
                &mut |_| Ok(()),
                &mut |_| Ok(()),
                &mut |_| Ok(()),
                &mut Vec::new(),
            )
            .expect("stream event should succeed");

        assert_eq!(
            client.latest_response_id().as_deref(),
            Some("resp_recorded")
        );
    }

    #[test]
    fn test_should_retry_status_for_server_errors() {
        assert!(OpenAIClient::should_retry_status(
            StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(OpenAIClient::should_retry_status(StatusCode::BAD_GATEWAY));
        assert!(OpenAIClient::should_retry_status(
            StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(OpenAIClient::should_retry_status(
            StatusCode::GATEWAY_TIMEOUT
        ));
        assert!(OpenAIClient::should_retry_status(
            StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(OpenAIClient::should_retry_status(
            StatusCode::REQUEST_TIMEOUT
        ));
    }

    #[test]
    fn test_should_not_retry_status_for_client_errors() {
        assert!(!OpenAIClient::should_retry_status(StatusCode::BAD_REQUEST));
        assert!(!OpenAIClient::should_retry_status(StatusCode::UNAUTHORIZED));
        assert!(!OpenAIClient::should_retry_status(StatusCode::FORBIDDEN));
        assert!(!OpenAIClient::should_retry_status(StatusCode::NOT_FOUND));
    }

    #[test]
    fn test_retry_after_from_headers_parses_seconds() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("5"));

        let delay = OpenAIClient::retry_after_from_headers(&headers);
        assert_eq!(delay, Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_compute_retry_delay_prefers_retry_after_but_caps() {
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30_000,
            backoff_multiplier: 2.0,
            jitter: false,
        };

        let retry_after = Some(Duration::from_secs(120));
        let delay = OpenAIClient::compute_retry_delay(&retry_config, 2, retry_after);
        assert_eq!(delay, Duration::from_secs(30));
    }

    #[test]
    fn test_compute_retry_delay_uses_backoff_when_no_retry_after() {
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 500,
            max_backoff_ms: 30_000,
            backoff_multiplier: 2.0,
            jitter: false,
        };

        let delay = OpenAIClient::compute_retry_delay(&retry_config, 3, None);
        assert_eq!(delay, Duration::from_millis(2000));
    }

    #[test]
    fn test_classify_stream_error_detects_unexpected_eof() {
        let error = anyhow!(
            "error decoding response body: error reading a body from connection: unexpected EOF during chunk size line"
        );
        let classification = OpenAIClient::classify_stream_error(&error);
        assert!(matches!(
            classification,
            Some(StreamRetryReason::Network { .. })
        ));
    }

    #[test]
    fn test_classify_stream_error_detects_connection_reset() {
        let error = anyhow!(
            "Failed to read stream chunk: error decoding response body: error reading a body from connection: Connection reset by peer (os error 54)"
        );
        let classification = OpenAIClient::classify_stream_error(&error);
        assert!(matches!(
            classification,
            Some(StreamRetryReason::Network { .. })
        ));
    }

    #[test]
    fn test_classify_stream_error_detects_api_error() {
        let error = anyhow!(
            "Retryable API error: Internal server error (code: Some(\"internal_server_error\"))"
        );
        let classification = OpenAIClient::classify_stream_error(&error);
        assert!(matches!(
            classification,
            Some(StreamRetryReason::ApiError { .. })
        ));
    }

    #[test]
    fn test_classify_stream_error_non_network() {
        let error = anyhow!("failed to parse JSON payload");
        assert!(OpenAIClient::classify_stream_error(&error).is_none());
    }
}
