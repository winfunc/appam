//! Shared provider selection and runtime client abstractions.
//!
//! The rest of Appam depends on this module instead of depending directly on a
//! specific provider client. That separation keeps provider quirks, auth, and
//! stream parsing contained in the provider submodules while the runtime works
//! in terms of unified messages and tool calls.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::unified::{UnifiedMessage, UnifiedTool, UnifiedToolCall};

/// Sanitized provider diagnostics captured for failed requests.
///
/// This payload is intentionally limited to request/response data that helps
/// operators debug provider failures without storing credentials. Provider
/// clients only populate this structure for failed requests and clear it after
/// successful completion.
///
/// # Security
///
/// The payload is expected to be safe for traces and persisted diagnostics, but
/// callers should still avoid attaching end-user secrets or unrelated prompt
/// bodies when forwarding it elsewhere.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderFailureCapture {
    /// Provider label used for the failing request.
    pub provider: String,
    /// Model identifier used for the failing request.
    pub model: String,
    /// Optional HTTP status code returned by the provider.
    pub http_status: Option<u16>,
    /// Serialized request payload sent to the provider.
    pub request_payload: String,
    /// Raw response payload or terminal error/stream payload.
    pub response_payload: String,
    /// Provider-native response identifier, when exposed.
    pub provider_response_id: Option<String>,
}

/// Enumerates the LLM backends Appam can target.
///
/// Determines which backend API to use for language model inference.
/// Each provider has its own configuration section and may support
/// different features.
///
/// # Note on Copy
///
/// This enum does not implement `Copy` because the `AzureOpenAI` and
/// `AzureAnthropic` variants
/// contains `String` fields. Use `.clone()` when necessary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LlmProvider {
    /// OpenRouter Chat Completions API.
    ///
    /// Supports: Tool calling, reasoning tokens, provider routing, automatic caching
    /// Endpoint: `https://openrouter.ai/api/v1/chat/completions`
    #[default]
    OpenRouterCompletions,

    /// OpenRouter Responses API.
    ///
    /// Supports: Tool calling, enhanced reasoning with effort levels, structured outputs
    /// Endpoint: `https://openrouter.ai/api/v1/responses`
    OpenRouterResponses,

    /// Anthropic Claude API (Messages API format)
    ///
    /// Supports: Tool calling, extended thinking, prompt caching, vision, server tools
    /// Endpoint: `https://api.anthropic.com/v1/messages`
    Anthropic,

    /// OpenAI Responses API
    ///
    /// Supports: Tool calling, reasoning (o-series models), structured outputs, service tiers
    /// Endpoint: `https://api.openai.com/v1/responses`
    OpenAI,

    /// OpenAI Codex subscription-backed Responses API.
    ///
    /// Uses ChatGPT OAuth credentials and the ChatGPT Codex backend rather than
    /// Platform API keys and `api.openai.com`.
    ///
    /// Supports: Tool calling, reasoning, and encrypted reasoning continuity
    /// for ChatGPT subscription-backed Codex models.
    /// Endpoint: `https://chatgpt.com/backend-api/codex/responses`
    #[serde(rename = "openai_codex")]
    OpenAICodex,

    /// Google Vertex AI Gemini API
    ///
    /// Supports: Streaming responses, function calling, thought signatures,
    /// and usage metadata via `generateContent` / `streamGenerateContent`.
    ///
    /// Endpoint (API key mode): `https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?alt=sse&key=...`
    /// Endpoint (project mode): `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse`
    Vertex,

    /// Azure OpenAI Responses API
    ///
    /// Routes requests through Azure's hosted OpenAI endpoints with Azure-specific
    /// authentication and URL construction.
    ///
    /// Supports: Same features as OpenAI, but accessed via Azure endpoints
    /// Endpoint: `https://{resource_name}.cognitiveservices.azure.com/openai/responses?api-version={api_version}`
    /// Auth: `api-key` header (not Bearer token)
    /// Env: `AZURE_OPENAI_API_KEY` (fallback to `OPENAI_API_KEY`)
    #[serde(rename = "azure_openai")]
    AzureOpenAI {
        /// Azure resource name (subdomain of your Azure OpenAI endpoint)
        resource_name: String,
        /// API version string (e.g., "2025-04-01-preview")
        api_version: String,
    },

    /// Azure-hosted Anthropic Messages API
    ///
    /// Routes requests through Azure-hosted Anthropic-compatible endpoints while
    /// preserving the Anthropic Messages wire format and Appam's existing
    /// Anthropic streaming/tool machinery.
    ///
    /// Supports: Same features as Anthropic Claude (tools, thinking, caching,
    /// SSE streaming), but accessed via Azure-hosted endpoints.
    /// Endpoint: `{base_url}/v1/messages`
    /// Auth: `x-api-key` or `Authorization: Bearer`
    #[serde(rename = "azure_anthropic")]
    AzureAnthropic {
        /// Full Azure Anthropic base URL without `/v1/messages`.
        base_url: String,
        /// Authentication method for Azure Anthropic requests.
        #[serde(default)]
        auth_method: crate::llm::anthropic::AzureAnthropicAuthMethod,
    },

    /// AWS Bedrock Claude API
    ///
    /// Routes requests through AWS Bedrock endpoints with Bedrock-specific
    /// authentication and URL construction.
    ///
    /// Supports: Same features as Anthropic Claude (tools, thinking, etc.)
    ///
    /// # Authentication Methods
    ///
    /// - **SigV4** (default): Uses AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
    ///   Supports streaming via `/invoke-with-response-stream` endpoint.
    /// - **Bearer Token**: Uses Bedrock API Keys (`AWS_BEARER_TOKEN_BEDROCK`).
    ///   Only supports non-streaming `/invoke` endpoint.
    Bedrock {
        /// AWS region for the Bedrock endpoint (e.g., "us-east-1")
        region: String,
        /// Bedrock model identifier (e.g., "us.anthropic.claude-opus-4-5-20251101-v1:0")
        model_id: String,
        /// Authentication method (default: SigV4 for streaming support)
        #[serde(default)]
        auth_method: crate::llm::anthropic::BedrockAuthMethod,
    },
}

impl std::fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenRouterCompletions => write!(f, "openrouter-completions"),
            Self::OpenRouterResponses => write!(f, "openrouter-responses"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAI => write!(f, "openai"),
            Self::OpenAICodex => write!(f, "openai-codex"),
            Self::Vertex => write!(f, "vertex"),
            Self::AzureOpenAI {
                resource_name,
                api_version,
            } => write!(f, "azure-openai[{}@{}]", resource_name, api_version),
            Self::AzureAnthropic {
                base_url,
                auth_method,
            } => write!(f, "azure-anthropic[{}:{}]", base_url, auth_method.as_str()),
            Self::Bedrock {
                region,
                model_id,
                auth_method,
            } => {
                let auth_str = match auth_method {
                    crate::llm::anthropic::BedrockAuthMethod::SigV4 => "sigv4",
                    crate::llm::anthropic::BedrockAuthMethod::BearerToken => "bearer",
                };
                write!(f, "bedrock[{}@{}:{}]", model_id, region, auth_str)
            }
        }
    }
}

impl std::str::FromStr for LlmProvider {
    type Err = anyhow::Error;

    /// Parse provider from string.
    ///
    /// Accepts:
    /// - "openrouter-completions" / "openrouter_completions" / "openrouter" → OpenRouterCompletions
    /// - "openrouter-responses" / "openrouterresponses" → OpenRouterResponses
    /// - "anthropic" → Anthropic
    /// - "openai" → OpenAI
    /// - "openai-codex" / "openai_codex" / "codex" → OpenAICodex
    /// - "vertex" / "google-vertex" / "google_vertex" → Vertex
    /// - "azure-openai" / "azure_openai" / "azure" → AzureOpenAI
    /// - "azure-anthropic" / "azure_anthropic" → AzureAnthropic
    /// - "bedrock" / "aws-bedrock" / "aws_bedrock" → Bedrock
    ///
    /// # AzureOpenAI Configuration
    ///
    /// When parsing "azure-openai", the resource_name and api_version are read from:
    /// - `AZURE_OPENAI_RESOURCE` env var (required for Azure)
    /// - `AZURE_OPENAI_API_VERSION` env var (defaults to "2025-04-01-preview")
    ///
    /// # Bedrock Configuration
    ///
    /// When parsing "bedrock", the region and model_id are read from:
    /// - `AWS_REGION` or `AWS_DEFAULT_REGION` env var (defaults to "us-east-1")
    /// - `AWS_BEDROCK_MODEL_ID` env var (defaults to "us.anthropic.claude-sonnet-4-5-20250514-v1:0")
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "openrouter" | "openrouter-completions" | "openroutercompletions" => {
                Ok(Self::OpenRouterCompletions)
            }
            "openrouter-responses" | "openrouterresponses" => Ok(Self::OpenRouterResponses),
            "anthropic" => Ok(Self::Anthropic),
            "openai" => Ok(Self::OpenAI),
            "openai-codex" | "openai_codex" | "codex" => Ok(Self::OpenAICodex),
            "vertex" | "google-vertex" | "google_vertex" => Ok(Self::Vertex),
            "azure-openai" | "azure_openai" | "azure" => {
                // Parse from environment variables
                let resource_name = std::env::var("AZURE_OPENAI_RESOURCE")
                    .unwrap_or_else(|_| "example-resource".to_string());
                let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
                    .unwrap_or_else(|_| "2025-04-01-preview".to_string());
                Ok(Self::AzureOpenAI {
                    resource_name,
                    api_version,
                })
            }
            "azure-anthropic" | "azure_anthropic" => {
                let base_url = if let Ok(base_url) = std::env::var("AZURE_ANTHROPIC_BASE_URL") {
                    if base_url.trim().is_empty() {
                        return Err(anyhow::anyhow!(
                            "AZURE_ANTHROPIC_BASE_URL must not be empty when APPAM_PROVIDER=azure-anthropic"
                        ));
                    }
                    base_url
                } else {
                    let resource = std::env::var("AZURE_ANTHROPIC_RESOURCE").map_err(|_| {
                        anyhow::anyhow!(
                            "Missing Azure Anthropic endpoint configuration. Set AZURE_ANTHROPIC_BASE_URL or AZURE_ANTHROPIC_RESOURCE."
                        )
                    })?;
                    crate::llm::anthropic::AzureAnthropicConfig::base_url_from_resource(
                        &resource,
                    )?
                };

                let auth_method = std::env::var("AZURE_ANTHROPIC_AUTH_METHOD")
                    .ok()
                    .map(|value| value.parse())
                    .transpose()?
                    .unwrap_or_default();

                Ok(Self::AzureAnthropic {
                    base_url,
                    auth_method,
                })
            }
            "bedrock" | "aws-bedrock" | "aws_bedrock" => {
                // Parse from environment variables
                let region = std::env::var("AWS_REGION")
                    .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                    .unwrap_or_else(|_| "us-east-1".to_string());
                let model_id = std::env::var("AWS_BEDROCK_MODEL_ID")
                    .unwrap_or_else(|_| "us.anthropic.claude-opus-4-5-20251101-v1:0".to_string());
                // Default to SigV4 for streaming support
                let auth_method = crate::llm::anthropic::BedrockAuthMethod::default();
                Ok(Self::Bedrock { region, model_id, auth_method })
            }
            _ => Err(anyhow::anyhow!(
                "Invalid provider: {}. Must be 'openrouter-completions', 'openrouter-responses', 'anthropic', 'openai', 'openai-codex', 'vertex', 'azure-openai', 'azure-anthropic', or 'bedrock'",
                s
            )),
        }
    }
}

impl LlmProvider {
    /// Normalized provider identifier used for pricing and usage aggregation.
    ///
    /// The pricing tables expect canonical provider names (`anthropic`, `openai`,
    /// `openrouter`, `vertex`) while the display implementation differentiates between
    /// OpenRouter APIs. Use this helper whenever we persist usage or look up
    /// per-provider configuration.
    ///
    /// # Note on Azure
    ///
    /// Azure OpenAI uses the same pricing model as OpenAI, so it returns "openai"
    /// for pricing purposes.
    ///
    /// Azure Anthropic uses Anthropic-compatible endpoints and should be priced
    /// using Anthropic model tables.
    ///
    /// # Note on Bedrock
    ///
    /// AWS Bedrock uses Anthropic Claude models with the same pricing model,
    /// so it returns "anthropic" for pricing purposes.
    pub fn pricing_key(&self) -> &'static str {
        match self {
            Self::Anthropic | Self::AzureAnthropic { .. } | Self::Bedrock { .. } => "anthropic",
            Self::OpenAI | Self::OpenAICodex | Self::AzureOpenAI { .. } => "openai",
            Self::Vertex => "vertex",
            Self::OpenRouterCompletions | Self::OpenRouterResponses => "openrouter",
        }
    }
}

/// Common streaming interface implemented by provider clients.
///
/// All provider clients (OpenRouter, Anthropic, etc.) implement this trait,
/// enabling the agent runtime to work with any provider through a unified API.
///
/// # Streaming Model
///
/// All methods use callback-based streaming:
/// - `on_content`: Called for each text content chunk
/// - `on_tool_calls`: Called when tool calls are finalized
/// - `on_reasoning`: Called for reasoning/thinking tokens
/// - `on_tool_calls_partial`: Called for incremental tool call updates
///
/// # Provider-Specific Features
///
/// While this trait provides a common interface, individual providers may
/// support additional features accessible through their specific configuration
/// (e.g., Anthropic's prompt caching, OpenRouter's reasoning effort levels).
///
/// # Error Handling
///
/// All methods return `Result` to propagate API errors, network failures,
/// or parsing issues. Implementations should provide detailed error context.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` for use in async contexts.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Stream a chat completion with tool calling support.
    ///
    /// Sends a conversation with optional tool definitions to the LLM and
    /// streams the response through the provided callbacks.
    ///
    /// # Parameters
    ///
    /// - `messages`: Conversation history in unified format
    /// - `tools`: Available tool specifications
    /// - `on_content`: Callback for each text content chunk
    /// - `on_tool_calls`: Callback when tool calls are finalized
    /// - `on_reasoning`: Callback for reasoning/thinking tokens (text only, for streaming display)
    /// - `on_tool_calls_partial`: Callback for incremental tool call updates
    /// - `on_content_block_complete`: Callback for complete content blocks (preserves signatures)
    /// - `on_usage`: Callback for token usage statistics (called at completion)
    ///
    /// # Conversation Flow
    ///
    /// 1. LLM generates text content → `on_content` called repeatedly
    /// 2. LLM decides to use tools → `on_tool_calls` called with complete arguments
    /// 3. Caller executes tools and appends results to messages
    /// 4. Repeat until LLM stops requesting tools
    /// 5. Response completes → `on_usage` called with token statistics
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Authentication fails
    /// - Network request fails
    /// - API returns an error response
    /// - Response parsing fails
    /// - Any callback returns an error
    #[allow(clippy::too_many_arguments)]
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
        on_usage: FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(super::unified::UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(super::unified::UnifiedUsage) -> Result<()> + Send;

    /// Get the provider name.
    ///
    /// Used for logging and debugging to identify which provider is being used.
    fn provider_name(&self) -> &str;
}

/// Runtime-selected provider client wrapper.
///
/// `DynamicLlmClient` keeps provider construction explicit while giving the
/// runtime a single concrete type it can store, log, and query for failure
/// diagnostics.
///
/// # Usage
///
/// ```rust,no_run
/// use appam::llm::provider::DynamicLlmClient;
/// use appam::llm::LlmProvider;
/// use appam::config::AppConfig;
///
/// let config = AppConfig::default();
/// let client = DynamicLlmClient::from_config(&config)?;
/// // Client now uses the configured provider
/// # Ok::<(), anyhow::Error>(())
/// ```
pub enum DynamicLlmClient {
    /// OpenRouter Chat Completions API client.
    OpenRouterCompletions(crate::llm::openrouter::completions::OpenRouterCompletionsClient),
    /// OpenRouter Responses API client.
    OpenRouterResponses(crate::llm::openrouter::responses::OpenRouterClient),
    /// Anthropic Messages API client
    Anthropic(crate::llm::anthropic::AnthropicClient),
    /// OpenAI Responses API client
    OpenAI(crate::llm::openai::OpenAIClient),
    /// OpenAI Codex subscription-backed Responses API client
    OpenAICodex(crate::llm::openai_codex::OpenAICodexClient),
    /// Google Vertex Gemini API client
    Vertex(crate::llm::vertex::VertexClient),
    /// Azure OpenAI Responses API client (uses OpenAI client with Azure config)
    AzureOpenAI {
        /// The underlying OpenAI client configured for Azure
        client: crate::llm::openai::OpenAIClient,
        /// Azure resource name for display/logging
        resource_name: String,
        /// API version for display/logging
        api_version: String,
    },
    /// Azure Anthropic Messages API client (uses Anthropic client with Azure config)
    AzureAnthropic {
        /// The underlying Anthropic client configured for Azure Anthropic.
        client: crate::llm::anthropic::AnthropicClient,
        /// Base URL for display/logging.
        base_url: String,
        /// Authentication method for display/logging.
        auth_method: crate::llm::anthropic::AzureAnthropicAuthMethod,
    },
    /// AWS Bedrock Claude API client (uses Anthropic client with Bedrock config)
    Bedrock {
        /// The underlying Anthropic client configured for Bedrock
        client: crate::llm::anthropic::AnthropicClient,
        /// AWS region for display/logging
        region: String,
        /// Bedrock model ID for display/logging
        model_id: String,
        /// Authentication method (SigV4 or Bearer)
        auth_method: crate::llm::anthropic::BedrockAuthMethod,
    },
}

impl DynamicLlmClient {
    /// Create a provider client from a fully resolved [`crate::config::AppConfig`].
    ///
    /// Selects the appropriate client implementation based on `config.provider`.
    ///
    /// # Errors
    ///
    /// Returns an error if client construction fails (e.g., missing API key).
    pub fn from_config(config: &crate::config::AppConfig) -> Result<Self> {
        match &config.provider {
            LlmProvider::OpenRouterCompletions => {
                let client = crate::llm::openrouter::completions::OpenRouterCompletionsClient::new(
                    config.openrouter.clone(),
                    config.openrouter.reasoning.clone(),
                    config.openrouter.provider_preferences.clone(),
                )?;
                Ok(Self::OpenRouterCompletions(client))
            }
            LlmProvider::OpenRouterResponses => {
                let client = crate::llm::openrouter::responses::OpenRouterClient::new(
                    config.openrouter.clone(),
                )?;
                Ok(Self::OpenRouterResponses(client))
            }
            LlmProvider::Anthropic => {
                let client = crate::llm::anthropic::AnthropicClient::new(config.anthropic.clone())?;
                Ok(Self::Anthropic(client))
            }
            LlmProvider::OpenAI => {
                let client = crate::llm::openai::OpenAIClient::new(config.openai.clone())?;
                Ok(Self::OpenAI(client))
            }
            LlmProvider::OpenAICodex => {
                let client =
                    crate::llm::openai_codex::OpenAICodexClient::new(config.openai_codex.clone())?;
                Ok(Self::OpenAICodex(client))
            }
            LlmProvider::Vertex => {
                let client = crate::llm::vertex::VertexClient::new(config.vertex.clone())?;
                Ok(Self::Vertex(client))
            }
            LlmProvider::AzureOpenAI {
                resource_name,
                api_version,
            } => {
                // Create OpenAI config with Azure-specific settings
                let mut azure_config = config.openai.clone();
                azure_config.azure = Some(crate::llm::openai::AzureConfig {
                    resource_name: resource_name.clone(),
                    api_version: api_version.clone(),
                });
                let client = crate::llm::openai::OpenAIClient::new(azure_config)?;
                Ok(Self::AzureOpenAI {
                    client,
                    resource_name: resource_name.clone(),
                    api_version: api_version.clone(),
                })
            }
            LlmProvider::AzureAnthropic {
                base_url,
                auth_method,
            } => {
                let mut azure_config = config.anthropic.clone();
                azure_config.azure = Some(crate::llm::anthropic::AzureAnthropicConfig {
                    base_url: base_url.clone(),
                    auth_method: auth_method.clone(),
                });
                let client = crate::llm::anthropic::AnthropicClient::new(azure_config)?;
                Ok(Self::AzureAnthropic {
                    client,
                    base_url: base_url.clone(),
                    auth_method: auth_method.clone(),
                })
            }
            LlmProvider::Bedrock {
                region,
                model_id,
                auth_method,
            } => {
                // Create Anthropic config with Bedrock-specific settings
                let mut bedrock_config = config.anthropic.clone();

                // For Bearer token auth, disable streaming since it's not supported
                if *auth_method == crate::llm::anthropic::BedrockAuthMethod::BearerToken {
                    bedrock_config.stream = false;
                }

                bedrock_config.bedrock = Some(crate::llm::anthropic::BedrockConfig {
                    region: region.clone(),
                    model_id: model_id.clone(),
                    auth_method: auth_method.clone(),
                    ..Default::default()
                });
                let client = crate::llm::anthropic::AnthropicClient::new(bedrock_config)?;
                Ok(Self::Bedrock {
                    client,
                    region: region.clone(),
                    model_id: model_id.clone(),
                    auth_method: auth_method.clone(),
                })
            }
        }
    }

    /// Return the currently selected provider descriptor.
    pub fn provider(&self) -> LlmProvider {
        match self {
            Self::OpenRouterCompletions(_) => LlmProvider::OpenRouterCompletions,
            Self::OpenRouterResponses(_) => LlmProvider::OpenRouterResponses,
            Self::Anthropic(_) => LlmProvider::Anthropic,
            Self::OpenAI(_) => LlmProvider::OpenAI,
            Self::OpenAICodex(_) => LlmProvider::OpenAICodex,
            Self::Vertex(_) => LlmProvider::Vertex,
            Self::AzureOpenAI {
                resource_name,
                api_version,
                ..
            } => LlmProvider::AzureOpenAI {
                resource_name: resource_name.clone(),
                api_version: api_version.clone(),
            },
            Self::AzureAnthropic {
                base_url,
                auth_method,
                ..
            } => LlmProvider::AzureAnthropic {
                base_url: base_url.clone(),
                auth_method: auth_method.clone(),
            },
            Self::Bedrock {
                region,
                model_id,
                auth_method,
                ..
            } => LlmProvider::Bedrock {
                region: region.clone(),
                model_id: model_id.clone(),
                auth_method: auth_method.clone(),
            },
        }
    }

    /// Return a stable lowercase provider label for logs and metrics.
    pub fn provider_name(&self) -> &str {
        match self {
            Self::OpenRouterCompletions(_) => "openrouter-completions",
            Self::OpenRouterResponses(_) => "openrouter-responses",
            Self::Anthropic(_) => "anthropic",
            Self::OpenAI(_) => "openai",
            Self::OpenAICodex(_) => "openai-codex",
            Self::Vertex(_) => "vertex",
            Self::AzureOpenAI { .. } => "azure-openai",
            Self::AzureAnthropic { .. } => "azure-anthropic",
            Self::Bedrock { .. } => "bedrock",
        }
    }

    /// Return the latest provider-native response ID, when the backend exposes one.
    ///
    /// OpenAI's Responses API uses the top-level response ID for
    /// `previous_response_id` continuation. Other providers either do not expose
    /// an equivalent concept or Appam does not currently model it.
    pub fn latest_response_id(&self) -> Option<String> {
        match self {
            Self::OpenAI(client) => client.latest_response_id(),
            Self::OpenAICodex(client) => client.latest_response_id(),
            Self::AzureOpenAI { client, .. } => client.latest_response_id(),
            _ => None,
        }
    }

    /// Update the provider-native continuation anchor when supported.
    ///
    /// Providers that do not model response continuation IDs ignore this call.
    pub fn set_previous_response_id(&self, response_id: Option<String>) {
        match self {
            Self::OpenAI(client) => client.set_previous_response_id(response_id),
            Self::AzureOpenAI { client, .. } => client.set_previous_response_id(response_id),
            _ => {}
        }
    }

    /// Retrieve and clear the most recent failed provider exchange, if any.
    ///
    /// Callers use this immediately after a request error to attach provider-
    /// level diagnostics to traces or persisted session records.
    pub fn take_last_failed_exchange(&self) -> Option<ProviderFailureCapture> {
        match self {
            Self::OpenAI(client) => client.take_last_failed_exchange(),
            Self::OpenAICodex(client) => client.take_last_failed_exchange(),
            Self::AzureOpenAI { client, .. } => client.take_last_failed_exchange(),
            Self::Anthropic(client) => client.take_last_failed_exchange(),
            Self::AzureAnthropic { client, .. } => client.take_last_failed_exchange(),
            Self::Bedrock { client, .. } => client.take_last_failed_exchange(),
            _ => None,
        }
    }
}

#[async_trait]
impl LlmClient for DynamicLlmClient {
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
        on_usage: FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(super::unified::UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(super::unified::UnifiedUsage) -> Result<()> + Send,
    {
        match self {
            Self::OpenRouterCompletions(client) => {
                client
                    .chat_with_tools_streaming(
                        messages,
                        tools,
                        on_content,
                        on_tool_calls,
                        on_reasoning,
                        on_tool_calls_partial,
                        on_content_block_complete,
                        on_usage,
                    )
                    .await
            }
            Self::OpenRouterResponses(client) => {
                client
                    .chat_with_tools_streaming(
                        messages,
                        tools,
                        on_content,
                        on_tool_calls,
                        on_reasoning,
                        on_tool_calls_partial,
                        on_content_block_complete,
                        on_usage,
                    )
                    .await
            }
            Self::Anthropic(client) => {
                client
                    .chat_with_tools_streaming(
                        messages,
                        tools,
                        on_content,
                        on_tool_calls,
                        on_reasoning,
                        on_tool_calls_partial,
                        on_content_block_complete,
                        on_usage,
                    )
                    .await
            }
            Self::OpenAI(client) => {
                client
                    .chat_with_tools_streaming(
                        messages,
                        tools,
                        on_content,
                        on_tool_calls,
                        on_reasoning,
                        on_tool_calls_partial,
                        on_content_block_complete,
                        on_usage,
                    )
                    .await
            }
            Self::OpenAICodex(client) => {
                client
                    .chat_with_tools_streaming(
                        messages,
                        tools,
                        on_content,
                        on_tool_calls,
                        on_reasoning,
                        on_tool_calls_partial,
                        on_content_block_complete,
                        on_usage,
                    )
                    .await
            }
            Self::Vertex(client) => {
                client
                    .chat_with_tools_streaming(
                        messages,
                        tools,
                        on_content,
                        on_tool_calls,
                        on_reasoning,
                        on_tool_calls_partial,
                        on_content_block_complete,
                        on_usage,
                    )
                    .await
            }
            Self::AzureOpenAI { client, .. } => {
                client
                    .chat_with_tools_streaming(
                        messages,
                        tools,
                        on_content,
                        on_tool_calls,
                        on_reasoning,
                        on_tool_calls_partial,
                        on_content_block_complete,
                        on_usage,
                    )
                    .await
            }
            Self::AzureAnthropic { client, .. } => {
                client
                    .chat_with_tools_streaming(
                        messages,
                        tools,
                        on_content,
                        on_tool_calls,
                        on_reasoning,
                        on_tool_calls_partial,
                        on_content_block_complete,
                        on_usage,
                    )
                    .await
            }
            Self::Bedrock { client, .. } => {
                client
                    .chat_with_tools_streaming(
                        messages,
                        tools,
                        on_content,
                        on_tool_calls,
                        on_reasoning,
                        on_tool_calls_partial,
                        on_content_block_complete,
                        on_usage,
                    )
                    .await
            }
        }
    }

    fn provider_name(&self) -> &str {
        match self {
            Self::OpenRouterCompletions(client) => client.provider_name(),
            Self::OpenRouterResponses(client) => client.provider_name(),
            Self::Anthropic(client) => client.provider_name(),
            Self::OpenAI(client) => client.provider_name(),
            Self::OpenAICodex(client) => client.provider_name(),
            Self::Vertex(client) => client.provider_name(),
            Self::AzureOpenAI { client, .. } => client.provider_name(),
            Self::AzureAnthropic { client, .. } => client.provider_name(),
            Self::Bedrock { client, .. } => client.provider_name(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pricing_key_normalization() {
        // Test that pricing_key returns normalized identifiers for pricing lookup
        assert_eq!(LlmProvider::Anthropic.pricing_key(), "anthropic");
        assert_eq!(LlmProvider::OpenAI.pricing_key(), "openai");
        assert_eq!(LlmProvider::OpenAICodex.pricing_key(), "openai");
        assert_eq!(LlmProvider::Vertex.pricing_key(), "vertex");

        // Both OpenRouter variants should normalize to "openrouter"
        assert_eq!(
            LlmProvider::OpenRouterCompletions.pricing_key(),
            "openrouter"
        );
        assert_eq!(LlmProvider::OpenRouterResponses.pricing_key(), "openrouter");

        // Azure OpenAI should normalize to "openai" (same pricing model)
        assert_eq!(
            LlmProvider::AzureOpenAI {
                resource_name: "test".to_string(),
                api_version: "2025-04-01".to_string(),
            }
            .pricing_key(),
            "openai"
        );

        assert_eq!(
            LlmProvider::AzureAnthropic {
                base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
                auth_method: crate::llm::anthropic::AzureAnthropicAuthMethod::XApiKey,
            }
            .pricing_key(),
            "anthropic"
        );

        // Bedrock should normalize to "anthropic" (same pricing model for Claude)
        assert_eq!(
            LlmProvider::Bedrock {
                region: "us-east-1".to_string(),
                model_id: "us.anthropic.claude-sonnet-4-5-20250514-v1:0".to_string(),
                auth_method: crate::llm::anthropic::BedrockAuthMethod::default(),
            }
            .pricing_key(),
            "anthropic"
        );
    }

    #[test]
    fn test_pricing_key_matches_pricing_store() {
        // Ensure pricing keys match the normalized provider names used by the
        // pricing store and embedded models.dev seed snapshot.
        let valid_providers = ["anthropic", "openai", "openrouter", "vertex"];

        assert!(valid_providers.contains(&LlmProvider::Anthropic.pricing_key()));
        assert!(valid_providers.contains(&LlmProvider::OpenAI.pricing_key()));
        assert!(valid_providers.contains(&LlmProvider::OpenAICodex.pricing_key()));
        assert!(valid_providers.contains(&LlmProvider::Vertex.pricing_key()));
        assert!(valid_providers.contains(&LlmProvider::OpenRouterCompletions.pricing_key()));
        assert!(valid_providers.contains(&LlmProvider::OpenRouterResponses.pricing_key()));
        assert!(valid_providers.contains(
            &LlmProvider::AzureOpenAI {
                resource_name: "test".to_string(),
                api_version: "2025-04-01".to_string(),
            }
            .pricing_key()
        ));
        assert!(valid_providers.contains(
            &LlmProvider::AzureAnthropic {
                base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
                auth_method: crate::llm::anthropic::AzureAnthropicAuthMethod::XApiKey,
            }
            .pricing_key()
        ));
        assert!(valid_providers.contains(
            &LlmProvider::Bedrock {
                region: "us-east-1".to_string(),
                model_id: "us.anthropic.claude-sonnet-4-5-20250514-v1:0".to_string(),
                auth_method: crate::llm::anthropic::BedrockAuthMethod::default(),
            }
            .pricing_key()
        ));
    }

    #[test]
    fn test_azure_openai_display() {
        let provider = LlmProvider::AzureOpenAI {
            resource_name: "my-resource".to_string(),
            api_version: "2025-04-01-preview".to_string(),
        };
        assert_eq!(
            format!("{}", provider),
            "azure-openai[my-resource@2025-04-01-preview]"
        );
    }

    #[test]
    fn test_openai_codex_display() {
        assert_eq!(LlmProvider::OpenAICodex.to_string(), "openai-codex");
    }

    #[test]
    fn test_azure_anthropic_display() {
        let provider = LlmProvider::AzureAnthropic {
            base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
            auth_method: crate::llm::anthropic::AzureAnthropicAuthMethod::BearerToken,
        };
        assert_eq!(
            format!("{}", provider),
            "azure-anthropic[https://example-resource.services.ai.azure.com/anthropic:bearer]"
        );
    }

    #[test]
    fn test_vertex_display_and_from_str() {
        assert_eq!(LlmProvider::Vertex.to_string(), "vertex");

        let parsed: LlmProvider = "vertex".parse().unwrap();
        assert_eq!(parsed, LlmProvider::Vertex);

        let parsed_alias: LlmProvider = "google-vertex".parse().unwrap();
        assert_eq!(parsed_alias, LlmProvider::Vertex);
    }

    #[test]
    fn test_openai_codex_from_str() {
        let provider: LlmProvider = "openai-codex".parse().unwrap();
        assert_eq!(provider, LlmProvider::OpenAICodex);

        let alias: LlmProvider = "codex".parse().unwrap();
        assert_eq!(alias, LlmProvider::OpenAICodex);
    }

    #[test]
    fn test_azure_openai_from_str() {
        // Note: This test uses env vars, so values may vary
        let provider: LlmProvider = "azure-openai".parse().unwrap();
        assert!(matches!(provider, LlmProvider::AzureOpenAI { .. }));
    }

    #[test]
    fn test_azure_anthropic_from_str_with_base_url() {
        let previous_base_url = std::env::var("AZURE_ANTHROPIC_BASE_URL").ok();
        let previous_resource = std::env::var("AZURE_ANTHROPIC_RESOURCE").ok();
        let previous_auth = std::env::var("AZURE_ANTHROPIC_AUTH_METHOD").ok();

        std::env::set_var(
            "AZURE_ANTHROPIC_BASE_URL",
            "https://example-resource.services.ai.azure.com/anthropic",
        );
        std::env::remove_var("AZURE_ANTHROPIC_RESOURCE");
        std::env::set_var("AZURE_ANTHROPIC_AUTH_METHOD", "bearer");

        let provider: LlmProvider = "azure-anthropic".parse().unwrap();
        assert_eq!(
            provider,
            LlmProvider::AzureAnthropic {
                base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
                auth_method: crate::llm::anthropic::AzureAnthropicAuthMethod::BearerToken,
            }
        );

        if let Some(value) = previous_base_url {
            std::env::set_var("AZURE_ANTHROPIC_BASE_URL", value);
        } else {
            std::env::remove_var("AZURE_ANTHROPIC_BASE_URL");
        }
        if let Some(value) = previous_resource {
            std::env::set_var("AZURE_ANTHROPIC_RESOURCE", value);
        } else {
            std::env::remove_var("AZURE_ANTHROPIC_RESOURCE");
        }
        if let Some(value) = previous_auth {
            std::env::set_var("AZURE_ANTHROPIC_AUTH_METHOD", value);
        } else {
            std::env::remove_var("AZURE_ANTHROPIC_AUTH_METHOD");
        }
    }

    #[test]
    fn test_bedrock_display() {
        let provider = LlmProvider::Bedrock {
            region: "us-east-1".to_string(),
            model_id: "us.anthropic.claude-sonnet-4-5-20250514-v1:0".to_string(),
            auth_method: crate::llm::anthropic::BedrockAuthMethod::SigV4,
        };
        assert_eq!(
            format!("{}", provider),
            "bedrock[us.anthropic.claude-sonnet-4-5-20250514-v1:0@us-east-1:sigv4]"
        );

        // Also test bearer token display
        let provider_bearer = LlmProvider::Bedrock {
            region: "us-west-2".to_string(),
            model_id: "us.anthropic.claude-opus-4-5-20251101-v1:0".to_string(),
            auth_method: crate::llm::anthropic::BedrockAuthMethod::BearerToken,
        };
        assert_eq!(
            format!("{}", provider_bearer),
            "bedrock[us.anthropic.claude-opus-4-5-20251101-v1:0@us-west-2:bearer]"
        );
    }

    #[test]
    fn test_bedrock_from_str() {
        // Note: This test uses env vars, so values may vary
        let provider: LlmProvider = "bedrock".parse().unwrap();
        assert!(matches!(provider, LlmProvider::Bedrock { .. }));

        // Also test alternative names
        let provider2: LlmProvider = "aws-bedrock".parse().unwrap();
        assert!(matches!(provider2, LlmProvider::Bedrock { .. }));

        let provider3: LlmProvider = "aws_bedrock".parse().unwrap();
        assert!(matches!(provider3, LlmProvider::Bedrock { .. }));
    }
}
