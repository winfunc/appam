//! Configuration for Anthropic Claude API client.
//!
//! Defines all configurable parameters for the Messages API, including
//! Anthropic-specific features like extended thinking, prompt caching,
//! and beta feature flags.

use super::types::CacheControl;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Configuration for the Anthropic Claude API client.
///
/// Contains connection details, model selection, and feature configuration.
/// API keys should be provided via environment variables and never logged.
///
/// # Examples
///
/// Basic configuration:
/// ```ignore
/// let config = AnthropicConfig {
///     model: "claude-sonnet-4-5".to_string(),
///     max_tokens: 4096,
///     ..Default::default()
/// };
/// ```
///
/// With extended thinking:
/// ```ignore
/// let config = AnthropicConfig {
///     model: "claude-sonnet-4-5".to_string(),
///     max_tokens: 16000,
///     thinking: Some(ThinkingConfig::adaptive()),  // or ThinkingConfig::enabled(10000)
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// API key for authentication (prefer env var ANTHROPIC_API_KEY)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Base URL for Anthropic API
    #[serde(default = "AnthropicConfig::default_base_url")]
    pub base_url: String,

    /// Model identifier (e.g., "claude-sonnet-4-5", "claude-opus-4-5", "claude-opus-4-1")
    ///
    /// Available models:
    /// - `claude-opus-4-5-20251101`: Latest Opus 4.5 (200K context)
    /// - `claude-sonnet-4-5-20250929`: Latest Sonnet (200K context, 1M beta)
    /// - `claude-sonnet-4-20250514`: Sonnet 4
    /// - `claude-opus-4-1-20250805`: Opus 4.1
    /// - `claude-opus-4-20250514`: Opus 4
    /// - `claude-haiku-4-5-20251001`: Latest Haiku
    /// - `claude-3-7-sonnet-20250219`: Sonnet 3.7
    #[serde(default = "AnthropicConfig::default_model")]
    pub model: String,

    /// Optional canonical Anthropic model identifier used only for pricing/accounting.
    ///
    /// This lets callers send provider-specific deployment names in `model`
    /// while still attributing usage to the canonical models.dev identifier.
    /// The primary use case is Azure-hosted Anthropic deployments whose
    /// operator-defined deployment names do not necessarily match Anthropic's
    /// public model slug.
    ///
    /// Examples:
    /// - request model: `claude-4-6-opus`
    /// - pricing model: `claude-opus-4-6`
    ///
    /// When omitted, usage accounting falls back to `model`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing_model: Option<String>,

    /// Maximum number of tokens to generate
    ///
    /// This is a STRICT limit (no auto-adjustment). Must be less than
    /// (200K context window - input tokens), or (1M with beta header).
    ///
    /// When using extended thinking, this includes the thinking budget.
    #[serde(default = "AnthropicConfig::default_max_tokens")]
    pub max_tokens: u32,

    /// Temperature for sampling (0.0-1.0)
    ///
    /// Lower values are more deterministic, higher values more creative.
    /// Default: 1.0 (Anthropic's default)
    ///
    /// Note: Incompatible with extended thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p nucleus sampling (0.0-1.0)
    ///
    /// Recommended: Use temperature OR top_p, not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling
    ///
    /// Only sample from the top K options for each token.
    /// Advanced use cases only.
    ///
    /// Note: Incompatible with extended thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Enable streaming responses
    #[serde(default = "AnthropicConfig::default_stream")]
    pub stream: bool,

    /// Custom stop sequences
    ///
    /// Model stops generating when any of these sequences is encountered.
    /// Response will have `stop_reason: "stop_sequence"`.
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Extended thinking configuration
    ///
    /// Enables Claude to show its reasoning process with configurable token budget.
    /// Requires `budget_tokens >= 1024` and `budget_tokens < max_tokens`.
    ///
    /// Incompatible with: temperature, top_k, forced tool use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Prompt caching configuration.
    ///
    /// Appam maps this high-level toggle differently depending on transport:
    ///
    /// - **Direct Anthropic / Azure Anthropic**: Uses Anthropic's top-level
    ///   `cache_control` request field so the API can place the breakpoint on
    ///   the last cacheable block automatically.
    /// - **AWS Bedrock**: Uses block-level `cache_control` checkpoints because
    ///   Bedrock's Anthropic InvokeModel integration expects cache checkpoints
    ///   inside supported `system`, `messages`, and `tools` fields.
    ///
    /// This split preserves the reference Anthropic SDK semantics on direct
    /// transports while still supporting Bedrock's provider-specific prompt
    /// caching format for compatible Claude models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caching: Option<CachingConfig>,

    /// Tool choice strategy
    ///
    /// Controls how Claude uses provided tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceConfig>,

    /// Effort level for controlling Claude's token spending.
    ///
    /// Controls how eagerly Claude spends tokens. Affects text, tool calls,
    /// and thinking. Use `Max` for deepest reasoning (Opus 4.6 only), `Low`
    /// for fast/cheap subagents. Serialized as `output_config.effort`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<EffortLevel>,

    /// Beta features
    ///
    /// Opt-in to beta APIs (fine-grained streaming, interleaved thinking, etc.)
    #[serde(default)]
    pub beta_features: BetaFeatures,

    /// Request metadata
    ///
    /// Optional metadata for tracking and analytics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<RequestMetadata>,

    /// Retry configuration for handling transient errors
    ///
    /// Enables automatic retry with exponential backoff for rate limits and overload errors.
    /// Set to `None` to disable retries completely.
    /// Default: Enabled with 5 retries, 1s initial backoff.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryConfig>,

    /// Network-level retry configuration for connection and timeout errors
    ///
    /// Handles transient network failures (connection timeouts, DNS errors) separately
    /// from API-level errors. Uses more aggressive defaults (fewer retries, faster backoff)
    /// for faster failure detection while maintaining reliability.
    /// Default: Enabled with 3 retries, 2s initial backoff.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network_retry: Option<NetworkRetryConfig>,

    /// Rate limiter configuration for proactive rate limit prevention
    ///
    /// Prevents rate limit errors by coordinating requests across all workers.
    /// Highly recommended for high-parallelism scenarios (15+ workers).
    /// Default: Disabled
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limiter: Option<RateLimiterConfig>,

    /// AWS Bedrock configuration (mutually exclusive with direct Anthropic API).
    ///
    /// When set, routes requests to AWS Bedrock endpoints with Bedrock-specific
    /// URL construction, authentication, and API versioning.
    ///
    /// This field is mutually exclusive with `azure`.
    ///
    /// Note: When using Bedrock, the `model` field is ignored in favor of
    /// `bedrock.model_id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bedrock: Option<BedrockConfig>,

    /// Azure-hosted Anthropic configuration (mutually exclusive with Bedrock).
    ///
    /// When set, routes requests through Azure-hosted Anthropic-compatible
    /// endpoints while preserving Anthropic's Messages API request and response
    /// shapes. This path reuses the same tool calling, streaming, thinking,
    /// prompt caching, retry, and usage accounting logic as the direct
    /// Anthropic transport.
    ///
    /// The `model` field continues to carry the Azure deployment name. Do not
    /// hardcode Azure host patterns here; instead provide a full `base_url`
    /// such as:
    ///
    /// - `https://example-resource.services.ai.azure.com/anthropic`
    /// - `https://example-resource.services.ai.azure.com/anthropic`
    ///
    /// This field is mutually exclusive with `bedrock`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azure: Option<AzureAnthropicConfig>,
}

impl AnthropicConfig {
    fn default_base_url() -> String {
        "https://api.anthropic.com".to_string()
    }

    fn default_model() -> String {
        "claude-sonnet-4-5".to_string()
    }

    fn default_max_tokens() -> u32 {
        4096
    }

    fn default_stream() -> bool {
        true
    }

    /// Validate configuration and emit warnings for defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Thinking budget >= max_tokens
    /// - Thinking budget < 1024
    /// - Temperature/top_k used with thinking
    /// - Incompatible beta features
    /// - Both Azure Anthropic and Bedrock are configured simultaneously
    /// - Azure Anthropic base URL cannot be normalized
    pub fn validate(&self) -> Result<()> {
        if self.bedrock.is_some() && self.azure.is_some() {
            return Err(anyhow!(
                "AnthropicConfig cannot enable both bedrock and azure transports at the same time"
            ));
        }

        if let Some(ref azure) = self.azure {
            azure.normalized_base_url()?;
        }

        // Validate thinking configuration
        if let Some(ref thinking) = self.thinking {
            if thinking.enabled {
                if thinking.budget_tokens >= self.max_tokens {
                    return Err(anyhow!(
                        "thinking.budget_tokens ({}) must be < max_tokens ({})",
                        thinking.budget_tokens,
                        self.max_tokens
                    ));
                }
                if thinking.budget_tokens < 1024 {
                    return Err(anyhow!(
                        "thinking.budget_tokens ({}) must be >= 1024",
                        thinking.budget_tokens
                    ));
                }

                // Check incompatibilities
                if self.temperature.is_some() {
                    return Err(anyhow!(
                        "Extended thinking is incompatible with temperature parameter"
                    ));
                }
                if self.top_k.is_some() {
                    return Err(anyhow!(
                        "Extended thinking is incompatible with top_k parameter"
                    ));
                }
            }
        }

        Ok(())
    }
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: Self::default_base_url(),
            model: Self::default_model(),
            pricing_model: None,
            max_tokens: Self::default_max_tokens(),
            temperature: None,
            top_p: None,
            top_k: None,
            stream: Self::default_stream(),
            stop_sequences: Vec::new(),
            thinking: None,
            caching: None,
            tool_choice: None,
            effort: None,
            beta_features: BetaFeatures::default(),
            metadata: None,
            retry: Some(RetryConfig::default()),
            network_retry: Some(NetworkRetryConfig::default()),
            rate_limiter: None,
            bedrock: None,
            azure: None,
        }
    }
}

/// Authentication method for Azure-hosted Anthropic endpoints.
///
/// Azure Anthropic deployments currently accept two authentication shapes that
/// are relevant to Appam:
///
/// - `XApiKey`: Uses the `x-api-key` header. This is the default because it
///   mirrors the Foundry SDK and works with Azure-issued API keys.
/// - `BearerToken`: Uses `Authorization: Bearer ...`. This is useful for
///   Microsoft Entra tokens and, in this repository's verified environment, the
///   same Azure key also works when supplied as a bearer token.
///
/// `api-key` is intentionally not modeled because it returned `401` during
/// live validation for the target Azure Anthropic resource.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AzureAnthropicAuthMethod {
    /// Send credentials using the `x-api-key` header.
    ///
    /// This is the default transport because it matches Anthropic's Foundry SDK
    /// and the validated Azure Anthropic behavior for API-key authentication.
    #[default]
    XApiKey,

    /// Send credentials using `Authorization: Bearer ...`.
    ///
    /// Use this for Microsoft Entra tokens or environments that explicitly
    /// require bearer authentication.
    BearerToken,
}

impl AzureAnthropicAuthMethod {
    /// Return the stable string form used in logs, docs, and display output.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(AzureAnthropicAuthMethod::XApiKey.as_str(), "x-api-key");
    /// assert_eq!(AzureAnthropicAuthMethod::BearerToken.as_str(), "bearer");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::XApiKey => "x-api-key",
            Self::BearerToken => "bearer",
        }
    }
}

impl std::str::FromStr for AzureAnthropicAuthMethod {
    type Err = anyhow::Error;

    /// Parse an Azure Anthropic auth method from common env-friendly strings.
    ///
    /// Accepted values:
    /// - `x_api_key`
    /// - `x-api-key`
    /// - `api_key`
    /// - `apikey`
    /// - `bearer`
    /// - `bearer_token`
    /// - `bearer-token`
    ///
    /// # Errors
    ///
    /// Returns an error when the string does not match a supported
    /// authentication method.
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "x_api_key" | "x-api-key" | "api_key" | "apikey" => Ok(Self::XApiKey),
            "bearer" | "bearer_token" | "bearer-token" => Ok(Self::BearerToken),
            _ => Err(anyhow!(
                "Invalid Azure Anthropic auth method: {}. Must be 'x_api_key' or 'bearer'",
                s
            )),
        }
    }
}

/// Azure Anthropic transport configuration.
///
/// This config isolates the Azure-specific transport concerns from the
/// Anthropic request body itself:
///
/// - `base_url` identifies the Azure Anthropic endpoint root
/// - `auth_method` controls whether Appam emits `x-api-key` or `Authorization`
///
/// The rest of the Anthropic settings remain shared with the direct Messages
/// API path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureAnthropicConfig {
    /// Base Azure Anthropic endpoint without the trailing `/v1/messages`.
    ///
    /// Recommended values:
    /// - `https://example-resource.services.ai.azure.com/anthropic`
    /// - `https://example-resource.services.ai.azure.com/anthropic`
    ///
    /// The normalizer accepts common user input variants such as:
    /// - a trailing slash
    /// - `/v1`
    /// - `/v1/messages`
    ///
    /// The normalized value is used to build requests as
    /// `{base_url}/v1/messages`.
    pub base_url: String,

    /// Authentication mode for Azure Anthropic requests.
    ///
    /// Default: `AzureAnthropicAuthMethod::XApiKey`
    #[serde(default)]
    pub auth_method: AzureAnthropicAuthMethod,
}

impl AzureAnthropicConfig {
    /// Build the Azure Anthropic base URL from a resource name.
    ///
    /// This helper intentionally uses the `services.ai.azure.com` host because
    /// it is the documented resource-derived format. Callers can still provide
    /// any other validated host pattern directly through `base_url`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let url = AzureAnthropicConfig::base_url_from_resource("example-resource").unwrap();
    /// assert_eq!(url, "https://example-resource.services.ai.azure.com/anthropic");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the resource string is empty after trimming.
    pub fn base_url_from_resource(resource: &str) -> Result<String> {
        let resource = resource.trim();
        if resource.is_empty() {
            return Err(anyhow!(
                "AZURE_ANTHROPIC_RESOURCE must not be empty when deriving the base URL"
            ));
        }

        Ok(format!(
            "https://{}.services.ai.azure.com/anthropic",
            resource
        ))
    }

    /// Normalize the Azure Anthropic base URL to a stable request root.
    ///
    /// This accepts user-provided values that may already include `v1` or the
    /// full `v1/messages` endpoint and strips them back to the transport root.
    ///
    /// # Returns
    ///
    /// A normalized base URL without a trailing slash.
    ///
    /// # Errors
    ///
    /// Returns an error if the URL is empty or cannot be parsed.
    pub fn normalized_base_url(&self) -> Result<String> {
        let trimmed = self.base_url.trim();
        if trimmed.is_empty() {
            return Err(anyhow!(
                "Azure Anthropic base_url must not be empty when azure transport is enabled"
            ));
        }

        let mut parsed = url::Url::parse(trimmed).map_err(|err| {
            anyhow!(
                "Invalid Azure Anthropic base_url '{}': {}",
                self.base_url,
                err
            )
        })?;

        parsed.set_query(None);
        parsed.set_fragment(None);

        let path = parsed.path().trim_end_matches('/');
        let normalized_path = if let Some(stripped) = path.strip_suffix("/v1/messages") {
            stripped
        } else if let Some(stripped) = path.strip_suffix("/v1") {
            stripped
        } else {
            path
        };

        let final_path = if normalized_path.is_empty() {
            "/".to_string()
        } else {
            normalized_path.to_string()
        };
        parsed.set_path(&final_path);

        let mut normalized = parsed.to_string();
        if normalized.ends_with('/') {
            normalized.pop();
        }

        Ok(normalized)
    }
}

impl Default for AzureAnthropicConfig {
    fn default() -> Self {
        Self {
            base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
            auth_method: AzureAnthropicAuthMethod::default(),
        }
    }
}

/// Extended thinking configuration.
///
/// Enables Claude to output its internal reasoning process before generating
/// the final answer, improving response quality for complex tasks.
///
/// # Token Budget (Legacy Models)
///
/// The `budget_tokens` parameter determines the maximum tokens Claude can use
/// for reasoning. Larger budgets enable more thorough analysis but increase
/// latency and cost.
///
/// Minimum: 1024 tokens
/// Maximum: Less than `max_tokens`
///
/// # Adaptive Thinking (Opus 4.6+)
///
/// For Claude Opus 4.6 and newer, use adaptive thinking instead of a fixed
/// budget. In adaptive mode, Claude dynamically decides when and how much to
/// think based on request complexity. Control thinking depth with the `effort`
/// parameter (see `EffortLevel`) rather than `budget_tokens`.
///
/// `budget_tokens` is deprecated on Claude Opus 4.6 and will be removed in a
/// future model release. Use `adaptive = true` with `EffortLevel` instead.
///
/// # Model Differences
///
/// - **Claude Opus 4.6**: Use adaptive thinking (recommended)
/// - **Claude Opus 4.5, Sonnet 4.5, etc.**: Use `enabled` with `budget_tokens`
/// - **Claude 3.7**: Returns full thinking output
///
/// # Incompatibilities
///
/// Extended thinking cannot be used with:
/// - `temperature` parameter
/// - `top_k` parameter
/// - Forced tool use (`tool_choice: {type: "any"}` or `{type: "tool"}`)
///
/// # Examples
///
/// ```ignore
/// // Adaptive thinking for Opus 4.6+
/// let thinking = ThinkingConfig::adaptive();
///
/// // Legacy fixed-budget thinking for older models
/// let thinking = ThinkingConfig {
///     enabled: true,
///     budget_tokens: 10000,
///     adaptive: false,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Enable extended thinking (legacy mode with fixed budget).
    ///
    /// Ignored when `adaptive` is true.
    pub enabled: bool,

    /// Maximum tokens for reasoning (min: 1024, must be < max_tokens).
    ///
    /// Only used when `adaptive` is false and `enabled` is true.
    /// Deprecated for Opus 4.6+; use adaptive thinking instead.
    pub budget_tokens: u32,

    /// Enable adaptive thinking (Opus 4.6+ recommended).
    ///
    /// When true, Claude dynamically decides when and how much to think
    /// based on request complexity. Automatically enables interleaved
    /// thinking (Claude can think between tool calls). Control thinking
    /// depth with `EffortLevel` instead of `budget_tokens`.
    ///
    /// Defaults to `false` for backward compatibility.
    #[serde(default)]
    pub adaptive: bool,
}

impl ThinkingConfig {
    /// Create an adaptive thinking config for Opus 4.6+.
    ///
    /// Adaptive thinking lets Claude dynamically decide when and how much
    /// to think. Pair with `EffortLevel` to control thinking depth.
    pub fn adaptive() -> Self {
        Self {
            enabled: false,
            budget_tokens: 0,
            adaptive: true,
        }
    }

    /// Create an enabled thinking config with the specified budget.
    ///
    /// Use for models before Opus 4.6 (Opus 4.5, Sonnet 4.5, etc.).
    pub fn enabled(budget_tokens: u32) -> Self {
        Self {
            enabled: true,
            budget_tokens,
            adaptive: false,
        }
    }

    /// Create a disabled thinking config.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            budget_tokens: 0,
            adaptive: false,
        }
    }
}

/// Prompt caching configuration.
///
/// This configuration intentionally keeps the public API transport-agnostic
/// while mapping to the provider-specific caching mechanism underneath.
///
/// For direct Anthropic and Azure Anthropic, Appam sends a single top-level
/// `cache_control` object and lets Anthropic attach the breakpoint to the last
/// cacheable block in the request.
///
/// For AWS Bedrock, Appam injects block-level `cache_control` checkpoints into
/// the supported Anthropic fields (`system`, `messages`, and `tools`) because
/// Bedrock's Anthropic InvokeModel shape expects explicit checkpoints rather
/// than Anthropic's top-level helper.
///
/// # Cache Behavior
///
/// - **Lifetime**: 5 minutes (default) or 1 hour
/// - **Minimum Size**: 1024 tokens (Sonnet/Opus), 4096 tokens (Haiku 4.5)
/// - **Prefix Matching**: Automatic prefix lookup (up to ~20 blocks)
/// - **Direct Anthropic / Azure Anthropic**: One top-level breakpoint applied
///   by Anthropic to the last cacheable block in the request
/// - **AWS Bedrock**: Automatic block-level checkpoints injected into the end
///   of the supported cacheable sections present in the request
///
/// # Transport Support
///
/// - **Direct Anthropic API**: Supported
/// - **Azure Anthropic**: Supported
/// - **AWS Bedrock**: Supported for compatible Claude models using Bedrock's
///   prompt caching checkpoint format
///
/// # Pricing
///
/// - **Cache writes**: 1.25x base input token price (5m), 2x (1h)
/// - **Cache reads**: 0.1x base input token price
/// - **Cache misses**: Regular input token price
///
/// # Examples
///
/// ```ignore
/// let caching = CachingConfig {
///     enabled: true,
///     ttl: CacheTTL::FiveMinutes,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable prompt caching
    pub enabled: bool,
    /// Cache time-to-live
    pub ttl: CacheTTL,
}

impl CachingConfig {
    /// Convert this high-level toggle into Anthropic's top-level cache control.
    ///
    /// # Purpose
    ///
    /// Anthropic's reference SDK exposes a request-level `cache_control` field
    /// that instructs the API to mark the last cacheable block in the request.
    /// Appam uses the same mechanism for its automatic caching path so that
    /// cached prompt prefixes behave consistently across requests without the
    /// client having to infer block ordering rules itself.
    ///
    /// # Returns
    ///
    /// Returns `Some(CacheControl)` when caching is enabled. Returns `None`
    /// when caching is disabled so callers can omit the field entirely.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let caching = CachingConfig::default();
    /// let marker = caching.top_level_cache_control().unwrap();
    /// assert_eq!(marker.cache_type, "ephemeral");
    /// assert_eq!(marker.ttl.as_deref(), Some("5m"));
    /// ```
    pub fn top_level_cache_control(&self) -> Option<CacheControl> {
        if !self.enabled {
            return None;
        }

        Some(CacheControl {
            cache_type: "ephemeral".to_string(),
            ttl: Some(self.ttl.as_str().to_string()),
        })
    }
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl: CacheTTL::FiveMinutes,
        }
    }
}

/// Cache time-to-live duration.
///
/// Determines how long cached content remains valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CacheTTL {
    /// 5-minute cache (default, refreshed on use)
    #[serde(rename = "5m")]
    #[default]
    FiveMinutes,
    /// 1-hour cache (2x write cost)
    #[serde(rename = "1h")]
    OneHour,
}

impl CacheTTL {
    /// Get the string representation for API requests.
    pub fn as_str(&self) -> &str {
        match self {
            Self::FiveMinutes => "5m",
            Self::OneHour => "1h",
        }
    }
}

/// Effort level for controlling Claude's token spending.
///
/// Controls how eagerly Claude spends tokens when responding to requests,
/// trading off between response thoroughness and token efficiency. Affects
/// all tokens: text responses, tool calls, and extended thinking.
///
/// # Supported Models
///
/// - **Claude Opus 4.6**: All levels including `Max`
/// - **Claude Opus 4.5**: `Low`, `Medium`, `High` (requires `effort-2025-11-24` beta on Bedrock)
///
/// # Levels
///
/// - `Max`: Absolute maximum capability, no constraints. Opus 4.6 only.
/// - `High`: Default. Deep reasoning, complex coding, agentic tasks.
/// - `Medium`: Balanced token usage. Good for agentic tasks needing speed/cost/perf balance.
/// - `Low`: Most efficient. Significant token savings for simple tasks or subagents.
///
/// # Examples
///
/// ```ignore
/// use appam::llm::anthropic::EffortLevel;
///
/// let effort = EffortLevel::Max;  // For deepest reasoning
/// let effort = EffortLevel::Low;  // For fast, cheap subagents
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EffortLevel {
    /// Most efficient: significant token savings, some capability reduction.
    Low,
    /// Balanced: moderate token savings with solid performance.
    Medium,
    /// Default: high capability, equivalent to not setting the parameter.
    #[default]
    High,
    /// Maximum capability with no constraints on token spending.
    /// Only available on Claude Opus 4.6; other models return an error.
    Max,
}

impl EffortLevel {
    /// Get the string representation for API requests.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

/// Tool choice configuration.
///
/// Controls how Claude uses the provided tools.
///
/// # Options
///
/// - **Auto**: Claude decides whether to use tools (default)
/// - **Any**: Claude must use at least one tool
/// - **Tool**: Force Claude to use a specific tool
/// - **None**: Claude cannot use any tools
///
/// # Parallel Tool Use
///
/// By default, Claude may use multiple tools in one response. Set
/// `disable_parallel` to true to force sequential tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolChoiceConfig {
    /// Let Claude decide whether to use tools (default)
    Auto {
        /// Disable parallel tool use
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    /// Claude must use at least one tool
    Any {
        /// Disable parallel tool use (forces exactly one tool)
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    /// Force Claude to use a specific tool
    Tool {
        /// Name of the tool to use
        name: String,
        /// Disable parallel tool use (forces exactly this tool)
        #[serde(default)]
        disable_parallel_tool_use: bool,
    },
    /// Claude cannot use any tools
    None,
}

impl Default for ToolChoiceConfig {
    fn default() -> Self {
        Self::Auto {
            disable_parallel_tool_use: false,
        }
    }
}

/// Beta feature flags.
///
/// Opt-in to Anthropic's beta APIs by enabling these features.
/// Each beta feature adds a corresponding header to API requests.
///
/// For direct Anthropic API: sent as `anthropic-beta` HTTP header.
/// For AWS Bedrock: sent as `anthropic_beta` array in the request body.
///
/// # Available Betas
///
/// - **fine_grained_tool_streaming**: Faster tool parameter streaming (GA on Opus 4.6)
/// - **interleaved_thinking**: Thinking between tool calls (GA on Opus 4.6 via adaptive)
/// - **context_management**: Automatic context editing
/// - **context_1m**: 1M token context window (Sonnet 4/4.5, tier 4+ only)
/// - **effort**: Effort parameter beta (required on Bedrock for Opus 4.5)
///
/// # Examples
///
/// ```ignore
/// let beta = BetaFeatures {
///     context_1m: true,
///     effort: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BetaFeatures {
    /// Enable fine-grained tool streaming (beta: fine-grained-tool-streaming-2025-05-14)
    ///
    /// Streams tool parameters without buffering/JSON validation, reducing latency.
    /// May produce invalid JSON if max_tokens is hit mid-parameter.
    /// GA on Opus 4.6 — header has no effect but is safe to include.
    #[serde(default)]
    pub fine_grained_tool_streaming: bool,

    /// Enable interleaved thinking (beta: interleaved-thinking-2025-05-14)
    ///
    /// Claude 4 models can think between tool calls for more sophisticated reasoning.
    /// GA on Opus 4.6 via adaptive thinking — header has no effect but is safe to include.
    #[serde(default)]
    pub interleaved_thinking: bool,

    /// Enable context management (beta: context-management-2025-06-27)
    ///
    /// Automatically clear old tool results when context grows beyond threshold.
    /// Compatible with Claude Sonnet 4.5 and Claude Haiku 4.5.
    #[serde(default)]
    pub context_management: bool,

    /// Enable 1M token context window (beta: context-1m-2025-08-07)
    ///
    /// Only available for Claude Sonnet 4 and 4.5, tier 4+ organizations.
    /// Requests > 200K tokens incur premium pricing (2x input, 1.5x output).
    #[serde(default)]
    pub context_1m: bool,

    /// Enable effort parameter beta (beta: effort-2025-11-24)
    ///
    /// Required on AWS Bedrock for Opus 4.5 to use the effort parameter.
    /// GA on Opus 4.6 direct API — header has no effect but is safe to include.
    #[serde(default)]
    pub effort: bool,
}

impl BetaFeatures {
    /// Get the list of beta header values to include in API requests.
    ///
    /// For direct Anthropic API: values are joined with commas in the
    /// `anthropic-beta` HTTP header.
    /// For Bedrock: values are sent as the `anthropic_beta` JSON array
    /// in the request body.
    pub fn to_header_values(&self) -> Vec<String> {
        self.to_header_values_for_model(None)
    }

    /// Get beta header values that are valid for the selected Anthropic model.
    ///
    /// # Arguments
    ///
    /// * `model` - Optional model identifier used to drop model-specific beta
    ///   flags that Anthropic rejects for other model families.
    ///
    /// # Returns
    ///
    /// A list of beta header values safe to include for `model`. Model-agnostic
    /// beta flags are preserved. `context-1m-2025-08-07` is included only for
    /// Claude Sonnet 4 and 4.5 because Anthropic rejects that beta for Opus.
    ///
    /// # Security
    ///
    /// This function does not inspect or log prompt content, credentials, or
    /// request bodies. It only filters static beta feature names from local
    /// configuration so provider requests fail closed on unsupported model
    /// feature combinations.
    pub fn to_header_values_for_model(&self, model: Option<&str>) -> Vec<String> {
        let mut values = Vec::new();

        if self.fine_grained_tool_streaming {
            values.push("fine-grained-tool-streaming-2025-05-14".to_string());
        }
        if self.interleaved_thinking {
            values.push("interleaved-thinking-2025-05-14".to_string());
        }
        if self.context_management {
            values.push("context-management-2025-06-27".to_string());
        }
        if self.context_1m && model.map(context_1m_supported_by_model).unwrap_or(true) {
            values.push("context-1m-2025-08-07".to_string());
        }
        if self.effort {
            values.push("effort-2025-11-24".to_string());
        }

        values
    }

    /// Check if any beta features are enabled.
    pub fn has_any(&self) -> bool {
        self.fine_grained_tool_streaming
            || self.interleaved_thinking
            || self.context_management
            || self.context_1m
            || self.effort
    }

    /// Returns whether any beta feature remains enabled for `model`.
    ///
    /// This is the model-aware counterpart to [`Self::has_any`]. Use it when
    /// deciding whether to emit provider beta headers for a concrete request.
    pub fn has_any_for_model(&self, model: &str) -> bool {
        !self.to_header_values_for_model(Some(model)).is_empty()
    }
}

fn context_1m_supported_by_model(model: &str) -> bool {
    let normalized = model.to_ascii_lowercase();
    if normalized.contains("claude-4-sonnet") {
        return true;
    }

    ["claude-sonnet-4-5", "claude-sonnet-4-20250514"]
        .iter()
        .any(|supported| normalized.contains(supported))
        || normalized
            .rsplit(['.', '/', ':'])
            .next()
            .is_some_and(|tail| tail == "claude-sonnet-4")
}

/// AWS Bedrock authentication method.
///
/// Determines how requests to Bedrock are authenticated.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BedrockAuthMethod {
    /// AWS SigV4 signing (default, supports streaming).
    ///
    /// Uses standard AWS credentials:
    /// - `AWS_ACCESS_KEY_ID` - AWS access key
    /// - `AWS_SECRET_ACCESS_KEY` - AWS secret key
    /// - `AWS_SESSION_TOKEN` - Optional session token for temporary credentials
    ///
    /// This method supports both streaming and non-streaming endpoints.
    #[default]
    SigV4,

    /// Bearer token authentication (non-streaming only).
    ///
    /// Uses Bedrock API Keys:
    /// - `AWS_BEARER_TOKEN_BEDROCK` - Bearer token
    ///
    /// Note: This method only works with the `/invoke` endpoint.
    /// The streaming endpoint requires SigV4 authentication.
    BearerToken,
}

/// AWS Bedrock configuration for Claude API.
///
/// When set, the Anthropic client will route requests to AWS Bedrock endpoints
/// instead of the direct Anthropic API.
///
/// # URL Format
///
/// Bedrock uses a different URL format:
/// - Non-streaming: `https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke`
/// - Streaming: `https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke-with-response-stream`
///
/// # Authentication
///
/// Two authentication methods are supported:
///
/// ## SigV4 (Default, supports streaming)
///
/// Uses standard AWS credentials from environment:
/// - `AWS_ACCESS_KEY_ID` - AWS access key ID
/// - `AWS_SECRET_ACCESS_KEY` - AWS secret access key
/// - `AWS_SESSION_TOKEN` - Optional session token for temporary credentials
///
/// ## Bearer Token (Non-streaming only)
///
/// Uses Bedrock API Keys:
/// - `AWS_BEARER_TOKEN_BEDROCK` - Bearer token
///
/// Note: Bearer tokens only work with the `/invoke` endpoint. For streaming,
/// you must use SigV4 authentication.
///
/// # Model ID Format
///
/// Bedrock uses different model identifiers with region prefix:
/// - `us.anthropic.claude-opus-4-5-20251101-v1:0`
/// - `us.anthropic.claude-sonnet-4-5-20250514-v1:0`
/// - `anthropic.claude-3-5-sonnet-20241022-v2:0`
///
/// # API Version
///
/// Bedrock requires `anthropic_version` in the request body (not as a header):
/// - Default: `bedrock-2023-05-31`
///
/// # Examples
///
/// ```ignore
/// use appam::llm::anthropic::{AnthropicConfig, BedrockConfig, BedrockAuthMethod};
///
/// // SigV4 authentication (default, supports streaming)
/// let config = AnthropicConfig {
///     bedrock: Some(BedrockConfig {
///         region: "us-east-1".to_string(),
///         model_id: "us.anthropic.claude-opus-4-5-20251101-v1:0".to_string(),
///         auth_method: BedrockAuthMethod::SigV4,
///         ..Default::default()
///     }),
///     ..Default::default()
/// };
///
/// // Bearer token authentication (non-streaming only)
/// let config = AnthropicConfig {
///     bedrock: Some(BedrockConfig {
///         region: "us-east-1".to_string(),
///         model_id: "us.anthropic.claude-opus-4-5-20251101-v1:0".to_string(),
///         auth_method: BedrockAuthMethod::BearerToken,
///         ..Default::default()
///     }),
///     stream: false, // Required for Bearer token
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockConfig {
    /// AWS region for the Bedrock endpoint (e.g., "us-east-1", "us-west-2").
    ///
    /// This determines the Bedrock runtime endpoint URL.
    /// Can also be set via `AWS_REGION` or `AWS_DEFAULT_REGION` environment variables.
    #[serde(default = "BedrockConfig::default_region")]
    pub region: String,

    /// Bedrock model identifier.
    ///
    /// Format: `{region-prefix}.anthropic.{model-name}-{version}:0`
    ///
    /// Examples:
    /// - `us.anthropic.claude-opus-4-5-20251101-v1:0`
    /// - `us.anthropic.claude-sonnet-4-5-20250514-v1:0`
    /// - `anthropic.claude-3-5-sonnet-20241022-v2:0`
    #[serde(default = "BedrockConfig::default_model_id")]
    pub model_id: String,

    /// Anthropic API version for Bedrock requests.
    ///
    /// This is sent in the request body (not as a header like direct Anthropic API).
    /// Default: `bedrock-2023-05-31`
    #[serde(default = "BedrockConfig::default_anthropic_version")]
    pub anthropic_version: String,

    /// Authentication method for Bedrock requests.
    ///
    /// - `SigV4` (default): AWS Signature Version 4, supports streaming
    /// - `BearerToken`: Bedrock API Keys, non-streaming only
    #[serde(default)]
    pub auth_method: BedrockAuthMethod,
}

impl BedrockConfig {
    fn default_region() -> String {
        std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string())
    }

    fn default_model_id() -> String {
        "us.anthropic.claude-sonnet-4-5-20250514-v1:0".to_string()
    }

    fn default_anthropic_version() -> String {
        "bedrock-2023-05-31".to_string()
    }

    /// Build the Bedrock streaming endpoint URL.
    ///
    /// Returns the URL for the invoke-with-response-stream endpoint.
    pub fn streaming_endpoint(&self) -> String {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke-with-response-stream",
            self.region,
            self.model_id.replace(':', "%3A")
        )
    }

    /// Build the Bedrock non-streaming endpoint URL.
    ///
    /// Returns the URL for the invoke endpoint.
    pub fn invoke_endpoint(&self) -> String {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke",
            self.region,
            self.model_id.replace(':', "%3A")
        )
    }
}

impl Default for BedrockConfig {
    fn default() -> Self {
        Self {
            region: Self::default_region(),
            model_id: Self::default_model_id(),
            anthropic_version: Self::default_anthropic_version(),
            auth_method: BedrockAuthMethod::default(),
        }
    }
}

/// Rate limiter configuration for proactive rate limit prevention.
///
/// Tracks actual token consumption from API responses in a sliding 60-second window
/// and blocks requests when the org-wide rate limit would be exceeded. This prevents
/// rate limit errors proactively rather than relying on reactive retry logic.
///
/// # How It Works
///
/// 1. Before each request, check if usage in last 60s is below threshold (80% of limit)
/// 2. If below threshold, proceed with request
/// 3. After response, record actual token usage (input + output from API response)
/// 4. Maintain sliding window, pruning consumptions older than 60 seconds
/// 5. If at threshold, block new requests until window clears
///
/// # Examples
///
/// Conservative (recommended for 15+ workers):
/// ```ignore
/// let limiter = RateLimiterConfig {
///     enabled: true,
///     tokens_per_minute: 1_800_000,  // 90% of 2M org limit
/// };
/// ```
///
/// Aggressive (for lower parallelism):
/// ```ignore
/// let limiter = RateLimiterConfig {
///     enabled: true,
///     tokens_per_minute: 1_950_000,  // 97.5% of 2M org limit
/// };
/// ```
///
/// Disabled (relies on retry logic only):
/// ```ignore
/// let config = AnthropicConfig {
///     rate_limiter: None,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterConfig {
    /// Enable global rate limiting
    ///
    /// When enabled, all Anthropic API requests across all agents/workers
    /// will coordinate to stay under the specified tokens per minute limit
    /// based on actual consumption tracking.
    /// Default: false (disabled)
    #[serde(default)]
    pub enabled: bool,

    /// Maximum tokens per minute (org-wide limit)
    ///
    /// Set to 90-95% of your actual org limit to leave buffer.
    /// For 2M tokens/min org limit, use 1_800_000.
    /// Default: 1_800_000
    #[serde(default = "RateLimiterConfig::default_tokens_per_minute")]
    pub tokens_per_minute: u32,
}

impl RateLimiterConfig {
    fn default_tokens_per_minute() -> u32 {
        1_800_000 // 90% of 2M limit
    }
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tokens_per_minute: Self::default_tokens_per_minute(),
        }
    }
}

/// Request metadata for tracking and analytics.
///
/// Optional metadata attached to API requests. Can include user IDs,
/// session identifiers, or other tracking information.
///
/// # Security
///
/// Never include PII (name, email, phone) in user_id. Use opaque identifiers
/// like UUIDs or hashes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// External user identifier (UUID, hash, or opaque ID)
    ///
    /// Anthropic may use this to detect abuse. Do not include PII.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// Retry configuration for handling transient API errors.
///
/// Implements exponential backoff with jitter for rate limit and overload errors.
///
/// # Retry Strategy
///
/// - **Retryable errors**: `rate_limit_error` (429), `overloaded_error` (529)
/// - **Retryable HTTP status codes**: 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout
/// - **Non-retryable errors**: All other error types fail immediately
/// - **Backoff calculation**: `min(initial_backoff * multiplier^(attempt - 1), max_backoff)`
/// - **Jitter**: Adds random variation to avoid thundering herd
///
/// # Examples
///
/// Default configuration with 5 retries:
/// ```ignore
/// let retry = RetryConfig::default();
/// assert_eq!(retry.max_retries, 5);
/// assert_eq!(retry.initial_backoff_ms, 1000);
/// ```
///
/// Custom configuration with longer backoff:
/// ```ignore
/// let retry = RetryConfig {
///     max_retries: 3,
///     initial_backoff_ms: 2000,
///     max_backoff_ms: 30000,
///     backoff_multiplier: 2.0,
///     jitter: true,
/// };
/// ```
///
/// Disable retries:
/// ```ignore
/// let config = AnthropicConfig {
///     retry: None, // No retries
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    ///
    /// After this many failed attempts, the error is returned to the caller.
    /// Default: 5
    #[serde(default = "RetryConfig::default_max_retries")]
    pub max_retries: u32,

    /// Initial backoff duration in milliseconds
    ///
    /// The first retry waits this long. Subsequent retries use exponential backoff.
    /// Default: 1000ms (1 second)
    #[serde(default = "RetryConfig::default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,

    /// Maximum backoff duration in milliseconds
    ///
    /// Backoff duration is capped at this value to prevent excessive wait times.
    /// Default: 60000ms (60 seconds)
    #[serde(default = "RetryConfig::default_max_backoff_ms")]
    pub max_backoff_ms: u64,

    /// Backoff multiplier for exponential growth
    ///
    /// Each retry multiplies the previous backoff by this factor.
    /// Default: 2.0 (doubles each time)
    #[serde(default = "RetryConfig::default_backoff_multiplier")]
    pub backoff_multiplier: f32,

    /// Enable random jitter to avoid thundering herd
    ///
    /// Adds ±50% random variation to backoff duration for better distribution
    /// of parallel workers (critical for 20+ workers hitting rate limits).
    /// Default: true
    #[serde(default = "RetryConfig::default_jitter")]
    pub jitter: bool,
}

impl RetryConfig {
    fn default_max_retries() -> u32 {
        5
    }

    fn default_initial_backoff_ms() -> u64 {
        1000
    }

    fn default_max_backoff_ms() -> u64 {
        60000
    }

    fn default_backoff_multiplier() -> f32 {
        2.0
    }

    fn default_jitter() -> bool {
        true
    }

    /// Calculate backoff duration for a given attempt number.
    ///
    /// # Arguments
    ///
    /// * `attempt` - Retry attempt number (1-indexed)
    ///
    /// # Returns
    ///
    /// Backoff duration in milliseconds, with optional jitter applied.
    ///
    /// # Formula
    ///
    /// ```text
    /// base_backoff = min(initial_backoff * multiplier^(attempt - 1), max_backoff)
    /// with_jitter = base_backoff * (1.0 + random(-0.5, 0.5))
    /// ```
    pub fn calculate_backoff(&self, attempt: u32) -> u64 {
        // Calculate exponential backoff: initial * multiplier^(attempt - 1)
        let exponent = (attempt as f32 - 1.0).max(0.0);
        let base_backoff =
            (self.initial_backoff_ms as f32) * self.backoff_multiplier.powf(exponent);

        // Cap at max_backoff
        let capped_backoff = base_backoff.min(self.max_backoff_ms as f32);

        // Apply jitter if enabled
        // For high-parallelism scenarios, we use ±50% jitter instead of ±10%
        // to better distribute 20+ workers and avoid thundering herd
        if self.jitter {
            let jitter_factor = 1.0 + rand::random_range(-0.5..0.5); // ±50%
            (capped_backoff * jitter_factor) as u64
        } else {
            capped_backoff as u64
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: Self::default_max_retries(),
            initial_backoff_ms: Self::default_initial_backoff_ms(),
            max_backoff_ms: Self::default_max_backoff_ms(),
            backoff_multiplier: Self::default_backoff_multiplier(),
            jitter: Self::default_jitter(),
        }
    }
}

/// Network-level retry configuration for connection and timeout errors.
///
/// Separate from API retry config to allow faster failure detection for
/// network issues while maintaining generous retries for API rate limits.
///
/// # Default Strategy
///
/// - Fewer retries (3 vs 8 for API errors)
/// - Faster backoff (2s initial vs 5s for API errors)
/// - Lower max backoff (30s vs 120s for API errors)
/// - Jitter enabled by default for parallel workers
///
/// # Examples
///
/// ```rust
/// use appam::llm::anthropic::config::NetworkRetryConfig;
///
/// let config = NetworkRetryConfig {
///     max_retries: 3,
///     initial_backoff_ms: 2000,
///     max_backoff_ms: 30000,
///     backoff_multiplier: 2.0,
///     jitter: true,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRetryConfig {
    /// Maximum number of retry attempts for network errors
    ///
    /// Default: 3 (fewer than API retries for faster failure)
    #[serde(default = "NetworkRetryConfig::default_max_retries")]
    pub max_retries: u32,

    /// Initial backoff duration in milliseconds for network errors
    ///
    /// Default: 2000ms (2 seconds, faster than API retries)
    #[serde(default = "NetworkRetryConfig::default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,

    /// Maximum backoff duration in milliseconds for network errors
    ///
    /// Default: 30000ms (30 seconds, lower than API retries)
    #[serde(default = "NetworkRetryConfig::default_max_backoff_ms")]
    pub max_backoff_ms: u64,

    /// Backoff multiplier for exponential growth
    ///
    /// Default: 2.0 (doubles each time)
    #[serde(default = "NetworkRetryConfig::default_backoff_multiplier")]
    pub backoff_multiplier: f32,

    /// Enable random jitter to avoid thundering herd
    ///
    /// Default: true
    #[serde(default = "NetworkRetryConfig::default_jitter")]
    pub jitter: bool,
}

impl NetworkRetryConfig {
    fn default_max_retries() -> u32 {
        3
    }

    fn default_initial_backoff_ms() -> u64 {
        2000
    }

    fn default_max_backoff_ms() -> u64 {
        30000
    }

    fn default_backoff_multiplier() -> f32 {
        2.0
    }

    fn default_jitter() -> bool {
        true
    }

    /// Calculate backoff duration for a given network retry attempt.
    ///
    /// Uses same exponential backoff algorithm as RetryConfig but with
    /// more aggressive defaults.
    ///
    /// # Arguments
    ///
    /// * `attempt` - Retry attempt number (1-indexed)
    ///
    /// # Returns
    ///
    /// Backoff duration in milliseconds, with optional jitter applied.
    pub fn calculate_backoff(&self, attempt: u32) -> u64 {
        let exponent = (attempt as f32 - 1.0).max(0.0);
        let base_backoff =
            (self.initial_backoff_ms as f32) * self.backoff_multiplier.powf(exponent);
        let capped_backoff = base_backoff.min(self.max_backoff_ms as f32);

        if self.jitter {
            let jitter_factor = 1.0 + rand::random_range(-0.5..0.5);
            (capped_backoff * jitter_factor) as u64
        } else {
            capped_backoff as u64
        }
    }
}

impl Default for NetworkRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: Self::default_max_retries(),
            initial_backoff_ms: Self::default_initial_backoff_ms(),
            max_backoff_ms: Self::default_max_backoff_ms(),
            backoff_multiplier: Self::default_backoff_multiplier(),
            jitter: Self::default_jitter(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AnthropicConfig::default();
        assert_eq!(config.model, "claude-sonnet-4-5");
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.base_url, "https://api.anthropic.com");
        assert!(config.stream);
        assert!(config.azure.is_none());
    }

    #[test]
    fn test_thinking_validation_budget_too_high() {
        let config = AnthropicConfig {
            max_tokens: 5000,
            thinking: Some(ThinkingConfig {
                enabled: true,
                budget_tokens: 6000, // > max_tokens
                adaptive: false,
            }),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_thinking_validation_budget_too_low() {
        let config = AnthropicConfig {
            thinking: Some(ThinkingConfig {
                enabled: true,
                budget_tokens: 512, // < 1024
                adaptive: false,
            }),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_thinking_incompatible_with_temperature() {
        let config = AnthropicConfig {
            thinking: Some(ThinkingConfig {
                enabled: true,
                budget_tokens: 2000,
                adaptive: false,
            }),
            temperature: Some(0.5),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_beta_features_headers() {
        let beta = BetaFeatures {
            fine_grained_tool_streaming: true,
            interleaved_thinking: true,
            ..Default::default()
        };

        let headers = beta.to_header_values();
        assert_eq!(headers.len(), 2);
        assert!(headers.contains(&"fine-grained-tool-streaming-2025-05-14".to_string()));
        assert!(headers.contains(&"interleaved-thinking-2025-05-14".to_string()));
    }

    #[test]
    fn beta_features_omit_1m_context_for_opus_models() {
        let beta = BetaFeatures {
            context_1m: true,
            fine_grained_tool_streaming: true,
            ..Default::default()
        };

        let headers = beta.to_header_values_for_model(Some("claude-opus-4-7"));

        assert!(headers.contains(&"fine-grained-tool-streaming-2025-05-14".to_string()));
        assert!(!headers.contains(&"context-1m-2025-08-07".to_string()));
        assert!(beta.has_any_for_model("claude-opus-4-7"));
    }

    #[test]
    fn beta_features_keep_1m_context_for_supported_sonnet_models() {
        let beta = BetaFeatures {
            context_1m: true,
            ..Default::default()
        };

        let sonnet_4_headers = beta.to_header_values_for_model(Some("claude-sonnet-4-20250514"));
        let sonnet_45_headers = beta.to_header_values_for_model(Some("claude-sonnet-4-5"));

        assert!(sonnet_4_headers.contains(&"context-1m-2025-08-07".to_string()));
        assert!(sonnet_45_headers.contains(&"context-1m-2025-08-07".to_string()));
        assert!(beta.has_any_for_model("claude-sonnet-4-5"));
    }

    #[test]
    fn beta_features_do_not_treat_sonnet_46_as_1m_context_supported() {
        let beta = BetaFeatures {
            context_1m: true,
            ..Default::default()
        };

        let headers = beta.to_header_values_for_model(Some("claude-sonnet-4-6"));

        assert!(headers.is_empty());
        assert!(!beta.has_any_for_model("claude-sonnet-4-6"));
    }

    #[test]
    fn test_cache_ttl_as_str() {
        assert_eq!(CacheTTL::FiveMinutes.as_str(), "5m");
        assert_eq!(CacheTTL::OneHour.as_str(), "1h");
    }

    #[test]
    fn test_retry_config_defaults() {
        let retry = RetryConfig::default();
        assert_eq!(retry.max_retries, 5);
        assert_eq!(retry.initial_backoff_ms, 1000);
        assert_eq!(retry.max_backoff_ms, 60000);
        assert_eq!(retry.backoff_multiplier, 2.0);
        assert!(retry.jitter);
    }

    #[test]
    fn test_retry_config_backoff_calculation() {
        let retry = RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 1000,
            max_backoff_ms: 60000,
            backoff_multiplier: 2.0,
            jitter: false, // Disable jitter for deterministic testing
        };

        // Attempt 1: 1000 * 2^0 = 1000ms
        assert_eq!(retry.calculate_backoff(1), 1000);

        // Attempt 2: 1000 * 2^1 = 2000ms
        assert_eq!(retry.calculate_backoff(2), 2000);

        // Attempt 3: 1000 * 2^2 = 4000ms
        assert_eq!(retry.calculate_backoff(3), 4000);

        // Attempt 4: 1000 * 2^3 = 8000ms
        assert_eq!(retry.calculate_backoff(4), 8000);

        // Attempt 5: 1000 * 2^4 = 16000ms
        assert_eq!(retry.calculate_backoff(5), 16000);
    }

    #[test]
    fn test_retry_config_backoff_capped_at_max() {
        let retry = RetryConfig {
            max_retries: 10,
            initial_backoff_ms: 1000,
            max_backoff_ms: 10000, // Cap at 10 seconds
            backoff_multiplier: 2.0,
            jitter: false,
        };

        // Attempt 5: 1000 * 2^4 = 16000ms, but capped at 10000ms
        assert_eq!(retry.calculate_backoff(5), 10000);

        // Attempt 10: Would be huge, but still capped at 10000ms
        assert_eq!(retry.calculate_backoff(10), 10000);
    }

    #[test]
    fn test_retry_config_with_jitter() {
        let retry = RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 1000,
            max_backoff_ms: 60000,
            backoff_multiplier: 2.0,
            jitter: true,
        };

        // With ±50% jitter, result should be within that range
        let backoff = retry.calculate_backoff(1);
        assert!(
            (500..=1500).contains(&backoff),
            "Backoff with jitter out of range: {}",
            backoff
        );

        // Multiple calls should produce different values with jitter
        let backoff1 = retry.calculate_backoff(2);
        let backoff2 = retry.calculate_backoff(2);
        // They might be the same due to randomness, but check range (2000 ± 50%)
        assert!(
            (1000..=3000).contains(&backoff1),
            "Backoff1 out of range: {}",
            backoff1
        );
        assert!(
            (1000..=3000).contains(&backoff2),
            "Backoff2 out of range: {}",
            backoff2
        );
    }

    #[test]
    fn test_anthropic_config_with_retry() {
        let config = AnthropicConfig::default();
        assert!(config.retry.is_some());

        let retry = config.retry.unwrap();
        assert_eq!(retry.max_retries, 5);
    }

    #[test]
    fn test_anthropic_config_without_retry() {
        let config = AnthropicConfig {
            retry: None,
            ..Default::default()
        };
        assert!(config.retry.is_none());
    }

    #[test]
    fn test_azure_anthropic_auth_method_parsing() {
        assert_eq!(
            "x_api_key".parse::<AzureAnthropicAuthMethod>().unwrap(),
            AzureAnthropicAuthMethod::XApiKey
        );
        assert_eq!(
            "x-api-key".parse::<AzureAnthropicAuthMethod>().unwrap(),
            AzureAnthropicAuthMethod::XApiKey
        );
        assert_eq!(
            "bearer".parse::<AzureAnthropicAuthMethod>().unwrap(),
            AzureAnthropicAuthMethod::BearerToken
        );
        assert_eq!(
            "bearer_token".parse::<AzureAnthropicAuthMethod>().unwrap(),
            AzureAnthropicAuthMethod::BearerToken
        );
        assert!("invalid".parse::<AzureAnthropicAuthMethod>().is_err());
    }

    #[test]
    fn test_azure_anthropic_base_url_from_resource() {
        let derived = AzureAnthropicConfig::base_url_from_resource("example-resource").unwrap();
        assert_eq!(
            derived,
            "https://example-resource.services.ai.azure.com/anthropic"
        );
    }

    #[test]
    fn test_azure_anthropic_base_url_normalization() {
        let config = AzureAnthropicConfig {
            base_url: "https://example-resource.services.ai.azure.com/anthropic/v1/messages/"
                .to_string(),
            auth_method: AzureAnthropicAuthMethod::XApiKey,
        };

        assert_eq!(
            config.normalized_base_url().unwrap(),
            "https://example-resource.services.ai.azure.com/anthropic"
        );
    }

    #[test]
    fn test_validate_rejects_bedrock_and_azure_together() {
        let config = AnthropicConfig {
            bedrock: Some(BedrockConfig::default()),
            azure: Some(AzureAnthropicConfig {
                base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
                auth_method: AzureAnthropicAuthMethod::XApiKey,
            }),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_caching_config_top_level_cache_control() {
        let marker = CachingConfig::default()
            .top_level_cache_control()
            .expect("enabled caching should produce a marker");

        assert_eq!(marker.cache_type, "ephemeral");
        assert_eq!(marker.ttl.as_deref(), Some("5m"));
    }
}
