//! Configuration for OpenAI Responses API client.
//!
//! Defines all configurable parameters for the OpenAI Responses API, including
//! reasoning configuration, structured outputs, service tiers, and conversation management.
//!
//! # Azure OpenAI Support
//!
//! This module also supports Azure OpenAI endpoints. When `azure` configuration is provided,
//! the client will:
//! - Use Azure-specific URL format: `https://{resource_name}.cognitiveservices.azure.com/openai/responses?api-version={api_version}`
//! - Use `api-key` header instead of `Authorization: Bearer` header
//! - Read API key from `AZURE_OPENAI_API_KEY` environment variable (fallback to `OPENAI_API_KEY`)

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Azure-specific configuration for OpenAI Responses API.
///
/// When set, the OpenAI client will route requests to Azure OpenAI endpoints
/// instead of the standard OpenAI API.
///
/// # URL Format
///
/// Azure OpenAI uses a different URL format:
/// `https://{resource_name}.cognitiveservices.azure.com/openai/responses?api-version={api_version}`
///
/// # Authentication
///
/// Azure uses the `api-key` header instead of `Authorization: Bearer`:
/// - Primary: `AZURE_OPENAI_API_KEY` environment variable
/// - Fallback: `OPENAI_API_KEY` environment variable
/// - Override: `config.api_key` field
///
/// # Examples
///
/// ```rust
/// use appam::llm::openai::{OpenAIConfig, AzureConfig};
///
/// let config = OpenAIConfig {
///     azure: Some(AzureConfig {
///         resource_name: "my-azure-resource".to_string(),
///         api_version: "2025-04-01-preview".to_string(),
///     }),
///     model: "gpt-5.1-codex".to_string(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureConfig {
    /// Azure resource name (the subdomain in your Azure OpenAI endpoint).
    ///
    /// For endpoint `https://example-resource.services.ai.azure.com/`,
    /// the resource name is `example-resource`.
    pub resource_name: String,

    /// API version string for Azure OpenAI.
    ///
    /// Common values:
    /// - `2025-04-01-preview`: Latest preview with Responses API support
    /// - `2024-12-01-preview`: Previous preview version
    ///
    /// Check Azure documentation for the latest supported versions.
    pub api_version: String,
}

/// Configuration for the OpenAI Responses API client.
///
/// Contains connection details, model selection, and feature configuration.
/// API keys should be provided via environment variables and never logged.
///
/// # Examples
///
/// Basic configuration:
/// ```rust
/// # use appam::llm::openai::OpenAIConfig;
/// let config = OpenAIConfig {
///     model: "gpt-5.5".to_string(),
///     max_output_tokens: Some(4096),
///     ..Default::default()
/// };
/// ```
///
/// With reasoning:
/// ```rust
/// # use appam::llm::openai::{OpenAIConfig, ReasoningConfig, ReasoningEffort, ReasoningSummary};
/// let config = OpenAIConfig {
///     model: "gpt-5.5".to_string(),
///     reasoning: Some(ReasoningConfig {
///         effort: Some(ReasoningEffort::High),
///         summary: Some(ReasoningSummary::Detailed),
///     }),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// API key for authentication (prefer env var OPENAI_API_KEY)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Base URL for OpenAI API
    #[serde(default = "OpenAIConfig::default_base_url")]
    pub base_url: String,

    /// Organization ID (optional, for multi-org accounts)
    #[serde(default)]
    pub organization: Option<String>,

    /// Project ID (optional)
    #[serde(default)]
    pub project: Option<String>,

    /// Model identifier (e.g., "gpt-4o", "o3-mini", "gpt-5")
    ///
    /// Available models:
    /// - `gpt-4o`: Latest GPT-4 Optimized (128K context)
    /// - `gpt-4o-2024-08-06`: Structured outputs support
    /// - `o3-mini`: Fast reasoning model
    /// - `o3`: Advanced reasoning model
    /// - `gpt-5`: Experimental (if available)
    #[serde(default = "OpenAIConfig::default_model")]
    pub model: String,

    /// Optional canonical model identifier used only for pricing/accounting.
    ///
    /// This allows callers to route requests to deployment-specific model names
    /// while still billing against a canonical model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing_model: Option<String>,

    /// Maximum output tokens
    #[serde(default)]
    pub max_output_tokens: Option<i32>,

    /// Whether provider-side parallel tool batching should be enabled.
    ///
    /// Appam keeps this disabled by default and only turns it on when the
    /// owning agent explicitly opts in.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    /// Temperature for sampling (0.0-2.0)
    ///
    /// Lower values are more deterministic, higher values more creative.
    /// Default: 1.0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p nucleus sampling (0.0-1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Enable streaming responses
    #[serde(default = "OpenAIConfig::default_stream")]
    pub stream: bool,

    /// Reasoning configuration (for o-series models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    /// Text output format (plain text or structured JSON)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_format: Option<TextFormatConfig>,

    /// Text verbosity level (low, medium, high)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_verbosity: Option<TextVerbosity>,

    /// Service tier selection
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    /// Conversation management
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationConfig>,

    /// Retry configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryConfig>,

    /// Store responses for later retrieval
    #[serde(default)]
    pub store: Option<bool>,

    /// Background processing mode
    #[serde(default)]
    pub background: Option<bool>,

    /// Metadata (max 16 key-value pairs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,

    /// Prompt caching key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,

    /// Safety identifier for user tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,

    /// Top logprobs (0-20)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<i32>,

    /// Azure-specific configuration (mutually exclusive with direct OpenAI).
    ///
    /// When set, routes requests to Azure OpenAI endpoints with Azure-specific
    /// URL construction and authentication.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azure: Option<AzureConfig>,
}

impl OpenAIConfig {
    fn default_base_url() -> String {
        "https://api.openai.com/v1".to_string()
    }

    fn default_model() -> String {
        "gpt-5.5".to_string()
    }

    fn default_stream() -> bool {
        true
    }

    /// Validate configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid.
    pub fn validate(&self) -> Result<()> {
        if let Some(temperature) = self.temperature {
            if !(0.0..=2.0).contains(&temperature) {
                bail!(
                    "OpenAI temperature must be between 0.0 and 2.0, got {}",
                    temperature
                );
            }
        }

        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                bail!("OpenAI top_p must be between 0.0 and 1.0, got {}", top_p);
            }
        }

        if let Some(top_logprobs) = self.top_logprobs {
            if !(0..=20).contains(&top_logprobs) {
                bail!(
                    "OpenAI top_logprobs must be between 0 and 20, got {}",
                    top_logprobs
                );
            }
        }

        if self.azure.is_none() && matches!(self.service_tier, Some(ServiceTier::Scale)) {
            bail!(
                "OpenAI service_tier = \"scale\" is no longer supported; use auto, default, flex, or priority"
            );
        }

        let normalized_model = normalize_openai_model(&self.model);
        let requested_effort = self
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort);
        let effective_effort = self
            .reasoning
            .as_ref()
            .map(|_| resolve_reasoning_effort_for_model(&normalized_model, requested_effort));

        if let Some(reasoning) = &self.reasoning {
            if matches!(effective_effort, Some(ReasoningEffort::None))
                && reasoning.summary.is_some()
            {
                bail!(
                    "OpenAI reasoning summaries are unavailable when reasoning.effort is \"none\""
                );
            }

            if let Some(effort) = reasoning.effort {
                if effort == ReasoningEffort::None
                    && !model_supports_none_reasoning(&normalized_model)
                {
                    bail!(
                        "Model {} does not support reasoning.effort = \"none\"",
                        normalized_model
                    );
                }

                if effort == ReasoningEffort::XHigh
                    && !model_supports_xhigh_reasoning(&normalized_model)
                {
                    bail!(
                        "Model {} does not support reasoning.effort = \"xhigh\"",
                        normalized_model
                    );
                }

                if model_requires_high_reasoning(&normalized_model)
                    && effort != ReasoningEffort::High
                {
                    bail!(
                        "Model {} only supports reasoning.effort = \"high\"",
                        normalized_model
                    );
                }
            }
        }

        if (self.temperature.is_some() || self.top_p.is_some() || self.top_logprobs.is_some())
            && !model_supports_sampling_parameters(&normalized_model, requested_effort)
        {
            bail!(
                "Model {} only supports temperature, top_p, and logprobs when sampling is enabled. \
                 GPT-5.5 requires reasoning.effort = \"none\", while older GPT-5/o-series models reject these fields.",
                normalized_model
            );
        }

        Ok(())
    }
}

/// Normalize OpenAI model identifiers for direct OpenAI and Azure usage.
///
/// # Purpose
///
/// Appam allows provider-prefixed model names such as `openai/gpt-5.5` in shared
/// agent configuration, but the OpenAI Responses API expects the bare model name.
/// This helper strips only the provider prefix and preserves the exact model
/// identifier so distinct models like `gpt-5.5` and `gpt-5.5-pro` are not
/// conflated.
///
/// # Examples
///
/// ```rust
/// # use appam::llm::openai::normalize_openai_model;
/// assert_eq!(normalize_openai_model("openai/gpt-5.5"), "gpt-5.5");
/// assert_eq!(normalize_openai_model("openai/gpt-5.5-pro"), "gpt-5.5-pro");
/// ```
pub fn normalize_openai_model(model: &str) -> String {
    model
        .strip_prefix("openai/")
        .unwrap_or(model)
        .trim()
        .to_string()
}

fn is_gpt_55_pro_model(model: &str) -> bool {
    let model = normalize_openai_model(model);
    model == "gpt-5.5-pro" || model.starts_with("gpt-5.5-pro-")
}

fn is_gpt_55_model(model: &str) -> bool {
    let model = normalize_openai_model(model);
    model == "gpt-5.5" || model.starts_with("gpt-5.5-") && !is_gpt_55_pro_model(&model)
}

fn is_gpt_52_default_none_model(model: &str) -> bool {
    let model = normalize_openai_model(model);
    model == "gpt-5.2" || model.starts_with("gpt-5.2-20")
}

/// Returns whether a model supports `reasoning.effort = "none"`.
///
/// GPT-5.5 defaults to `none` and accepts temperature/top-p/logprobs only in
/// that mode. GPT-5.2 retains the same compatibility rules for the direct model
/// alias, while other GPT-5 variants use older reasoning behavior.
pub fn model_supports_none_reasoning(model: &str) -> bool {
    is_gpt_55_model(model) || is_gpt_52_default_none_model(model)
}

fn model_requires_high_reasoning(model: &str) -> bool {
    let model = normalize_openai_model(model);
    model == "gpt-5-pro" || model.starts_with("gpt-5-pro-")
}

fn model_defaults_to_high_reasoning(model: &str) -> bool {
    is_gpt_55_pro_model(model) || model_requires_high_reasoning(model)
}

/// Returns whether sampling-oriented parameters can be sent for a model.
///
/// OpenAI's current GPT-5.5 compatibility rules only allow `temperature`,
/// `top_p`, and `top_logprobs` when GPT-5.5 is running with
/// `reasoning.effort = "none"` (explicitly or via the model default). Older
/// GPT-5 family models and o-series reasoning models reject these fields.
pub fn model_supports_sampling_parameters(
    model: &str,
    requested_effort: Option<ReasoningEffort>,
) -> bool {
    let normalized = normalize_openai_model(model);

    if model_supports_none_reasoning(&normalized) {
        return matches!(
            requested_effort.unwrap_or(ReasoningEffort::None),
            ReasoningEffort::None
        );
    }

    if normalized.starts_with("gpt-5")
        || normalized.starts_with("o1")
        || normalized.starts_with("o3")
        || normalized.starts_with("o4")
    {
        return false;
    }

    true
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: Self::default_base_url(),
            organization: None,
            project: None,
            model: Self::default_model(),
            pricing_model: None,
            max_output_tokens: Some(4096),
            parallel_tool_calls: Some(false),
            temperature: None,
            top_p: None,
            stream: Self::default_stream(),
            reasoning: None,
            text_format: None,
            text_verbosity: None,
            service_tier: None,
            conversation: None,
            retry: Some(RetryConfig::default()),
            store: None,
            background: None,
            metadata: None,
            prompt_cache_key: None,
            safety_identifier: None,
            top_logprobs: None,
            azure: None,
        }
    }
}

/// Reasoning configuration for o-series and gpt-5 models.
///
/// Enables enhanced reasoning with configurable effort levels and summary generation.
/// Reasoning models "think before they answer", producing an internal chain of thought
/// before responding to the user.
///
/// # Effort Levels
///
/// - `Low`: Minimal reasoning, faster responses, lower token usage
/// - `Medium`: Balanced reasoning (default), good for most tasks
/// - `High`: Deep reasoning for complex problems, slower but more thorough
///
/// # Summary Verbosity
///
/// - `Auto`: Automatically select appropriate verbosity (default)
/// - `Concise`: Brief summaries of reasoning process
/// - `Detailed`: Comprehensive summaries with full reasoning breakdown
///
/// # Token Budget
///
/// Reserve at least 25,000 tokens for reasoning and outputs when starting.
/// Use `max_output_tokens` to control the total token budget (reasoning + output).
///
/// # Examples
///
/// Basic configuration:
/// ```rust
/// # use appam::llm::openai::{ReasoningConfig, ReasoningEffort, ReasoningSummary};
/// let reasoning = ReasoningConfig {
///     effort: Some(ReasoningEffort::High),
///     summary: Some(ReasoningSummary::Detailed),
/// };
/// ```
///
/// Using builder methods:
/// ```rust
/// # use appam::llm::openai::ReasoningConfig;
/// let reasoning = ReasoningConfig::high_effort();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Reasoning effort level (low, medium, high)
    ///
    /// Controls how many reasoning tokens the model generates before responding.
    /// Higher effort levels produce more thorough reasoning but consume more tokens
    /// and take longer to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,

    /// Summary verbosity (auto, concise, detailed)
    ///
    /// Controls the level of detail in the reasoning summary returned with the response.
    /// Note: Reasoning summaries may require organization verification for some models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            effort: Some(ReasoningEffort::Medium),
            summary: Some(ReasoningSummary::Auto),
        }
    }
}

impl ReasoningConfig {
    /// Create a configuration for automatic reasoning with default settings.
    ///
    /// Uses medium effort and auto summary verbosity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use appam::llm::openai::ReasoningConfig;
    /// let reasoning = ReasoningConfig::auto();
    /// ```
    pub fn auto() -> Self {
        Self {
            effort: Some(ReasoningEffort::Medium),
            summary: Some(ReasoningSummary::Auto),
        }
    }

    /// Create a high-effort reasoning configuration with detailed summaries.
    ///
    /// Best for complex problems requiring deep analysis. Slower and more expensive,
    /// but produces more thorough results.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use appam::llm::openai::ReasoningConfig;
    /// let reasoning = ReasoningConfig::high_effort();
    /// ```
    pub fn high_effort() -> Self {
        Self {
            effort: Some(ReasoningEffort::High),
            summary: Some(ReasoningSummary::Detailed),
        }
    }

    /// Create an extra-high-effort reasoning configuration with detailed summaries.
    ///
    /// Maximum reasoning effort for the most complex problems. Only supported by
    /// selected models such as GPT-5.5 and legacy codex variants that expose
    /// `xhigh`. Provides the deepest analysis at significantly higher token cost.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use appam::llm::openai::ReasoningConfig;
    /// let reasoning = ReasoningConfig::xhigh_effort();
    /// ```
    pub fn xhigh_effort() -> Self {
        Self {
            effort: Some(ReasoningEffort::XHigh),
            summary: Some(ReasoningSummary::Detailed),
        }
    }

    /// Create a configuration that keeps GPT-5.5 in its lowest-latency
    /// `reasoning.effort = "none"` mode.
    ///
    /// This is the mode required when sending `temperature`, `top_p`, or
    /// `top_logprobs` to GPT-5.5.
    pub fn no_reasoning() -> Self {
        Self {
            effort: Some(ReasoningEffort::None),
            summary: None,
        }
    }

    /// Create a low-latency reasoning configuration with concise summaries.
    ///
    /// Optimized for speed with minimal reasoning overhead. Best for simpler tasks
    /// where fast response time is more important than deep analysis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use appam::llm::openai::ReasoningConfig;
    /// let reasoning = ReasoningConfig::low_latency();
    /// ```
    pub fn low_latency() -> Self {
        Self {
            effort: Some(ReasoningEffort::Low),
            summary: Some(ReasoningSummary::Concise),
        }
    }

    /// Create a minimal-effort reasoning configuration.
    ///
    /// This is the lowest non-zero reasoning mode exposed by the current
    /// OpenAI Responses API and is useful when you want some deliberate
    /// reasoning without paying the latency cost of `low` or above.
    pub fn minimal() -> Self {
        Self {
            effort: Some(ReasoningEffort::Minimal),
            summary: Some(ReasoningSummary::Concise),
        }
    }

    /// Create a configuration with custom effort and summary settings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use appam::llm::openai::{ReasoningConfig, ReasoningEffort, ReasoningSummary};
    /// let reasoning = ReasoningConfig::custom(ReasoningEffort::High, ReasoningSummary::Concise);
    /// ```
    pub fn custom(effort: ReasoningEffort, summary: ReasoningSummary) -> Self {
        Self {
            effort: Some(effort),
            summary: Some(summary),
        }
    }
}

/// Reasoning effort level for o-series and gpt-5 models.
///
/// Controls how much computational effort the model puts into reasoning.
/// Higher effort levels generate more reasoning tokens and take longer,
/// but often produce better results for complex tasks.
///
/// # API Mapping
///
/// Serializes to lowercase strings: "none", "minimal", "low", "medium",
/// "high", "xhigh"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disable deliberate reasoning for the lowest-latency GPT-5.5 mode.
    ///
    /// This is also the compatibility mode required when using sampling-oriented
    /// parameters such as `temperature`, `top_p`, or `top_logprobs`.
    None,
    /// Minimal reasoning for very low-latency deliberation.
    ///
    /// Useful when you want the model to spend a small amount of reasoning
    /// effort without fully disabling reasoning.
    Minimal,
    /// Light reasoning for simple tasks
    ///
    /// Fastest option with minimal token usage. Suitable for straightforward
    /// questions that don't require deep analysis.
    Low,
    /// Balanced reasoning (default)
    ///
    /// Good balance between speed and thoroughness. Suitable for most tasks.
    #[default]
    Medium,
    /// Deep reasoning for complex problems
    ///
    /// Most thorough option with highest token usage. Best for complex problems
    /// requiring careful analysis, multi-step planning, or detailed code generation.
    High,
    /// Extra-high reasoning for the most complex problems
    ///
    /// Maximum reasoning effort available. Supported by GPT-5.5 and selected
    /// legacy codex models. Provides the deepest analysis at the cost of
    /// significantly more tokens and longer generation time.
    XHigh,
}

/// Get the default reasoning effort level for a given model.
///
/// Returns the recommended reasoning effort based on the model's capabilities:
/// - `gpt-5.5`: None (current default mode)
/// - `gpt-5.5-pro`: High
/// - `gpt-5-pro`: High (legacy high-only model)
/// - `gpt-5.1-codex-max`, `gpt-5.2-codex`, `gpt-5.3-codex`: XHigh
/// - Other GPT-5/o-series reasoning models: High
///
/// # Arguments
///
/// * `model` - The model identifier string
///
/// # Returns
///
/// The recommended `ReasoningEffort` for the model
///
/// # Examples
///
/// ```rust
/// # use appam::llm::openai::default_reasoning_effort_for_model;
/// let effort = default_reasoning_effort_for_model("gpt-5.5");
/// assert!(matches!(effort, appam::llm::openai::ReasoningEffort::None));
///
/// let effort = default_reasoning_effort_for_model("gpt-5.1-codex-max");
/// assert!(matches!(effort, appam::llm::openai::ReasoningEffort::XHigh));
/// ```
pub fn default_reasoning_effort_for_model(model: &str) -> ReasoningEffort {
    if model_supports_none_reasoning(model) {
        ReasoningEffort::None
    } else if model_defaults_to_high_reasoning(model) {
        ReasoningEffort::High
    } else if model_supports_xhigh_reasoning(model) {
        ReasoningEffort::XHigh
    } else {
        ReasoningEffort::High
    }
}

/// Returns whether a model supports `ReasoningEffort::XHigh`.
///
/// # Purpose
///
/// OpenAI does not uniformly support extra-high reasoning across all model
/// variants. This helper centralizes capability checks so callers can apply
/// a consistent fallback policy.
///
/// # Arguments
///
/// * `model` - The model identifier string (for example, "gpt-5.3-codex")
///
/// # Returns
///
/// `true` when the model supports `XHigh`, otherwise `false`.
///
/// # Supported Models
///
/// `XHigh` is currently supported for:
/// - `gpt-5.5`
/// - `gpt-5.5-pro`
/// - `gpt-5.1-codex-max`
/// - `gpt-5.2-codex`
/// - `gpt-5.3-codex`
///
/// # Security and Reliability Notes
///
/// This function performs an exact allowlist match (fail-closed). Unknown model
/// identifiers are treated as unsupported and therefore do not receive `XHigh`.
pub fn model_supports_xhigh_reasoning(model: &str) -> bool {
    let model = normalize_openai_model(model);

    matches!(
        model.as_str(),
        "gpt-5.5" | "gpt-5.1-codex-max" | "gpt-5.2-codex" | "gpt-5.3-codex"
    ) || model.starts_with("gpt-5.5-")
}

/// Resolves the effective reasoning effort for a specific model.
///
/// # Purpose
///
/// Computes the final effort by combining:
/// 1. The caller-provided effort (if any), or model default when omitted
/// 2. A capability guard that downgrades unsupported `XHigh` requests to `High`
///
/// This ensures `XHigh` is only used by models that support it, while preserving
/// explicit non-`XHigh` requests (low, medium, high).
///
/// # Arguments
///
/// * `model` - The model identifier string
/// * `requested_effort` - Optional caller-provided effort override
///
/// # Returns
///
/// The final `ReasoningEffort` to send in the OpenAI request.
///
/// # Edge Cases
///
/// - If `requested_effort` is `None`, the model default is used.
/// - If `requested_effort` is `Some(XHigh)` but the model is unsupported,
///   this function downgrades to `High`.
/// - If `requested_effort` is `Some(None)` but the model is unsupported,
///   this function upgrades to the model default.
/// - Unknown model names are treated as unsupported for `XHigh`.
pub fn resolve_reasoning_effort_for_model(
    model: &str,
    requested_effort: Option<ReasoningEffort>,
) -> ReasoningEffort {
    let normalized = normalize_openai_model(model);
    let selected =
        requested_effort.unwrap_or_else(|| default_reasoning_effort_for_model(&normalized));

    if selected == ReasoningEffort::None && !model_supports_none_reasoning(&normalized) {
        default_reasoning_effort_for_model(&normalized)
    } else if selected == ReasoningEffort::XHigh && !model_supports_xhigh_reasoning(&normalized) {
        ReasoningEffort::High
    } else {
        selected
    }
}

/// Reasoning summary verbosity for o-series and gpt-5 models.
///
/// Controls the level of detail in the reasoning summary returned with the response.
/// The raw reasoning tokens are not exposed via the API, but summaries provide
/// insight into the model's thought process.
///
/// # API Mapping
///
/// Serializes to lowercase strings: "auto", "concise", "detailed"
///
/// # Organization Verification
///
/// Some models may require organization verification before using detailed summarizers.
/// Check the platform settings page for verification status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    /// Automatically select appropriate verbosity
    ///
    /// The system will choose the best summarizer for the model being used.
    /// This is equivalent to "detailed" for most reasoning models.
    #[default]
    Auto,
    /// Brief summaries of reasoning process
    ///
    /// Provides a concise overview of the model's reasoning without extensive detail.
    Concise,
    /// Comprehensive summaries with full reasoning breakdown
    ///
    /// Provides detailed insight into the model's thought process, including
    /// intermediate steps and decision-making logic.
    Detailed,
}

/// Text verbosity level.
///
/// Controls the level of detail in model responses.
///
/// # Verbosity Levels
///
/// - `Low`: Concise, brief responses with minimal detail
/// - `Medium`: Balanced responses with moderate detail (default)
/// - `High`: Detailed, comprehensive responses with full explanations
///
/// # Examples
///
/// ```rust
/// # use appam::llm::openai::TextVerbosity;
/// let verbosity = TextVerbosity::High;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TextVerbosity {
    /// Low verbosity - concise responses
    Low,
    /// Medium verbosity - balanced responses (default)
    #[default]
    Medium,
    /// High verbosity - detailed responses
    High,
}

/// Text format configuration (structured outputs).
///
/// Defines the output format for text responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormatConfig {
    /// Plain text output (default)
    Text,
    /// JSON object output (legacy)
    #[serde(rename = "json_object")]
    JsonObject,
    /// JSON schema output (structured outputs)
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// Schema name (a-z, A-Z, 0-9, underscores, dashes, max 64 chars)
        name: String,
        /// Optional description
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        /// JSON Schema object
        schema: serde_json::Value,
        /// Enable strict mode (recommended)
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// Service tier for request prioritization.
///
/// Determines latency and throughput characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    /// Automatic tier selection
    #[default]
    Auto,
    /// Default tier
    Default,
    /// Flexible tier (lower cost, variable latency)
    Flex,
    /// Scale tier (high throughput)
    Scale,
    /// Priority tier (lowest latency)
    Priority,
}

/// Conversation management configuration.
///
/// Enables multi-turn conversation continuity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationConfig {
    /// Conversation ID for multi-turn conversations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Previous response ID for continuity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
}

/// Retry configuration for handling transient errors.
///
/// Implements exponential backoff with jitter for rate limit and server errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    #[serde(default = "RetryConfig::default_max_retries")]
    pub max_retries: u32,

    /// Initial backoff duration in milliseconds
    #[serde(default = "RetryConfig::default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,

    /// Maximum backoff duration in milliseconds
    #[serde(default = "RetryConfig::default_max_backoff_ms")]
    pub max_backoff_ms: u64,

    /// Backoff multiplier for exponential growth
    #[serde(default = "RetryConfig::default_backoff_multiplier")]
    pub backoff_multiplier: f32,

    /// Enable random jitter to avoid thundering herd
    #[serde(default = "RetryConfig::default_jitter")]
    pub jitter: bool,
}

impl RetryConfig {
    fn default_max_retries() -> u32 {
        3
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
    pub fn calculate_backoff(&self, attempt: u32) -> u64 {
        let exponent = (attempt as f32 - 1.0).max(0.0);
        let base_backoff =
            (self.initial_backoff_ms as f32) * self.backoff_multiplier.powf(exponent);
        let capped_backoff = base_backoff.min(self.max_backoff_ms as f32);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OpenAIConfig::default();
        assert_eq!(config.model, "gpt-5.5");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert!(config.stream);
        assert_eq!(config.max_output_tokens, Some(4096));
    }

    #[test]
    fn test_normalize_openai_model_strips_provider_prefix() {
        assert_eq!(normalize_openai_model("openai/gpt-5.5"), "gpt-5.5");
        assert_eq!(normalize_openai_model("openai/gpt-5.5-pro"), "gpt-5.5-pro");
    }

    #[test]
    fn test_default_reasoning_effort_uses_gpt55_none_mode() {
        assert_eq!(
            default_reasoning_effort_for_model("gpt-5.5"),
            ReasoningEffort::None
        );
        assert_eq!(
            default_reasoning_effort_for_model("gpt-5.5-pro"),
            ReasoningEffort::High
        );
        assert_eq!(
            default_reasoning_effort_for_model("gpt-5-pro"),
            ReasoningEffort::High
        );
    }

    #[test]
    fn test_model_supports_sampling_parameters_only_for_gpt55_none() {
        assert!(model_supports_sampling_parameters("gpt-5.5", None));
        assert!(model_supports_sampling_parameters(
            "gpt-5.5",
            Some(ReasoningEffort::None)
        ));
        assert!(!model_supports_sampling_parameters(
            "gpt-5.5",
            Some(ReasoningEffort::High)
        ));
        assert!(!model_supports_sampling_parameters(
            "gpt-5-mini",
            Some(ReasoningEffort::High)
        ));
        assert!(!model_supports_sampling_parameters("gpt-5.5-pro", None));
    }

    #[test]
    fn test_validate_rejects_sampling_with_gpt55_reasoning() {
        let config = OpenAIConfig {
            model: "gpt-5.5".to_string(),
            reasoning: Some(ReasoningConfig::high_effort()),
            temperature: Some(0.7),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_allows_sampling_with_gpt55_none_reasoning() {
        let config = OpenAIConfig {
            model: "gpt-5.5".to_string(),
            reasoning: Some(ReasoningConfig::no_reasoning()),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_logprobs: Some(5),
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_summary_when_reasoning_is_none() {
        let config = OpenAIConfig {
            model: "gpt-5.5".to_string(),
            reasoning: Some(ReasoningConfig {
                effort: Some(ReasoningEffort::None),
                summary: Some(ReasoningSummary::Detailed),
            }),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_scale_service_tier_for_direct_openai() {
        let config = OpenAIConfig {
            service_tier: Some(ServiceTier::Scale),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_allows_scale_service_tier_for_azure() {
        let config = OpenAIConfig {
            service_tier: Some(ServiceTier::Scale),
            azure: Some(AzureConfig {
                resource_name: "example".to_string(),
                api_version: "2025-04-01-preview".to_string(),
            }),
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_none_reasoning_for_gpt55_pro() {
        let config = OpenAIConfig {
            model: "gpt-5.5-pro".to_string(),
            reasoning: Some(ReasoningConfig {
                effort: Some(ReasoningEffort::None),
                summary: None,
            }),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_allows_xhigh_reasoning_for_gpt55_pro() {
        let config = OpenAIConfig {
            model: "gpt-5.5-pro".to_string(),
            reasoning: Some(ReasoningConfig {
                effort: Some(ReasoningEffort::XHigh),
                summary: Some(ReasoningSummary::Detailed),
            }),
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_reasoning_config_minimal_builder_uses_minimal_effort() {
        let reasoning = ReasoningConfig::minimal();

        assert_eq!(reasoning.effort, Some(ReasoningEffort::Minimal));
        assert_eq!(reasoning.summary, Some(ReasoningSummary::Concise));
    }

    #[test]
    fn test_validate_rejects_summary_when_gpt55_default_resolves_to_none() {
        let config = OpenAIConfig {
            model: "gpt-5.5".to_string(),
            reasoning: Some(ReasoningConfig {
                effort: None,
                summary: Some(ReasoningSummary::Detailed),
            }),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_retry_config_defaults() {
        let retry = RetryConfig::default();
        assert_eq!(retry.max_retries, 3);
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
            jitter: false,
        };

        assert_eq!(retry.calculate_backoff(1), 1000);
        assert_eq!(retry.calculate_backoff(2), 2000);
        assert_eq!(retry.calculate_backoff(3), 4000);
        assert_eq!(retry.calculate_backoff(4), 8000);
    }

    #[test]
    fn test_retry_config_with_jitter() {
        let retry = RetryConfig {
            jitter: true,
            ..Default::default()
        };

        let backoff = retry.calculate_backoff(1);
        assert!((500..=1500).contains(&backoff));
    }
}
