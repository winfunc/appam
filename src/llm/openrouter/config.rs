//! Shared configuration for OpenRouter APIs (Completions and Responses).
//!
//! Contains API key management, base URL configuration, reasoning settings,
//! and provider routing preferences shared across both APIs.

use serde::{Deserialize, Serialize};

/// Reasoning effort level.
///
/// Controls how much computational effort the model puts into reasoning.
/// Higher effort levels produce more thorough reasoning but consume more tokens.
///
/// **Applies to:** Both Completions and Responses APIs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Minimal reasoning for simple tasks
    Minimal,
    /// Light reasoning
    Low,
    /// Balanced reasoning (default)
    #[default]
    Medium,
    /// Deep reasoning for complex problems
    High,
}

/// Reasoning summary verbosity.
///
/// **Applies to:** Responses API only
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SummaryVerbosity {
    /// Automatic verbosity selection
    #[default]
    Auto,
    /// Concise summaries
    Concise,
    /// Detailed summaries
    Detailed,
}

/// Unified reasoning configuration for both Completions and Responses APIs.
///
/// This configuration works across both OpenRouter APIs:
/// - **Completions API**: Uses `enabled`, `effort`, `max_tokens`, `exclude`
/// - **Responses API**: Uses `enabled`, `effort`, `max_tokens`, `summary`
///
/// Fields that don't apply to a specific API are ignored during serialization.
///
/// # Examples
///
/// ## Completions API with high effort
/// ```no_run
/// use appam::llm::openrouter::{ReasoningConfig, ReasoningEffort};
///
/// let config = ReasoningConfig {
///     effort: Some(ReasoningEffort::High),
///     max_tokens: Some(32000),
///     exclude: Some(false),
///     ..Default::default()
/// };
/// ```
///
/// ## Responses API with detailed summary
/// ```no_run
/// use appam::llm::openrouter::{ReasoningConfig, ReasoningEffort, SummaryVerbosity};
///
/// let config = ReasoningConfig {
///     effort: Some(ReasoningEffort::High),
///     summary: Some(SummaryVerbosity::Detailed),
///     max_tokens: Some(63999),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Enable reasoning (default: true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,

    /// Reasoning effort level (OpenAI-style: high/medium/low)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,

    /// Maximum reasoning tokens (Anthropic-style: specific count)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Exclude reasoning from response (Completions API only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,

    /// Summary verbosity (Responses API only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<SummaryVerbosity>,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            enabled: Some(true), // Enabled by default
            effort: Some(ReasoningEffort::Medium),
            max_tokens: None,
            exclude: Some(false), // Include reasoning by default
            summary: Some(SummaryVerbosity::Auto),
        }
    }
}

impl ReasoningConfig {
    /// Create a high-effort reasoning configuration with custom token budget.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Maximum reasoning tokens (e.g., 32000 for Anthropic, 63999 for extended thinking)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use appam::llm::openrouter::ReasoningConfig;
    ///
    /// let config = ReasoningConfig::high_effort(32000);
    /// ```
    pub fn high_effort(max_tokens: u32) -> Self {
        Self {
            enabled: Some(true),
            effort: Some(ReasoningEffort::High),
            max_tokens: Some(max_tokens),
            exclude: Some(false),
            summary: Some(SummaryVerbosity::Detailed),
        }
    }

    /// Create a reasoning configuration with reasoning excluded from response.
    ///
    /// The model still uses reasoning internally, but it's not returned in the response.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use appam::llm::openrouter::ReasoningConfig;
    ///
    /// let config = ReasoningConfig::excluded();
    /// ```
    pub fn excluded() -> Self {
        Self {
            enabled: Some(true),
            effort: Some(ReasoningEffort::Medium),
            max_tokens: None,
            exclude: Some(true), // Exclude from response
            summary: None,
        }
    }
}

/// Base configuration for OpenRouter API client.
///
/// Contains connection details, authentication, and model selection
/// shared across both Completions and Responses APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    /// API key for authentication (prefer env var OPENROUTER_API_KEY)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Base URL for OpenRouter API
    #[serde(default = "OpenRouterConfig::default_base_url")]
    pub base_url: String,

    /// Model identifier (e.g., "openai/gpt-5", "anthropic/claude-sonnet-4-5")
    #[serde(default = "OpenRouterConfig::default_model")]
    pub model: String,

    /// Optional HTTP-Referer header for attribution on openrouter.ai rankings
    #[serde(default)]
    pub http_referer: Option<String>,

    /// Optional X-Title header for attribution on openrouter.ai rankings
    #[serde(default)]
    pub x_title: Option<String>,

    /// Whether to enable SSE streaming
    #[serde(default = "OpenRouterConfig::default_stream")]
    pub stream: bool,

    /// Maximum output tokens
    #[serde(default)]
    pub max_output_tokens: Option<u32>,

    /// Temperature (0.0-2.0)
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Top-p sampling (0.0-1.0)
    #[serde(default)]
    pub top_p: Option<f32>,

    // API-specific configuration
    /// Reasoning configuration (both APIs)
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,

    /// Provider routing preferences (Completions API only)
    #[serde(default)]
    pub provider_preferences: Option<ProviderPreferences>,

    /// Prompt transforms (Completions API only)
    #[serde(default)]
    pub transforms: Option<Vec<String>>,

    /// Fallback models (Completions API only)
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

impl OpenRouterConfig {
    /// Default base URL for OpenRouter API.
    pub fn default_base_url() -> String {
        "https://openrouter.ai/api/v1".to_string()
    }

    /// Default model (GPT-5).
    pub fn default_model() -> String {
        "openai/gpt-5".to_string()
    }

    /// Default streaming enabled.
    pub fn default_stream() -> bool {
        true
    }
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: Self::default_base_url(),
            model: Self::default_model(),
            http_referer: None,
            x_title: None,
            stream: Self::default_stream(),
            max_output_tokens: Some(9000),
            temperature: None,
            top_p: None,
            reasoning: None,
            provider_preferences: None,
            transforms: None,
            models: None,
        }
    }
}

/// Provider sorting strategy for Completions API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderSort {
    /// Sort by price (lowest first)
    Price,
    /// Sort by throughput (highest first)
    Throughput,
    /// Sort by latency (lowest first)
    Latency,
}

/// Data collection policy for provider routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataCollection {
    /// Allow providers that may store data
    Allow,
    /// Only use providers that don't collect data
    Deny,
}

/// Quantization level for model filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationLevel {
    /// Integer 4-bit quantization
    Int4,
    /// Integer 8-bit quantization
    Int8,
    /// Floating point 4-bit quantization
    Fp4,
    /// Floating point 6-bit quantization
    Fp6,
    /// Floating point 8-bit quantization
    Fp8,
    /// Floating point 16-bit quantization
    Fp16,
    /// Brain floating point 16-bit quantization
    Bf16,
    /// Floating point 32-bit quantization
    Fp32,
    /// Unknown quantization
    Unknown,
}

/// Maximum price constraints for provider routing.
///
/// All prices are in dollars per million tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPrice {
    /// Maximum price per million prompt tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<f64>,

    /// Maximum price per million completion tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<f64>,

    /// Maximum price per request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<f64>,

    /// Maximum price per image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<f64>,
}

/// Provider routing preferences for Completions API.
///
/// Controls how OpenRouter selects and routes requests to model providers,
/// including fallback behavior, data policies, and cost constraints.
///
/// # Examples
///
/// ## Prioritize Anthropic with fallbacks
/// ```no_run
/// use appam::llm::openrouter::{ProviderPreferences, DataCollection};
///
/// let prefs = ProviderPreferences {
///     order: Some(vec!["anthropic".to_string(), "openai".to_string()]),
///     allow_fallbacks: Some(true),
///     data_collection: Some(DataCollection::Deny),
///     zdr: Some(true),
///     ..Default::default()
/// };
/// ```
///
/// ## Cost-optimized routing
/// ```no_run
/// use appam::llm::openrouter::{ProviderPreferences, ProviderSort, MaxPrice};
///
/// let prefs = ProviderPreferences {
///     sort: Some(ProviderSort::Price),
///     max_price: Some(MaxPrice {
///         prompt: Some(1.0),  // $1/M tokens
///         completion: Some(3.0),  // $3/M tokens
///         request: None,
///         image: None,
///     }),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPreferences {
    /// Provider order (e.g., ["anthropic", "openai"])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,

    /// Allow fallback providers if primary fails
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,

    /// Only route to providers supporting all request parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,

    /// Data collection policy (allow/deny)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,

    /// Zero Data Retention enforcement
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,

    /// Whitelist providers (only use these)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,

    /// Blacklist providers (never use these)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,

    /// Filter by quantization levels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<QuantizationLevel>>,

    /// Sort providers by criteria
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<ProviderSort>,

    /// Maximum price constraints
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_price: Option<MaxPrice>,
}

impl Default for ProviderPreferences {
    fn default() -> Self {
        Self {
            order: None,
            allow_fallbacks: Some(true),
            require_parameters: None,
            data_collection: None,
            zdr: None,
            only: None,
            ignore: None,
            quantizations: None,
            sort: None,
            max_price: None,
        }
    }
}
