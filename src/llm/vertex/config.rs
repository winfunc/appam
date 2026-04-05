//! Configuration for the Google Vertex AI provider.
//!
//! This configuration is intentionally aligned with the rest of appam's
//! provider configs: credentials are optional in the struct (to encourage
//! environment-variable usage), and request-level parameters are explicit and
//! typed for predictable behavior.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Function-calling mode for Vertex requests.
///
/// Vertex supports three function-calling behaviors:
/// - `Auto`: Model decides between plain text and tool calls.
/// - `Any`: Model must return a function call.
/// - `None`: Tool calling disabled for this request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum VertexFunctionCallingMode {
    /// Model can choose between text or function calls.
    #[default]
    Auto,
    /// Model must produce function calls.
    Any,
    /// Model is prohibited from function calls.
    None,
}

/// Optional thinking configuration for supported Gemini models.
///
/// The exact effect depends on model family and API revision. Unknown fields
/// are omitted from the request to preserve fail-closed behavior.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VertexThinkingConfig {
    /// Optional thinking level hint (for APIs that accept string levels).
    ///
    /// Typical values include `LOW`, `MEDIUM`, or `HIGH`.
    #[serde(rename = "thinkingLevel", skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<String>,

    /// Whether to include thought content in responses when supported.
    #[serde(rename = "includeThoughts", skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
}

/// Retry configuration for Vertex API requests.
///
/// Reuses OpenAI retry semantics so all providers share consistent backoff
/// behavior and DX.
pub type VertexRetryConfig = crate::llm::openai::RetryConfig;

/// Configuration for the Vertex AI client.
///
/// # Authentication
///
/// Supported auth modes:
/// - API key query param (`key=...`) via `api_key` or env fallbacks
/// - Bearer token via `access_token` or env fallback
///
/// Resolution order:
/// - API key: `config.api_key` → `GOOGLE_VERTEX_API_KEY` → `GOOGLE_API_KEY` → `GEMINI_API_KEY`
/// - Bearer: `config.access_token` → `GOOGLE_VERTEX_ACCESS_TOKEN`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexConfig {
    /// Optional API key for key-based authentication.
    #[serde(default)]
    pub api_key: Option<String>,

    /// Optional OAuth bearer token for Authorization header.
    #[serde(default)]
    pub access_token: Option<String>,

    /// Base URL for Vertex API requests.
    ///
    /// Default: `https://aiplatform.googleapis.com`.
    #[serde(default = "VertexConfig::default_base_url")]
    pub base_url: String,

    /// Gemini model identifier (for example, `gemini-2.5-flash`).
    #[serde(default = "VertexConfig::default_model")]
    pub model: String,

    /// Optional Google Cloud project ID.
    ///
    /// When set, project-scoped endpoints are used.
    #[serde(default)]
    pub project_id: Option<String>,

    /// Vertex location/region used for project-scoped endpoints.
    #[serde(default = "VertexConfig::default_location")]
    pub location: String,

    /// Whether to call streaming endpoint.
    #[serde(default = "VertexConfig::default_stream")]
    pub stream: bool,

    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling top-p.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Function-calling mode.
    #[serde(default)]
    pub function_calling_mode: VertexFunctionCallingMode,

    /// Optional allow-list of function names in `ANY` mode.
    #[serde(default)]
    pub allowed_function_names: Option<Vec<String>>,

    /// Stream function-call arguments as partial updates.
    #[serde(default)]
    pub stream_function_call_arguments: bool,

    /// Optional thinking configuration.
    #[serde(default)]
    pub thinking: Option<VertexThinkingConfig>,

    /// Retry policy for transient failures.
    #[serde(default)]
    pub retry: Option<VertexRetryConfig>,
}

impl VertexConfig {
    fn default_base_url() -> String {
        "https://aiplatform.googleapis.com".to_string()
    }

    fn default_model() -> String {
        "gemini-2.5-flash".to_string()
    }

    fn default_location() -> String {
        "us-central1".to_string()
    }

    fn default_stream() -> bool {
        true
    }

    /// Validate provider configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when static configuration is malformed, such as empty
    /// model names, invalid temperature ranges, or missing required fields for
    /// project-scoped routing.
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            return Err(anyhow!("Vertex model cannot be empty"));
        }

        if self.location.trim().is_empty() {
            return Err(anyhow!("Vertex location cannot be empty"));
        }

        if let Some(temperature) = self.temperature {
            if !(0.0..=2.0).contains(&temperature) {
                return Err(anyhow!(
                    "Vertex temperature must be within [0.0, 2.0], got {}",
                    temperature
                ));
            }
        }

        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(anyhow!(
                    "Vertex top_p must be within [0.0, 1.0], got {}",
                    top_p
                ));
            }
        }

        Ok(())
    }
}

impl Default for VertexConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            access_token: None,
            base_url: Self::default_base_url(),
            model: Self::default_model(),
            project_id: None,
            location: Self::default_location(),
            stream: Self::default_stream(),
            max_output_tokens: Some(4096),
            temperature: None,
            top_p: None,
            top_k: None,
            function_calling_mode: VertexFunctionCallingMode::Auto,
            allowed_function_names: None,
            stream_function_call_arguments: false,
            thinking: None,
            retry: Some(VertexRetryConfig::default()),
        }
    }
}
