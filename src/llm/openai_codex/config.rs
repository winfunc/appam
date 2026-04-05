//! Configuration for the OpenAI Codex subscription provider.
//!
//! The Codex transport shares model families and reasoning controls with the
//! direct OpenAI Responses API, but it authenticates with ChatGPT OAuth access
//! tokens and targets the ChatGPT backend. This configuration keeps those
//! transport-specific concerns separate from `OpenAIConfig`.

use std::path::PathBuf;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::llm::openai::{
    model_supports_sampling_parameters, normalize_openai_model, ReasoningConfig, ReasoningEffort,
    RetryConfig, TextVerbosity,
};

/// Configuration for the OpenAI Codex subscription-backed provider.
///
/// # Authentication precedence
///
/// Runtime authentication is resolved in this order:
///
/// 1. `access_token` set directly on this config
/// 2. `OPENAI_CODEX_ACCESS_TOKEN`
/// 3. Cached OAuth credentials in `auth_file`
///
/// The cached credential flow is intended for trusted local developer machines.
/// Do not copy or expose the auth file in untrusted environments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICodexConfig {
    /// Explicit ChatGPT OAuth access token override.
    ///
    /// This is primarily useful for tests or for callers that already manage the
    /// ChatGPT credential lifecycle outside Appam.
    #[serde(default)]
    pub access_token: Option<String>,

    /// Base backend URL for the Codex transport.
    ///
    /// The client appends `/codex/responses` when building requests.
    #[serde(default = "OpenAICodexConfig::default_base_url")]
    pub base_url: String,

    /// Model identifier to send to the Codex backend.
    ///
    /// Examples include `gpt-5.4`, `gpt-5.3-codex`, and `gpt-5.2-codex`.
    #[serde(default = "OpenAICodexConfig::default_model")]
    pub model: String,

    /// Optional canonical model identifier used for pricing/accounting only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing_model: Option<String>,

    /// Maximum output tokens for the response.
    #[serde(default)]
    pub max_output_tokens: Option<i32>,

    /// Optional temperature override for supported models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Optional top-p override for supported models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Whether streaming should be requested.
    ///
    /// Appam's LLM interface is streaming-first, so this defaults to `true`.
    #[serde(default = "OpenAICodexConfig::default_stream")]
    pub stream: bool,

    /// OpenAI-compatible reasoning configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    /// OpenAI-compatible text verbosity configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_verbosity: Option<TextVerbosity>,

    /// Retry policy for transient network and backend failures.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryConfig>,

    /// Path to the local OAuth credential cache.
    #[serde(default = "OpenAICodexConfig::default_auth_file")]
    pub auth_file: PathBuf,

    /// Codex originator header used for backend diagnostics.
    ///
    /// This value is intentionally stable and non-sensitive. It helps OpenAI
    /// distinguish Appam-originated traffic without exposing user information.
    #[serde(default = "OpenAICodexConfig::default_originator")]
    pub originator: String,
}

impl OpenAICodexConfig {
    fn default_base_url() -> String {
        "https://chatgpt.com/backend-api".to_string()
    }

    fn default_model() -> String {
        "gpt-5.4".to_string()
    }

    fn default_stream() -> bool {
        true
    }

    fn default_auth_file() -> PathBuf {
        super::auth::default_auth_file_path()
    }

    fn default_originator() -> String {
        "pi".to_string()
    }

    /// Validate the Codex configuration before creating a client.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    ///
    /// - sampling values are out of range
    /// - reasoning summaries are requested for `reasoning.effort = none`
    /// - sampling parameters are requested for models that reject them
    pub fn validate(&self) -> Result<()> {
        if let Some(temperature) = self.temperature {
            if !(0.0..=2.0).contains(&temperature) {
                bail!(
                    "OpenAI Codex temperature must be between 0.0 and 2.0, got {}",
                    temperature
                );
            }
        }

        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                bail!(
                    "OpenAI Codex top_p must be between 0.0 and 1.0, got {}",
                    top_p
                );
            }
        }

        let requested_effort = self
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort);
        let resolved_effort = self
            .reasoning
            .as_ref()
            .map(|_| resolve_reasoning_effort_for_codex_model(&self.model, requested_effort));

        if matches!(resolved_effort, Some(ReasoningEffort::None))
            && self
                .reasoning
                .as_ref()
                .and_then(|reasoning| reasoning.summary)
                .is_some()
        {
            bail!(
                "OpenAI Codex reasoning summaries are not available when reasoning.effort resolves to \"none\""
            );
        }

        if (self.temperature.is_some() || self.top_p.is_some())
            && !model_supports_sampling_parameters(
                &normalize_openai_model(&self.model),
                resolved_effort,
            )
        {
            bail!(
                "OpenAI Codex sampling parameters are not supported for model {} with the configured reasoning effort",
                normalize_openai_model(&self.model)
            );
        }

        Ok(())
    }
}

impl Default for OpenAICodexConfig {
    fn default() -> Self {
        Self {
            access_token: None,
            base_url: Self::default_base_url(),
            model: Self::default_model(),
            pricing_model: None,
            max_output_tokens: Some(4096),
            temperature: None,
            top_p: None,
            stream: Self::default_stream(),
            reasoning: None,
            text_verbosity: None,
            retry: Some(RetryConfig::default()),
            auth_file: Self::default_auth_file(),
            originator: Self::default_originator(),
        }
    }
}

/// Resolve the effective reasoning effort for a Codex model.
///
/// The ChatGPT Codex backend is close to the public OpenAI Responses API, but a
/// few legacy Codex aliases have stricter effort compatibility. This helper
/// first applies the standard OpenAI model-aware resolution and then clamps
/// unsupported Codex-specific cases to values the backend accepts.
pub fn resolve_reasoning_effort_for_codex_model(
    model: &str,
    requested_effort: Option<ReasoningEffort>,
) -> ReasoningEffort {
    let normalized = normalize_openai_model(model);
    let selected =
        crate::llm::openai::resolve_reasoning_effort_for_model(&normalized, requested_effort);

    match normalized.as_str() {
        model
            if (model.starts_with("gpt-5.2")
                || model.starts_with("gpt-5.3")
                || model.starts_with("gpt-5.4"))
                && selected == ReasoningEffort::Minimal =>
        {
            ReasoningEffort::Low
        }
        "gpt-5.1" if selected == ReasoningEffort::XHigh => ReasoningEffort::High,
        "gpt-5.1-codex-mini" => match selected {
            ReasoningEffort::High | ReasoningEffort::XHigh => ReasoningEffort::High,
            _ => ReasoningEffort::Medium,
        },
        _ => selected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OpenAICodexConfig::default();
        assert_eq!(config.base_url, "https://chatgpt.com/backend-api");
        assert_eq!(config.model, "gpt-5.4");
        assert_eq!(config.max_output_tokens, Some(4096));
        assert!(config.stream);
    }

    #[test]
    fn test_validate_rejects_invalid_sampling_ranges() {
        let config = OpenAICodexConfig {
            temperature: Some(9.0),
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = OpenAICodexConfig {
            top_p: Some(9.0),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_reasoning_summary_for_none_mode() {
        let config = OpenAICodexConfig {
            model: "gpt-5.4".to_string(),
            reasoning: Some(ReasoningConfig {
                effort: Some(ReasoningEffort::None),
                summary: Some(crate::llm::openai::ReasoningSummary::Detailed),
            }),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_allows_sampling_with_none_reasoning() {
        let config = OpenAICodexConfig {
            model: "gpt-5.4".to_string(),
            reasoning: Some(ReasoningConfig::no_reasoning()),
            temperature: Some(0.5),
            top_p: Some(0.9),
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_sampling_with_high_reasoning() {
        let config = OpenAICodexConfig {
            model: "gpt-5.4".to_string(),
            reasoning: Some(ReasoningConfig::high_effort()),
            temperature: Some(0.5),
            ..Default::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_resolve_reasoning_effort_clamps_codex_specific_models() {
        assert_eq!(
            resolve_reasoning_effort_for_codex_model("gpt-5.4", Some(ReasoningEffort::Minimal)),
            ReasoningEffort::Low
        );
        assert_eq!(
            resolve_reasoning_effort_for_codex_model("gpt-5.1", Some(ReasoningEffort::XHigh)),
            ReasoningEffort::High
        );
        assert_eq!(
            resolve_reasoning_effort_for_codex_model(
                "gpt-5.1-codex-mini",
                Some(ReasoningEffort::Low)
            ),
            ReasoningEffort::Medium
        );
    }
}
