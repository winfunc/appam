//! Provider-agnostic configuration for server-side context compaction.
//!
//! Long-running agent sessions eventually outgrow the model's context window.
//! Both Anthropic and OpenAI now expose *server-side* compaction: when the
//! rendered context crosses a caller-supplied token threshold, the provider
//! itself summarizes older conversation content and continues from the
//! compacted context. The summary comes back as a dedicated item in the
//! response and must be replayed on subsequent requests.
//!
//! Appam maps one [`CompactionConfig`] onto both wire formats:
//!
//! | Provider | Request field | Replayed artifact |
//! |---|---|---|
//! | Anthropic (direct, Bedrock, Azure) | `context_management.edits[{"type": "compact_20260112", ...}]` + beta `compact-2026-01-12` | `compaction` content block (human-readable summary) |
//! | OpenAI Responses (direct, Azure) | `context_management: [{"type": "compaction", "compact_threshold": N}]` | `compaction` output item (opaque `encrypted_content`) |
//!
//! # Design decisions
//!
//! - **Server-side, not client-side.** Appam never issues a separate
//!   summarization request, so there is no double-billed "shadow" call and no
//!   client-side prompt to maintain. The provider decides when the threshold
//!   is crossed using its own authoritative token accounting.
//! - **No context loss in appam.** The full transcript stays in the session
//!   history; only the *provider* prunes pre-compaction content at render
//!   time. Sessions remain replayable and auditable after compaction.
//! - **Accurate accounting.** Anthropic reports the compaction pass in a new
//!   `usage.iterations` array whose tokens are *excluded* from top-level
//!   usage. Appam surfaces those tokens via
//!   [`UnifiedUsage::compaction_input_tokens`](crate::llm::unified::UnifiedUsage)
//!   and bills them at the model's normal input/output rates.
//!
//! # Provider constraints
//!
//! - Anthropic requires a trigger of at least 50,000 input tokens and only
//!   supports compaction on Claude Sonnet 4.6+ / Opus 4.6+ / Claude 5 models.
//! - OpenAI requires a threshold of at least 1,000 tokens.
//! - Providers without a server-side compaction API (OpenRouter, Vertex,
//!   OpenAI Codex) ignore this configuration; the runtime logs a warning.
//!
//! Appam clamps out-of-range thresholds to the provider minimum and logs a
//! warning instead of failing the request.

use serde::{Deserialize, Serialize};

/// Anthropic beta identifier that enables server-side compaction.
///
/// Sent as the `anthropic-beta` HTTP header on direct/Azure Anthropic
/// requests and inside the `anthropic_beta` JSON body array on Bedrock.
pub const ANTHROPIC_COMPACTION_BETA: &str = "compact-2026-01-12";

/// Anthropic `context_management` edit type for compaction.
pub const ANTHROPIC_COMPACTION_EDIT_TYPE: &str = "compact_20260112";

/// Minimum trigger threshold accepted by Anthropic (tokens).
pub const ANTHROPIC_MIN_TRIGGER_TOKENS: u64 = 50_000;

/// Minimum compaction threshold accepted by OpenAI (tokens).
pub const OPENAI_MIN_TRIGGER_TOKENS: u64 = 1_000;

/// Configuration for provider-side automatic context compaction.
///
/// Attach this to a provider configuration (or use
/// `AgentBuilder::enable_auto_compaction`) to have the provider summarize
/// older conversation content once the context crosses `trigger_tokens`.
///
/// # Examples
///
/// ```rust,no_run
/// use appam::llm::compaction::CompactionConfig;
///
/// // Compact once the rendered context exceeds ~120K tokens.
/// let config = CompactionConfig::with_trigger_tokens(120_000);
///
/// // Provider default threshold (Anthropic: 150K tokens).
/// let default_trigger = CompactionConfig::enabled();
/// # let _ = (config, default_trigger);
/// ```
///
/// # Edge cases
///
/// - A threshold below the provider minimum is clamped upward with a warning
///   (Anthropic: 50,000 tokens; OpenAI: 1,000 tokens).
/// - `instructions` is Anthropic-only and *completely replaces* the default
///   summarization prompt when set. OpenAI ignores it.
/// - When `enabled` is `false` the configuration is inert everywhere.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Whether provider-side compaction is requested.
    #[serde(default)]
    pub enabled: bool,

    /// Token threshold at which the provider triggers compaction.
    ///
    /// `None` uses the provider default (Anthropic: 150,000 input tokens).
    /// OpenAI has no documented default, so appam sends the Anthropic-style
    /// default of 150,000 when unset to keep behavior predictable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_tokens: Option<u64>,

    /// Custom summarization instructions (Anthropic only).
    ///
    /// Replaces Anthropic's default summarization prompt entirely. Useful to
    /// pin what must survive compaction (code snippets, decisions, TODOs) or
    /// to forbid tool use during the internal summarization step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

impl CompactionConfig {
    /// Default trigger used when the caller does not specify one.
    ///
    /// Matches Anthropic's documented server default so both providers
    /// behave consistently out of the box.
    pub const DEFAULT_TRIGGER_TOKENS: u64 = 150_000;

    /// Enable compaction with the provider-default trigger threshold.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            trigger_tokens: None,
            instructions: None,
        }
    }

    /// Enable compaction with an explicit trigger threshold in tokens.
    pub fn with_trigger_tokens(trigger_tokens: u64) -> Self {
        Self {
            enabled: true,
            trigger_tokens: Some(trigger_tokens),
            instructions: None,
        }
    }

    /// Set custom summarization instructions (Anthropic only), consuming style.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Whether this configuration should activate compaction on requests.
    pub fn is_active(&self) -> bool {
        self.enabled
    }

    /// Resolve the effective trigger threshold for a provider minimum.
    ///
    /// Falls back to [`Self::DEFAULT_TRIGGER_TOKENS`] when unset and clamps
    /// values below `provider_min` upward, logging a warning so the caller
    /// can correct their configuration.
    pub fn effective_trigger_tokens(&self, provider_min: u64) -> u64 {
        let requested = self.trigger_tokens.unwrap_or(Self::DEFAULT_TRIGGER_TOKENS);
        if requested < provider_min {
            tracing::warn!(
                requested_trigger_tokens = requested,
                provider_min_tokens = provider_min,
                "Compaction trigger below provider minimum; clamping upward"
            );
            provider_min
        } else {
            requested
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_by_default() {
        let config = CompactionConfig::default();
        assert!(!config.is_active());
        assert!(config.trigger_tokens.is_none());
    }

    #[test]
    fn test_enabled_with_threshold() {
        let config = CompactionConfig::with_trigger_tokens(120_000);
        assert!(config.is_active());
        assert_eq!(config.trigger_tokens, Some(120_000));
    }

    #[test]
    fn test_effective_trigger_clamps_to_provider_minimum() {
        let config = CompactionConfig::with_trigger_tokens(10_000);
        assert_eq!(
            config.effective_trigger_tokens(ANTHROPIC_MIN_TRIGGER_TOKENS),
            ANTHROPIC_MIN_TRIGGER_TOKENS
        );
        assert_eq!(
            config.effective_trigger_tokens(OPENAI_MIN_TRIGGER_TOKENS),
            10_000
        );
    }

    #[test]
    fn test_effective_trigger_defaults_when_unset() {
        let config = CompactionConfig::enabled();
        assert_eq!(
            config.effective_trigger_tokens(OPENAI_MIN_TRIGGER_TOKENS),
            CompactionConfig::DEFAULT_TRIGGER_TOKENS
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let config = CompactionConfig::with_trigger_tokens(75_000)
            .instructions("Preserve code snippets and decisions.");
        let json = serde_json::to_string(&config).unwrap();
        let parsed: CompactionConfig = serde_json::from_str(&json).unwrap();
        assert!(parsed.enabled);
        assert_eq!(parsed.trigger_tokens, Some(75_000));
        assert_eq!(
            parsed.instructions.as_deref(),
            Some("Preserve code snippets and decisions.")
        );
    }
}
