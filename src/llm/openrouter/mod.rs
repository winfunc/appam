//! OpenRouter API clients (Completions and Responses).
//!
//! This module provides two OpenRouter API client implementations:
//! - **Completions API** (`completions`): Standard, stable Chat Completions API with provider routing
//! - **Responses API** (`responses`): Alpha API with advanced reasoning features
//!
//! Both clients implement the unified `LlmClient` trait and support automatic reasoning
//! preservation, tool calling, and streaming.
//!
//! # Examples
//!
//! ## Using Completions API with provider routing
//! ```no_run
//! use appam::llm::openrouter::{OpenRouterConfig, ReasoningConfig, ProviderPreferences};
//! use appam::llm::openrouter::completions::OpenRouterCompletionsClient;
//! use appam::prelude::*;
//!
//! let cfg = OpenRouterConfig {
//!     model: "anthropic/claude-sonnet-4-5".to_string(),
//!     ..Default::default()
//! };
//!
//! let reasoning = ReasoningConfig::high_effort(32000);
//!
//! let provider_prefs = ProviderPreferences {
//!     order: Some(vec!["anthropic".to_string()]),
//!     ..Default::default()
//! };
//!
//! let client = OpenRouterCompletionsClient::new(
//!     cfg,
//!     Some(reasoning),
//!     Some(provider_prefs)
//! )?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Using Responses API with reasoning
//! ```no_run
//! use appam::llm::openrouter::{OpenRouterConfig, ReasoningConfig};
//! use appam::llm::openrouter::responses::OpenRouterClient;
//!
//! let cfg = OpenRouterConfig {
//!     model: "anthropic/claude-sonnet-4-5".to_string(),
//!     reasoning: Some(ReasoningConfig::high_effort(63999)),
//!     ..Default::default()
//! };
//!
//! let client = OpenRouterClient::new(cfg)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

// Submodules
pub mod completions;
pub mod config;
pub mod responses;
pub mod types;

// Re-export commonly used types for convenience
pub use config::{
    DataCollection, MaxPrice, OpenRouterConfig, ProviderPreferences, ProviderSort,
    QuantizationLevel, ReasoningConfig, ReasoningEffort, SummaryVerbosity,
};

pub use types::{
    CompletionMessage, CompletionRequest, ErrorResponse, ReasoningDetail, ReasoningFormat,
    ToolCall, ToolCallFunction, ToolChoice, ToolSpec, Usage,
};

pub use completions::OpenRouterCompletionsClient;
pub use responses::OpenRouterClient as OpenRouterResponsesClient;
