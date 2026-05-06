//! OpenAI Responses API client implementation.
//!
//! Provides a complete client for the OpenAI Responses API with support for:
//! - Streaming responses via Server-Sent Events
//! - Tool calling with parallel execution
//! - Reasoning for o-series models (o3-mini, o3)
//! - Structured outputs via JSON schema
//! - Prompt caching with cache keys
//! - Service tier selection for latency/cost control
//! - Conversation continuity and state management
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```rust,no_run
//! use appam::llm::openai::{OpenAIClient, OpenAIConfig};
//! use appam::llm::unified::UnifiedMessage;
//! use appam::llm::LlmClient;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = OpenAIConfig::default();
//!     let client = OpenAIClient::new(config)?;
//!
//!     let messages = vec![UnifiedMessage::user("What is 2+2?")];
//!
//!     client.chat_with_tools_streaming(
//!         &messages,
//!         &[],
//!         |chunk| { print!("{}", chunk); Ok(()) },
//!         |_| Ok(()),
//!         |_| Ok(()),
//!         |_| Ok(()),
//!         |_| Ok(()),
//!         |_| Ok(()),
//!     ).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! With reasoning (GPT-5.5 and other reasoning-capable models):
//!
//! ```rust,no_run
//! use appam::llm::openai::{OpenAIClient, OpenAIConfig, ReasoningConfig, ReasoningEffort, ReasoningSummary};
//!
//! let config = OpenAIConfig {
//!     model: "gpt-5.5".to_string(),
//!     reasoning: Some(ReasoningConfig {
//!         effort: Some(ReasoningEffort::High),
//!         summary: Some(ReasoningSummary::Detailed),
//!     }),
//!     ..Default::default()
//! };
//! ```
//!
//! Using convenience methods:
//!
//! ```rust,no_run
//! use appam::llm::openai::{OpenAIConfig, ReasoningConfig};
//!
//! let config = OpenAIConfig {
//!     model: "gpt-5.5".to_string(),
//!     reasoning: Some(ReasoningConfig::high_effort()),
//!     ..Default::default()
//! };
//! ```

pub mod client;
pub mod config;
pub mod convert;
pub mod streaming;
pub mod types;

// Re-exports for convenience
pub use client::OpenAIClient;
pub use config::{
    default_reasoning_effort_for_model, model_supports_none_reasoning,
    model_supports_sampling_parameters, model_supports_xhigh_reasoning, normalize_openai_model,
    resolve_reasoning_effort_for_model, AzureConfig, ConversationConfig, OpenAIConfig,
    ReasoningConfig, ReasoningEffort, ReasoningSummary, RetryConfig, ServiceTier, TextFormatConfig,
    TextVerbosity,
};
pub use types::{
    InputItem, MessageRole, OutputItem, Response, ResponseCreateParams, ResponseInput,
    ResponseStatus, ResponseTextConfig, ResponseTextFormat, Tool, ToolChoice,
};
