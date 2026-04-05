//! Anthropic Claude Messages API client implementation.
//!
//! This module implements the official Anthropic Messages API with full support for:
//! - Extended thinking with configurable token budgets
//! - Prompt caching with 5-minute and 1-hour TTL
//! - Vision (images via base64/URL)
//! - Documents (PDFs and plain text)
//! - Server tools (web search, bash, code execution, text editor)
//! - Client tools with parallel execution
//! - Server-Sent Events (SSE) streaming
//!
//! # API Documentation
//!
//! Official API reference: <https://docs.anthropic.com/en/api/messages>
//!
//! # Features
//!
//! ## Extended Thinking
//!
//! Claude can show its step-by-step reasoning process before answering:
//!
//! ```ignore
//! // Adaptive thinking for Opus 4.6+ (recommended)
//! let config = AnthropicConfig {
//!     thinking: Some(ThinkingConfig::adaptive()),
//!     ..Default::default()
//! };
//!
//! // Legacy fixed-budget thinking for older models
//! let config = AnthropicConfig {
//!     thinking: Some(ThinkingConfig::enabled(10000)),
//!     ..Default::default()
//! };
//! ```
//!
//! ## Prompt Caching
//!
//! Enable Anthropic's automatic prompt caching helper to cache the request
//! prefix through the last cacheable block:
//!
//! ```ignore
//! let config = AnthropicConfig {
//!     caching: Some(CachingConfig {
//!         enabled: true,
//!         ttl: CacheTTL::FiveMinutes,
//!     }),
//!     ..Default::default()
//! };
//! ```
//!
//! Appam maps this to Anthropic's top-level `cache_control` request field on
//! the direct Anthropic and Azure Anthropic transports. On AWS Bedrock,
//! Appam injects block-level `cache_control` checkpoints because Bedrock's
//! Anthropic InvokeModel shape expects explicit cache checkpoints in supported
//! fields instead of Anthropic's top-level helper.
//!
//! ## Vision & Documents
//!
//! Process images and PDFs as part of the conversation:
//!
//! ```ignore
//! use appam::llm::unified::{UnifiedMessage, UnifiedContentBlock, ImageSource};
//!
//! let msg = UnifiedMessage {
//!     role: UnifiedRole::User,
//!     content: vec![
//!         UnifiedContentBlock::Image {
//!             source: ImageSource::Url { url: "https://example.com/image.jpg".to_string() },
//!             detail: Some("high".to_string()),
//!         },
//!         UnifiedContentBlock::Text { text: "Describe this image.".to_string() },
//!     ],
//!     id: None,
//!     timestamp: None,
//! };
//! ```

pub mod caching;
pub mod client;
pub mod config;
pub mod convert;
pub mod rate_limiter;
pub mod streaming;
pub mod thinking;
pub mod tools;
pub mod types;
pub mod vision;

// Re-exports
pub use client::AnthropicClient;
pub use config::{
    AnthropicConfig, AzureAnthropicAuthMethod, AzureAnthropicConfig, BedrockAuthMethod,
    BedrockConfig, BetaFeatures, CacheTTL, CachingConfig, EffortLevel, RateLimiterConfig,
    RequestMetadata, RetryConfig, ThinkingConfig, ToolChoiceConfig,
};
pub use convert::{from_unified_messages, from_unified_tools};
pub use rate_limiter::RateLimiter;
pub use streaming::StreamEvent;
pub use types::{ContentBlock, ErrorResponse, Message, MessageRole, Tool, ToolChoice, Usage};
