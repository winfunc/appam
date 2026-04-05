//! Google Vertex AI Gemini client implementation.
//!
//! This module provides a first-class integration with Vertex AI's
//! `generateContent` and `streamGenerateContent` APIs. It supports:
//! - Streaming text output (SSE via `alt=sse`)
//! - Function/tool calling
//! - Multi-turn conversation history
//! - Thought-signature preservation for tool-calling continuity
//! - Usage metadata extraction
//!
//! The implementation follows the same unified interfaces as the other
//! providers, so switching between Vertex, Anthropic, OpenAI, and OpenRouter
//! does not require runtime-flow changes.

pub mod client;
pub mod config;
pub mod convert;
pub mod types;

pub use client::VertexClient;
pub use config::{VertexConfig, VertexFunctionCallingMode, VertexThinkingConfig};
