//! Rust agent orchestration library for tool-using, long-horizon, traceable AI systems.
//!
//! Appam is designed for agent workloads where the hard parts are operational:
//! repeated tool use across multiple turns, streaming UX, session persistence,
//! trace capture, and provider portability. The shared runtime stays provider
//! agnostic while the provider modules own wire-format quirks, auth, retry, and
//! streaming details.
//!
//! # What Appam Gives You
//!
//! - Rust-first agent construction through [`AgentBuilder`], [`RuntimeAgent`],
//!   and the shortcut constructors in [`agent::quick`]
//! - typed and untyped tool definitions via [`Tool`], [`AsyncTool`],
//!   [`ToolRegistry`], [`tool`], and [`Schema`]
//! - streaming-first execution through [`agent::streaming::StreamEvent`],
//!   [`agent::streaming_builder::StreamBuilder`], and built-in consumers
//! - durable session history through [`SessionHistory`]
//! - portable LLM routing across Anthropic, OpenAI, OpenAI Codex, OpenRouter,
//!   Vertex, Azure, and Bedrock via [`LlmProvider`] and [`DynamicLlmClient`]
//!
//! # Recommended Entry Points
//!
//! Use [`Agent::quick`](crate::agent::quick::Agent::quick) when you want the
//! smallest working setup:
//!
//! ```no_run
//! use appam::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let agent = Agent::quick(
//!         "anthropic/claude-sonnet-4-5",
//!         "You are a concise Rust assistant.",
//!         vec![],
//!     )?;
//!
//!     agent
//!         .stream("Explain ownership in Rust in three sentences.")
//!         .on_content(|text| print!("{text}"))
//!         .run()
//!         .await?;
//!
//!     println!();
//!     Ok(())
//! }
//! ```
//!
//! Use [`AgentBuilder`] when you need explicit runtime control:
//!
//! ```no_run
//! use appam::prelude::*;
//! use std::sync::Arc;
//! # use serde_json::{json, Value};
//! # struct EchoTool;
//! # impl Tool for EchoTool {
//! #     fn name(&self) -> &str { "echo" }
//! #     fn spec(&self) -> Result<ToolSpec> {
//! #         Ok(serde_json::from_value(json!({
//! #             "type": "function",
//! #             "function": {
//! #                 "name": "echo",
//! #                 "description": "Echo a message",
//! #                 "parameters": {
//! #                     "type": "object",
//! #                     "properties": {
//! #                         "message": {"type": "string"}
//! #                     },
//! #                     "required": ["message"]
//! #                 }
//! #             }
//! #         }))?)
//! #     }
//! #     fn execute(&self, args: Value) -> Result<Value> {
//! #         Ok(json!({ "output": args["message"].clone() }))
//! #     }
//! # }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let agent = AgentBuilder::new("assistant")
//!         .provider(LlmProvider::OpenAI)
//!         .model("openai/gpt-5.4")
//!         .system_prompt("You are a careful assistant. Use tools before guessing.")
//!         .with_tool(Arc::new(EchoTool))
//!         .enable_history()
//!         .build()?;
//!
//!     agent.run("Say hello via the echo tool.").await?;
//!     Ok(())
//! }
//! ```
//!
//! Use [`TomlAgent`] when configuration lives on disk and needs to be extended
//! with Rust tools at runtime:
//!
//! ```no_run
//! use appam::prelude::*;
//! use std::sync::Arc;
//! # use serde_json::{json, Value};
//! # struct CustomTool;
//! # impl Tool for CustomTool {
//! #     fn name(&self) -> &str { "custom" }
//! #     fn spec(&self) -> Result<ToolSpec> {
//! #         Ok(serde_json::from_value(json!({
//! #             "type": "function",
//! #             "function": {
//! #                 "name": "custom",
//! #                 "description": "Example custom tool",
//! #                 "parameters": {"type": "object", "properties": {}}
//! #             }
//! #         }))?)
//! #     }
//! #     fn execute(&self, _: Value) -> Result<Value> { Ok(json!({"ok": true})) }
//! # }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let agent = TomlAgent::from_file("agent.toml")?
//!         .with_additional_tool(Arc::new(CustomTool));
//!
//!     agent.run("Use the custom tool if it helps.").await?;
//!     Ok(())
//! }
//! ```
//!
//! # Architecture
//!
//! Appam keeps the crate split along subsystem boundaries:
//!
//! - [`agent`] owns runtime orchestration, continuation logic, stream events,
//!   and session history
//! - [`tools`] owns schemas, execution traits, managed state, and runtime lookup
//! - [`llm`] owns provider-neutral message types plus provider-specific clients
//! - [`config`] owns global configuration, env overrides, and builder APIs
//! - [`logging`] owns tracing setup and persisted session logs
//!
//! # Security and Operational Notes
//!
//! - Tool calls are model output and must be treated as untrusted input.
//! - Managed state access fails closed when state was not registered.
//! - The legacy web API surface remains intentionally disabled; the trace
//!   visualizer helpers in [`web`] remain available.
//! - Provider-specific secrets are never meant to be logged or embedded in
//!   trace artifacts.
//!
//! # Docs.rs Navigation
//!
//! If you are new to the crate, the most useful pages are usually:
//!
//! - [`prelude`] for the ergonomic import surface
//! - [`AgentBuilder`] and [`RuntimeAgent`] for programmatic agents
//! - [`Tool`] and [`AsyncTool`] for tool authoring
//! - [`tool`] and [`Schema`] for macro-driven tool definitions
//! - [`StreamBuilder`](crate::agent::streaming_builder::StreamBuilder) for
//!   closure-based streaming
//! - [`SessionHistory`] for durable sessions

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_inception)]

pub mod agent;
pub mod config;
pub mod http;
pub mod llm;
pub mod logging;
pub mod tools;
pub mod web;

// TODO: Implement these modules in future phases
// pub mod interface;

// Re-export commonly used types for convenience
pub use agent::history::SessionHistory;
pub use agent::{Agent, AgentBuilder, RuntimeAgent, Session, TomlAgent};
pub use config::{
    load_config_from_env, load_global_config, AgentConfigBuilder, AppConfig, AppConfigBuilder,
    HistoryConfig, LogFormat, LoggingConfig, TraceFormat,
};
pub use llm::{
    DynamicLlmClient, LlmClient, LlmProvider, UnifiedMessage, UnifiedTool, UnifiedToolCall,
};
pub use tools::{AsyncTool, SessionState, State, Tool, ToolConcurrency, ToolContext, ToolRegistry};

// Re-export procedural macros
pub use appam_macros::{tool, Schema};
pub use async_trait::async_trait;

/// Convenient import surface for the most common Appam workflows.
///
/// The prelude intentionally favors the Rust-first SDK path: agent builders,
/// quick constructors, streaming types, tool traits, macro helpers, and the
/// most common `anyhow`, `serde`, and `tokio` re-exports used by examples.
///
/// Import it when you want to prototype quickly or write concise examples:
///
/// ```
/// use appam::prelude::*;
/// ```
///
/// For library code that exposes Appam types in its own public API, prefer
/// importing only the specific items you need so your dependency surface stays
/// explicit.
///
/// # Examples
///
/// ```no_run
/// use appam::prelude::*;
/// use anyhow::Result;
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     // Use Agent::quick() for one-liner creation
///     let agent = Agent::quick(
///         "anthropic/claude-sonnet-4-5",
///         "You are helpful.",
///         vec![],
///     )?;
///
///     // Use closure-based streaming
///     agent
///         .stream("Hello")
///         .on_content(|text| print!("{}", text))
///         .run()
///         .await?;
///
///     Ok(())
/// }
/// ```
pub mod prelude {
    // Core agent types
    pub use crate::agent::Agent as AgentTrait;
    pub use crate::agent::{AgentBuilder, RuntimeAgent, Session, TomlAgent};

    // Quick constructors and shortcuts
    pub use crate::agent::quick::{
        Agent, AgentBuilderAsyncToolExt, AgentBuilderToolExt, AgentQuick,
    };

    // Streaming types
    pub use crate::agent::streaming::{StreamConsumer, StreamEvent};
    pub use crate::agent::streaming_builder::StreamBuilder;

    // Consumers
    pub use crate::agent::consumers::{
        CallbackConsumer, ChannelConsumer, ConsoleConsumer, TraceConsumer,
    };

    // Error types
    pub use crate::agent::errors::{analyze_tool_error, ToolExecutionError};

    // History
    pub use crate::agent::history::{SessionHistory, SessionSummary};

    // Configuration
    pub use crate::config::{
        load_config_from_env, load_global_config, AgentConfigBuilder, AppConfig, AppConfigBuilder,
        HistoryConfig, LogFormat, LoggingConfig, TraceFormat,
    };

    // LLM types
    pub use crate::llm::{
        ChatMessage, DynamicLlmClient, LlmClient, LlmProvider, Role, ToolSpec, UnifiedMessage,
        UnifiedTool, UnifiedToolCall,
    };

    // Tool system
    pub use crate::tools::register::{ClosureTool, ToolRegistryExt};
    pub use crate::tools::{
        AsyncTool, SessionState, State, Tool, ToolConcurrency, ToolContext, ToolRegistry,
    };

    // Procedural macros
    pub use appam_macros::{tool, Schema};

    // Re-export common external types for convenience
    pub use anyhow::{anyhow, bail, Context, Result};
    pub use serde::{Deserialize, Serialize};
    pub use serde_json::{json, Value};
    pub use std::sync::Arc;
    pub use tokio;
}
