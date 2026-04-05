//! Appam: AI Agent Framework
//!
//! A comprehensive framework for building AI agents with minimal configuration.
//! Create powerful agents by writing TOML configs, tool definitions, and tool
//! implementations in Rust or Python.
//!
//! # Overview
//!
//! Appam provides:
//! - **Agent system**: Define agents with system prompts and tool sets
//! - **Tool framework**: Implement tools in Rust or Python with automatic loading
//! - **LLM integration**: Streaming OpenRouter client with tool calling
//! - **OpenAI Codex auth**: Local OAuth cache for ChatGPT Codex subscription access
//! - **Configuration**: Hierarchical TOML-based configuration
//! - **Interfaces**: Built-in TUI and web API
//! - **Logging**: Structured tracing and session transcripts
//!
//! # Quick Start
//!
//! ## Option 1: Pure Rust SDK (No TOML Required)
//!
//! Build agents entirely in Rust with the builder API:
//!
//! ```no_run
//! use appam::prelude::*;
//! use anyhow::Result;
//! use std::sync::Arc;
//! # use serde_json::{json, Value};
//! # struct MyTool;
//! # impl Tool for MyTool {
//! #     fn name(&self) -> &str { "my_tool" }
//! #     fn spec(&self) -> Result<ToolSpec> {
//! #         Ok(serde_json::from_value(json!({
//! #             "type": "function",
//! #             "function": {
//! #                 "name": "my_tool",
//! #                 "description": "Example tool",
//! #                 "parameters": {"type": "object", "properties": {}}
//! #             }
//! #         }))?)
//! #     }
//! #     fn execute(&self, _: Value) -> Result<Value> { Ok(json!({"output": "result"})) }
//! # }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let agent = AgentBuilder::new("my-agent")
//!         .model("openai/gpt-4o-mini")
//!         .system_prompt("You are a helpful AI assistant.")
//!         .with_tool(Arc::new(MyTool))
//!         .build()?;
//!
//!     agent.run("Hello!").await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Option 2: TOML Configuration
//!
//! Create an agent configuration file (`agent.toml`):
//!
//! ```toml
//! [agent]
//! name = "assistant"
//! model = "openai/gpt-5"
//! system_prompt = "prompt.txt"
//!
//! [[tools]]
//! name = "bash"
//! schema = "tools/bash.json"
//! implementation = { type = "python", script = "tools/bash.py" }
//! ```
//!
//! Load and run the agent:
//!
//! ```no_run
//! use appam::prelude::*;
//! use anyhow::Result;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let agent = TomlAgent::from_file("agent.toml")?;
//!     agent.run("What can you do?").await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Option 3: Hybrid Approach
//!
//! Load from TOML and extend with Rust tools:
//!
//! ```no_run
//! # use appam::prelude::*;
//! # use anyhow::Result;
//! # use std::sync::Arc;
//! # use serde_json::{json, Value};
//! # struct CustomTool;
//! # impl Tool for CustomTool {
//! #     fn name(&self) -> &str { "custom" }
//! #     fn spec(&self) -> Result<ToolSpec> {
//! #         Ok(serde_json::from_value(json!({
//! #             "type": "function",
//! #             "function": {
//! #                 "name": "custom",
//! #                 "description": "Custom tool",
//! #                 "parameters": {"type": "object", "properties": {}}
//! #             }
//! #         }))?)
//! #     }
//! #     fn execute(&self, _: Value) -> Result<Value> { Ok(json!({"output": "result"})) }
//! # }
//! # async fn example() -> Result<()> {
//! let agent = TomlAgent::from_file("agent.toml")?
//!     .with_additional_tool(Arc::new(CustomTool));
//!
//! agent.run("Use custom tool").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! ## Agents
//!
//! Agents are defined by:
//! - A **system prompt** that establishes behavior and capabilities
//! - A **tool set** that provides executable functions
//! - A **model** that powers the agent's reasoning
//!
//! The `Agent` trait provides the core interface. Use `TomlAgent` to load
//! agents from configuration files.
//!
//! ## Tools
//!
//! Tools are executable functions exposed to the LLM. Each tool has:
//! - A **JSON schema** defining parameters and description
//! - An **implementation** in Rust or Python
//!
//! Tools can be implemented as:
//! - **Rust**: Native performance, type safety, full access to Rust ecosystem
//! - **Python**: Easy prototyping, access to Python libraries
//!
//! ## Configuration
//!
//! Configuration is hierarchical:
//! 1. Default values (hardcoded)
//! 2. Global config (`appam.toml`)
//! 3. Agent config (per-agent TOML file)
//! 4. Environment variables (highest priority)
//!
//! ## Interfaces
//!
//! Run agents via:
//! - **TUI**: Interactive terminal interface with rich widgets
//! - **CLI**: Simple streaming output
//! - **Web API**: RESTful API with Server-Sent Events streaming
//!
//! # Examples
//!
//! ## Creating a Python Tool
//!
//! Define the schema (`echo.json`):
//!
//! ```json
//! {
//!   "type": "function",
//!   "function": {
//!     "name": "echo",
//!     "description": "Echo back the input message",
//!     "parameters": {
//!       "type": "object",
//!       "properties": {
//!         "message": {
//!           "type": "string",
//!           "description": "Message to echo"
//!         }
//!       },
//!       "required": ["message"]
//!     }
//!   }
//! }
//! ```
//!
//! Implement the tool (`echo.py`):
//!
//! ```python
//! def execute(args):
//!     """Echo the input message."""
//!     return {"output": args.get("message", "")}
//! ```
//!
//! Register in agent config:
//!
//! ```toml
//! [[tools]]
//! name = "echo"
//! schema = "tools/echo.json"
//! implementation = { type = "python", script = "tools/echo.py" }
//! ```
//!
//! ## Programmatic Agent Creation
//!
//! ```no_run
//! use appam::agent::{Agent, TomlAgent};
//! use appam::tools::{Tool, ToolRegistry};
//! use std::sync::Arc;
//!
//! async fn create_custom_agent() {
//!     let agent = TomlAgent::from_file("agent.toml")
//!         .unwrap()
//!         .with_model("anthropic/claude-3.5-sonnet");
//!
//!     // Run with custom prompt
//!     agent.run("Analyze this codebase").await.unwrap();
//! }
//! ```

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
pub use tools::{Tool, ToolRegistry};

// Re-export procedural macros
pub use appam_macros::{tool, Schema};

/// Prelude module for convenient imports.
///
/// Import everything you need with:
///
/// ```
/// use appam::prelude::*;
/// ```
/// Prelude module for convenient imports
///
/// Import everything you need with `use appam::prelude::*;`
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
    pub use crate::agent::{AgentBuilder, RuntimeAgent, Session, TomlAgent}; // The trait

    // Quick constructors and shortcuts (NEW!)
    pub use crate::agent::quick::{Agent, AgentBuilderToolExt, AgentQuick};

    // Streaming types
    pub use crate::agent::streaming::{StreamConsumer, StreamEvent};
    pub use crate::agent::streaming_builder::StreamBuilder;

    // Consumers
    pub use crate::agent::consumers::{
        CallbackConsumer, ChannelConsumer, ConsoleConsumer, TraceConsumer,
    };

    // Error types (NEW!)
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
    pub use crate::tools::{Tool, ToolRegistry};

    // Procedural macros - the star of the DX improvements! (NEW!)
    pub use appam_macros::{tool, Schema};

    // Re-export common external types for convenience
    pub use anyhow::{anyhow, bail, Context, Result};
    pub use serde::{Deserialize, Serialize};
    pub use serde_json::{json, Value};
    pub use std::sync::Arc;
    pub use tokio;
}
