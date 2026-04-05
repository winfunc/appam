//! Tool system for agent capabilities.
//!
//! This module provides the core `Tool` trait and supporting infrastructure for
//! loading, registering, and executing tools from both Rust and Python implementations.

pub mod builtin;
pub mod loader;
#[cfg(feature = "python")]
pub mod python;
pub mod register;
pub mod registry;

use anyhow::Result;
use serde_json::Value;

use crate::llm::ToolSpec;

/// Core trait for tools that can be invoked by agents.
///
/// Tools are executable functions exposed to the LLM via JSON schemas. Each
/// tool has a unique name, provides a specification for the LLM, and implements
/// execution logic.
///
/// # Examples
///
/// ```
/// use appam::tools::Tool;
/// use appam::llm::ToolSpec;
/// use serde_json::{json, Value};
/// use anyhow::Result;
///
/// struct EchoTool;
///
/// impl Tool for EchoTool {
///     fn name(&self) -> &str {
///         "echo"
///     }
///
///     fn spec(&self) -> Result<ToolSpec> {
///         Ok(serde_json::from_value(json!({
///             "type": "function",
///             "function": {
///                 "name": "echo",
///                 "description": "Echo back the input message",
///                 "parameters": {
///                     "type": "object",
///                     "properties": {
///                         "message": {
///                             "type": "string",
///                             "description": "Message to echo"
///                         }
///                     },
///                     "required": ["message"]
///                 }
///             }
///         }))?)
///     }
///
///     fn execute(&self, args: Value) -> Result<Value> {
///         let msg = args["message"].as_str().unwrap_or("");
///         Ok(json!({ "output": msg }))
///     }
/// }
/// ```
pub trait Tool: Send + Sync {
    /// Return the unique, stable function name for this tool.
    ///
    /// This name must match the name in the tool specification and is used
    /// for routing LLM tool calls to the correct implementation.
    fn name(&self) -> &str;

    /// Return the tool specification for the LLM.
    ///
    /// The specification includes the function signature, parameter schema,
    /// and description. This is typically loaded from a JSON file to maintain
    /// a single source of truth.
    ///
    /// # Errors
    ///
    /// Returns an error if the specification cannot be loaded or parsed.
    fn spec(&self) -> Result<ToolSpec>;

    /// Execute the tool with the given arguments.
    ///
    /// Arguments are provided as a JSON value matching the schema from `spec()`.
    /// The tool should validate inputs, perform its operation, and return a
    /// JSON result.
    ///
    /// # Errors
    ///
    /// Returns an error if arguments are invalid, execution fails, or results
    /// cannot be serialized.
    ///
    /// # Security
    ///
    /// Tool implementations must validate all inputs, avoid shell injection,
    /// and respect sandbox boundaries. Never trust arguments from the LLM.
    fn execute(&self, args: Value) -> Result<Value>;
}

pub use registry::ToolRegistry;
