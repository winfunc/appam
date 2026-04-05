//! Tool registration helpers for simplified tool creation.
//!
//! Provides convenience functions for registering tools without full trait
//! implementations, including closure-based tools.

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use super::{Tool, ToolRegistry};
use crate::llm::ToolSpec;

/// A tool created from a closure or function.
///
/// This allows quick tool creation without implementing the full Tool trait.
pub struct ClosureTool {
    name: String,
    spec: ToolSpec,
    execute_fn: Box<dyn Fn(Value) -> Result<Value> + Send + Sync>,
}

impl ClosureTool {
    /// Create a new closure-based tool.
    ///
    /// # Parameters
    ///
    /// - `name`: Tool name
    /// - `spec`: Tool specification (JSON schema)
    /// - `execute_fn`: Function that executes the tool
    ///
    /// # Examples
    ///
    /// ```
    /// # use appam::llm::ToolSpec;
    /// # use appam::tools::register::ClosureTool;
    /// # use serde_json::{json, Value};
    /// # use anyhow::Result;
    /// let tool = ClosureTool::new(
    ///     "echo",
    ///     serde_json::from_value(json!({
    ///         "type": "function",
    ///         "name": "echo",
    ///         "description": "Echo tool",
    ///         "parameters": {
    ///             "type": "object",
    ///             "properties": {
    ///                 "message": {"type": "string"}
    ///             }
    ///         }
    ///     })).unwrap(),
    ///     |args: Value| {
    ///         Ok(json!({"output": args["message"]}))
    ///     }
    /// );
    /// ```
    pub fn new<F>(name: impl Into<String>, spec: ToolSpec, execute_fn: F) -> Self
    where
        F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            spec,
            execute_fn: Box::new(execute_fn),
        }
    }
}

impl Tool for ClosureTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec(&self) -> Result<ToolSpec> {
        Ok(self.spec.clone())
    }

    fn execute(&self, args: Value) -> Result<Value> {
        (self.execute_fn)(args)
    }
}

/// Extension trait for ToolRegistry to register closure-based tools.
pub trait ToolRegistryExt {
    /// Register a tool using a closure.
    ///
    /// This is a convenience method for quickly adding simple tools without
    /// implementing the full Tool trait.
    ///
    /// # Examples
    ///
    /// ```
    /// use appam::tools::{ToolRegistry, register::ToolRegistryExt};
    /// use serde_json::{json, Value};
    ///
    /// let registry = ToolRegistry::new();
    /// registry.register_fn(
    ///     "echo",
    ///     serde_json::from_value(json!({
    ///         "type": "function",
    ///         "name": "echo",
    ///         "description": "Echo tool",
    ///         "parameters": {
    ///             "type": "object",
    ///             "properties": {
    ///                 "message": {"type": "string"}
    ///             }
    ///         }
    ///     })).unwrap(),
    ///     |args: Value| {
    ///         Ok(json!({"output": args["message"]}))
    ///     }
    /// );
    /// ```
    fn register_fn<F>(&self, name: impl Into<String>, spec: ToolSpec, execute_fn: F)
    where
        F: Fn(Value) -> Result<Value> + Send + Sync + 'static;
}

impl ToolRegistryExt for ToolRegistry {
    fn register_fn<F>(&self, name: impl Into<String>, spec: ToolSpec, execute_fn: F)
    where
        F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    {
        let tool = ClosureTool::new(name, spec, execute_fn);
        self.register(Arc::new(tool));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_closure_tool() {
        let tool = ClosureTool::new(
            "test",
            serde_json::from_value(json!({
                "type": "function",
                "name": "test",
                "description": "Test tool",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }))
            .unwrap(),
            |_args: Value| Ok(json!({"result": "success"})),
        );

        assert_eq!(tool.name(), "test");
        let result = tool.execute(json!({})).unwrap();
        assert_eq!(result["result"], "success");
    }

    #[test]
    fn test_register_fn() {
        let registry = ToolRegistry::new();

        registry.register_fn(
            "echo",
            serde_json::from_value(json!({
                "type": "function",
                "name": "echo",
                "description": "Echo tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    }
                }
            }))
            .unwrap(),
            |args: Value| Ok(json!({"output": args["message"]})),
        );

        let result = registry
            .execute("echo", json!({"message": "hello"}))
            .unwrap();
        assert_eq!(result["output"], "hello");
    }
}
