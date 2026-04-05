//! Dynamic tool registry for runtime tool management.
//!
//! Replaces the hardcoded match-based tool resolution with a flexible registry
//! that supports loading tools from TOML configurations.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Result};

use super::Tool;

/// Thread-safe registry for tool implementations.
///
/// The registry maps tool names to boxed trait objects, allowing dynamic
/// lookup and execution. Tools can be registered at runtime from agent
/// configurations.
///
/// # Thread Safety
///
/// The registry uses an `RwLock` to allow concurrent reads (tool lookups)
/// while serializing writes (tool registration).
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Arc<RwLock<HashMap<String, Arc<dyn Tool>>>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tool_count", &self.len())
            .finish()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create a new empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a tool in the registry.
    ///
    /// If a tool with the same name already exists, it will be replaced.
    ///
    /// # Examples
    ///
    /// ```
    /// use appam::tools::{ToolRegistry, Tool};
    /// use std::sync::Arc;
    ///
    /// let registry = ToolRegistry::new();
    /// // registry.register(Arc::new(MyTool));
    /// ```
    pub fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        let mut tools = self.tools.write().unwrap();
        tools.insert(name, tool);
    }

    /// Resolve a tool by name.
    ///
    /// Returns a cloned Arc to the tool implementation, or None if the tool
    /// is not registered.
    ///
    /// # Examples
    ///
    /// ```
    /// use appam::tools::ToolRegistry;
    ///
    /// let registry = ToolRegistry::new();
    /// let tool = registry.resolve("bash");
    /// ```
    pub fn resolve(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let tools = self.tools.read().unwrap();
        tools.get(name).cloned()
    }

    /// List all registered tool names.
    ///
    /// Returns a sorted vector of tool names currently in the registry.
    pub fn list(&self) -> Vec<String> {
        let tools = self.tools.read().unwrap();
        let mut names: Vec<String> = tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        let tools = self.tools.read().unwrap();
        tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Unregister a tool by name.
    ///
    /// Returns the unregistered tool if it existed, or None if not found.
    pub fn unregister(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let mut tools = self.tools.write().unwrap();
        tools.remove(name)
    }

    /// Clear all tools from the registry.
    pub fn clear(&self) {
        let mut tools = self.tools.write().unwrap();
        tools.clear();
    }

    /// Create a registry pre-populated with built-in tools.
    ///
    /// This is a convenience method for creating a registry with all standard
    /// tools (bash, read_file, write_file, etc.) already registered.
    pub fn with_builtins() -> Self {
        // Register built-in tools here once they're implemented
        // registry.register(Arc::new(builtin::BashTool::new()));
        // etc.

        Self::new()
    }

    /// Register multiple tools at once.
    ///
    /// # Examples
    ///
    /// ```
    /// use appam::tools::ToolRegistry;
    /// use std::sync::Arc;
    ///
    /// let registry = ToolRegistry::new();
    /// // registry.register_many(vec![
    /// //     Arc::new(Tool1),
    /// //     Arc::new(Tool2),
    /// // ]);
    /// ```
    pub fn register_many(&self, tools: Vec<Arc<dyn Tool>>) {
        for tool in tools {
            self.register(tool);
        }
    }

    /// Execute a tool by name with the given arguments.
    ///
    /// This is a convenience method that combines resolution and execution.
    ///
    /// # Errors
    ///
    /// Returns an error if the tool is not found or execution fails.
    pub fn execute(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        let tool = self
            .resolve(name)
            .ok_or_else(|| anyhow!("Tool not found: {}", name))?;

        tool.execute(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolSpec;
    use serde_json::json;

    struct MockTool {
        name: String,
    }

    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn spec(&self) -> Result<ToolSpec> {
            Ok(serde_json::from_value(json!({
                "type": "function",
                "name": self.name,
                "description": "Mock tool",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }))?)
        }

        fn execute(&self, _args: serde_json::Value) -> Result<serde_json::Value> {
            Ok(json!({"success": true}))
        }
    }

    #[test]
    fn test_registry_basic_operations() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());

        let tool = Arc::new(MockTool {
            name: "test".to_string(),
        });
        registry.register(tool.clone());

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.resolve("test").is_some());
        assert!(registry.resolve("nonexistent").is_none());

        let names = registry.list();
        assert_eq!(names, vec!["test"]);

        registry.clear();
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_execute() {
        let registry = ToolRegistry::new();
        let tool = Arc::new(MockTool {
            name: "test".to_string(),
        });
        registry.register(tool);

        let result = registry.execute("test", json!({})).unwrap();
        assert_eq!(result, json!({"success": true}));

        let error = registry.execute("nonexistent", json!({}));
        assert!(error.is_err());
    }
}
