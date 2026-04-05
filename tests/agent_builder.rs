//! Integration tests for AgentBuilder.

use anyhow::Result;
use appam::agent::Agent; // Explicit import for trait methods
use appam::prelude::*;
use serde_json::{json, Value};
use std::sync::Arc;

struct TestTool {
    name: String,
}

impl Tool for TestTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec(&self) -> Result<ToolSpec> {
        Ok(ToolSpec {
            type_field: "function".to_string(),
            name: self.name.clone(),
            description: "Test tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }),
            strict: None,
        })
    }

    fn execute(&self, _args: Value) -> Result<Value> {
        Ok(json!({"output": "test result", "success": true}))
    }
}

#[test]
fn test_agent_builder_basic() {
    let result = AgentBuilder::new("test-agent")
        .system_prompt("You are a test assistant.")
        .build();

    assert!(result.is_ok());
    let agent = result.unwrap();
    assert_eq!(agent.name(), "test-agent");
}

#[test]
fn test_agent_builder_with_model() {
    let agent = AgentBuilder::new("test-agent")
        .model("test-model")
        .system_prompt("Test prompt")
        .build()
        .unwrap();

    assert_eq!(agent.model(), "test-model");
}

#[test]
fn test_agent_builder_with_tool() {
    let agent = AgentBuilder::new("test-agent")
        .system_prompt("Test prompt")
        .with_tool(Arc::new(TestTool {
            name: "test_tool".to_string(),
        }))
        .build()
        .unwrap();

    let tools = agent.available_tools().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "test_tool");
}

#[test]
fn test_agent_builder_with_multiple_tools() {
    let agent = AgentBuilder::new("test-agent")
        .system_prompt("Test prompt")
        .with_tools(vec![
            Arc::new(TestTool {
                name: "tool1".to_string(),
            }),
            Arc::new(TestTool {
                name: "tool2".to_string(),
            }),
        ])
        .build()
        .unwrap();

    let tools = agent.available_tools().unwrap();
    assert_eq!(tools.len(), 2);
}

#[test]
fn test_agent_builder_missing_prompt() {
    let result = AgentBuilder::new("test-agent").build();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("System prompt"));
}

#[test]
fn test_agent_builder_with_prompt_file() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"Test prompt from file").unwrap();
    file.flush().unwrap();

    let agent = AgentBuilder::new("test-agent")
        .system_prompt_file(file.path())
        .build()
        .unwrap();

    assert_eq!(agent.system_prompt().unwrap(), "Test prompt from file");
}

#[test]
fn test_agent_builder_with_custom_registry() {
    let registry = Arc::new(ToolRegistry::new());
    registry.register(Arc::new(TestTool {
        name: "pre_registered".to_string(),
    }));

    let agent = AgentBuilder::new("test-agent")
        .system_prompt("Test prompt")
        .with_registry(registry)
        .with_tool(Arc::new(TestTool {
            name: "added_later".to_string(),
        }))
        .build()
        .unwrap();

    let tools = agent.available_tools().unwrap();
    assert_eq!(tools.len(), 2);
}

#[test]
fn test_agent_execute_tool() {
    let agent = AgentBuilder::new("test-agent")
        .system_prompt("Test prompt")
        .with_tool(Arc::new(TestTool {
            name: "test_tool".to_string(),
        }))
        .build()
        .unwrap();

    let result = agent
        .execute_tool("test_tool", json!({"input": "test"}))
        .unwrap();
    assert_eq!(result["output"], "test result");
    assert_eq!(result["success"], true);
}

#[test]
fn test_runtime_agent_modification() {
    let mut agent = AgentBuilder::new("test-agent")
        .system_prompt("Original prompt")
        .build()
        .unwrap();

    agent.set_system_prompt("New prompt");
    assert_eq!(agent.system_prompt().unwrap(), "New prompt");

    agent.add_tool(Arc::new(TestTool {
        name: "dynamic_tool".to_string(),
    }));

    let tools = agent.available_tools().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "dynamic_tool");
}
