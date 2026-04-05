//! Integration tests for hybrid TOML + programmatic configuration.
#![cfg(feature = "python")]

use anyhow::Result;
use appam::prelude::*;
use serde_json::{json, Value};
use std::io::Write;
use std::sync::Arc;
use tempfile::TempDir;

struct CustomTool {
    name: String,
}

impl Tool for CustomTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec(&self) -> Result<ToolSpec> {
        Ok(serde_json::from_value(json!({
            "type": "function",
            "name": self.name,
            "description": "Custom tool added programmatically",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }))?)
    }

    fn execute(&self, _args: Value) -> Result<Value> {
        Ok(json!({"output": "custom tool result", "success": true}))
    }
}

fn create_test_agent_files() -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().unwrap();
    let base = dir.path();

    // Create agent config
    let config_path = base.join("agent.toml");
    let mut config_file = std::fs::File::create(&config_path).unwrap();
    config_file
        .write_all(
            br#"
[agent]
name = "test_agent"
model = "openai/gpt-4"
system_prompt = "prompt.txt"

[[tools]]
name = "echo"
schema = "echo.json"
implementation = { type = "python", script = "echo.py" }
"#,
        )
        .unwrap();

    // Create system prompt
    let prompt_path = base.join("prompt.txt");
    std::fs::write(&prompt_path, "You are a test assistant.").unwrap();

    // Create tool schema
    let schema_path = base.join("echo.json");
    std::fs::write(
        &schema_path,
        r#"{"type":"function","name":"echo","description":"Echo tool","parameters":{"type":"object","properties":{"message":{"type":"string"}}}}"#,
    )
    .unwrap();

    // Create tool script
    let script_path = base.join("echo.py");
    std::fs::write(
        &script_path,
        r#"
def execute(args):
    return {"output": args.get("message", "")}
"#,
    )
    .unwrap();

    (dir, config_path)
}

#[test]
fn test_toml_agent_with_additional_tool() {
    let (_dir, config_path) = create_test_agent_files();

    let agent = TomlAgent::from_file(&config_path)
        .unwrap()
        .with_additional_tool(Arc::new(CustomTool {
            name: "custom".to_string(),
        }));

    let tools = agent.available_tools().unwrap();
    assert_eq!(tools.len(), 2); // echo + custom
    assert!(tools.iter().any(|t| t.name == "echo"));
    assert!(tools.iter().any(|t| t.name == "custom"));
}

#[test]
fn test_toml_agent_with_multiple_tools() {
    let (_dir, config_path) = create_test_agent_files();

    let agent = TomlAgent::from_file(&config_path)
        .unwrap()
        .with_additional_tools(vec![
            Arc::new(CustomTool {
                name: "custom1".to_string(),
            }),
            Arc::new(CustomTool {
                name: "custom2".to_string(),
            }),
        ]);

    let tools = agent.available_tools().unwrap();
    assert_eq!(tools.len(), 3); // echo + custom1 + custom2
}

#[test]
fn test_toml_agent_with_model_override() {
    let (_dir, config_path) = create_test_agent_files();

    let agent = TomlAgent::from_file(&config_path).unwrap();
    assert_eq!(agent.model(), "openai/gpt-4");

    let agent_overridden = agent.with_model("anthropic/claude-3.5-sonnet");
    assert_eq!(agent_overridden.model(), "anthropic/claude-3.5-sonnet");
}

#[test]
fn test_toml_agent_with_prompt_override() {
    let (_dir, config_path) = create_test_agent_files();

    let agent = TomlAgent::from_file(&config_path)
        .unwrap()
        .with_system_prompt_override("New custom prompt");

    assert_eq!(agent.system_prompt().unwrap(), "New custom prompt");
}

#[test]
fn test_toml_agent_execute_custom_tool() {
    let (_dir, config_path) = create_test_agent_files();

    let agent = TomlAgent::from_file(&config_path)
        .unwrap()
        .with_additional_tool(Arc::new(CustomTool {
            name: "custom".to_string(),
        }));

    let result = agent.execute_tool("custom", json!({})).unwrap();
    assert_eq!(result["output"], "custom tool result");
}

#[test]
fn test_hybrid_configuration_chaining() {
    let (_dir, config_path) = create_test_agent_files();

    let agent = TomlAgent::from_file(&config_path)
        .unwrap()
        .with_model("new-model")
        .with_additional_tool(Arc::new(CustomTool {
            name: "tool1".to_string(),
        }))
        .with_additional_tool(Arc::new(CustomTool {
            name: "tool2".to_string(),
        }))
        .with_system_prompt_override("Overridden prompt");

    assert_eq!(agent.model(), "new-model");
    assert_eq!(agent.system_prompt().unwrap(), "Overridden prompt");
    let tools = agent.available_tools().unwrap();
    assert_eq!(tools.len(), 3); // echo + tool1 + tool2
}
