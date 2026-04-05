//! Agent configuration structures.
//!
//! Defines the TOML schema for agent definitions, including metadata, model
//! selection, system prompts, and tool configurations.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::tools::loader::ToolConfig;

/// Agent configuration loaded from TOML.
///
/// Defines all settings needed to instantiate and run an agent: name, model,
/// system prompt, and tools.
///
/// # Example TOML
///
/// ```toml
/// [agent]
/// name = "assistant"
/// model = "openai/gpt-5"
/// system_prompt = "prompts/assistant.txt"
/// description = "A helpful AI assistant"
///
/// [[tools]]
/// name = "bash"
/// schema = "tools/bash.json"
/// implementation = { type = "rust", module = "appam::tools::builtin::bash" }
///
/// [[tools]]
/// name = "analyze_code"
/// schema = "tools/analyze.json"
/// implementation = { type = "python", script = "tools/analyze.py" }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent metadata and settings
    pub agent: AgentMetadata,
    /// List of tool configurations
    #[serde(default)]
    pub tools: Vec<ToolConfig>,
}

/// Agent metadata.
///
/// Core agent identification and configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    /// Agent name (must be unique)
    pub name: String,
    /// LLM model to use (e.g., "openai/gpt-5", "anthropic/claude-3.5-sonnet")
    #[serde(default)]
    pub model: Option<String>,
    /// Path to system prompt file (relative to agent config directory)
    pub system_prompt: PathBuf,
    /// Optional description of agent capabilities
    #[serde(default)]
    pub description: Option<String>,
    /// Optional version identifier
    #[serde(default)]
    pub version: Option<String>,
}

impl AgentConfig {
    /// Load agent configuration from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        let config: AgentConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Validate the agent configuration.
    ///
    /// Checks that required fields are present and that referenced files exist.
    /// Does not validate tool schemas or implementations.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate(&self, base_dir: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let base = base_dir.as_ref();

        // Check system prompt exists
        let prompt_path = if self.agent.system_prompt.is_absolute() {
            self.agent.system_prompt.clone()
        } else {
            base.join(&self.agent.system_prompt)
        };

        if !prompt_path.exists() {
            anyhow::bail!("System prompt file not found: {}", prompt_path.display());
        }

        // Validate tool names are unique
        let mut seen_names = std::collections::HashSet::new();
        for tool in &self.tools {
            if !seen_names.insert(&tool.name) {
                anyhow::bail!("Duplicate tool name: {}", tool.name);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parse_agent_config() {
        let toml = r#"
[agent]
name = "test_agent"
model = "openai/gpt-5"
system_prompt = "prompts/test.txt"
description = "A test agent"
version = "1.0.0"

[[tools]]
name = "echo"
schema = "tools/echo.json"
implementation = { type = "python", script = "tools/echo.py" }
"#;

        let config: AgentConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.agent.name, "test_agent");
        assert_eq!(config.agent.model.as_deref(), Some("openai/gpt-5"));
        assert_eq!(config.tools.len(), 1);
        assert_eq!(config.tools[0].name, "echo");
    }

    #[test]
    fn test_validate_missing_prompt() {
        let dir = TempDir::new().unwrap();
        let toml = r#"
[agent]
name = "test"
system_prompt = "nonexistent.txt"
"#;

        let config: AgentConfig = toml::from_str(toml).unwrap();
        let result = config.validate(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_duplicate_tools() {
        let toml = r#"
[agent]
name = "test"
system_prompt = "prompt.txt"

[[tools]]
name = "duplicate"
schema = "tools/a.json"
implementation = { type = "python", script = "a.py" }

[[tools]]
name = "duplicate"
schema = "tools/b.json"
implementation = { type = "python", script = "b.py" }
"#;

        let config: AgentConfig = toml::from_str(toml).unwrap();

        let dir = TempDir::new().unwrap();
        let prompt_path = dir.path().join("prompt.txt");
        std::fs::write(&prompt_path, "test").unwrap();

        let result = config.validate(dir.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Duplicate tool name"));
    }
}
