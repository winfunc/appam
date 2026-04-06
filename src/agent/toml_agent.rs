//! TOML-based agent implementation.
//!
//! Provides an agent that loads its configuration, system prompt, and tools
//! from a TOML file, enabling zero-code agent creation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use tracing::info;

use super::{config::AgentConfig, Agent};
use crate::llm::ToolSpec;
use crate::tools::{
    loader::load_tools, registry::ToolRegistry, AsyncTool, ToolConcurrency, ToolContext,
};

/// Agent loaded from a TOML configuration file.
///
/// This agent dynamically loads its system prompt, model settings, and tool
/// definitions from a TOML file, making it easy to create new agents without
/// writing code.
///
/// # Configuration
///
/// The agent expects a TOML file with the following structure:
///
/// ```toml
/// [agent]
/// name = "my_agent"
/// model = "openai/gpt-5"
/// system_prompt = "prompts/my_prompt.txt"
///
/// [[tools]]
/// name = "tool1"
/// schema = "tools/tool1.json"
/// implementation = { type = "python", script = "tools/tool1.py" }
/// ```
///
/// Paths are resolved relative to the directory containing the TOML file.
///
/// # Examples
///
/// ```no_run
/// use appam::agent::{Agent, TomlAgent};
/// use anyhow::Result;
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let agent = TomlAgent::from_file("agents/assistant.toml")?;
///     agent.run("What can you do?").await?;
///     Ok(())
/// }
/// ```
pub struct TomlAgent {
    /// Agent configuration
    config: AgentConfig,
    /// Base directory for resolving relative paths
    base_dir: PathBuf,
    /// Loaded system prompt
    system_prompt: String,
    /// Tool registry with loaded tools
    registry: Arc<ToolRegistry>,
    /// Optional model override (from global config or CLI)
    model_override: Option<String>,
}

impl TomlAgent {
    /// Load an agent from a TOML configuration file.
    ///
    /// Reads the configuration, validates it, loads the system prompt, and
    /// instantiates all tools (both Rust and Python).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The config file cannot be read or parsed
    /// - The system prompt file is missing
    /// - Any tool fails to load
    /// - Configuration validation fails
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading agent from TOML");

        let config = AgentConfig::from_file(path)
            .with_context(|| format!("Failed to load agent config: {}", path.display()))?;

        let base_dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        config
            .validate(&base_dir)
            .context("Agent configuration validation failed")?;

        // Load system prompt
        let prompt_path = if config.agent.system_prompt.is_absolute() {
            config.agent.system_prompt.clone()
        } else {
            base_dir.join(&config.agent.system_prompt)
        };

        let system_prompt = std::fs::read_to_string(&prompt_path)
            .with_context(|| format!("Failed to read system prompt: {}", prompt_path.display()))?;

        // Load tools
        let registry = Arc::new(ToolRegistry::new());
        load_tools(&config.tools, &base_dir, &registry).context("Failed to load tools")?;

        info!(
            agent = %config.agent.name,
            tools = registry.len(),
            "Agent loaded successfully"
        );

        Ok(Self {
            config,
            base_dir,
            system_prompt,
            registry,
            model_override: None,
        })
    }

    /// Create an agent with a model override.
    ///
    /// This allows runtime model selection that overrides the configuration.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_override = Some(model.into());
        self
    }

    /// Get the model to use for this agent.
    ///
    /// Returns the override if set, otherwise the model from config, or a
    /// default if neither is specified.
    pub fn model(&self) -> String {
        self.model_override
            .clone()
            .or_else(|| self.config.agent.model.clone())
            .unwrap_or_else(|| "openai/gpt-5".to_string())
    }

    /// Get the base directory for this agent.
    ///
    /// Used for resolving relative paths in tool configurations.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Get the tool registry for this agent.
    pub fn registry(&self) -> Arc<ToolRegistry> {
        Arc::clone(&self.registry)
    }

    /// Get the agent configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Add an additional tool to the agent's registry.
    ///
    /// This allows extending TOML-configured agents with programmatically
    /// defined tools.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::TomlAgent;
    /// # use std::sync::Arc;
    /// # use appam::tools::Tool;
    /// # struct CustomTool;
    /// # impl Tool for CustomTool {
    /// #     fn name(&self) -> &str { "custom" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// let agent = TomlAgent::from_file("agent.toml")?
    ///     .with_additional_tool(Arc::new(CustomTool));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn with_additional_tool(self, tool: Arc<dyn crate::tools::Tool>) -> Self {
        self.registry.register(tool);
        self
    }

    /// Add an additional async/context-aware tool to the agent's registry.
    pub fn with_additional_async_tool(self, tool: Arc<dyn AsyncTool>) -> Self {
        self.registry.register_async(tool);
        self
    }

    /// Add multiple additional tools at once.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::TomlAgent;
    /// # use std::sync::Arc;
    /// # use appam::tools::Tool;
    /// # struct Tool1;
    /// # struct Tool2;
    /// # impl Tool for Tool1 {
    /// #     fn name(&self) -> &str { "tool1" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// # impl Tool for Tool2 {
    /// #     fn name(&self) -> &str { "tool2" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// let agent = TomlAgent::from_file("agent.toml")?
    ///     .with_additional_tools(vec![Arc::new(Tool1), Arc::new(Tool2)]);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn with_additional_tools(self, tools: Vec<Arc<dyn crate::tools::Tool>>) -> Self {
        for tool in tools {
            self.registry.register(tool);
        }
        self
    }

    /// Add multiple additional async/context-aware tools at once.
    pub fn with_additional_async_tools(self, tools: Vec<Arc<dyn AsyncTool>>) -> Self {
        for tool in tools {
            self.registry.register_async(tool);
        }
        self
    }

    /// Override the system prompt with a new one.
    ///
    /// Replaces the prompt loaded from the TOML file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::TomlAgent;
    /// let agent = TomlAgent::from_file("agent.toml")?
    ///     .with_system_prompt_override("You are a specialized assistant.");
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn with_system_prompt_override(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

#[async_trait]
impl Agent for TomlAgent {
    fn name(&self) -> &str {
        &self.config.agent.name
    }

    fn system_prompt(&self) -> Result<String> {
        Ok(self.system_prompt.clone())
    }

    fn available_tools(&self) -> Result<Vec<ToolSpec>> {
        self.registry.specs()
    }

    fn execute_tool(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        self.registry.execute(name, args)
    }

    async fn execute_tool_with_context(
        &self,
        name: &str,
        ctx: ToolContext,
        args: serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.registry.execute_with_context(ctx, name, args).await
    }

    fn tool_concurrency(&self, name: &str) -> ToolConcurrency {
        self.registry
            .concurrency(name)
            .unwrap_or(ToolConcurrency::SerialOnly)
    }

    // Uses default run implementation from runtime module
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "python")]
    use std::io::Write;
    #[cfg(feature = "python")]
    use tempfile::TempDir;

    #[cfg(feature = "python")]
    fn create_test_agent_files() -> (TempDir, PathBuf) {
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
model = "openai/gpt-5"
system_prompt = "prompt.txt"
description = "Test agent"

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
            r#"{"type": "function", "name": "echo", "description": "Echo tool", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}}"#,
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

    #[cfg(feature = "python")]
    #[test]
    fn test_load_toml_agent() {
        let (_dir, config_path) = create_test_agent_files();
        let agent = TomlAgent::from_file(&config_path).unwrap();

        assert_eq!(agent.name(), "test_agent");
        assert_eq!(agent.model(), "openai/gpt-5");
        assert!(agent.system_prompt().unwrap().contains("test assistant"));

        let tools = agent.available_tools().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_agent_with_model_override() {
        let (_dir, config_path) = create_test_agent_files();
        let agent = TomlAgent::from_file(&config_path)
            .unwrap()
            .with_model("anthropic/claude-3.5-sonnet");

        assert_eq!(agent.model(), "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn test_load_nonexistent_config() {
        let result = TomlAgent::from_file("nonexistent.toml");
        assert!(result.is_err());
    }
}
