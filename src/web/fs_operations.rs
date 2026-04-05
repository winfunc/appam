//! File system operations for agent management.
//!
//! Provides safe, validated file operations for creating, updating, and deleting
//! agent configurations, tools, and prompts. Includes path validation to prevent
//! directory traversal and other security issues.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use tracing::{debug, info};

use crate::agent::config::AgentConfig;
use crate::tools::loader::ToolImplementation;

#[cfg(test)]
use crate::agent::config::AgentMetadata;

/// Validate an agent name for safe file system use.
///
/// Agent names must:
/// - Be non-empty
/// - Contain only alphanumeric characters, hyphens, and underscores
/// - Not start with a dot
/// - Not contain path separators
///
/// # Errors
///
/// Returns an error if the name is invalid or potentially unsafe.
pub fn validate_agent_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("Agent name cannot be empty");
    }

    if name.starts_with('.') {
        bail!("Agent name cannot start with a dot");
    }

    if name.contains('/') || name.contains('\\') {
        bail!("Agent name cannot contain path separators");
    }

    if name.contains("..") {
        bail!("Agent name cannot contain '..'");
    }

    // Allow alphanumeric, hyphen, underscore
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        bail!("Agent name can only contain alphanumeric characters, hyphens, and underscores");
    }

    Ok(())
}

/// Validate a tool name for safe file system use.
///
/// Uses the same rules as agent names.
pub fn validate_tool_name(name: &str) -> Result<()> {
    validate_agent_name(name).context("Invalid tool name")
}

/// Create an agent directory structure.
///
/// Creates:
/// - agents/{agent_name}/
/// - agents/{agent_name}/tools/
///
/// # Errors
///
/// Returns an error if the directory already exists or cannot be created.
pub fn create_agent_directory(base_dir: &Path, agent_name: &str) -> Result<PathBuf> {
    validate_agent_name(agent_name)?;

    let agent_dir = base_dir.join(agent_name);

    if agent_dir.exists() {
        bail!("Agent directory already exists: {}", agent_dir.display());
    }

    fs::create_dir_all(&agent_dir)
        .with_context(|| format!("Failed to create agent directory: {}", agent_dir.display()))?;

    let tools_dir = agent_dir.join("tools");
    fs::create_dir_all(&tools_dir)
        .with_context(|| format!("Failed to create tools directory: {}", tools_dir.display()))?;

    info!(
        agent = %agent_name,
        path = %agent_dir.display(),
        "Created agent directory"
    );

    Ok(agent_dir)
}

/// Write agent TOML configuration file.
///
/// Serializes the agent configuration and writes it to the specified path,
/// or generates a path based on the agent name if no path is provided.
///
/// # Parameters
///
/// - `agent_dir`: Base directory for the agent
/// - `config`: Agent configuration to write
/// - `toml_path`: Optional specific file path to write to (preserves original filename)
///
/// # Errors
///
/// Returns an error if serialization or file writing fails.
pub fn write_agent_toml_to_path(config: &AgentConfig, toml_path: &Path) -> Result<()> {
    // Build TOML content manually to ensure proper inline table formatting
    let mut toml_content = String::new();

    // Write [agent] section
    toml_content.push_str("[agent]\n");
    toml_content.push_str(&format!("name = \"{}\"\n", config.agent.name));

    if let Some(ref model) = config.agent.model {
        toml_content.push_str(&format!("model = \"{}\"\n", model));
    }

    toml_content.push_str(&format!(
        "system_prompt = \"{}\"\n",
        config.agent.system_prompt.display()
    ));

    if let Some(ref desc) = config.agent.description {
        if !desc.is_empty() {
            toml_content.push_str(&format!("description = \"{}\"\n", desc));
        }
    }

    if let Some(ref version) = config.agent.version {
        if !version.is_empty() {
            toml_content.push_str(&format!("version = \"{}\"\n", version));
        }
    }

    // Write [[tools]] sections with inline table format
    for tool in &config.tools {
        toml_content.push_str("\n[[tools]]\n");
        toml_content.push_str(&format!("name = \"{}\"\n", tool.name));
        toml_content.push_str(&format!("schema = \"{}\"\n", tool.schema.display()));

        // Write implementation as inline table
        match &tool.implementation {
            ToolImplementation::Python { script } => {
                toml_content.push_str(&format!(
                    "implementation = {{ type = \"python\", script = \"{}\" }}\n",
                    script.display()
                ));
            }
            ToolImplementation::Rust { module } => {
                toml_content.push_str(&format!(
                    "implementation = {{ type = \"rust\", module = \"{}\" }}\n",
                    module
                ));
            }
        }
    }

    fs::write(toml_path, toml_content)
        .with_context(|| format!("Failed to write TOML file: {}", toml_path.display()))?;

    debug!(path = %toml_path.display(), "Wrote agent TOML");
    Ok(())
}

/// Write agent TOML configuration file to agent directory.
///
/// Creates a new TOML file with the agent's name.
/// Use this when creating new agents.
///
/// # Errors
///
/// Returns an error if serialization or file writing fails.
pub fn write_agent_toml(agent_dir: &Path, config: &AgentConfig) -> Result<()> {
    let toml_path = agent_dir.join(format!("{}.toml", config.agent.name));
    write_agent_toml_to_path(config, &toml_path)
}

/// Write system prompt file.
///
/// Writes the prompt content to {agent_dir}/prompt.txt
///
/// # Errors
///
/// Returns an error if file writing fails.
pub fn write_system_prompt(agent_dir: &Path, content: &str) -> Result<()> {
    let prompt_path = agent_dir.join("prompt.txt");

    fs::write(&prompt_path, content)
        .with_context(|| format!("Failed to write prompt file: {}", prompt_path.display()))?;

    debug!(path = %prompt_path.display(), "Wrote system prompt");
    Ok(())
}

/// Tool file information for writing.
#[derive(Debug, Clone, Serialize)]
pub struct ToolFileInfo {
    /// The name of the tool.
    pub name: String,
    /// The JSON schema for the tool.
    pub schema_json: String,
    /// Optional Python implementation code.
    pub python_code: Option<String>,
}

/// Write tool files (JSON schema and Python implementation).
///
/// Creates:
/// - {agent_dir}/tools/{tool_name}.json
/// - {agent_dir}/tools/{tool_name}.py (if Python implementation)
///
/// # Errors
///
/// Returns an error if files cannot be written or JSON is invalid.
pub fn write_tool_files(agent_dir: &Path, tool: &ToolFileInfo) -> Result<()> {
    validate_tool_name(&tool.name)?;

    let tools_dir = agent_dir.join("tools");
    fs::create_dir_all(&tools_dir)
        .with_context(|| format!("Failed to create tools directory: {}", tools_dir.display()))?;

    // Validate JSON before writing
    serde_json::from_str::<serde_json::Value>(&tool.schema_json)
        .context("Invalid JSON schema for tool")?;

    // Write JSON schema
    let schema_path = tools_dir.join(format!("{}.json", tool.name));
    fs::write(&schema_path, &tool.schema_json)
        .with_context(|| format!("Failed to write tool schema: {}", schema_path.display()))?;

    debug!(path = %schema_path.display(), "Wrote tool schema");

    // Write Python implementation if provided
    if let Some(code) = &tool.python_code {
        let script_path = tools_dir.join(format!("{}.py", tool.name));
        fs::write(&script_path, code)
            .with_context(|| format!("Failed to write tool script: {}", script_path.display()))?;

        debug!(path = %script_path.display(), "Wrote tool script");
    }

    Ok(())
}

/// Delete a tool's files from an agent.
///
/// Removes both the JSON schema and Python script (if exists).
///
/// # Errors
///
/// Returns an error if files cannot be deleted.
pub fn delete_tool_files(agent_dir: &Path, tool_name: &str) -> Result<()> {
    validate_tool_name(tool_name)?;

    let tools_dir = agent_dir.join("tools");

    let schema_path = tools_dir.join(format!("{}.json", tool_name));
    if schema_path.exists() {
        fs::remove_file(&schema_path)
            .with_context(|| format!("Failed to delete tool schema: {}", schema_path.display()))?;
        debug!(path = %schema_path.display(), "Deleted tool schema");
    }

    let script_path = tools_dir.join(format!("{}.py", tool_name));
    if script_path.exists() {
        fs::remove_file(&script_path)
            .with_context(|| format!("Failed to delete tool script: {}", script_path.display()))?;
        debug!(path = %script_path.display(), "Deleted tool script");
    }

    Ok(())
}

/// Delete an agent directory and all its contents.
///
/// # Errors
///
/// Returns an error if the directory cannot be deleted.
pub fn delete_agent_directory(agent_dir: &Path) -> Result<()> {
    if !agent_dir.exists() {
        bail!("Agent directory does not exist: {}", agent_dir.display());
    }

    fs::remove_dir_all(agent_dir)
        .with_context(|| format!("Failed to delete agent directory: {}", agent_dir.display()))?;

    info!(path = %agent_dir.display(), "Deleted agent directory");
    Ok(())
}

/// Duplicate an agent directory with a new name.
///
/// Copies all files and updates the agent name in the TOML configuration.
///
/// # Errors
///
/// Returns an error if copying fails or the destination already exists.
pub fn duplicate_agent_directory(
    base_dir: &Path,
    src_agent_name: &str,
    dst_agent_name: &str,
) -> Result<PathBuf> {
    validate_agent_name(src_agent_name)?;
    validate_agent_name(dst_agent_name)?;

    let src_dir = base_dir.join(src_agent_name);
    let dst_dir = base_dir.join(dst_agent_name);

    if !src_dir.exists() {
        bail!(
            "Source agent directory does not exist: {}",
            src_dir.display()
        );
    }

    if dst_dir.exists() {
        bail!(
            "Destination agent directory already exists: {}",
            dst_dir.display()
        );
    }

    // Copy directory recursively
    copy_dir_recursive(&src_dir, &dst_dir)?;

    // Update the TOML file with new agent name
    let src_toml_path = dst_dir.join(format!("{}.toml", src_agent_name));

    if src_toml_path.exists() {
        // Read and parse TOML
        let toml_content =
            fs::read_to_string(&src_toml_path).context("Failed to read source TOML")?;

        let mut config: AgentConfig =
            toml::from_str(&toml_content).context("Failed to parse source TOML")?;

        // Update agent name
        config.agent.name = dst_agent_name.to_string();

        // Write updated TOML
        write_agent_toml(&dst_dir, &config)?;

        // Remove old TOML file
        fs::remove_file(&src_toml_path).context("Failed to remove old TOML file")?;
    }

    info!(
        src = %src_dir.display(),
        dst = %dst_dir.display(),
        "Duplicated agent directory"
    );

    Ok(dst_dir)
}

/// Copy a directory recursively.
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)
        .with_context(|| format!("Failed to create directory: {}", dst.display()))?;

    for entry in
        fs::read_dir(src).with_context(|| format!("Failed to read directory: {}", src.display()))?
    {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path).with_context(|| {
                format!(
                    "Failed to copy file from {} to {}",
                    src_path.display(),
                    dst_path.display()
                )
            })?;
        }
    }

    Ok(())
}

/// Read agent configuration including all associated files.
///
/// Returns the raw TOML content, parsed config, prompt, and tool details.
///
/// # Errors
///
/// Returns an error if files cannot be read or parsed.
pub fn read_agent_config(agent_dir: &Path, agent_name: &str) -> Result<AgentConfigData> {
    validate_agent_name(agent_name)?;

    // Verify directory exists
    if !agent_dir.exists() {
        bail!("Agent directory does not exist: {}", agent_dir.display());
    }

    if !agent_dir.is_dir() {
        bail!("Path is not a directory: {}", agent_dir.display());
    }

    // Search for the TOML file that contains this agent name
    // The filename might not match the agent name (e.g., simple_agent.toml vs simple_assistant)
    let mut toml_path = None;
    let mut toml_content = String::new();

    for entry in fs::read_dir(agent_dir)
        .with_context(|| format!("Failed to read directory: {}", agent_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            let content = fs::read_to_string(&path)?;

            // Try to parse and check if this is the right agent
            if let Ok(config) = toml::from_str::<AgentConfig>(&content) {
                if config.agent.name == agent_name {
                    toml_path = Some(path);
                    toml_content = content;
                    break;
                }
            }
        }
    }

    let toml_path = toml_path
        .ok_or_else(|| anyhow::anyhow!("No TOML file found for agent '{}'", agent_name))?;

    let config: AgentConfig = toml::from_str(&toml_content)
        .with_context(|| format!("Failed to parse TOML: {}", toml_path.display()))?;

    // Read prompt
    let prompt_path = agent_dir.join(&config.agent.system_prompt);
    let prompt_content = fs::read_to_string(&prompt_path)
        .with_context(|| format!("Failed to read prompt: {}", prompt_path.display()))?;

    // Read tool details
    let mut tools_details = Vec::new();
    for tool_config in &config.tools {
        let schema_path = agent_dir.join(&tool_config.schema);
        let schema_json = fs::read_to_string(&schema_path)
            .with_context(|| format!("Failed to read tool schema: {}", schema_path.display()))?;

        let (implementation_type, code) = match &tool_config.implementation {
            ToolImplementation::Python { script } => {
                let script_path = agent_dir.join(script);
                let python_code = fs::read_to_string(&script_path).with_context(|| {
                    format!("Failed to read tool script: {}", script_path.display())
                })?;
                ("python".to_string(), Some(python_code))
            }
            ToolImplementation::Rust { module } => (format!("rust:{}", module), None),
        };

        tools_details.push(ToolDetailData {
            name: tool_config.name.clone(),
            schema_json,
            implementation_type,
            code,
        });
    }

    Ok(AgentConfigData {
        name: agent_name.to_string(),
        config_toml: toml_content,
        config_parsed: config,
        prompt_content,
        tools_details,
        base_path: agent_dir.to_string_lossy().to_string(),
        toml_file_path: toml_path.to_string_lossy().to_string(),
    })
}

/// Complete agent configuration data.
#[derive(Debug, Clone, Serialize)]
pub struct AgentConfigData {
    /// The name of the agent.
    pub name: String,
    /// The raw TOML configuration content.
    pub config_toml: String,
    /// The parsed agent configuration.
    pub config_parsed: AgentConfig,
    /// The content of the system prompt file.
    pub prompt_content: String,
    /// Details of all tools configured for the agent.
    pub tools_details: Vec<ToolDetailData>,
    /// The base directory path of the agent.
    pub base_path: String,
    /// The full path to the TOML configuration file.
    pub toml_file_path: String,
}

/// Tool detail data.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDetailData {
    /// The name of the tool.
    pub name: String,
    /// The JSON schema for the tool.
    pub schema_json: String,
    /// The implementation type (e.g., "python", "rust:module_name").
    pub implementation_type: String,
    /// Optional implementation code for Python tools.
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_validate_agent_name() {
        assert!(validate_agent_name("my_agent").is_ok());
        assert!(validate_agent_name("my-agent-123").is_ok());
        assert!(validate_agent_name("").is_err());
        assert!(validate_agent_name(".hidden").is_err());
        assert!(validate_agent_name("../etc/passwd").is_err());
        assert!(validate_agent_name("agent/name").is_err());
        assert!(validate_agent_name("agent name").is_err());
    }

    #[test]
    fn test_create_agent_directory() {
        let temp_dir = TempDir::new().unwrap();
        let agent_dir = create_agent_directory(temp_dir.path(), "test_agent").unwrap();

        assert!(agent_dir.exists());
        assert!(agent_dir.join("tools").exists());
    }

    #[test]
    fn test_write_agent_toml() {
        let temp_dir = TempDir::new().unwrap();
        let agent_dir = temp_dir.path().join("test_agent");
        fs::create_dir(&agent_dir).unwrap();

        let config = AgentConfig {
            agent: AgentMetadata {
                name: "test_agent".to_string(),
                model: Some("openai/gpt-5".to_string()),
                system_prompt: PathBuf::from("prompt.txt"),
                description: Some("Test agent".to_string()),
                version: Some("1.0.0".to_string()),
            },
            tools: vec![],
        };

        write_agent_toml(&agent_dir, &config).unwrap();

        let toml_path = agent_dir.join("test_agent.toml");
        assert!(toml_path.exists());
    }

    #[test]
    fn test_write_system_prompt() {
        let temp_dir = TempDir::new().unwrap();
        let agent_dir = temp_dir.path().join("test_agent");
        fs::create_dir(&agent_dir).unwrap();

        write_system_prompt(&agent_dir, "You are a helpful assistant.").unwrap();

        let prompt_path = agent_dir.join("prompt.txt");
        assert!(prompt_path.exists());

        let content = fs::read_to_string(&prompt_path).unwrap();
        assert_eq!(content, "You are a helpful assistant.");
    }

    #[test]
    fn test_write_tool_files() {
        let temp_dir = TempDir::new().unwrap();
        let agent_dir = temp_dir.path().join("test_agent");
        fs::create_dir_all(agent_dir.join("tools")).unwrap();

        let tool = ToolFileInfo {
            name: "echo".to_string(),
            schema_json: r#"{"type":"function","name":"echo","description":"Echo tool","parameters":{"type":"object","properties":{}}}"#.to_string(),
            python_code: Some("def execute(args):\n    return args".to_string()),
        };

        write_tool_files(&agent_dir, &tool).unwrap();

        assert!(agent_dir.join("tools/echo.json").exists());
        assert!(agent_dir.join("tools/echo.py").exists());
    }
}
