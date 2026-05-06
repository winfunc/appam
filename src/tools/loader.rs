//! Tool loading from TOML configurations.
//!
//! Provides utilities for dynamically loading tools based on agent configuration,
//! supporting both Rust (via module paths) and Python (via script files) implementations.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

#[cfg(feature = "python")]
use super::python::PythonTool;
use super::{registry::ToolRegistry, Tool};
use crate::llm::ToolSpec;

/// Tool configuration from TOML.
///
/// Defines how a tool should be loaded, including its schema and implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Tool name (must match function name in schema)
    pub name: String,
    /// Path to JSON schema file
    pub schema: PathBuf,
    /// Implementation details
    pub implementation: ToolImplementation,
}

/// Tool implementation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolImplementation {
    /// Experimental Rust implementation via module path.
    ///
    /// This legacy TOML surface is currently disabled. Rust tools should be
    /// registered programmatically through `AgentBuilder`, `RuntimeAgent`, or a
    /// pre-populated `ToolRegistry`.
    Rust {
        /// Module path (e.g., "appam::tools::builtin::bash")
        module: String,
    },
    /// Python implementation via script file
    Python {
        /// Path to Python script
        script: PathBuf,
    },
}

/// Load tools from a vector of tool configurations.
///
/// Reads schema files, instantiates tool implementations (Rust or Python),
/// and registers them in the provided registry.
///
/// # Parameters
///
/// - `configs`: Tool configurations from agent TOML
/// - `base_path`: Base directory for resolving relative paths
/// - `registry`: Target registry for loaded tools
///
/// # Errors
///
/// Returns an error if any tool fails to load (schema not found, Python script
/// missing, Rust module not registered, etc.).
pub fn load_tools(configs: &[ToolConfig], base_path: &Path, registry: &ToolRegistry) -> Result<()> {
    info!("Loading {} tools", configs.len());

    for config in configs {
        load_tool(config, base_path, registry)
            .with_context(|| format!("Failed to load tool: {}", config.name))?;
    }

    info!("Successfully loaded {} tools", configs.len());
    Ok(())
}

/// Load a single tool from its configuration.
fn load_tool(config: &ToolConfig, base_path: &Path, registry: &ToolRegistry) -> Result<()> {
    debug!(tool = %config.name, "Loading tool");

    // Resolve and load schema
    let schema_path = if config.schema.is_absolute() {
        config.schema.clone()
    } else {
        base_path.join(&config.schema)
    };

    let schema_content = std::fs::read_to_string(&schema_path)
        .with_context(|| format!("Failed to read schema: {}", schema_path.display()))?;

    let spec: ToolSpec = serde_json::from_str(&schema_content)
        .with_context(|| format!("Failed to parse schema: {}", schema_path.display()))?;

    // Validate that tool name matches schema
    if spec.name != config.name {
        return Err(anyhow!(
            "Tool name mismatch: config says '{}' but schema says '{}'",
            config.name,
            spec.name
        ));
    }

    // Load implementation
    let tool: Arc<dyn Tool> = match &config.implementation {
        ToolImplementation::Rust { module } => load_rust_tool(&config.name, module, spec)?,
        #[cfg(feature = "python")]
        ToolImplementation::Python { script } => {
            let script_path = if script.is_absolute() {
                script.clone()
            } else {
                base_path.join(script)
            };

            Arc::new(PythonTool::new(config.name.clone(), script_path, spec)?)
        }
        #[cfg(not(feature = "python"))]
        ToolImplementation::Python { .. } => {
            return Err(anyhow!(
                "Python tool '{}' cannot be loaded: Python support not enabled. Rebuild with --features python",
                config.name
            ));
        }
    };

    registry.register(tool);
    debug!(tool = %config.name, "Tool loaded successfully");

    Ok(())
}

/// Load a Rust tool by module path.
///
/// This function attempts to resolve a Rust module string to an actual tool
/// implementation. Since we can't dynamically load Rust code at runtime, this
/// function checks against a registry of known built-in tools.
///
/// For custom Rust tools, users should either:
/// 1. Add them to the built-in registry in `builtin/mod.rs`
/// 2. Manually register them before loading agent configurations
/// 3. Use the Python bridge for dynamic loading
fn load_rust_tool(name: &str, module: &str, spec: ToolSpec) -> Result<Arc<dyn Tool>> {
    let _ = (name, module, spec);

    Err(anyhow!(
        "TOML-configured Rust tool loading is experimental and currently disabled. \
         Register Rust tools programmatically instead."
    ))
}

/// Resolve a built-in Rust tool by module path.
///
/// This legacy hook is intentionally disabled until the TOML Rust-tool surface
/// is redesigned.
#[allow(dead_code)]
fn resolve_builtin_rust_tool(_name: &str, _module: &str, _spec: ToolSpec) -> Option<Arc<dyn Tool>> {
    // Try to resolve from built-in tools
    None
}

/// Validate tool configurations without loading.
///
/// Checks that all schema files exist and are valid JSON, and that Python
/// scripts exist. Does not instantiate tools or register them.
///
/// # Errors
///
/// Returns the first validation error encountered.
pub fn validate_tool_configs(configs: &[ToolConfig], base_path: &Path) -> Result<()> {
    #[allow(clippy::never_loop)]
    for config in configs {
        // Check schema exists and is valid
        let schema_path = if config.schema.is_absolute() {
            config.schema.clone()
        } else {
            base_path.join(&config.schema)
        };

        if !schema_path.exists() {
            return Err(anyhow!(
                "Schema file not found for tool '{}': {}",
                config.name,
                schema_path.display()
            ));
        }

        let schema_content = std::fs::read_to_string(&schema_path)?;
        let spec: ToolSpec = serde_json::from_str(&schema_content)
            .with_context(|| format!("Invalid schema for tool '{}'", config.name))?;

        if spec.name != config.name {
            return Err(anyhow!(
                "Tool name mismatch in '{}': config says '{}' but schema says '{}'",
                schema_path.display(),
                config.name,
                spec.name
            ));
        }

        // Check implementation exists
        match &config.implementation {
            #[cfg(feature = "python")]
            ToolImplementation::Python { script } => {
                let script_path = if script.is_absolute() {
                    script.clone()
                } else {
                    base_path.join(script)
                };

                if !script_path.exists() {
                    return Err(anyhow!(
                        "Python script not found for tool '{}': {}",
                        config.name,
                        script_path.display()
                    ));
                }
            }
            #[cfg(not(feature = "python"))]
            ToolImplementation::Python { .. } => {
                return Err(anyhow!(
                    "Python tool '{}' cannot be validated: Python support not enabled",
                    config.name
                ));
            }
            ToolImplementation::Rust { module } => {
                let _ = module;
                return Err(anyhow!(
                    "Rust tool '{}' uses a disabled experimental TOML feature. Register Rust tools programmatically instead.",
                    config.name
                ));
            }
        }
    }

    Ok(())
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_env() -> (TempDir, PathBuf, PathBuf) {
        let dir = TempDir::new().unwrap();
        let base = dir.path().to_path_buf();

        let schema_path = base.join("echo.json");
        let mut schema_file = std::fs::File::create(&schema_path).unwrap();
        schema_file
            .write_all(
                json!({
                    "type": "function",
                    "name": "echo",
                    "description": "Echo tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        }
                    }
                })
                .to_string()
                .as_bytes(),
            )
            .unwrap();

        let script_path = base.join("echo.py");
        let mut script_file = std::fs::File::create(&script_path).unwrap();
        script_file
            .write_all(b"def execute(args):\n    return {'output': args.get('message', '')}")
            .unwrap();

        (dir, schema_path, script_path)
    }

    #[test]
    fn test_validate_tool_configs() {
        let (_dir, schema_path, _script_path) = create_test_env();
        let base = schema_path.parent().unwrap();

        let configs = vec![ToolConfig {
            name: "echo".to_string(),
            schema: "echo.json".into(),
            implementation: ToolImplementation::Python {
                script: "echo.py".into(),
            },
        }];

        assert!(validate_tool_configs(&configs, base).is_ok());
    }

    #[test]
    fn test_validate_missing_schema() {
        let dir = TempDir::new().unwrap();
        let configs = vec![ToolConfig {
            name: "echo".to_string(),
            schema: "nonexistent.json".into(),
            implementation: ToolImplementation::Python {
                script: "echo.py".into(),
            },
        }];

        let result = validate_tool_configs(&configs, dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_python_tool() {
        let (_dir, schema_path, _script_path) = create_test_env();
        let base = schema_path.parent().unwrap();

        let registry = ToolRegistry::new();
        let configs = vec![ToolConfig {
            name: "echo".to_string(),
            schema: "echo.json".into(),
            implementation: ToolImplementation::Python {
                script: "echo.py".into(),
            },
        }];

        load_tools(&configs, base, &registry).unwrap();

        assert_eq!(registry.len(), 1);
        assert!(registry.resolve("echo").is_some());
    }
}
