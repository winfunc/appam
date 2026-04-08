//! Fluent builders for global and TOML-style configuration objects.
//!
//! These builders are primarily convenience layers around the public config
//! structs. They are useful when a caller wants TOML-shaped configuration data
//! from Rust code without hand-constructing every nested struct.

use std::path::PathBuf;

use super::{
    AppConfig, HistoryConfig, LogFormat, LoggingConfig, RateLimitConfig, TraceFormat, WebConfig,
};
use crate::agent::config::{AgentConfig, AgentMetadata};
use crate::llm::openrouter::OpenRouterConfig;
use crate::tools::loader::{ToolConfig, ToolImplementation};

/// Fluent builder for [`AppConfig`].
///
/// This builder targets the OpenRouter-first global config shape because that
/// is the historical default configuration surface. Provider-specific runtime
/// overrides are still better expressed through [`crate::agent::AgentBuilder`]
/// when constructing a concrete agent instance.
///
/// # Examples
///
/// ```no_run
/// use appam::config::AppConfigBuilder;
/// use anyhow::Result;
///
/// # fn main() -> Result<()> {
/// let config = AppConfigBuilder::new()
///     .openrouter_api_key("sk-...")
///     .model("openai/gpt-5")
///     .logs_dir("./logs")
///     .log_level("debug")
///     .build();
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct AppConfigBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    logs_dir: Option<PathBuf>,
    log_level: Option<String>,
    human_console: Option<bool>,
    log_format: Option<LogFormat>,
    enable_logs: Option<bool>,
    enable_traces: Option<bool>,
    trace_format: Option<TraceFormat>,
    history_enabled: Option<bool>,
    history_db_path: Option<PathBuf>,
    history_auto_save: Option<bool>,
    history_max_sessions: Option<usize>,
    web_host: Option<String>,
    web_port: Option<u16>,
    web_cors: Option<bool>,
    rate_limit_rpm: Option<u64>,
    rate_limit_burst: Option<u32>,
}

impl AppConfigBuilder {
    /// Create a new application config builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the OpenRouter API key.
    pub fn openrouter_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the OpenRouter base URL.
    pub fn openrouter_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the default LLM model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the logs directory.
    pub fn logs_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.logs_dir = Some(dir.into());
        self
    }

    /// Set the log level (trace, debug, info, warn, error).
    pub fn log_level(mut self, level: impl Into<String>) -> Self {
        self.log_level = Some(level.into());
        self
    }

    /// Enable or disable human-readable console output.
    pub fn human_console(mut self, enabled: bool) -> Self {
        self.human_console = Some(enabled);
        self
    }

    /// Set the log file format.
    pub fn log_format(mut self, format: LogFormat) -> Self {
        self.log_format = Some(format);
        self
    }

    /// Enable or disable Appam logs (tracing framework: run-*.log, run-*.jsonl).
    ///
    /// When disabled, logs will only be written to console output.
    /// This controls framework logs, not agent session traces.
    pub fn enable_logs(mut self, enabled: bool) -> Self {
        self.enable_logs = Some(enabled);
        self
    }

    /// Enable or disable agent traces (session files: session-*.jsonl, session-*.json).
    ///
    /// When disabled, no session trace or log files will be written.
    /// This controls agent conversation traces, not framework logs.
    pub fn enable_traces(mut self, enabled: bool) -> Self {
        self.enable_traces = Some(enabled);
        self
    }

    /// Set the trace detail level.
    pub fn trace_format(mut self, format: TraceFormat) -> Self {
        self.trace_format = Some(format);
        self
    }

    /// Enable session history with custom database path.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::config::AppConfigBuilder;
    /// let config = AppConfigBuilder::new()
    ///     .enable_history("data/my_sessions.db")
    ///     .build();
    /// ```
    pub fn enable_history(mut self, db_path: impl Into<PathBuf>) -> Self {
        self.history_enabled = Some(true);
        self.history_db_path = Some(db_path.into());
        self
    }

    /// Enable or disable automatic session saving.
    pub fn history_auto_save(mut self, enabled: bool) -> Self {
        self.history_auto_save = Some(enabled);
        self
    }

    /// Set maximum number of sessions to keep in history.
    pub fn history_max_sessions(mut self, max: usize) -> Self {
        self.history_max_sessions = Some(max);
        self
    }

    /// Set the web server host.
    pub fn web_host(mut self, host: impl Into<String>) -> Self {
        self.web_host = Some(host.into());
        self
    }

    /// Set the web server port.
    pub fn web_port(mut self, port: u16) -> Self {
        self.web_port = Some(port);
        self
    }

    /// Enable or disable CORS.
    pub fn web_cors(mut self, enabled: bool) -> Self {
        self.web_cors = Some(enabled);
        self
    }

    /// Set rate limit requests per minute.
    pub fn rate_limit_rpm(mut self, rpm: u64) -> Self {
        self.rate_limit_rpm = Some(rpm);
        self
    }

    /// Set rate limit burst size.
    pub fn rate_limit_burst(mut self, burst: u32) -> Self {
        self.rate_limit_burst = Some(burst);
        self
    }

    /// Build the application configuration.
    pub fn build(self) -> AppConfig {
        let mut openrouter = OpenRouterConfig::default();
        if let Some(key) = self.api_key {
            openrouter.api_key = Some(key);
        }
        if let Some(url) = self.base_url {
            openrouter.base_url = url;
        }
        if let Some(model) = self.model {
            openrouter.model = model;
        }

        let mut logging = LoggingConfig::default();
        if let Some(dir) = self.logs_dir {
            logging.logs_dir = dir;
        }
        if let Some(level) = self.log_level {
            logging.level = level;
        }
        if let Some(human) = self.human_console {
            logging.human_console = human;
        }
        if let Some(format) = self.log_format {
            logging.log_format = format;
        }
        if let Some(enabled) = self.enable_logs {
            logging.enable_logs = enabled;
        }
        if let Some(enabled) = self.enable_traces {
            logging.enable_traces = enabled;
        }
        if let Some(format) = self.trace_format {
            logging.trace_format = format;
        }

        let mut history = HistoryConfig::default();
        if let Some(enabled) = self.history_enabled {
            history.enabled = enabled;
        }
        if let Some(db_path) = self.history_db_path {
            history.db_path = db_path;
        }
        if let Some(auto_save) = self.history_auto_save {
            history.auto_save = auto_save;
        }
        if let Some(max) = self.history_max_sessions {
            history.max_sessions = Some(max);
        }

        let web = if self.web_host.is_some()
            || self.web_port.is_some()
            || self.web_cors.is_some()
            || self.rate_limit_rpm.is_some()
            || self.rate_limit_burst.is_some()
        {
            let mut web_config = WebConfig::default();
            if let Some(host) = self.web_host {
                web_config.host = host;
            }
            if let Some(port) = self.web_port {
                web_config.port = port;
            }
            if let Some(cors) = self.web_cors {
                web_config.cors = cors;
            }

            if self.rate_limit_rpm.is_some() || self.rate_limit_burst.is_some() {
                let mut rate_limit = RateLimitConfig::default();
                if let Some(rpm) = self.rate_limit_rpm {
                    rate_limit.requests_per_minute = rpm;
                }
                if let Some(burst) = self.rate_limit_burst {
                    rate_limit.burst = burst;
                }
                web_config.rate_limit = Some(rate_limit);
            }

            Some(web_config)
        } else {
            None
        };

        AppConfig {
            provider: crate::llm::LlmProvider::default(),
            openrouter,
            anthropic: crate::llm::anthropic::AnthropicConfig::default(),
            openai: crate::llm::openai::OpenAIConfig::default(),
            openai_codex: crate::llm::openai_codex::OpenAICodexConfig::default(),
            vertex: crate::llm::vertex::VertexConfig::default(),
            logging,
            history,
            web,
        }
    }
}

/// Builder for agent configuration.
///
/// Provides a fluent API for constructing `AgentConfig` programmatically,
/// which can then be saved to TOML or used directly.
///
/// # Examples
///
/// ```no_run
/// use appam::config::AgentConfigBuilder;
/// use anyhow::Result;
///
/// # fn main() -> Result<()> {
/// let config = AgentConfigBuilder::new("my-agent")
///     .model("openai/gpt-5")
///     .system_prompt("prompt.txt")
///     .description("A helpful assistant")
///     .add_python_tool("echo", "tools/echo.json", "tools/echo.py")
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct AgentConfigBuilder {
    name: String,
    model: Option<String>,
    system_prompt: Option<PathBuf>,
    description: Option<String>,
    version: Option<String>,
    tools: Vec<ToolConfig>,
}

impl AgentConfigBuilder {
    /// Create a new agent config builder with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            model: None,
            system_prompt: None,
            description: None,
            version: None,
            tools: Vec::new(),
        }
    }

    /// Set the LLM model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the system prompt file path.
    pub fn system_prompt(mut self, path: impl Into<PathBuf>) -> Self {
        self.system_prompt = Some(path.into());
        self
    }

    /// Set the agent description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the agent version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Add a Python tool.
    ///
    /// # Parameters
    ///
    /// - `name`: Tool name
    /// - `schema`: Path to JSON schema file
    /// - `script`: Path to Python script
    pub fn add_python_tool(
        mut self,
        name: impl Into<String>,
        schema: impl Into<PathBuf>,
        script: impl Into<PathBuf>,
    ) -> Self {
        self.tools.push(ToolConfig {
            name: name.into(),
            schema: schema.into(),
            implementation: ToolImplementation::Python {
                script: script.into(),
            },
        });
        self
    }

    /// Add a Rust tool.
    ///
    /// # Parameters
    ///
    /// - `name`: Tool name
    /// - `schema`: Path to JSON schema file
    /// - `module`: Rust module path (e.g., "appam::tools::builtin::bash")
    pub fn add_rust_tool(
        mut self,
        name: impl Into<String>,
        schema: impl Into<PathBuf>,
        module: impl Into<String>,
    ) -> Self {
        self.tools.push(ToolConfig {
            name: name.into(),
            schema: schema.into(),
            implementation: ToolImplementation::Rust {
                module: module.into(),
            },
        });
        self
    }

    /// Add a tool configuration directly.
    pub fn add_tool(mut self, tool: ToolConfig) -> Self {
        self.tools.push(tool);
        self
    }

    /// Build the agent configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the system prompt path is not set.
    pub fn build(self) -> anyhow::Result<AgentConfig> {
        let system_prompt = self
            .system_prompt
            .ok_or_else(|| anyhow::anyhow!("System prompt path must be set"))?;

        Ok(AgentConfig {
            agent: AgentMetadata {
                name: self.name,
                model: self.model,
                system_prompt,
                description: self.description,
                version: self.version,
            },
            tools: self.tools,
        })
    }

    /// Save the configuration to a TOML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration cannot be serialized or written.
    pub fn save_to_file(self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let config = self.build()?;
        let toml_str = toml::to_string_pretty(&config)?;
        std::fs::write(path, toml_str)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_config_builder() {
        let config = AppConfigBuilder::new()
            .openrouter_api_key("test-key")
            .model("openai/gpt-5")
            .logs_dir("./logs")
            .log_level("debug")
            .build();

        assert_eq!(config.openrouter.api_key.as_deref(), Some("test-key"));
        assert_eq!(config.openrouter.model, "openai/gpt-5");
        assert_eq!(config.logging.logs_dir, PathBuf::from("./logs"));
        assert_eq!(config.logging.level, "debug");
    }

    #[test]
    fn test_app_config_builder_with_web() {
        let config = AppConfigBuilder::new()
            .web_host("127.0.0.1")
            .web_port(8080)
            .web_cors(false)
            .rate_limit_rpm(100)
            .rate_limit_burst(20)
            .build();

        let web = config.web.expect("Web config should be set");
        assert_eq!(web.host, "127.0.0.1");
        assert_eq!(web.port, 8080);
        assert!(!web.cors);

        let rate_limit = web.rate_limit.expect("Rate limit should be set");
        assert_eq!(rate_limit.requests_per_minute, 100);
        assert_eq!(rate_limit.burst, 20);
    }

    #[test]
    fn test_agent_config_builder() {
        let config = AgentConfigBuilder::new("test-agent")
            .model("openai/gpt-5")
            .system_prompt("prompt.txt")
            .description("Test agent")
            .version("1.0.0")
            .build()
            .unwrap();

        assert_eq!(config.agent.name, "test-agent");
        assert_eq!(config.agent.model.as_deref(), Some("openai/gpt-5"));
        assert_eq!(config.agent.description.as_deref(), Some("Test agent"));
        assert_eq!(config.agent.version.as_deref(), Some("1.0.0"));
    }

    #[test]
    fn test_agent_config_builder_with_tools() {
        let config = AgentConfigBuilder::new("test-agent")
            .system_prompt("prompt.txt")
            .add_python_tool("echo", "echo.json", "echo.py")
            .add_rust_tool("bash", "bash.json", "appam::tools::builtin::bash")
            .build()
            .unwrap();

        assert_eq!(config.tools.len(), 2);
        assert_eq!(config.tools[0].name, "echo");
        assert_eq!(config.tools[1].name, "bash");
    }

    #[test]
    fn test_agent_config_builder_missing_prompt() {
        let result = AgentConfigBuilder::new("test-agent").build();
        assert!(result.is_err());
    }
}
