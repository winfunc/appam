//! Global configuration types and precedence helpers.
//!
//! Appam keeps configuration resolution explicit and deterministic. The shared
//! precedence order across the crate is:
//!
//! 1. built-in defaults
//! 2. global configuration files
//! 3. per-agent overrides
//! 4. environment variables
//!
//! The types in this module model the global layer and expose helpers for
//! loading and mutating it safely before the runtime constructs a provider
//! client.

pub mod builder;
pub use builder::{AgentConfigBuilder, AppConfigBuilder};

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::llm::anthropic::AnthropicConfig;
use crate::llm::openai::OpenAIConfig;
use crate::llm::openai_codex::OpenAICodexConfig;
use crate::llm::openrouter::{OpenRouterConfig, ReasoningConfig};
use crate::llm::vertex::{VertexConfig, VertexThinkingConfig};
use crate::llm::LlmProvider;

/// Root configuration object for Appam.
///
/// This type holds provider selection plus the provider-specific configuration
/// blocks consumed by the runtime. It also carries cross-cutting concerns such
/// as logging, history, and the legacy web configuration surface.
///
/// Most users either construct it through [`AppConfigBuilder`], load it from a
/// TOML file with [`load_global_config`], or start from
/// [`load_config_from_env`].
///
/// # Provider Selection
///
/// The `provider` field determines which LLM backend to use. Each provider
/// has its own configuration section:
///
/// ```toml
/// provider = "anthropic"  # or "openrouter", "openai"
///
/// [anthropic]
/// model = "claude-sonnet-4-5"
/// max_tokens = 4096
///
/// [openrouter]
/// model = "openai/gpt-5"
/// max_output_tokens = 9000
///
/// [openai]
/// model = "gpt-5.5"
/// max_output_tokens = 4096
///
/// [openai_codex]
/// model = "gpt-5.5"
///
/// [vertex]
/// model = "gemini-2.5-flash"
/// location = "us-central1"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    /// LLM provider selection
    #[serde(default)]
    pub provider: LlmProvider,

    /// OpenRouter API configuration
    #[serde(default)]
    pub openrouter: OpenRouterConfig,

    /// Anthropic API configuration
    #[serde(default)]
    pub anthropic: AnthropicConfig,

    /// OpenAI API configuration
    #[serde(default)]
    pub openai: OpenAIConfig,

    /// OpenAI Codex subscription provider configuration
    #[serde(default)]
    pub openai_codex: OpenAICodexConfig,

    /// Google Vertex API configuration
    #[serde(default)]
    pub vertex: VertexConfig,

    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Session history configuration
    #[serde(default)]
    pub history: HistoryConfig,
    /// Web API configuration (optional)
    #[serde(default)]
    pub web: Option<WebConfig>,
}

/// Log output format.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    /// Human-readable .log file
    Plain,
    /// Structured .jsonl file
    Json,
    /// Both plain and JSON formats
    #[default]
    Both,
}

/// Trace detail level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum TraceFormat {
    /// Essential information only
    Compact,
    /// Full details including reasoning
    #[default]
    Detailed,
}

/// Logging and trace-output configuration.
///
/// `LoggingConfig` controls Appam's own structured logs plus conversation trace
/// capture. The two are intentionally separate so callers can keep traces
/// without verbose framework logs, or vice versa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Directory for log files and session transcripts
    #[serde(default = "LoggingConfig::default_logs_dir")]
    pub logs_dir: PathBuf,
    /// Enable human-readable console output
    #[serde(default = "LoggingConfig::default_human_console")]
    pub human_console: bool,
    /// Log level (trace, debug, info, warn, error)
    #[serde(default = "LoggingConfig::default_level")]
    pub level: String,
    /// Log file format
    #[serde(default)]
    pub log_format: LogFormat,
    /// Enable Appam logs (tracing framework logs: run-*.log, run-*.jsonl)
    ///
    /// When disabled, logs will only be written to console output.
    /// This controls the tracing framework logs, not agent session traces.
    #[serde(
        default = "LoggingConfig::default_enable_logs",
        alias = "file_logging_enabled"
    )]
    pub enable_logs: bool,
    /// Enable agent traces (session files: session-*.jsonl, session-*.json)
    ///
    /// When disabled, no session trace or log files will be written.
    /// This controls agent conversation traces, not framework logs.
    #[serde(
        default = "LoggingConfig::default_enable_traces",
        alias = "trace_enabled"
    )]
    pub enable_traces: bool,
    /// Trace detail level
    #[serde(default)]
    pub trace_format: TraceFormat,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            logs_dir: Self::default_logs_dir(),
            human_console: Self::default_human_console(),
            level: Self::default_level(),
            log_format: LogFormat::default(),
            enable_logs: Self::default_enable_logs(),
            enable_traces: Self::default_enable_traces(),
            trace_format: TraceFormat::default(),
        }
    }
}

impl LoggingConfig {
    fn default_logs_dir() -> PathBuf {
        PathBuf::from("logs")
    }

    fn default_human_console() -> bool {
        true
    }

    fn default_level() -> String {
        "info".to_string()
    }

    fn default_enable_logs() -> bool {
        false
    }

    fn default_enable_traces() -> bool {
        false
    }
}

/// Persistent session-history configuration.
///
/// History uses SQLite so sessions can be resumed, inspected, or listed across
/// process restarts. When disabled, the higher-level history API remains
/// available but behaves as a no-op surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryConfig {
    /// Enable session history persistence
    #[serde(default)]
    pub enabled: bool,
    /// Database file path
    #[serde(default = "HistoryConfig::default_db_path")]
    pub db_path: PathBuf,
    /// Automatically save sessions after completion
    #[serde(default = "HistoryConfig::default_auto_save")]
    pub auto_save: bool,
    /// Maximum number of sessions to keep (None = unlimited)
    #[serde(default)]
    pub max_sessions: Option<usize>,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            db_path: Self::default_db_path(),
            auto_save: Self::default_auto_save(),
            max_sessions: None,
        }
    }
}

impl HistoryConfig {
    fn default_db_path() -> PathBuf {
        PathBuf::from("data/sessions.db")
    }

    fn default_auto_save() -> bool {
        true
    }
}

/// Configuration for the legacy Axum-based web surface.
///
/// The full web API is intentionally disabled at runtime today, but the
/// configuration type remains part of the public API for backward compatibility
/// and for the trace visualizer helpers under [`crate::web`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConfig {
    /// Host address to bind to
    #[serde(default = "WebConfig::default_host")]
    pub host: String,
    /// Port to listen on
    #[serde(default = "WebConfig::default_port")]
    pub port: u16,
    /// Enable CORS
    #[serde(default = "WebConfig::default_cors")]
    pub cors: bool,
    /// Rate limiting configuration
    #[serde(default)]
    pub rate_limit: Option<RateLimitConfig>,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            host: Self::default_host(),
            port: Self::default_port(),
            cors: Self::default_cors(),
            rate_limit: None,
        }
    }
}

impl WebConfig {
    fn default_host() -> String {
        "0.0.0.0".to_string()
    }

    fn default_port() -> u16 {
        3000
    }

    fn default_cors() -> bool {
        true
    }
}

/// Rate-limiting settings for the legacy web surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute per IP
    #[serde(default = "RateLimitConfig::default_requests_per_minute")]
    pub requests_per_minute: u64,
    /// Burst size
    #[serde(default = "RateLimitConfig::default_burst")]
    pub burst: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: Self::default_requests_per_minute(),
            burst: Self::default_burst(),
        }
    }
}

impl RateLimitConfig {
    fn default_requests_per_minute() -> u64 {
        60
    }

    fn default_burst() -> u32 {
        10
    }
}

/// Load configuration from built-in defaults plus environment variables.
///
/// This helper intentionally skips all file I/O. Use it when a program wants
/// an explicit config object driven only by process environment.
///
/// # Environment Variables
///
/// ## Provider Selection
/// - `APPAM_PROVIDER`: Override provider (anthropic, openrouter, openai, openai-codex, vertex, azure-openai, azure-anthropic, bedrock)
///
/// ## OpenRouter
/// - `OPENROUTER_API_KEY`: API key for OpenRouter
/// - `OPENROUTER_MODEL`: Model identifier
/// - `OPENROUTER_BASE_URL`: API base URL
///
/// ## Anthropic
/// - `ANTHROPIC_API_KEY`: API key for Anthropic
/// - `ANTHROPIC_MODEL`: Model identifier
/// - `ANTHROPIC_BASE_URL`: API base URL
/// - `AZURE_ANTHROPIC_MODEL`: Azure Anthropic deployment/model override
/// - `AZURE_ANTHROPIC_BASE_URL`: Full Azure Anthropic base URL
/// - `AZURE_ANTHROPIC_RESOURCE`: Azure Anthropic resource name used to derive the base URL
/// - `AZURE_ANTHROPIC_AUTH_METHOD`: `x_api_key` or `bearer`
///
/// ## OpenAI / Azure OpenAI
/// - `OPENAI_API_KEY`: API key for direct OpenAI requests
/// - `OPENAI_MODEL`: Model identifier (for example `gpt-5.5`)
/// - `OPENAI_BASE_URL`: API base URL override
/// - `OPENAI_ORGANIZATION`: Optional organization header
/// - `OPENAI_PROJECT`: Optional project header
/// - `AZURE_OPENAI_MODEL`: Azure deployment/model override
///
/// ## OpenAI Codex
/// - `OPENAI_CODEX_MODEL`: Codex model identifier
/// - `OPENAI_CODEX_BASE_URL`: Codex backend base URL override
/// - `OPENAI_CODEX_ACCESS_TOKEN`: Explicit ChatGPT OAuth access token override
/// - `OPENAI_CODEX_AUTH_FILE`: Auth cache file path override
///
/// ## Vertex
/// - `GOOGLE_VERTEX_API_KEY`: API key for Vertex/Gemini
/// - `GOOGLE_VERTEX_ACCESS_TOKEN`: OAuth bearer token for Vertex
/// - `GOOGLE_VERTEX_MODEL`: Model identifier
/// - `GOOGLE_VERTEX_LOCATION`: Vertex location
/// - `GOOGLE_VERTEX_PROJECT`: Google Cloud project ID
/// - `GOOGLE_VERTEX_BASE_URL`: API base URL
/// - `GOOGLE_VERTEX_INCLUDE_THOUGHTS`: Enable thought blocks in responses (`true`/`false`)
/// - `GOOGLE_VERTEX_THINKING_LEVEL`: Optional thinking level hint (`LOW`/`MEDIUM`/`HIGH`)
///
/// ## Logging
/// - `APPAM_LOG_LEVEL`: Logging level
/// - `APPAM_LOGS_DIR`: Logs directory
/// - `APPAM_TRACE_ENABLED`: Enable trace files
///
/// ## History
/// - `APPAM_HISTORY_ENABLED`: Enable session history
/// - `APPAM_HISTORY_DB_PATH`: Database file path
///
/// # Errors
///
/// Returns an error if environment variables contain invalid values.
pub fn load_config_from_env() -> Result<AppConfig> {
    let mut cfg = AppConfig::default();
    apply_env_overrides(&mut cfg)?;
    Ok(cfg)
}

/// Load global configuration from a TOML file, then apply env overrides.
///
/// Reads configuration from the specified path, applies environment variable
/// overrides, and validates provider-specific features.
///
/// # Arguments
///
/// - `path`: Path to the TOML config file. Must be explicitly provided.
///
/// # Environment Variables
///
/// Environment variables override file settings. See `load_config_from_env()` for details.
///
/// # Errors
///
/// Returns an error if:
/// - The config file cannot be read
/// - The config file contains invalid TOML
/// - Environment variables contain invalid values
pub fn load_global_config(path: &Path) -> Result<AppConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;
    let mut cfg: AppConfig =
        toml::from_str(&content).with_context(|| format!("Invalid TOML in {}", path.display()))?;

    apply_env_overrides(&mut cfg)?;
    Ok(cfg)
}

/// Apply environment-variable overrides in place.
///
/// The mutation is explicit so callers can load configuration from multiple
/// sources first, then apply environment precedence once at the end.
///
/// # Errors
///
/// Returns an error if environment variables contain invalid values.
fn apply_env_overrides(cfg: &mut AppConfig) -> Result<()> {
    // Provider selection
    if let Ok(provider) = std::env::var("APPAM_PROVIDER") {
        if !provider.trim().is_empty() {
            cfg.provider = provider.parse().context("Invalid APPAM_PROVIDER value")?;
        }
    }

    // OpenRouter overrides
    if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
        if !key.trim().is_empty() {
            cfg.openrouter.api_key = Some(key);
        }
    }

    if let Ok(model) = std::env::var("OPENROUTER_MODEL") {
        if !model.trim().is_empty() {
            cfg.openrouter.model = model;
        }
    }

    if let Ok(base_url) = std::env::var("OPENROUTER_BASE_URL") {
        if !base_url.trim().is_empty() {
            cfg.openrouter.base_url = base_url;
        }
    }

    // Anthropic overrides
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        if !key.trim().is_empty() {
            cfg.anthropic.api_key = Some(key);
        }
    }

    if let Ok(model) = std::env::var("ANTHROPIC_MODEL") {
        if !model.trim().is_empty() {
            cfg.anthropic.model = model;
        }
    }

    if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
        if !base_url.trim().is_empty() {
            cfg.anthropic.base_url = base_url;
        }
    }

    if let Ok(model) = std::env::var("AZURE_ANTHROPIC_MODEL") {
        if !model.trim().is_empty() {
            cfg.anthropic.model = model;
        }
    }

    let azure_base_url = std::env::var("AZURE_ANTHROPIC_BASE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(Ok)
        .or_else(|| {
            std::env::var("AZURE_ANTHROPIC_RESOURCE")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(|resource| {
                    crate::llm::anthropic::AzureAnthropicConfig::base_url_from_resource(&resource)
                })
        })
        .transpose()?;

    let azure_auth_method = std::env::var("AZURE_ANTHROPIC_AUTH_METHOD")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.parse())
        .transpose()?;

    if azure_base_url.is_some() || azure_auth_method.is_some() {
        let mut azure = cfg.anthropic.azure.clone().unwrap_or_default();
        if let Some(base_url) = azure_base_url {
            azure.base_url = base_url;
        }
        if let Some(auth_method) = azure_auth_method {
            azure.auth_method = auth_method;
        }
        cfg.anthropic.azure = Some(azure);
    }

    // OpenAI overrides
    if let Ok(model) = std::env::var("OPENAI_MODEL") {
        if !model.trim().is_empty() {
            cfg.openai.model = model;
        }
    }

    if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
        if !base_url.trim().is_empty() {
            cfg.openai.base_url = base_url;
        }
    }

    if let Ok(organization) = std::env::var("OPENAI_ORGANIZATION") {
        if !organization.trim().is_empty() {
            cfg.openai.organization = Some(organization);
        }
    }

    if let Ok(project) = std::env::var("OPENAI_PROJECT") {
        if !project.trim().is_empty() {
            cfg.openai.project = Some(project);
        }
    }

    if let Ok(model) = std::env::var("AZURE_OPENAI_MODEL") {
        if !model.trim().is_empty() {
            cfg.openai.model = model;
        }
    }

    if let Ok(model) = std::env::var("OPENAI_CODEX_MODEL") {
        if !model.trim().is_empty() {
            cfg.openai_codex.model = model;
        }
    }

    if let Ok(base_url) = std::env::var("OPENAI_CODEX_BASE_URL") {
        if !base_url.trim().is_empty() {
            cfg.openai_codex.base_url = base_url;
        }
    }

    if let Ok(access_token) = std::env::var("OPENAI_CODEX_ACCESS_TOKEN") {
        if !access_token.trim().is_empty() {
            cfg.openai_codex.access_token = Some(access_token);
        }
    }

    if let Ok(auth_file) = std::env::var("OPENAI_CODEX_AUTH_FILE") {
        if !auth_file.trim().is_empty() {
            cfg.openai_codex.auth_file = PathBuf::from(auth_file);
        }
    }

    // Vertex overrides
    if let Ok(key) = std::env::var("GOOGLE_VERTEX_API_KEY") {
        if !key.trim().is_empty() {
            cfg.vertex.api_key = Some(key);
        }
    }

    if let Ok(token) = std::env::var("GOOGLE_VERTEX_ACCESS_TOKEN") {
        if !token.trim().is_empty() {
            cfg.vertex.access_token = Some(token);
        }
    }

    if let Ok(model) = std::env::var("GOOGLE_VERTEX_MODEL") {
        if !model.trim().is_empty() {
            cfg.vertex.model = model;
        }
    }

    if let Ok(location) = std::env::var("GOOGLE_VERTEX_LOCATION") {
        if !location.trim().is_empty() {
            cfg.vertex.location = location;
        }
    }

    if let Ok(project) = std::env::var("GOOGLE_VERTEX_PROJECT") {
        if !project.trim().is_empty() {
            cfg.vertex.project_id = Some(project);
        }
    }

    if let Ok(base_url) = std::env::var("GOOGLE_VERTEX_BASE_URL") {
        if !base_url.trim().is_empty() {
            cfg.vertex.base_url = base_url;
        }
    }

    if let Ok(include_thoughts) = std::env::var("GOOGLE_VERTEX_INCLUDE_THOUGHTS") {
        if !include_thoughts.trim().is_empty() {
            let enabled = include_thoughts.eq_ignore_ascii_case("true")
                || include_thoughts == "1"
                || include_thoughts.eq_ignore_ascii_case("yes");

            let thinking = cfg
                .vertex
                .thinking
                .get_or_insert_with(VertexThinkingConfig::default);
            thinking.include_thoughts = Some(enabled);
        }
    }

    if let Ok(thinking_level) = std::env::var("GOOGLE_VERTEX_THINKING_LEVEL") {
        if !thinking_level.trim().is_empty() {
            let thinking = cfg
                .vertex
                .thinking
                .get_or_insert_with(VertexThinkingConfig::default);
            thinking.thinking_level = Some(thinking_level);
        }
    }

    if let Ok(level) = std::env::var("APPAM_LOG_LEVEL") {
        if !level.trim().is_empty() {
            cfg.logging.level = level;
        }
    }

    if let Ok(logs_dir) = std::env::var("APPAM_LOGS_DIR") {
        if !logs_dir.trim().is_empty() {
            cfg.logging.logs_dir = PathBuf::from(logs_dir);
        }
    }

    if let Ok(log_format) = std::env::var("APPAM_LOG_FORMAT") {
        match log_format.to_lowercase().as_str() {
            "plain" => cfg.logging.log_format = LogFormat::Plain,
            "json" => cfg.logging.log_format = LogFormat::Json,
            "both" => cfg.logging.log_format = LogFormat::Both,
            _ => {} // Keep existing value
        }
    }

    if let Ok(trace_format) = std::env::var("APPAM_TRACE_FORMAT") {
        match trace_format.to_lowercase().as_str() {
            "compact" => cfg.logging.trace_format = TraceFormat::Compact,
            "detailed" => cfg.logging.trace_format = TraceFormat::Detailed,
            _ => {} // Keep existing value
        }
    }

    if let Ok(enable_traces) = std::env::var("APPAM_ENABLE_TRACES") {
        cfg.logging.enable_traces = enable_traces.to_lowercase() == "true";
    }

    if let Ok(enable_logs) = std::env::var("APPAM_ENABLE_LOGS") {
        cfg.logging.enable_logs = enable_logs.to_lowercase() == "true";
    }

    if let Ok(history_enabled) = std::env::var("APPAM_HISTORY_ENABLED") {
        cfg.history.enabled = history_enabled.to_lowercase() == "true";
    }

    if let Ok(history_db_path) = std::env::var("APPAM_HISTORY_DB_PATH") {
        if !history_db_path.trim().is_empty() {
            cfg.history.db_path = PathBuf::from(history_db_path);
        }
    }

    if cfg.openrouter.reasoning.is_none() {
        cfg.openrouter.reasoning = Some(ReasoningConfig::default());
    }

    // Validate provider-specific features and emit warnings
    validate_provider_features(cfg)?;

    Ok(())
}

/// Validate provider-specific features and warn about incompatibilities.
///
/// Checks if Anthropic-specific features are configured with OpenRouter provider
/// (and vice versa), emitting warnings to guide users toward compatible configurations.
///
/// # Errors
///
/// Returns an error for critical incompatibilities (currently none).
fn validate_provider_features(cfg: &AppConfig) -> Result<()> {
    use tracing::warn;

    match cfg.provider {
        LlmProvider::OpenRouterCompletions | LlmProvider::OpenRouterResponses => {
            // Warn about Anthropic-specific features
            if cfg.anthropic.thinking.is_some() {
                warn!(
                    "Extended thinking is Anthropic-specific and will be ignored with OpenRouter provider. \
                     Consider using OpenRouter's 'reasoning' configuration instead."
                );
            }

            if cfg.anthropic.caching.as_ref().is_some_and(|c| c.enabled) {
                warn!(
                    "Prompt caching is Anthropic-specific and will be ignored with OpenRouter provider. \
                     OpenRouter supports automatic prompt caching (no manual configuration needed)."
                );
            }

            if cfg.anthropic.beta_features.has_any() {
                warn!(
                    "Anthropic beta features (fine-grained streaming, interleaved thinking, etc.) \
                     are not supported with OpenRouter provider and will be ignored."
                );
            }

            if cfg.anthropic.tool_choice.is_some() {
                warn!(
                    "Anthropic tool_choice configuration detected but using OpenRouter provider. \
                     Tool choice will use OpenRouter's format (auto/none/required)."
                );
            }
        }
        LlmProvider::Anthropic | LlmProvider::AzureAnthropic { .. } => {
            // Warn about OpenRouter-specific features
            if cfg
                .openrouter
                .reasoning
                .as_ref()
                .is_some_and(|r| r.enabled == Some(true))
            {
                warn!(
                    "OpenRouter reasoning configuration detected but using an Anthropic-compatible provider. \
                     Use Anthropic's 'thinking' configuration for extended reasoning instead."
                );
            }

            if cfg.openrouter.http_referer.is_some() || cfg.openrouter.x_title.is_some() {
                warn!(
                    "OpenRouter attribution headers (http_referer, x_title) are not used \
                     with Anthropic-compatible providers and will be ignored."
                );
            }
        }
        LlmProvider::OpenAI | LlmProvider::OpenAICodex | LlmProvider::AzureOpenAI { .. } => {
            // Warn about provider-specific features
            if cfg.anthropic.thinking.is_some() {
                warn!(
                    "Anthropic extended thinking configuration detected but using OpenAI provider. \
                     Use OpenAI's 'reasoning' configuration for o-series models instead."
                );
            }

            if cfg.anthropic.caching.as_ref().is_some_and(|c| c.enabled) {
                warn!(
                    "Anthropic prompt caching configuration detected but using OpenAI provider. \
                     OpenAI uses 'prompt_cache_key' for caching."
                );
            }

            if cfg.openrouter.http_referer.is_some() || cfg.openrouter.x_title.is_some() {
                warn!(
                    "OpenRouter attribution headers (http_referer, x_title) are not used \
                     with OpenAI-compatible providers and will be ignored."
                );
            }
        }
        LlmProvider::Bedrock { .. } => {
            // Warn about OpenRouter-specific features
            if cfg
                .openrouter
                .reasoning
                .as_ref()
                .is_some_and(|r| r.enabled == Some(true))
            {
                warn!(
                    "OpenRouter reasoning configuration detected but using Bedrock provider. \
                     Use Anthropic's 'thinking' configuration for extended reasoning instead."
                );
            }

            if cfg.openrouter.http_referer.is_some() || cfg.openrouter.x_title.is_some() {
                warn!(
                    "OpenRouter attribution headers (http_referer, x_title) are not used \
                     with Bedrock provider and will be ignored."
                );
            }
        }
        LlmProvider::Vertex => {
            if cfg.anthropic.thinking.is_some() {
                warn!(
                    "Anthropic extended thinking configuration detected but using Vertex provider. \
                     Configure [vertex.thinking] instead."
                );
            }

            if cfg
                .openrouter
                .reasoning
                .as_ref()
                .is_some_and(|r| r.enabled == Some(true))
            {
                warn!(
                    "OpenRouter reasoning configuration detected but using Vertex provider. \
                     Configure [vertex.thinking] if the selected Gemini model supports it."
                );
            }

            if cfg.openai.service_tier.is_some() {
                warn!(
                    "OpenAI service_tier configuration detected but using Vertex provider; it will be ignored."
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::io::Write;
    use std::sync::Mutex;
    use tempfile::NamedTempFile;

    static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    #[test]
    fn test_load_default_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_model = std::env::var("OPENROUTER_MODEL").ok();
        let prev_level = std::env::var("APPAM_LOG_LEVEL").ok();
        std::env::remove_var("OPENROUTER_MODEL");
        std::env::remove_var("APPAM_LOG_LEVEL");

        let cfg = load_config_from_env().unwrap();
        assert_eq!(cfg.openrouter.model, "openai/gpt-5");
        assert_eq!(cfg.logging.level, "info");

        if let Some(value) = prev_model {
            std::env::set_var("OPENROUTER_MODEL", value);
        }
        if let Some(value) = prev_level {
            std::env::set_var("APPAM_LOG_LEVEL", value);
        }
    }

    #[test]
    fn test_load_config_from_file() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_model = std::env::var("OPENROUTER_MODEL").ok();
        let prev_level = std::env::var("APPAM_LOG_LEVEL").ok();
        std::env::remove_var("OPENROUTER_MODEL");
        std::env::remove_var("APPAM_LOG_LEVEL");

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(
            br#"
[openrouter]
model = "anthropic/claude-3.5-sonnet"
base_url = "https://example.com/api/v1"

[logging]
level = "debug"
logs_dir = "/tmp/logs"
"#,
        )
        .unwrap();
        file.flush().unwrap();

        let cfg = load_global_config(file.path()).unwrap();
        assert_eq!(cfg.openrouter.model, "anthropic/claude-3.5-sonnet");
        assert_eq!(cfg.logging.level, "debug");
        assert_eq!(cfg.logging.logs_dir, PathBuf::from("/tmp/logs"));

        if let Some(value) = prev_model {
            std::env::set_var("OPENROUTER_MODEL", value);
        } else {
            std::env::remove_var("OPENROUTER_MODEL");
        }
        if let Some(value) = prev_level {
            std::env::set_var("APPAM_LOG_LEVEL", value);
        } else {
            std::env::remove_var("APPAM_LOG_LEVEL");
        }
    }

    #[test]
    fn test_env_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("OPENROUTER_MODEL", "test-model");
        std::env::set_var("APPAM_LOG_LEVEL", "trace");

        let cfg = load_config_from_env().unwrap();
        assert_eq!(cfg.openrouter.model, "test-model");
        assert_eq!(cfg.logging.level, "trace");

        std::env::remove_var("OPENROUTER_MODEL");
        std::env::remove_var("APPAM_LOG_LEVEL");
    }

    #[test]
    fn test_vertex_env_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_api_key = std::env::var("GOOGLE_VERTEX_API_KEY").ok();
        let prev_model = std::env::var("GOOGLE_VERTEX_MODEL").ok();
        let prev_location = std::env::var("GOOGLE_VERTEX_LOCATION").ok();
        let prev_project = std::env::var("GOOGLE_VERTEX_PROJECT").ok();
        let prev_include_thoughts = std::env::var("GOOGLE_VERTEX_INCLUDE_THOUGHTS").ok();
        let prev_thinking_level = std::env::var("GOOGLE_VERTEX_THINKING_LEVEL").ok();

        std::env::set_var("GOOGLE_VERTEX_API_KEY", "vertex-test-key");
        std::env::set_var("GOOGLE_VERTEX_MODEL", "gemini-2.5-pro");
        std::env::set_var("GOOGLE_VERTEX_LOCATION", "europe-west4");
        std::env::set_var("GOOGLE_VERTEX_PROJECT", "vertex-project");
        std::env::set_var("GOOGLE_VERTEX_INCLUDE_THOUGHTS", "true");
        std::env::set_var("GOOGLE_VERTEX_THINKING_LEVEL", "HIGH");

        let cfg = load_config_from_env().unwrap();
        assert_eq!(cfg.vertex.api_key.as_deref(), Some("vertex-test-key"));
        assert_eq!(cfg.vertex.model, "gemini-2.5-pro");
        assert_eq!(cfg.vertex.location, "europe-west4");
        assert_eq!(cfg.vertex.project_id.as_deref(), Some("vertex-project"));
        assert_eq!(
            cfg.vertex
                .thinking
                .as_ref()
                .and_then(|t| t.include_thoughts),
            Some(true)
        );
        assert_eq!(
            cfg.vertex
                .thinking
                .as_ref()
                .and_then(|t| t.thinking_level.as_deref()),
            Some("HIGH")
        );

        if let Some(value) = prev_api_key {
            std::env::set_var("GOOGLE_VERTEX_API_KEY", value);
        } else {
            std::env::remove_var("GOOGLE_VERTEX_API_KEY");
        }

        if let Some(value) = prev_model {
            std::env::set_var("GOOGLE_VERTEX_MODEL", value);
        } else {
            std::env::remove_var("GOOGLE_VERTEX_MODEL");
        }

        if let Some(value) = prev_location {
            std::env::set_var("GOOGLE_VERTEX_LOCATION", value);
        } else {
            std::env::remove_var("GOOGLE_VERTEX_LOCATION");
        }

        if let Some(value) = prev_project {
            std::env::set_var("GOOGLE_VERTEX_PROJECT", value);
        } else {
            std::env::remove_var("GOOGLE_VERTEX_PROJECT");
        }

        if let Some(value) = prev_include_thoughts {
            std::env::set_var("GOOGLE_VERTEX_INCLUDE_THOUGHTS", value);
        } else {
            std::env::remove_var("GOOGLE_VERTEX_INCLUDE_THOUGHTS");
        }

        if let Some(value) = prev_thinking_level {
            std::env::set_var("GOOGLE_VERTEX_THINKING_LEVEL", value);
        } else {
            std::env::remove_var("GOOGLE_VERTEX_THINKING_LEVEL");
        }
    }

    #[test]
    fn test_openai_env_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_api_key = std::env::var("OPENAI_API_KEY").ok();
        let prev_model = std::env::var("OPENAI_MODEL").ok();
        let prev_base_url = std::env::var("OPENAI_BASE_URL").ok();
        let prev_org = std::env::var("OPENAI_ORGANIZATION").ok();
        let prev_project = std::env::var("OPENAI_PROJECT").ok();
        let prev_azure_model = std::env::var("AZURE_OPENAI_MODEL").ok();

        std::env::set_var("OPENAI_API_KEY", "openai-test-key");
        std::env::set_var("OPENAI_MODEL", "gpt-5.5");
        std::env::set_var("OPENAI_BASE_URL", "https://example.openai.test/v1");
        std::env::set_var("OPENAI_ORGANIZATION", "org_test");
        std::env::set_var("OPENAI_PROJECT", "proj_test");
        std::env::set_var("AZURE_OPENAI_MODEL", "gpt-5.5-azure");

        let cfg = load_config_from_env().unwrap();
        assert!(cfg.openai.api_key.is_none());
        assert_eq!(cfg.openai.model, "gpt-5.5-azure");
        assert_eq!(cfg.openai.base_url, "https://example.openai.test/v1");
        assert_eq!(cfg.openai.organization.as_deref(), Some("org_test"));
        assert_eq!(cfg.openai.project.as_deref(), Some("proj_test"));

        if let Some(value) = prev_api_key {
            std::env::set_var("OPENAI_API_KEY", value);
        } else {
            std::env::remove_var("OPENAI_API_KEY");
        }

        if let Some(value) = prev_model {
            std::env::set_var("OPENAI_MODEL", value);
        } else {
            std::env::remove_var("OPENAI_MODEL");
        }

        if let Some(value) = prev_base_url {
            std::env::set_var("OPENAI_BASE_URL", value);
        } else {
            std::env::remove_var("OPENAI_BASE_URL");
        }

        if let Some(value) = prev_org {
            std::env::set_var("OPENAI_ORGANIZATION", value);
        } else {
            std::env::remove_var("OPENAI_ORGANIZATION");
        }

        if let Some(value) = prev_project {
            std::env::set_var("OPENAI_PROJECT", value);
        } else {
            std::env::remove_var("OPENAI_PROJECT");
        }

        if let Some(value) = prev_azure_model {
            std::env::set_var("AZURE_OPENAI_MODEL", value);
        } else {
            std::env::remove_var("AZURE_OPENAI_MODEL");
        }
    }

    #[test]
    fn test_openai_codex_env_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_model = std::env::var("OPENAI_CODEX_MODEL").ok();
        let prev_base_url = std::env::var("OPENAI_CODEX_BASE_URL").ok();
        let prev_access_token = std::env::var("OPENAI_CODEX_ACCESS_TOKEN").ok();
        let prev_auth_file = std::env::var("OPENAI_CODEX_AUTH_FILE").ok();

        std::env::set_var("OPENAI_CODEX_MODEL", "gpt-5.3-codex");
        std::env::set_var(
            "OPENAI_CODEX_BASE_URL",
            "https://chatgpt.example.test/backend-api",
        );
        std::env::set_var("OPENAI_CODEX_ACCESS_TOKEN", "test-access-token");
        std::env::set_var(
            "OPENAI_CODEX_AUTH_FILE",
            "/tmp/appam-openai-codex-auth.json",
        );

        let cfg = load_config_from_env().unwrap();
        assert_eq!(cfg.openai_codex.model, "gpt-5.3-codex");
        assert_eq!(
            cfg.openai_codex.base_url,
            "https://chatgpt.example.test/backend-api"
        );
        assert_eq!(
            cfg.openai_codex.access_token.as_deref(),
            Some("test-access-token")
        );
        assert_eq!(
            cfg.openai_codex.auth_file,
            PathBuf::from("/tmp/appam-openai-codex-auth.json")
        );

        if let Some(value) = prev_model {
            std::env::set_var("OPENAI_CODEX_MODEL", value);
        } else {
            std::env::remove_var("OPENAI_CODEX_MODEL");
        }
        if let Some(value) = prev_base_url {
            std::env::set_var("OPENAI_CODEX_BASE_URL", value);
        } else {
            std::env::remove_var("OPENAI_CODEX_BASE_URL");
        }
        if let Some(value) = prev_access_token {
            std::env::set_var("OPENAI_CODEX_ACCESS_TOKEN", value);
        } else {
            std::env::remove_var("OPENAI_CODEX_ACCESS_TOKEN");
        }
        if let Some(value) = prev_auth_file {
            std::env::set_var("OPENAI_CODEX_AUTH_FILE", value);
        } else {
            std::env::remove_var("OPENAI_CODEX_AUTH_FILE");
        }
    }

    #[test]
    fn test_azure_anthropic_env_overrides() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_model = std::env::var("AZURE_ANTHROPIC_MODEL").ok();
        let prev_base_url = std::env::var("AZURE_ANTHROPIC_BASE_URL").ok();
        let prev_resource = std::env::var("AZURE_ANTHROPIC_RESOURCE").ok();
        let prev_auth_method = std::env::var("AZURE_ANTHROPIC_AUTH_METHOD").ok();

        std::env::set_var("AZURE_ANTHROPIC_MODEL", "claude-opus-4-6");
        std::env::set_var(
            "AZURE_ANTHROPIC_BASE_URL",
            "https://example-resource.services.ai.azure.com/anthropic/v1/messages",
        );
        std::env::remove_var("AZURE_ANTHROPIC_RESOURCE");
        std::env::set_var("AZURE_ANTHROPIC_AUTH_METHOD", "bearer");

        let cfg = load_config_from_env().unwrap();
        assert_eq!(cfg.anthropic.model, "claude-opus-4-6");
        let azure = cfg
            .anthropic
            .azure
            .expect("azure config should be populated");
        assert_eq!(
            azure.base_url,
            "https://example-resource.services.ai.azure.com/anthropic/v1/messages"
        );
        assert_eq!(
            azure.auth_method,
            crate::llm::anthropic::AzureAnthropicAuthMethod::BearerToken
        );

        if let Some(value) = prev_model {
            std::env::set_var("AZURE_ANTHROPIC_MODEL", value);
        } else {
            std::env::remove_var("AZURE_ANTHROPIC_MODEL");
        }
        if let Some(value) = prev_base_url {
            std::env::set_var("AZURE_ANTHROPIC_BASE_URL", value);
        } else {
            std::env::remove_var("AZURE_ANTHROPIC_BASE_URL");
        }
        if let Some(value) = prev_resource {
            std::env::set_var("AZURE_ANTHROPIC_RESOURCE", value);
        } else {
            std::env::remove_var("AZURE_ANTHROPIC_RESOURCE");
        }
        if let Some(value) = prev_auth_method {
            std::env::set_var("AZURE_ANTHROPIC_AUTH_METHOD", value);
        } else {
            std::env::remove_var("AZURE_ANTHROPIC_AUTH_METHOD");
        }
    }

    #[test]
    fn test_backward_compatible_field_names() {
        // Test that old field names still work via serde aliases
        let toml_with_old_names = r#"
[logging]
file_logging_enabled = true
trace_enabled = true
"#;
        let cfg: AppConfig = toml::from_str(toml_with_old_names).unwrap();
        assert!(cfg.logging.enable_logs);
        assert!(cfg.logging.enable_traces);

        // Test that new field names work
        let toml_with_new_names = r#"
[logging]
enable_logs = true
enable_traces = true
"#;
        let cfg: AppConfig = toml::from_str(toml_with_new_names).unwrap();
        assert!(cfg.logging.enable_logs);
        assert!(cfg.logging.enable_traces);
    }
}
