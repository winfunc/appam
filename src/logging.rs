//! Structured runtime logging plus persisted session transcripts.
//!
//! Appam keeps tracing logs and per-session artifacts separate so operators can
//! choose how much runtime detail to retain. This module initializes the global
//! `tracing` subscriber and provides a helper for writing standalone session
//! snapshots.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use chrono::Utc;
use serde::Serialize;
use tracing::subscriber::set_global_default;
use tracing_appender::non_blocking::{self, WorkerGuard};
use tracing_appender::rolling;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{fmt, EnvFilter};

use crate::config::{LogFormat, LoggingConfig};

/// Guards for non-blocking appenders that must remain alive for logging to work.
///
/// Stores non-blocking writer guards to prevent premature drop, which would
/// cause log events to be lost.
pub struct LoggingGuards {
    _log_guard: Option<WorkerGuard>,
    _json_guard: Option<WorkerGuard>,
}

/// Initialize the global `tracing` subscriber for Appam.
///
/// Sets up a tracing subscriber with:
/// - Optional human-readable console output
/// - File logging (plain .log, JSON .jsonl, or both)
/// - Environment-based log level filtering
///
/// The log format is controlled by `logging.log_format`:
/// - `LogFormat::Plain`: Human-readable .log file
/// - `LogFormat::Json`: Structured .jsonl file
/// - `LogFormat::Both`: Both formats simultaneously
///
/// Returns the path to the logs directory for session transcript storage.
///
/// # Errors
///
/// Returns an error if the logs directory cannot be created or the subscriber
/// cannot be initialized.
///
/// # Examples
///
/// ```no_run
/// use appam::config::{LogFormat, LoggingConfig};
/// use appam::logging::init_logging;
///
/// let mut config = LoggingConfig::default();
/// config.log_format = LogFormat::Both;
/// let logs_dir = init_logging(&config).unwrap();
/// ```
pub fn init_logging(logging: &LoggingConfig) -> Result<PathBuf> {
    // Ensure directory exists only if file logging is enabled
    if logging.enable_logs {
        fs::create_dir_all(&logging.logs_dir).with_context(|| {
            format!(
                "Failed to create logs directory: {}",
                logging.logs_dir.display()
            )
        })?;
    }

    // Human console layer (optional)
    let console_layer = if logging.human_console {
        Some(fmt::layer().with_target(false).with_level(true))
    } else {
        None
    };

    let now = Utc::now().format("%Y%m%dT%H%M%SZ");

    // Plain text file layer (optional, only if file logging is enabled)
    let (plain_layer, log_guard) = if logging.enable_logs
        && (logging.log_format == LogFormat::Plain || logging.log_format == LogFormat::Both)
    {
        let filename = format!("run-{}.log", now);
        let file_appender = rolling::never(&logging.logs_dir, &filename);
        let (non_blocking, guard) = non_blocking::NonBlockingBuilder::default()
            .lossy(false)
            .finish(file_appender);

        let layer = fmt::layer()
            .with_target(false)
            .with_level(true)
            .with_ansi(false) // No color codes in file
            .with_writer(move || non_blocking.clone());

        (Some(layer), Some(guard))
    } else {
        (None, None)
    };

    // JSON file layer (optional, only if file logging is enabled)
    let (json_layer, json_guard) = if logging.enable_logs
        && (logging.log_format == LogFormat::Json || logging.log_format == LogFormat::Both)
    {
        let filename = format!("run-{}.jsonl", now);
        let file_appender = rolling::never(&logging.logs_dir, &filename);
        let (non_blocking, guard) = non_blocking::NonBlockingBuilder::default()
            .lossy(false)
            .finish(file_appender);

        let layer = fmt::layer()
            .json()
            .with_current_span(true)
            .with_span_list(true)
            .with_target(false)
            .with_level(true)
            .with_writer(move || non_blocking.clone());

        (Some(layer), Some(guard))
    } else {
        (None, None)
    };

    // Store guards to prevent premature drop
    static LOG_GUARDS: OnceLock<LoggingGuards> = OnceLock::new();
    let _ = LOG_GUARDS.set(LoggingGuards {
        _log_guard: log_guard,
        _json_guard: json_guard,
    });

    // Build filter from config and environment
    let filter_directive = std::env::var("RUST_LOG").unwrap_or_else(|_| logging.level.clone());
    let env_filter =
        EnvFilter::try_new(&filter_directive).unwrap_or_else(|_| EnvFilter::new("info"));

    // Compose and set subscriber
    let subscriber = tracing_subscriber::registry()
        .with(env_filter)
        .with(plain_layer)
        .with(json_layer)
        .with(console_layer);

    // Try to set global subscriber - ignore error if already set
    // This allows multiple agent sessions to run without conflict
    match set_global_default(subscriber) {
        Ok(_) => tracing::debug!("Tracing subscriber initialized"),
        Err(_) => {
            // Subscriber already set by main or previous session - this is fine
            tracing::debug!("Tracing subscriber already initialized, skipping");
        }
    }

    Ok(logging.logs_dir.clone())
}

/// Write a complete session snapshot to a JSON file.
///
/// Persists the full conversation history, metadata, and timestamps for a
/// single agent interaction. Session logs enable:
/// - Debugging and troubleshooting
/// - Evaluation and quality analysis
/// - Compliance and auditability
/// - Replay and testing
///
/// Session files are written to `logs/session-<uuid>.json`.
///
/// # Security
///
/// Session logs may contain user data and API keys should never be included.
/// Ensure logs are stored securely and access is restricted.
///
/// # Errors
///
/// Returns an error if the file cannot be written or serialization fails.
pub fn write_session_log<T: Serialize>(
    logs_dir: &Path,
    session_id: &str,
    session: &T,
) -> Result<PathBuf> {
    fs::create_dir_all(logs_dir)
        .with_context(|| format!("Failed to create logs directory: {}", logs_dir.display()))?;

    let file_path = logs_dir.join(format!("session-{}.json", session_id));
    let content = serde_json::to_vec_pretty(session).context("Failed to serialize session log")?;

    fs::write(&file_path, content)
        .with_context(|| format!("Failed to write session log: {}", file_path.display()))?;

    Ok(file_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn test_write_session_log() {
        let dir = TempDir::new().unwrap();
        let session = json!({
            "session_id": "test-123",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        });

        let result = write_session_log(dir.path(), "test-123", &session);
        assert!(result.is_ok());

        let log_path = result.unwrap();
        assert!(log_path.exists());

        let content = fs::read_to_string(log_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["session_id"], "test-123");
    }
}
