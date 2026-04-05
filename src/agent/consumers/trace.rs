//! Real-time session trace consumer.
//!
//! Writes structured JSON traces to `.jsonl` files as agent events occur,
//! enabling real-time monitoring and post-execution analysis.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::Utc;
use serde_json::json;

use crate::config::TraceFormat;

use super::super::streaming::{StreamConsumer, StreamEvent};

/// Real-time trace consumer that writes events to `.jsonl` files.
///
/// Each event is written as a separate JSON line with timestamp, elapsed time,
/// event type, and event-specific data. Files are flushed immediately for
/// real-time visibility.
///
/// # Format
///
/// ```json
/// {"timestamp":"2025-10-24T10:30:45.123Z","elapsed_ms":1523.45,"type":"content","data":{...}}
/// ```
///
/// # Examples
///
/// ```no_run
/// use appam::agent::consumers::TraceConsumer;
/// use appam::config::TraceFormat;
/// use std::path::Path;
///
/// # fn main() -> anyhow::Result<()> {
/// let trace = TraceConsumer::new(
///     Path::new("logs"),
///     "session-123",
///     TraceFormat::Detailed
/// )?;
/// # Ok(())
/// # }
/// ```
pub struct TraceConsumer {
    writer: Arc<Mutex<BufWriter<File>>>,
    format: TraceFormat,
    start_time: Instant,
}

impl TraceConsumer {
    /// Create a new trace consumer.
    ///
    /// Creates a `.jsonl` trace file at `<logs_dir>/session-<session_id>.jsonl`.
    /// The file is opened in append mode to support session continuation.
    ///
    /// # Parameters
    ///
    /// - `logs_dir`: Directory for trace files
    /// - `session_id`: Unique session identifier
    /// - `format`: Trace detail level (Compact or Detailed)
    ///
    /// # Errors
    ///
    /// Returns an error if the trace file cannot be created or opened.
    pub fn new(logs_dir: &Path, session_id: &str, format: TraceFormat) -> Result<Self> {
        std::fs::create_dir_all(logs_dir)
            .with_context(|| format!("Failed to create logs directory: {}", logs_dir.display()))?;

        let trace_path = logs_dir.join(format!("session-{}.jsonl", session_id));

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&trace_path)
            .with_context(|| format!("Failed to create trace file: {}", trace_path.display()))?;

        let writer = BufWriter::new(file);

        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
            format,
            start_time: Instant::now(),
        })
    }

    /// Write a trace entry to the file.
    ///
    /// Each entry is a JSON object on a single line with timestamp, elapsed time,
    /// event type, and event-specific data.
    fn write_entry(&self, event_type: &str, data: serde_json::Value) -> Result<()> {
        let elapsed_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        let timestamp = Utc::now();

        let entry = json!({
            "timestamp": timestamp.to_rfc3339(),
            "elapsed_ms": elapsed_ms,
            "type": event_type,
            "data": data,
        });

        let mut writer = self.writer.lock().unwrap();
        serde_json::to_writer(&mut *writer, &entry).context("Failed to serialize trace entry")?;
        writeln!(writer).context("Failed to write newline")?;
        writer.flush().context("Failed to flush trace file")?;

        Ok(())
    }
}

impl StreamConsumer for TraceConsumer {
    fn on_event(&self, event: &StreamEvent) -> Result<()> {
        match event {
            StreamEvent::SessionStarted { session_id } => {
                self.write_entry(
                    "session_started",
                    json!({
                        "session_id": session_id,
                    }),
                )?;
            }

            StreamEvent::Content { content } => {
                self.write_entry(
                    "content",
                    json!({
                        "content": content,
                    }),
                )?;
            }

            StreamEvent::Reasoning { content } => {
                // Only include reasoning in detailed traces
                if self.format == TraceFormat::Detailed {
                    self.write_entry(
                        "reasoning",
                        json!({
                            "content": content,
                        }),
                    )?;
                }
            }

            StreamEvent::ToolCallStarted {
                tool_name,
                arguments,
            } => {
                self.write_entry(
                    "tool_call_started",
                    json!({
                        "tool_name": tool_name,
                        "arguments": arguments,
                    }),
                )?;
            }

            StreamEvent::ToolCallCompleted {
                tool_name,
                result,
                success,
                duration_ms,
            } => {
                let data = if self.format == TraceFormat::Detailed {
                    json!({
                        "tool_name": tool_name,
                        "result": result,
                        "success": success,
                        "duration_ms": duration_ms,
                    })
                } else {
                    // Compact: omit full result
                    json!({
                        "tool_name": tool_name,
                        "success": success,
                        "duration_ms": duration_ms,
                    })
                };

                self.write_entry("tool_call_completed", data)?;
            }

            StreamEvent::ToolCallFailed { tool_name, error } => {
                self.write_entry(
                    "tool_call_failed",
                    json!({
                        "tool_name": tool_name,
                        "error": error,
                    }),
                )?;
            }

            StreamEvent::TurnCompleted => {
                self.write_entry("turn_completed", json!({}))?;
            }

            StreamEvent::Done => {
                self.write_entry(
                    "done",
                    json!({
                        "total_elapsed_ms": self.start_time.elapsed().as_secs_f64() * 1000.0,
                    }),
                )?;
            }

            StreamEvent::UsageUpdate { snapshot } => {
                self.write_entry(
                    "usage_update",
                    json!({
                        "total_tokens": snapshot.total_tokens(),
                        "total_cost_usd": snapshot.total_cost_usd,
                        "request_count": snapshot.request_count,
                        "total_input_tokens": snapshot.total_input_tokens,
                        "total_output_tokens": snapshot.total_output_tokens,
                        "total_cache_creation_tokens": snapshot.total_cache_creation_tokens,
                        "total_cache_read_tokens": snapshot.total_cache_read_tokens,
                        "total_reasoning_tokens": snapshot.total_reasoning_tokens,
                    }),
                )?;
            }

            StreamEvent::Error {
                message,
                failure_kind,
                provider,
                model,
                http_status,
                request_payload,
                response_payload,
                provider_response_id,
            } => {
                self.write_entry(
                    "error",
                    json!({
                        "message": message,
                        "failure_kind": failure_kind,
                        "provider": provider,
                        "model": model,
                        "http_status": http_status,
                        "request_payload": request_payload,
                        "response_payload": response_payload,
                        "provider_response_id": provider_response_id,
                    }),
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TraceFormat;
    use tempfile::TempDir;

    #[test]
    fn test_trace_consumer_creation() {
        let dir = TempDir::new().unwrap();
        let trace = TraceConsumer::new(dir.path(), "test-123", TraceFormat::Detailed);
        assert!(trace.is_ok());

        let trace_file = dir.path().join("session-test-123.jsonl");
        assert!(trace_file.exists());
    }

    #[test]
    fn test_trace_consumer_writes() {
        let dir = TempDir::new().unwrap();
        let trace = TraceConsumer::new(dir.path(), "test-456", TraceFormat::Detailed).unwrap();

        // Write some events
        trace
            .on_event(&StreamEvent::SessionStarted {
                session_id: "test-456".to_string(),
            })
            .unwrap();

        trace
            .on_event(&StreamEvent::Content {
                content: "Hello".to_string(),
            })
            .unwrap();

        trace.on_event(&StreamEvent::Done).unwrap();

        // Read file and verify
        let trace_file = dir.path().join("session-test-456.jsonl");
        let content = std::fs::read_to_string(trace_file).unwrap();

        assert!(content.contains("session_started"));
        assert!(content.contains("content"));
        assert!(content.contains("done"));
        assert!(content.contains("Hello"));

        // Verify each line is valid JSON
        for line in content.lines() {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(parsed["timestamp"].is_string());
            assert!(parsed["elapsed_ms"].is_number());
            assert!(parsed["type"].is_string());
        }
    }

    #[test]
    fn test_compact_vs_detailed() {
        let dir = TempDir::new().unwrap();

        // Test compact format - should exclude reasoning
        let compact = TraceConsumer::new(dir.path(), "compact", TraceFormat::Compact).unwrap();
        compact
            .on_event(&StreamEvent::Reasoning {
                content: "thinking...".to_string(),
            })
            .unwrap();

        let compact_file = dir.path().join("session-compact.jsonl");
        let compact_content = std::fs::read_to_string(compact_file).unwrap();
        assert!(!compact_content.contains("reasoning"));

        // Test detailed format - should include reasoning
        let detailed = TraceConsumer::new(dir.path(), "detailed", TraceFormat::Detailed).unwrap();
        detailed
            .on_event(&StreamEvent::Reasoning {
                content: "thinking...".to_string(),
            })
            .unwrap();

        let detailed_file = dir.path().join("session-detailed.jsonl");
        let detailed_content = std::fs::read_to_string(detailed_file).unwrap();
        assert!(detailed_content.contains("reasoning"));
        assert!(detailed_content.contains("thinking..."));
    }

    #[test]
    fn test_tool_call_tracing() {
        let dir = TempDir::new().unwrap();
        let trace = TraceConsumer::new(dir.path(), "tools", TraceFormat::Detailed).unwrap();

        trace
            .on_event(&StreamEvent::ToolCallStarted {
                tool_name: "test_tool".to_string(),
                arguments: r#"{"arg": "value"}"#.to_string(),
            })
            .unwrap();

        trace
            .on_event(&StreamEvent::ToolCallCompleted {
                tool_name: "test_tool".to_string(),
                result: json!({"output": "success"}),
                success: true,
                duration_ms: 123.45,
            })
            .unwrap();

        let trace_file = dir.path().join("session-tools.jsonl");
        let content = std::fs::read_to_string(trace_file).unwrap();

        assert!(content.contains("tool_call_started"));
        assert!(content.contains("tool_call_completed"));
        assert!(content.contains("test_tool"));
        assert!(content.contains("123.45"));
    }
}
