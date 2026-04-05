//! Parsers for trace file formats.
//!
//! Provides parsing and normalization of both consolidated `.json` and streaming
//! `.jsonl` trace formats into a unified data structure.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::trace_models::{TraceData, TraceEvent};

/// Parse a trace file and return normalized trace data.
///
/// Automatically detects the format based on file extension and delegates
/// to the appropriate parser.
///
/// # Arguments
///
/// * `path` - Path to the trace file
///
/// # Returns
///
/// Normalized `TraceData` structure
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_trace_file(path: &Path) -> Result<TraceData> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .context("Invalid file extension")?;

    match extension {
        "json" => parse_json_trace(path),
        "jsonl" => parse_jsonl_trace(path),
        _ => anyhow::bail!("Unsupported trace file format: {}", extension),
    }
}

/// Parse a consolidated JSON trace file.
///
/// Expects a JSON object with the following structure:
/// ```json
/// {
///   "session_id": "...",
///   "agent_name": "...",
///   "model": "...",
///   "messages": [...],
///   "started_at": "...",
///   "ended_at": "..."
/// }
/// ```
///
/// This format is written by `write_session_log()` in `logging.rs`.
pub fn parse_json_trace(path: &Path) -> Result<TraceData> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read trace file: {}", path.display()))?;

    let json: Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON trace: {}", path.display()))?;

    let session_id = json["session_id"].as_str().unwrap_or("unknown").to_string();

    let mut trace = TraceData::new(session_id.clone());

    // Extract metadata
    trace.agent_name = json["agent_name"].as_str().map(String::from);
    trace.model = json["model"].as_str().map(String::from);

    if let Some(started) = json["started_at"].as_str() {
        trace.started_at = DateTime::parse_from_rfc3339(started)
            .ok()
            .map(|dt| dt.with_timezone(&Utc));
    }

    if let Some(ended) = json["ended_at"].as_str() {
        trace.ended_at = DateTime::parse_from_rfc3339(ended)
            .ok()
            .map(|dt| dt.with_timezone(&Utc));
    }

    // Convert messages to events
    if let Some(messages) = json["messages"].as_array() {
        let start_time = trace.started_at.unwrap_or_else(Utc::now);

        for (idx, message) in messages.iter().enumerate() {
            let elapsed_ms = idx as f64 * 100.0; // Approximate timing

            let timestamp = message["timestamp"]
                .as_str()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(start_time);

            // Determine event type from message role
            let event_type = if let Some(role) = message["role"].as_str() {
                match role {
                    "user" => "user_message",
                    "assistant" => {
                        if message["tool_calls"].is_array() {
                            "tool_calls"
                        } else if message["reasoning"].is_string() {
                            "reasoning"
                        } else {
                            "assistant_message"
                        }
                    }
                    "tool" => "tool_result",
                    _ => "message",
                }
            } else {
                "message"
            };

            trace.add_event(TraceEvent::new(
                timestamp,
                elapsed_ms,
                event_type.to_string(),
                message.clone(),
            ));
        }
    }

    trace.calculate_duration();
    Ok(trace)
}

/// Parse a streaming JSONL trace file.
///
/// Expects one JSON object per line with the structure:
/// ```json
/// {"timestamp": "...", "elapsed_ms": 123.45, "type": "event_type", "data": {...}}
/// ```
///
/// This format is written by `TraceConsumer` in `agent/consumers/trace.rs`.
pub fn parse_jsonl_trace(path: &Path) -> Result<TraceData> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open trace file: {}", path.display()))?;

    let reader = BufReader::new(file);
    let session_id = extract_session_id_from_path(path)?;

    let mut trace = TraceData::new(session_id);

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "Failed to read line {} from trace file: {}",
                line_num + 1,
                path.display()
            )
        })?;

        if line.trim().is_empty() {
            continue;
        }

        let entry: Value = serde_json::from_str(&line).with_context(|| {
            format!(
                "Failed to parse JSON on line {} in trace file: {}",
                line_num + 1,
                path.display()
            )
        })?;

        // Extract timestamp
        let timestamp = entry["timestamp"]
            .as_str()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        // Extract elapsed time
        let elapsed_ms = entry["elapsed_ms"].as_f64().unwrap_or(0.0);

        // Extract event type
        let event_type = entry["type"].as_str().unwrap_or("unknown").to_string();

        // Extract data payload
        let data = entry["data"].clone();

        // Extract metadata from specific events
        if event_type == "session_started" {
            if let Some(sid) = data["session_id"].as_str() {
                trace.session_id = sid.to_string();
            }
        }

        trace.add_event(TraceEvent::new(timestamp, elapsed_ms, event_type, data));
    }

    trace.sort_events();
    trace.calculate_duration();

    Ok(trace)
}

/// Extract session ID from trace file path.
///
/// Extracts the session UUID from filenames like `session-{uuid}.json[l]`.
fn extract_session_id_from_path(path: &Path) -> Result<String> {
    let filename = path
        .file_stem()
        .and_then(|s| s.to_str())
        .context("Invalid filename")?;

    // Remove "session-" prefix if present
    let session_id = filename.strip_prefix("session-").unwrap_or(filename);

    Ok(session_id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_extract_session_id() {
        let path = Path::new("session-abc123.json");
        let id = extract_session_id_from_path(path).unwrap();
        assert_eq!(id, "abc123");
    }

    #[test]
    fn test_parse_jsonl_trace() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("jsonl");

        writeln!(
            temp_file,
            r#"{{"timestamp":"2025-01-01T00:00:00Z","elapsed_ms":0.0,"type":"session_started","data":{{"session_id":"test-123"}}}}"#
        )
        .unwrap();
        writeln!(
            temp_file,
            r#"{{"timestamp":"2025-01-01T00:00:01Z","elapsed_ms":1000.0,"type":"content","data":{{"content":"Hello"}}}}"#
        )
        .unwrap();
        temp_file.flush().unwrap();

        // Copy to file with .jsonl extension for test
        std::fs::copy(temp_file.path(), &path).unwrap();

        let trace = parse_jsonl_trace(&path).unwrap();

        assert_eq!(trace.session_id, "test-123");
        assert_eq!(trace.events.len(), 2);
        assert_eq!(trace.events[0].event_type, "session_started");
        assert_eq!(trace.events[1].event_type, "content");
    }

    #[test]
    fn test_parse_json_trace() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("json");

        let json_content = r#"{
            "session_id": "test-456",
            "agent_name": "test-agent",
            "model": "gpt-4",
            "started_at": "2025-01-01T00:00:00Z",
            "ended_at": "2025-01-01T00:01:00Z",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "2025-01-01T00:00:00Z"
                },
                {
                    "role": "assistant",
                    "content": "Hi there",
                    "timestamp": "2025-01-01T00:00:30Z"
                }
            ]
        }"#;

        temp_file.write_all(json_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        std::fs::copy(temp_file.path(), &path).unwrap();

        let trace = parse_json_trace(&path).unwrap();

        assert_eq!(trace.session_id, "test-456");
        assert_eq!(trace.agent_name, Some("test-agent".to_string()));
        assert_eq!(trace.model, Some("gpt-4".to_string()));
        assert_eq!(trace.events.len(), 2);
    }
}
