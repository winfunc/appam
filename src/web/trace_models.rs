//! Data models for trace visualization.
//!
//! Defines structures for representing trace files and their contents in a
//! unified format, supporting both consolidated `.json` and streaming `.jsonl` formats.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Represents a trace file discovered in the logs directory.
///
/// Contains metadata about the file without loading its full contents,
/// enabling efficient listing and sorting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceFile {
    /// Unique identifier extracted from filename (session ID)
    pub id: String,
    /// Full filename (e.g., "session-abc123.json")
    pub filename: String,
    /// Trace file format
    pub format: TraceFileFormat,
    /// File size in bytes
    pub size: u64,
    /// Last modification timestamp
    pub modified: DateTime<Utc>,
    /// Full file path (not serialized to client)
    #[serde(skip)]
    pub path: PathBuf,
}

/// Format of a trace file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TraceFileFormat {
    /// Consolidated JSON format (session-*.json)
    Json,
    /// Streaming JSONL format (session-*.jsonl)
    Jsonl,
}

/// Unified representation of trace data from either format.
///
/// Normalizes both `.json` and `.jsonl` formats into a common structure
/// for consistent frontend rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceData {
    /// Session identifier
    pub session_id: String,
    /// Agent name (if available)
    pub agent_name: Option<String>,
    /// Model identifier (if available)
    pub model: Option<String>,
    /// All events in chronological order
    pub events: Vec<TraceEvent>,
    /// Session start time
    pub started_at: Option<DateTime<Utc>>,
    /// Session end time
    pub ended_at: Option<DateTime<Utc>>,
    /// Total duration in milliseconds
    pub total_duration_ms: Option<f64>,
}

/// A single event in a trace.
///
/// Represents any activity during an agent session: content generation,
/// reasoning, tool calls, errors, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Elapsed time from session start in milliseconds
    pub elapsed_ms: f64,
    /// Event type identifier
    pub event_type: String,
    /// Event-specific data
    pub data: serde_json::Value,
}

impl TraceEvent {
    /// Create a new trace event.
    pub fn new(
        timestamp: DateTime<Utc>,
        elapsed_ms: f64,
        event_type: String,
        data: serde_json::Value,
    ) -> Self {
        Self {
            timestamp,
            elapsed_ms,
            event_type,
            data,
        }
    }
}

impl TraceData {
    /// Create a new empty trace data structure.
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            agent_name: None,
            model: None,
            events: Vec::new(),
            started_at: None,
            ended_at: None,
            total_duration_ms: None,
        }
    }

    /// Add an event to the trace.
    pub fn add_event(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    /// Sort events by elapsed time (should already be sorted, but ensures ordering).
    pub fn sort_events(&mut self) {
        self.events.sort_by(|a, b| {
            a.elapsed_ms
                .partial_cmp(&b.elapsed_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Calculate total duration from events.
    pub fn calculate_duration(&mut self) {
        if let (Some(first), Some(last)) = (self.events.first(), self.events.last()) {
            self.total_duration_ms = Some(last.elapsed_ms - first.elapsed_ms);
            self.started_at = Some(first.timestamp);
            self.ended_at = Some(last.timestamp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_trace_event_creation() {
        let event = TraceEvent::new(
            Utc::now(),
            100.0,
            "content".to_string(),
            json!({"content": "test"}),
        );

        assert_eq!(event.event_type, "content");
        assert_eq!(event.elapsed_ms, 100.0);
    }

    #[test]
    fn test_trace_data_duration() {
        let mut trace = TraceData::new("test-session".to_string());
        let now = Utc::now();

        trace.add_event(TraceEvent::new(now, 0.0, "start".to_string(), json!({})));

        trace.add_event(TraceEvent::new(now, 1000.0, "end".to_string(), json!({})));

        trace.calculate_duration();

        assert_eq!(trace.total_duration_ms, Some(1000.0));
        assert!(trace.started_at.is_some());
        assert!(trace.ended_at.is_some());
    }
}
