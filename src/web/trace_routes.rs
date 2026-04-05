//! API routes for trace visualization.
//!
//! Provides endpoints for listing trace files and retrieving trace data
//! in a normalized format.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json, Response},
    routing::get,
    Router,
};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error};

use super::trace_models::{TraceData, TraceFile, TraceFileFormat};
use super::trace_parser::parse_trace_file;

/// Shared state for trace routes.
#[derive(Clone)]
pub struct TraceState {
    /// Directory containing trace files
    pub traces_dir: PathBuf,
}

impl TraceState {
    /// Create a new trace state.
    pub fn new(traces_dir: PathBuf) -> Self {
        Self { traces_dir }
    }
}

/// Create the trace routes router.
pub fn trace_routes() -> Router<Arc<TraceState>> {
    Router::new()
        .route("/", get(serve_trace_ui))
        .route("/api/traces", get(list_traces))
        .route("/api/traces/:id", get(get_trace))
}

/// Serve the trace visualizer UI.
///
/// Returns the embedded HTML file containing the complete trace visualizer
/// interface with all CSS and JavaScript inline.
///
/// GET /
async fn serve_trace_ui() -> impl IntoResponse {
    Html(include_str!("static/trace_visualizer.html"))
}

/// List all available trace files.
///
/// Scans the traces directory for session-*.json and session-*.jsonl files,
/// returning metadata sorted by modification time (newest first).
///
/// GET /api/traces
async fn list_traces(
    State(state): State<Arc<TraceState>>,
) -> Result<Json<Vec<TraceFile>>, ApiError> {
    debug!(
        traces_dir = %state.traces_dir.display(),
        "Listing trace files"
    );

    let mut traces = Vec::new();

    // Read directory entries
    let entries = std::fs::read_dir(&state.traces_dir).map_err(|e| {
        error!(error = %e, "Failed to read traces directory");
        ApiError::Internal(format!("Failed to read traces directory: {}", e))
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            error!(error = %e, "Failed to read directory entry");
            ApiError::Internal(format!("Failed to read directory entry: {}", e))
        })?;

        let path = entry.path();

        // Only process session trace files
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if !filename.starts_with("session-") {
            continue;
        }

        let extension = path.extension().and_then(|e| e.to_str());
        let format = match extension {
            Some("json") => TraceFileFormat::Json,
            Some("jsonl") => TraceFileFormat::Jsonl,
            _ => continue, // Skip non-trace files
        };

        // Extract session ID from filename
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("session-"))
            .unwrap_or("")
            .to_string();

        if id.is_empty() {
            continue;
        }

        // Get file metadata
        let metadata = entry.metadata().map_err(|e| {
            error!(error = %e, "Failed to read file metadata");
            ApiError::Internal(format!("Failed to read file metadata: {}", e))
        })?;

        let size = metadata.len();
        let modified = metadata
            .modified()
            .ok()
            .map(|t| {
                let datetime: chrono::DateTime<chrono::Utc> = t.into();
                datetime
            })
            .unwrap_or_else(chrono::Utc::now);

        traces.push(TraceFile {
            id,
            filename: filename.to_string(),
            format,
            size,
            modified,
            path: path.clone(),
        });
    }

    // Sort by modification time (newest first)
    traces.sort_by(|a, b| b.modified.cmp(&a.modified));

    debug!(count = traces.len(), "Found trace files");

    Ok(Json(traces))
}

/// Get a specific trace by ID.
///
/// Loads and parses the trace file, automatically detecting the format
/// and normalizing to a unified structure.
///
/// GET /api/traces/:id
async fn get_trace(
    State(state): State<Arc<TraceState>>,
    Path(id): Path<String>,
) -> Result<Json<TraceData>, ApiError> {
    debug!(session_id = %id, "Fetching trace");

    // Try both .json and .jsonl extensions
    let json_path = state.traces_dir.join(format!("session-{}.json", id));
    let jsonl_path = state.traces_dir.join(format!("session-{}.jsonl", id));

    let path = if json_path.exists() {
        json_path
    } else if jsonl_path.exists() {
        jsonl_path
    } else {
        return Err(ApiError::NotFound(format!("Trace not found: {}", id)));
    };

    debug!(path = %path.display(), "Loading trace file");

    let trace = parse_trace_file(&path).map_err(|e| {
        error!(error = %e, "Failed to parse trace file");
        ApiError::Internal(format!("Failed to parse trace: {}", e))
    })?;

    Ok(Json(trace))
}

/// API error types for trace routes.
#[derive(Debug)]
pub enum ApiError {
    /// Resource not found
    NotFound(String),
    /// Internal server error
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_trace_state_creation() {
        let dir = TempDir::new().unwrap();
        let state = TraceState::new(dir.path().to_path_buf());
        assert_eq!(state.traces_dir, dir.path());
    }
}
