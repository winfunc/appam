//! SQLite-based trace consumer for real-time event storage.
//!
//! Writes structured trace events to SQLite as agent events occur,
//! enabling persistent storage, querying, and analysis of agent sessions.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Error, Result};
use chrono::Utc;
use serde_json::json;
#[allow(unused_imports)]
use sqlx::Row;
use sqlx::SqlitePool;
use tokio::sync::Mutex as AsyncMutex;
use tracing::warn;

use super::super::streaming::{StreamConsumer, StreamEvent};

/// Pending event being accumulated before writing to database.
///
/// Consecutive events of the same type (e.g., Content chunks) are accumulated
/// into a single database entry for efficiency and readability.
struct PendingEvent {
    /// Type of event being accumulated
    event_type: String,
    /// Accumulated data (concatenated content for streaming types)
    accumulated_data: serde_json::Value,
    /// Timestamp of the first chunk
    start_timestamp: chrono::DateTime<chrono::Utc>,
    /// Elapsed milliseconds at the first chunk
    start_elapsed_ms: f64,
}

/// SQLite-based trace consumer that writes events to database.
///
/// Each event is written to the `agent_traces` table with sequence number,
/// timestamp, and event-specific data. Complete sessions are also stored
/// in the `agent_sessions` table for full replay capability.
///
/// # Examples
///
/// ```no_run
/// use appam::agent::consumers::SqliteTraceConsumer;
/// use sqlx::SqlitePool;
/// use std::sync::Arc;
///
/// # async fn example(pool: SqlitePool) -> anyhow::Result<()> {
/// let trace = SqliteTraceConsumer::new(
///     Arc::new(pool),
///     "session-123".to_string(),
///     "enrichment-agent".to_string(),
///     "claude-haiku-4-5-20251001".to_string(),
///     "index".to_string(),
///     1,
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct SqliteTraceConsumer {
    pool: Arc<SqlitePool>,
    session_id: String,
    agent_name: String,
    model: String,
    job_type: String,
    job_version: i64,
    start_time: Instant,
    sequence_counter: Arc<Mutex<i64>>,
    events: Arc<Mutex<Vec<serde_json::Value>>>,
    pending_event: Arc<AsyncMutex<Option<PendingEvent>>>,
}

impl SqliteTraceConsumer {
    /// Create a new SQLite trace consumer.
    ///
    /// # Parameters
    ///
    /// - `pool`: SQLite connection pool
    /// - `session_id`: Unique session identifier
    /// - `agent_name`: Name of the agent (enrichment-agent, hunter-agent, picker-agent)
    /// - `model`: LLM model identifier (e.g., claude-haiku-4-5-20251001)
    /// - `job_type`: Type of job (index, scan, pick)
    /// - `job_version`: Version number of the job
    pub fn new(
        pool: Arc<SqlitePool>,
        session_id: String,
        agent_name: String,
        model: String,
        job_type: String,
        job_version: i64,
    ) -> Self {
        Self {
            pool,
            session_id,
            agent_name,
            model,
            job_type,
            job_version,
            start_time: Instant::now(),
            sequence_counter: Arc::new(Mutex::new(0)),
            events: Arc::new(Mutex::new(Vec::new())),
            pending_event: Arc::new(AsyncMutex::new(None)),
        }
    }

    /// Reserve the next per-session sequence number for an externally managed row.
    ///
    /// Some callers persist bootstrap events, such as `system_prompt`, before the
    /// streaming runtime emits `session_started`. Those rows still live in the
    /// same `agent_traces` table and must consume sequence numbers from the same
    /// monotonic counter used by streamed events; otherwise the first streamed
    /// insert collides with the caller-managed row and emits a misleading warning
    /// even though the session continues normally.
    ///
    /// # Returns
    ///
    /// Returns the next available sequence number and advances the shared counter
    /// so subsequent streamed events continue after the reserved slot.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::consumers::SqliteTraceConsumer;
    /// # fn example(consumer: &SqliteTraceConsumer) {
    /// let sequence_number = consumer.reserve_sequence_number();
    /// assert!(sequence_number >= 0);
    /// # }
    /// ```
    pub fn reserve_sequence_number(&self) -> i64 {
        let mut counter = self.sequence_counter.lock().unwrap();
        let sequence_number = *counter;
        *counter += 1;
        sequence_number
    }

    /// Record the raw event stream in memory for later session persistence.
    ///
    /// This snapshot is captured synchronously before background SQLite writes are
    /// spawned so callers can reliably retrieve `get_events()` immediately after a
    /// session completes or fails.
    fn record_event_snapshot(&self, event_type: &str, data: serde_json::Value) {
        let elapsed_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        let timestamp = Utc::now();
        let mut events = self.events.lock().unwrap();
        events.push(json!({
            "timestamp": timestamp.to_rfc3339(),
            "elapsed_ms": elapsed_ms,
            "type": event_type,
            "data": data,
        }));
    }

    /// Emit a structured warning for trace persistence failures.
    ///
    /// The original implementation only logged the outer `anyhow` context, which
    /// hid the underlying SQLite failure and made repeated warnings difficult to
    /// diagnose from the TUI. Logging the full error chain plus session metadata
    /// keeps the failure actionable without surfacing request payloads or other
    /// sensitive content.
    fn warn_trace_persistence_failure(&self, operation: &'static str, error: &Error) {
        warn!(
            session_id = %self.session_id,
            agent_name = %self.agent_name,
            job_type = %self.job_type,
            job_version = self.job_version,
            model = %self.model,
            operation = operation,
            error = %format!("{error:#}"),
            "Trace persistence warning"
        );
    }

    /// Write a trace entry to the database.
    ///
    /// Each entry is a JSON object with timestamp, elapsed time,
    /// event type, and event-specific data.
    async fn write_entry(&self, event_type: &str, data: serde_json::Value) -> Result<()> {
        let elapsed_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        let timestamp = Utc::now();

        let sequence_number = self.reserve_sequence_number();

        // Serialize event data
        let event_data_json = serde_json::to_string(&data)?;

        // Write to database
        sqlx::query(
            r#"
            INSERT INTO agent_traces (
                session_id, agent_name, model, job_type, job_version,
                sequence_number, timestamp, elapsed_ms, event_type,
                event_data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&self.session_id)
        .bind(&self.agent_name)
        .bind(&self.model)
        .bind(&self.job_type)
        .bind(self.job_version)
        .bind(sequence_number)
        .bind(timestamp.to_rfc3339())
        .bind(elapsed_ms)
        .bind(event_type)
        .bind(&event_data_json)
        .bind(Utc::now().to_rfc3339())
        .execute(&*self.pool)
        .await
        .context("Failed to write trace entry to database")?;

        Ok(())
    }

    /// Accumulate an event or flush if type changes.
    ///
    /// This method handles the consolidation logic:
    /// - For streaming types (content, reasoning): accumulates consecutive events
    /// - When event type changes: flushes pending and starts new accumulation
    /// - Returns true if the event was accumulated, false if it should be written immediately
    async fn accumulate_or_flush(&self, event_type: &str, data: serde_json::Value) -> Result<bool> {
        let elapsed_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        let timestamp = Utc::now();

        // Determine if this is a streaming event type that should be accumulated
        let is_streaming_type = matches!(event_type, "content" | "reasoning");

        if !is_streaming_type {
            // Non-streaming events: flush pending and write immediately
            self.flush_pending_event().await?;
            return Ok(false);
        }

        // Handle streaming types - atomically check, flush if needed, and accumulate
        // Extract pending event if it needs flushing (different type)
        let to_flush = {
            let mut pending_guard = self.pending_event.lock().await;

            match pending_guard.as_ref() {
                Some(pending) if pending.event_type != event_type => {
                    // Different type: take it out to flush, then we'll add new one
                    pending_guard.take()
                }
                Some(_pending) => {
                    // Same type: accumulate in place
                    if let (Some(existing_content), Some(new_content)) = (
                        pending_guard.as_ref().and_then(|p| {
                            p.accumulated_data.get("content").and_then(|v| v.as_str())
                        }),
                        data.get("content").and_then(|v| v.as_str()),
                    ) {
                        // Modify in place - we still have the lock
                        let concatenated = format!("{}{}", existing_content, new_content);
                        if let Some(pending_mut) = pending_guard.as_mut() {
                            pending_mut.accumulated_data = json!({
                                "content": concatenated
                            });
                        }
                    }
                    return Ok(true);
                }
                None => {
                    // No pending: start new accumulation
                    *pending_guard = Some(PendingEvent {
                        event_type: event_type.to_string(),
                        accumulated_data: data.clone(),
                        start_timestamp: timestamp,
                        start_elapsed_ms: elapsed_ms,
                    });
                    return Ok(true);
                }
            }
        }; // Lock released here, with the old pending event extracted

        // Flush the old event (lock is released, so other tasks can proceed)
        if let Some(to_flush) = to_flush {
            self.flush_event(to_flush).await?;
        }

        // Now acquire lock again and set the new pending event
        {
            let mut pending_guard = self.pending_event.lock().await;
            *pending_guard = Some(PendingEvent {
                event_type: event_type.to_string(),
                accumulated_data: data.clone(),
                start_timestamp: timestamp,
                start_elapsed_ms: elapsed_ms,
            });
        }

        Ok(true)
    }

    /// Flush a specific event to the database.
    ///
    /// This is a helper that writes a single event without touching the pending_event mutex.
    async fn flush_event(&self, pending: PendingEvent) -> Result<()> {
        // Write the consolidated entry using the original timestamp and elapsed time
        let sequence_number = self.reserve_sequence_number();

        // Serialize event data
        let event_data_json = serde_json::to_string(&pending.accumulated_data)?;

        // Write to database
        sqlx::query(
            r#"
            INSERT INTO agent_traces (
                session_id, agent_name, model, job_type, job_version,
                sequence_number, timestamp, elapsed_ms, event_type,
                event_data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&self.session_id)
        .bind(&self.agent_name)
        .bind(&self.model)
        .bind(&self.job_type)
        .bind(self.job_version)
        .bind(sequence_number)
        .bind(pending.start_timestamp.to_rfc3339())
        .bind(pending.start_elapsed_ms)
        .bind(&pending.event_type)
        .bind(&event_data_json)
        .bind(Utc::now().to_rfc3339())
        .execute(&*self.pool)
        .await
        .context("Failed to write consolidated trace entry to database")?;

        Ok(())
    }

    /// Flush any pending accumulated event to the database.
    ///
    /// This writes the currently buffered event (if any) and clears the buffer.
    /// Used when event type changes or session ends to ensure all events are persisted.
    async fn flush_pending_event(&self) -> Result<()> {
        let pending = {
            let mut guard = self.pending_event.lock().await;
            guard.take()
        };

        if let Some(pending) = pending {
            self.flush_event(pending).await?;
        }

        Ok(())
    }

    /// Get all stored events for this session.
    ///
    /// Used to create the final session record.
    pub fn get_events(&self) -> Vec<serde_json::Value> {
        self.events.lock().unwrap().clone()
    }
}

impl StreamConsumer for SqliteTraceConsumer {
    fn on_event(&self, event: &StreamEvent) -> Result<()> {
        // Get runtime handle for spawning async tasks
        let rt = tokio::runtime::Handle::try_current()
            .context("No tokio runtime available for SqliteTraceConsumer")?;

        // Clone necessary data for spawning
        let pool = Arc::clone(&self.pool);
        let session_id = self.session_id.clone();
        let agent_name = self.agent_name.clone();
        let model = self.model.clone();
        let job_type = self.job_type.clone();
        let job_version = self.job_version;
        let start_time = self.start_time;
        let sequence_counter = Arc::clone(&self.sequence_counter);
        let events = Arc::clone(&self.events);
        let pending_event = Arc::clone(&self.pending_event);

        let (snapshot_type, snapshot_data) = match event {
            StreamEvent::SessionStarted { session_id } => (
                "session_started",
                json!({
                    "session_id": session_id,
                }),
            ),
            StreamEvent::Content { content } => (
                "content",
                json!({
                    "content": content,
                }),
            ),
            StreamEvent::Reasoning { content } => (
                "reasoning",
                json!({
                    "content": content,
                }),
            ),
            StreamEvent::ToolCallStarted {
                tool_name,
                arguments,
            } => (
                "tool_call_started",
                json!({
                    "tool_name": tool_name,
                    "arguments": arguments,
                }),
            ),
            StreamEvent::ToolCallCompleted {
                tool_name,
                result,
                success,
                duration_ms,
            } => (
                "tool_call_completed",
                json!({
                    "tool_name": tool_name,
                    "result": result,
                    "success": success,
                    "duration_ms": duration_ms,
                }),
            ),
            StreamEvent::ToolCallFailed { tool_name, error } => (
                "tool_call_failed",
                json!({
                    "tool_name": tool_name,
                    "error": error,
                }),
            ),
            StreamEvent::TurnCompleted => ("turn_completed", json!({})),
            StreamEvent::UsageUpdate { snapshot } => (
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
            ),
            StreamEvent::Done => (
                "done",
                json!({
                    "total_elapsed_ms": self.start_time.elapsed().as_secs_f64() * 1000.0,
                }),
            ),
            StreamEvent::Error {
                message,
                failure_kind,
                provider,
                model,
                http_status,
                request_payload,
                response_payload,
                provider_response_id,
            } => (
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
            ),
        };
        self.record_event_snapshot(snapshot_type, snapshot_data);

        // Spawn async task for each event type
        match event {
            StreamEvent::SessionStarted {
                session_id: event_session_id,
            } => {
                let event_session_id = event_session_id.clone();
                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };
                    // Flush pending before writing non-streaming event
                    if let Err(e) = consumer.flush_pending_event().await {
                        consumer.warn_trace_persistence_failure("flush_pending_event", &e);
                    }
                    if let Err(e) = consumer
                        .write_entry(
                            "session_started",
                            json!({
                                "session_id": event_session_id,
                            }),
                        )
                        .await
                    {
                        consumer.warn_trace_persistence_failure("write_session_started", &e);
                    }
                });
            }

            StreamEvent::Content { content } => {
                let content = content.clone();
                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };
                    // Use accumulation for streaming content
                    if let Err(e) = consumer
                        .accumulate_or_flush(
                            "content",
                            json!({
                                "content": content,
                            }),
                        )
                        .await
                    {
                        consumer.warn_trace_persistence_failure("accumulate_content", &e);
                    }
                });
            }

            StreamEvent::Reasoning { content } => {
                let content = content.clone();
                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };
                    // Use accumulation for streaming reasoning
                    if let Err(e) = consumer
                        .accumulate_or_flush(
                            "reasoning",
                            json!({
                                "content": content,
                            }),
                        )
                        .await
                    {
                        consumer.warn_trace_persistence_failure("accumulate_reasoning", &e);
                    }
                });
            }

            StreamEvent::ToolCallStarted {
                tool_name,
                arguments,
            } => {
                let tool_name = tool_name.clone();
                let arguments = arguments.clone();
                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };
                    // Flush pending before writing non-streaming event
                    if let Err(e) = consumer.flush_pending_event().await {
                        consumer.warn_trace_persistence_failure("flush_pending_event", &e);
                    }
                    if let Err(e) = consumer
                        .write_entry(
                            "tool_call_started",
                            json!({
                                "tool_name": tool_name,
                                "arguments": arguments,
                            }),
                        )
                        .await
                    {
                        consumer.warn_trace_persistence_failure("write_tool_call_started", &e);
                    }
                });
            }

            StreamEvent::ToolCallCompleted {
                tool_name,
                result,
                success,
                duration_ms,
            } => {
                let tool_name = tool_name.clone();
                let result = result.clone();
                let success = *success;
                let duration_ms = *duration_ms;
                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };
                    // Flush pending before writing non-streaming event
                    if let Err(e) = consumer.flush_pending_event().await {
                        consumer.warn_trace_persistence_failure("flush_pending_event", &e);
                    }
                    if let Err(e) = consumer
                        .write_entry(
                            "tool_call_completed",
                            json!({
                                "tool_name": tool_name,
                                "result": result,
                                "success": success,
                                "duration_ms": duration_ms,
                            }),
                        )
                        .await
                    {
                        consumer.warn_trace_persistence_failure("write_tool_call_completed", &e);
                    }
                });
            }

            StreamEvent::ToolCallFailed { tool_name, error } => {
                let tool_name = tool_name.clone();
                let error = error.clone();
                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };
                    // Flush pending before writing non-streaming event
                    if let Err(e) = consumer.flush_pending_event().await {
                        consumer.warn_trace_persistence_failure("flush_pending_event", &e);
                    }
                    if let Err(e) = consumer
                        .write_entry(
                            "tool_call_failed",
                            json!({
                                "tool_name": tool_name,
                                "error": error,
                            }),
                        )
                        .await
                    {
                        consumer.warn_trace_persistence_failure("write_tool_call_failed", &e);
                    }
                });
            }

            StreamEvent::TurnCompleted => {
                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };
                    // Flush pending at turn boundary
                    if let Err(e) = consumer.flush_pending_event().await {
                        consumer.warn_trace_persistence_failure("flush_pending_event", &e);
                    }
                    if let Err(e) = consumer.write_entry("turn_completed", json!({})).await {
                        consumer.warn_trace_persistence_failure("write_turn_completed", &e);
                    }
                });
            }

            StreamEvent::Done => {
                let total_elapsed_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };
                    // Flush pending before writing final event
                    if let Err(e) = consumer.flush_pending_event().await {
                        consumer.warn_trace_persistence_failure("flush_pending_event", &e);
                    }
                    if let Err(e) = consumer
                        .write_entry(
                            "done",
                            json!({
                                "total_elapsed_ms": total_elapsed_ms,
                            }),
                        )
                        .await
                    {
                        consumer.warn_trace_persistence_failure("write_done", &e);
                    }
                });
            }

            StreamEvent::UsageUpdate { snapshot } => {
                let event_data = serde_json::json!({
                    "total_tokens": snapshot.total_tokens(),
                    "total_cost_usd": snapshot.total_cost_usd,
                    "request_count": snapshot.request_count,
                    "total_input_tokens": snapshot.total_input_tokens,
                    "total_output_tokens": snapshot.total_output_tokens,
                    "total_cache_creation_tokens": snapshot.total_cache_creation_tokens,
                    "total_cache_read_tokens": snapshot.total_cache_read_tokens,
                    "total_reasoning_tokens": snapshot.total_reasoning_tokens,
                });

                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };

                    if let Err(error) = consumer.write_entry("usage_update", event_data).await {
                        consumer.warn_trace_persistence_failure("write_usage_update", &error);
                    }
                });
            }

            StreamEvent::Error {
                message,
                failure_kind,
                provider,
                model: failure_model,
                http_status,
                request_payload,
                response_payload,
                provider_response_id,
            } => {
                let message = message.clone();
                let failure_kind = failure_kind.map(|kind| kind.to_string());
                let provider = provider.clone();
                let model_name = failure_model.clone();
                let http_status = *http_status;
                let request_payload = request_payload.clone();
                let response_payload = response_payload.clone();
                let provider_response_id = provider_response_id.clone();
                rt.spawn(async move {
                    let consumer = SqliteTraceConsumer {
                        pool,
                        session_id,
                        agent_name,
                        model,
                        job_type,
                        job_version,
                        start_time,
                        sequence_counter,
                        events,
                        pending_event,
                    };
                    // Flush pending before writing error event
                    if let Err(e) = consumer.flush_pending_event().await {
                        consumer.warn_trace_persistence_failure("flush_pending_event", &e);
                    }
                    if let Err(e) = consumer
                        .write_entry(
                            "error",
                            json!({
                                "message": message,
                                "failure_kind": failure_kind,
                                "provider": provider,
                                "model": model_name,
                                "http_status": http_status,
                                "request_payload": request_payload,
                                "response_payload": response_payload,
                                "provider_response_id": provider_response_id,
                            }),
                        )
                        .await
                    {
                        consumer.warn_trace_persistence_failure("write_error", &e);
                    }
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use tempfile::TempDir;

    async fn setup_test_db() -> (SqlitePool, TempDir) {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");

        let options = SqliteConnectOptions::new()
            .filename(&db_path)
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap();

        // Create schema
        sqlx::query(
            r#"
            CREATE TABLE agent_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                model TEXT NOT NULL,
                job_type TEXT NOT NULL,
                job_version INTEGER NOT NULL,
                sequence_number INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                elapsed_ms REAL NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();

        (pool, dir)
    }

    #[tokio::test]
    async fn test_sqlite_trace_consumer() {
        let (pool, _dir) = setup_test_db().await;
        let consumer = SqliteTraceConsumer::new(
            Arc::new(pool.clone()),
            "test-session".to_string(),
            "test-agent".to_string(),
            "test-model".to_string(),
            "test".to_string(),
            1,
        );

        // Write some events
        consumer
            .on_event(&StreamEvent::SessionStarted {
                session_id: "test-session".to_string(),
            })
            .unwrap();

        consumer
            .on_event(&StreamEvent::Content {
                content: "Hello".to_string(),
            })
            .unwrap();

        consumer.on_event(&StreamEvent::Done).unwrap();

        // Wait for async tasks to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify events were written
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM agent_traces")
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(count, 3);

        // Verify sequence order
        let rows = sqlx::query(
            "SELECT event_type, sequence_number FROM agent_traces ORDER BY sequence_number",
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get::<String, _>("event_type"), "session_started");
        assert_eq!(rows[0].get::<i64, _>("sequence_number"), 0);
        assert_eq!(rows[1].get::<String, _>("event_type"), "content");
        assert_eq!(rows[1].get::<i64, _>("sequence_number"), 1);
        assert_eq!(rows[2].get::<String, _>("event_type"), "done");
        assert_eq!(rows[2].get::<i64, _>("sequence_number"), 2);
    }

    #[tokio::test]
    async fn test_external_sequence_reservation_prevents_collisions() {
        let (pool, _dir) = setup_test_db().await;
        let consumer = SqliteTraceConsumer::new(
            Arc::new(pool.clone()),
            "test-session".to_string(),
            "test-agent".to_string(),
            "test-model".to_string(),
            "test".to_string(),
            1,
        );

        let reserved_sequence = consumer.reserve_sequence_number();
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            INSERT INTO agent_traces (
                session_id, agent_name, model, job_type, job_version,
                sequence_number, timestamp, elapsed_ms, event_type,
                event_data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind("test-session")
        .bind("test-agent")
        .bind("test-model")
        .bind("test")
        .bind(1_i64)
        .bind(reserved_sequence)
        .bind(&now)
        .bind(0.0_f64)
        .bind("system_prompt")
        .bind(r#"{"content":"bootstrap"}"#)
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();

        consumer
            .on_event(&StreamEvent::SessionStarted {
                session_id: "test-session".to_string(),
            })
            .unwrap();
        consumer.on_event(&StreamEvent::Done).unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let rows = sqlx::query(
            "SELECT event_type, sequence_number FROM agent_traces ORDER BY sequence_number",
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get::<String, _>("event_type"), "system_prompt");
        assert_eq!(rows[0].get::<i64, _>("sequence_number"), 0);
        assert_eq!(rows[1].get::<String, _>("event_type"), "session_started");
        assert_eq!(rows[1].get::<i64, _>("sequence_number"), 1);
        assert_eq!(rows[2].get::<String, _>("event_type"), "done");
        assert_eq!(rows[2].get::<i64, _>("sequence_number"), 2);
    }

    #[tokio::test]
    async fn test_content_consolidation() {
        let (pool, _dir) = setup_test_db().await;
        let consumer = SqliteTraceConsumer::new(
            Arc::new(pool.clone()),
            "test-session".to_string(),
            "test-agent".to_string(),
            "test-model".to_string(),
            "test".to_string(),
            1,
        );

        // Send multiple consecutive content events
        consumer
            .on_event(&StreamEvent::Content {
                content: "Hello ".to_string(),
            })
            .unwrap();

        consumer
            .on_event(&StreamEvent::Content {
                content: "world".to_string(),
            })
            .unwrap();

        consumer
            .on_event(&StreamEvent::Content {
                content: "!".to_string(),
            })
            .unwrap();

        // Send a different event type to trigger flush
        consumer.on_event(&StreamEvent::Done).unwrap();

        // Wait for async tasks to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify only 2 events written (1 consolidated content + 1 done)
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM agent_traces")
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(count, 2, "Expected 2 events (consolidated content + done)");

        // Verify the content was consolidated
        let content_row = sqlx::query(
            "SELECT event_type, event_data FROM agent_traces WHERE event_type = 'content'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        let event_data: String = content_row.get("event_data");
        let data: serde_json::Value = serde_json::from_str(&event_data).unwrap();

        assert_eq!(
            data["content"].as_str().unwrap(),
            "Hello world!",
            "Content should be concatenated"
        );
    }

    #[tokio::test]
    async fn test_reasoning_consolidation() {
        let (pool, _dir) = setup_test_db().await;
        let consumer = SqliteTraceConsumer::new(
            Arc::new(pool.clone()),
            "test-session".to_string(),
            "test-agent".to_string(),
            "test-model".to_string(),
            "test".to_string(),
            1,
        );

        // Send multiple consecutive reasoning events
        consumer
            .on_event(&StreamEvent::Reasoning {
                content: "Thinking... ".to_string(),
            })
            .unwrap();

        consumer
            .on_event(&StreamEvent::Reasoning {
                content: "about this".to_string(),
            })
            .unwrap();

        // Send a different event type to trigger flush
        consumer
            .on_event(&StreamEvent::ToolCallStarted {
                tool_name: "test_tool".to_string(),
                arguments: "{}".to_string(),
            })
            .unwrap();

        // Wait for async tasks to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify 2 events written (1 consolidated reasoning + 1 tool_call_started)
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM agent_traces")
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(
            count, 2,
            "Expected 2 events (consolidated reasoning + tool call)"
        );

        // Verify the reasoning was consolidated
        let reasoning_row = sqlx::query(
            "SELECT event_type, event_data FROM agent_traces WHERE event_type = 'reasoning'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        let event_data: String = reasoning_row.get("event_data");
        let data: serde_json::Value = serde_json::from_str(&event_data).unwrap();

        assert_eq!(
            data["content"].as_str().unwrap(),
            "Thinking... about this",
            "Reasoning should be concatenated"
        );
    }

    #[tokio::test]
    async fn test_mixed_event_types() {
        let (pool, _dir) = setup_test_db().await;
        let consumer = SqliteTraceConsumer::new(
            Arc::new(pool.clone()),
            "test-session".to_string(),
            "test-agent".to_string(),
            "test-model".to_string(),
            "test".to_string(),
            1,
        );

        // Content -> Reasoning -> Content -> Done
        // Should create 4 separate entries (content not consecutive at the end)
        consumer
            .on_event(&StreamEvent::Content {
                content: "First ".to_string(),
            })
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        consumer
            .on_event(&StreamEvent::Content {
                content: "content".to_string(),
            })
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        consumer
            .on_event(&StreamEvent::Reasoning {
                content: "Thinking".to_string(),
            })
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        consumer
            .on_event(&StreamEvent::Content {
                content: "Second content".to_string(),
            })
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        consumer.on_event(&StreamEvent::Done).unwrap();

        // Wait for async tasks to complete (longer delay for multiple operations)
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify 4 events written (content + reasoning + content + done)
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM agent_traces")
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(
            count, 4,
            "Expected 4 events (two content blocks, reasoning, done)"
        );

        // Verify the sequence and content
        let rows = sqlx::query(
            "SELECT event_type, event_data, sequence_number FROM agent_traces ORDER BY sequence_number",
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        // First consolidated content
        assert_eq!(rows[0].get::<String, _>("event_type"), "content");
        let data0: serde_json::Value =
            serde_json::from_str(&rows[0].get::<String, _>("event_data")).unwrap();
        assert_eq!(data0["content"].as_str().unwrap(), "First content");

        // Reasoning
        assert_eq!(rows[1].get::<String, _>("event_type"), "reasoning");

        // Second content
        assert_eq!(rows[2].get::<String, _>("event_type"), "content");
        let data2: serde_json::Value =
            serde_json::from_str(&rows[2].get::<String, _>("event_data")).unwrap();
        assert_eq!(data2["content"].as_str().unwrap(), "Second content");

        // Done
        assert_eq!(rows[3].get::<String, _>("event_type"), "done");
    }

    #[tokio::test]
    async fn test_usage_update_persists_model_and_usage_fields() {
        let (pool, _dir) = setup_test_db().await;
        let consumer = SqliteTraceConsumer::new(
            Arc::new(pool.clone()),
            "usage-session".to_string(),
            "threat-hunter-agent".to_string(),
            "test-model-1".to_string(),
            "threat_hunter".to_string(),
            42,
        );

        consumer
            .on_event(&StreamEvent::UsageUpdate {
                snapshot: crate::llm::usage::AggregatedUsage {
                    total_input_tokens: 10,
                    total_output_tokens: 20,
                    total_cache_creation_tokens: 3,
                    total_cache_read_tokens: 4,
                    total_reasoning_tokens: 5,
                    total_cost_usd: 0.75,
                    request_count: 2,
                },
            })
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let row = sqlx::query(
            "SELECT model, event_data FROM agent_traces WHERE event_type = 'usage_update' LIMIT 1",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(row.get::<String, _>("model"), "test-model-1");

        let event_data: serde_json::Value =
            serde_json::from_str(&row.get::<String, _>("event_data")).unwrap();
        assert_eq!(event_data["total_tokens"].as_u64().unwrap(), 30);
        assert_eq!(event_data["total_cost_usd"].as_f64().unwrap(), 0.75);
        assert_eq!(event_data["request_count"].as_u64().unwrap(), 2);
        assert_eq!(event_data["total_input_tokens"].as_u64().unwrap(), 10);
        assert_eq!(event_data["total_output_tokens"].as_u64().unwrap(), 20);
    }

    #[tokio::test]
    async fn test_usage_update_preserves_sequence_numbering() {
        let (pool, _dir) = setup_test_db().await;
        let consumer = SqliteTraceConsumer::new(
            Arc::new(pool.clone()),
            "usage-sequence-session".to_string(),
            "hunter-agent".to_string(),
            "test-model-2".to_string(),
            "hunter".to_string(),
            7,
        );

        consumer
            .on_event(&StreamEvent::SessionStarted {
                session_id: "usage-sequence-session".to_string(),
            })
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        consumer
            .on_event(&StreamEvent::UsageUpdate {
                snapshot: crate::llm::usage::AggregatedUsage {
                    total_input_tokens: 111,
                    total_output_tokens: 222,
                    total_cache_creation_tokens: 0,
                    total_cache_read_tokens: 0,
                    total_reasoning_tokens: 0,
                    total_cost_usd: 1.23,
                    request_count: 1,
                },
            })
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        consumer.on_event(&StreamEvent::Done).unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let rows = sqlx::query(
            "SELECT event_type, sequence_number FROM agent_traces ORDER BY sequence_number",
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get::<String, _>("event_type"), "session_started");
        assert_eq!(rows[0].get::<i64, _>("sequence_number"), 0);
        assert_eq!(rows[1].get::<String, _>("event_type"), "usage_update");
        assert_eq!(rows[1].get::<i64, _>("sequence_number"), 1);
        assert_eq!(rows[2].get::<String, _>("event_type"), "done");
        assert_eq!(rows[2].get::<i64, _>("sequence_number"), 2);
    }
}
