//! Session history management with SQLite persistence.
//!
//! Provides session storage and retrieval for conversation continuation across
//! agent runs. This module is separate from the web-specific session store and
//! designed for use in any context (CLI, SDK, web, etc.).

use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use tracing::{debug, info};

use crate::config::HistoryConfig;
use crate::llm::ChatMessage;

use super::Session;

/// Summary of a session for listing operations.
///
/// Provides essential metadata without loading full message history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Unique session identifier
    pub id: String,
    /// Agent name
    pub agent_name: String,
    /// Model used
    pub model: String,
    /// Number of messages in session
    pub message_count: usize,
    /// Number of turns (user-assistant exchanges)
    pub turn_count: i32,
    /// Number of tool calls made
    pub tool_call_count: i32,
    /// Session creation time
    pub started_at: DateTime<Utc>,
    /// Last activity time
    pub updated_at: DateTime<Utc>,
    /// Session end time (if completed)
    pub ended_at: Option<DateTime<Utc>>,
}

/// Session history manager with SQLite persistence.
///
/// Manages agent session storage and retrieval for conversation continuation.
/// When history is disabled, operations are no-ops and return empty results.
///
/// # Examples
///
/// ```no_run
/// use appam::agent::history::SessionHistory;
/// use appam::config::HistoryConfig;
///
/// # async fn example() -> anyhow::Result<()> {
/// let mut config = HistoryConfig::default();
/// config.enabled = true;
/// config.db_path = "data/sessions.db".into();
///
/// let history = SessionHistory::new(config).await?;
///
/// // List all sessions
/// let sessions = history.list_sessions().await?;
/// for summary in sessions {
///     println!("Session {}: {} messages", summary.id, summary.message_count);
/// }
/// # Ok(())
/// # }
/// ```
pub struct SessionHistory {
    store: Option<SessionStore>,
    config: HistoryConfig,
}

impl SessionHistory {
    /// Create a new session history manager.
    ///
    /// If history is disabled in config, the store is initialized as `None`
    /// and all operations become no-ops.
    ///
    /// # Errors
    ///
    /// Returns an error if history is enabled but the database cannot be
    /// initialized.
    pub async fn new(config: HistoryConfig) -> Result<Self> {
        let store = if config.enabled {
            Some(SessionStore::new(&config.db_path).await?)
        } else {
            None
        };

        Ok(Self { store, config })
    }

    /// Check if history is enabled.
    pub fn is_enabled(&self) -> bool {
        self.store.is_some()
    }

    /// Create a new session entry.
    ///
    /// If history is disabled, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be written to the database.
    pub async fn create_session(&self, session: &Session) -> Result<()> {
        if let Some(store) = &self.store {
            store.create_session(session).await?;
            debug!(session_id = %session.session_id, "Session created in history");
        }
        Ok(())
    }

    /// Load a session by ID.
    ///
    /// Returns `None` if history is disabled or the session doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn load_session(&self, session_id: &str) -> Result<Option<Session>> {
        if let Some(store) = &self.store {
            store.get_session(session_id).await
        } else {
            Ok(None)
        }
    }

    /// Save or update a session.
    ///
    /// If history is disabled, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be written to the database.
    pub async fn save_session(&self, session: &Session) -> Result<()> {
        if let Some(store) = &self.store {
            store.upsert_session(session).await?;
            debug!(session_id = %session.session_id, "Session saved to history");
        }
        Ok(())
    }

    /// Append messages to an existing session.
    ///
    /// This is more efficient than loading, modifying, and saving the entire
    /// session when you only need to add messages.
    ///
    /// # Errors
    ///
    /// Returns an error if the session doesn't exist or cannot be updated.
    pub async fn append_messages(&self, session_id: &str, messages: &[ChatMessage]) -> Result<()> {
        if let Some(store) = &self.store {
            store.append_messages(session_id, messages).await?;
        }
        Ok(())
    }

    /// List all sessions.
    ///
    /// Returns session summaries ordered by most recent activity.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        if let Some(store) = &self.store {
            store.list_sessions().await
        } else {
            Ok(Vec::new())
        }
    }

    /// List sessions for a specific agent.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn list_agent_sessions(&self, agent_name: &str) -> Result<Vec<SessionSummary>> {
        if let Some(store) = &self.store {
            store.list_agent_sessions(agent_name).await
        } else {
            Ok(Vec::new())
        }
    }

    /// Delete a session by ID.
    ///
    /// Returns `true` if a session was deleted, `false` if it didn't exist or
    /// history is disabled.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn delete_session(&self, session_id: &str) -> Result<bool> {
        if let Some(store) = &self.store {
            store.delete_session(session_id).await
        } else {
            Ok(false)
        }
    }

    /// Get the number of stored sessions.
    pub async fn session_count(&self) -> Result<usize> {
        if let Some(store) = &self.store {
            store.session_count().await
        } else {
            Ok(0)
        }
    }

    /// Clean up old sessions beyond the configured maximum.
    ///
    /// If `max_sessions` is configured, deletes oldest sessions to stay within
    /// the limit. Returns the number of sessions deleted.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn enforce_max_sessions(&self) -> Result<usize> {
        if let Some(store) = &self.store {
            if let Some(max) = self.config.max_sessions {
                return store.enforce_max_sessions(max).await;
            }
        }
        Ok(0)
    }
}

/// Internal SQLite session store.
///
/// This is similar to `src/web/session.rs` but enhanced for general use with
/// additional metadata fields.
struct SessionStore {
    pool: SqlitePool,
}

impl SessionStore {
    /// Initialize a new session store with database at the given path.
    async fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let db_path = db_path.as_ref();

        // Create parent directory if needed
        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        info!(path = %db_path.display(), "Initializing session history database");

        // Connect to database
        let options = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .context("Failed to connect to session history database")?;

        // Initialize schema
        Self::init_schema(&pool).await?;

        info!("Session history database initialized");

        Ok(Self { pool })
    }

    /// Initialize database schema with enhanced metadata.
    async fn init_schema(pool: &SqlitePool) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS session_history (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                model TEXT NOT NULL,
                messages TEXT NOT NULL,
                metadata TEXT,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ended_at TEXT,
                turn_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0
            )
            "#,
        )
        .execute(pool)
        .await?;

        // Create indexes for common queries
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_history_updated ON session_history(updated_at DESC)",
        )
        .execute(pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_history_agent ON session_history(agent_name)")
            .execute(pool)
            .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_history_started ON session_history(started_at DESC)",
        )
        .execute(pool)
        .await?;

        // Migration: Add usage column for token/cost tracking
        let _ = sqlx::query(
            r#"
            ALTER TABLE session_history
            ADD COLUMN usage TEXT
            "#,
        )
        .execute(pool)
        .await;
        // Ignore error if column already exists

        Ok(())
    }

    /// Create a new session in the database.
    async fn create_session(&self, session: &Session) -> Result<()> {
        let messages_json = serde_json::to_string(&session.messages)?;
        let usage_json = session
            .usage
            .as_ref()
            .and_then(|u| serde_json::to_string(u).ok());
        let (turn_count, tool_call_count) = Self::compute_counts(&session.messages);

        sqlx::query(
            r#"
            INSERT INTO session_history (
                id, agent_name, model, messages, metadata,
                started_at, updated_at, ended_at, turn_count, tool_call_count, usage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&session.session_id)
        .bind(&session.agent_name)
        .bind(&session.model)
        .bind(&messages_json)
        .bind(Option::<String>::None) // metadata placeholder
        .bind(session.started_at.map(|t| t.to_rfc3339()))
        .bind(
            session
                .ended_at
                .map(|t| t.to_rfc3339())
                .unwrap_or_else(|| Utc::now().to_rfc3339()),
        )
        .bind(session.ended_at.map(|t| t.to_rfc3339()))
        .bind(turn_count)
        .bind(tool_call_count)
        .bind(usage_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update an existing session or insert if it doesn't exist.
    async fn upsert_session(&self, session: &Session) -> Result<()> {
        let messages_json = serde_json::to_string(&session.messages)?;
        let usage_json = session
            .usage
            .as_ref()
            .and_then(|u| serde_json::to_string(u).ok());
        let (turn_count, tool_call_count) = Self::compute_counts(&session.messages);

        // Try update first
        let result = sqlx::query(
            r#"
            UPDATE session_history
            SET messages = ?, updated_at = ?, ended_at = ?, turn_count = ?, tool_call_count = ?, usage = ?
            WHERE id = ?
            "#,
        )
        .bind(&messages_json)
        .bind(
            session
                .ended_at
                .map(|t| t.to_rfc3339())
                .unwrap_or_else(|| Utc::now().to_rfc3339()),
        )
        .bind(session.ended_at.map(|t| t.to_rfc3339()))
        .bind(turn_count)
        .bind(tool_call_count)
        .bind(usage_json)
        .bind(&session.session_id)
        .execute(&self.pool)
        .await?;

        // If no rows affected, insert
        if result.rows_affected() == 0 {
            self.create_session(session).await?;
        }

        Ok(())
    }

    /// Get a session by ID.
    async fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        let row = sqlx::query(
            r#"
            SELECT id, agent_name, model, messages, started_at, updated_at, ended_at, usage
            FROM session_history
            WHERE id = ?
            "#,
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => {
                let messages_json: String = row.try_get("messages")?;
                let messages: Vec<ChatMessage> = serde_json::from_str(&messages_json)?;

                let started_at_str: Option<String> = row.try_get("started_at")?;
                let ended_at_str: Option<String> = row.try_get("ended_at")?;

                // Deserialize usage if present
                let usage_json: Option<String> = row.try_get("usage").ok().flatten();
                let usage = usage_json.and_then(|json| serde_json::from_str(&json).ok());

                Ok(Some(Session {
                    session_id: row.try_get("id")?,
                    agent_name: row.try_get("agent_name")?,
                    model: row.try_get("model")?,
                    messages,
                    started_at: started_at_str
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.into()),
                    ended_at: ended_at_str
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.into()),
                    usage,
                }))
            }
            None => Ok(None),
        }
    }

    /// Append messages to an existing session.
    async fn append_messages(&self, session_id: &str, new_messages: &[ChatMessage]) -> Result<()> {
        // Load existing session
        let mut session = self
            .get_session(session_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        // Append new messages
        session.messages.extend_from_slice(new_messages);

        // Update in database
        self.upsert_session(&session).await
    }

    /// Delete a session by ID.
    async fn delete_session(&self, session_id: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM session_history WHERE id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    /// List all sessions.
    async fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let rows = sqlx::query(
            r#"
            SELECT id, agent_name, model, messages, started_at, updated_at, ended_at,
                   turn_count, tool_call_count
            FROM session_history
            ORDER BY updated_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Self::rows_to_summaries(rows)
    }

    /// List sessions for a specific agent.
    async fn list_agent_sessions(&self, agent_name: &str) -> Result<Vec<SessionSummary>> {
        let rows = sqlx::query(
            r#"
            SELECT id, agent_name, model, messages, started_at, updated_at, ended_at,
                   turn_count, tool_call_count
            FROM session_history
            WHERE agent_name = ?
            ORDER BY updated_at DESC
            "#,
        )
        .bind(agent_name)
        .fetch_all(&self.pool)
        .await?;

        Self::rows_to_summaries(rows)
    }

    /// Get session count.
    async fn session_count(&self) -> Result<usize> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM session_history")
            .fetch_one(&self.pool)
            .await?;

        let count: i64 = row.try_get("count")?;
        Ok(count as usize)
    }

    /// Enforce maximum session limit by deleting oldest sessions.
    async fn enforce_max_sessions(&self, max_sessions: usize) -> Result<usize> {
        // Get current count
        let count = self.session_count().await?;

        if count <= max_sessions {
            return Ok(0);
        }

        let to_delete = count - max_sessions;

        // Delete oldest sessions
        let result = sqlx::query(
            r#"
            DELETE FROM session_history
            WHERE id IN (
                SELECT id FROM session_history
                ORDER BY updated_at ASC
                LIMIT ?
            )
            "#,
        )
        .bind(to_delete as i64)
        .execute(&self.pool)
        .await?;

        let deleted = result.rows_affected() as usize;
        info!(
            deleted = deleted,
            max = max_sessions,
            "Enforced session limit"
        );

        Ok(deleted)
    }

    /// Convert database rows to session summaries.
    fn rows_to_summaries(rows: Vec<sqlx::sqlite::SqliteRow>) -> Result<Vec<SessionSummary>> {
        let mut summaries = Vec::new();

        for row in rows {
            let messages_json: String = row.try_get("messages")?;
            let messages: Vec<ChatMessage> = serde_json::from_str(&messages_json)?;

            let started_at_str: Option<String> = row.try_get("started_at")?;
            let updated_at_str: String = row.try_get("updated_at")?;
            let ended_at_str: Option<String> = row.try_get("ended_at")?;

            summaries.push(SessionSummary {
                id: row.try_get("id")?,
                agent_name: row.try_get("agent_name")?,
                model: row.try_get("model")?,
                message_count: messages.len(),
                turn_count: row.try_get("turn_count")?,
                tool_call_count: row.try_get("tool_call_count")?,
                started_at: started_at_str
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.into())
                    .unwrap_or_else(Utc::now),
                updated_at: DateTime::parse_from_rfc3339(&updated_at_str)?.into(),
                ended_at: ended_at_str
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.into()),
            });
        }

        Ok(summaries)
    }

    /// Compute turn and tool call counts from messages.
    fn compute_counts(messages: &[ChatMessage]) -> (i32, i32) {
        let mut turn_count = 0;
        let mut tool_call_count = 0;

        for msg in messages {
            if msg.role == crate::llm::Role::Assistant && msg.content.is_some() {
                turn_count += 1;
            }
            if let Some(tool_calls) = &msg.tool_calls {
                tool_call_count += tool_calls.len() as i32;
            }
        }

        (turn_count, tool_call_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_history_disabled() {
        let config = HistoryConfig::default(); // disabled by default
        let history = SessionHistory::new(config).await.unwrap();

        assert!(!history.is_enabled());
        assert_eq!(history.session_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_history_enabled() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = HistoryConfig {
            enabled: true,
            db_path: temp_file.path().to_owned(),
            ..HistoryConfig::default()
        };

        let history = SessionHistory::new(config).await.unwrap();
        assert!(history.is_enabled());
    }

    #[tokio::test]
    async fn test_create_and_load_session() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = HistoryConfig {
            enabled: true,
            db_path: temp_file.path().to_owned(),
            ..HistoryConfig::default()
        };

        let history = SessionHistory::new(config).await.unwrap();

        let session = Session {
            session_id: "test-123".to_string(),
            agent_name: "test-agent".to_string(),
            model: "gpt-5".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                name: None,
                tool_call_id: None,
                content: Some("Hello".to_string()),
                tool_calls: None,
                reasoning: None,
                raw_content_blocks: None,
                tool_metadata: None,
                timestamp: Some(Utc::now()),
                id: None,
                provider_response_id: None,
                status: None,
            }],
            started_at: Some(Utc::now()),
            ended_at: None,
            usage: None,
        };

        history.create_session(&session).await.unwrap();

        let loaded = history.load_session("test-123").await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().session_id, "test-123");
    }

    #[tokio::test]
    async fn test_list_sessions() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = HistoryConfig {
            enabled: true,
            db_path: temp_file.path().to_owned(),
            ..HistoryConfig::default()
        };

        let history = SessionHistory::new(config).await.unwrap();

        // Create multiple sessions
        for i in 0..3 {
            let session = Session {
                session_id: format!("test-{}", i),
                agent_name: "test-agent".to_string(),
                model: "gpt-5".to_string(),
                messages: vec![],
                started_at: Some(Utc::now()),
                ended_at: None,
                usage: None,
            };
            history.create_session(&session).await.unwrap();
        }

        let sessions = history.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 3);
    }
}
