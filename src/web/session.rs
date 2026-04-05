//! Session management for conversation history with SQLite persistence.
//!
//! Provides persistent storage for active chat sessions. Sessions track full
//! conversation history and survive server restarts.

use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use tracing::{debug, info};
use uuid::Uuid;

use crate::llm::ChatMessage;

/// A chat session with conversation history.
///
/// Sessions maintain the full message history for a conversation with an agent,
/// allowing conversations to be continued across multiple API requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier
    pub id: String,
    /// Agent name for this session
    pub agent_name: String,
    /// Model used
    pub model: String,
    /// Full conversation history
    pub messages: Vec<ChatMessage>,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Last activity time
    pub updated_at: DateTime<Utc>,
}

impl Session {
    /// Create a new session for an agent.
    pub fn new(agent_name: String, model: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            agent_name,
            model,
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a message to the session history.
    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
        self.updated_at = Utc::now();
    }

    /// Add multiple messages to the session history.
    pub fn add_messages(&mut self, messages: Vec<ChatMessage>) {
        self.messages.extend(messages);
        self.updated_at = Utc::now();
    }

    /// Get session summary.
    pub fn summary(&self) -> SessionSummary {
        SessionSummary {
            id: self.id.clone(),
            agent_name: self.agent_name.clone(),
            model: self.model.clone(),
            message_count: self.messages.len(),
            created_at: self.created_at,
            updated_at: self.updated_at,
        }
    }
}

/// Compact session summary for list endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// The unique session identifier.
    pub id: String,
    /// The name of the agent used in this session.
    pub agent_name: String,
    /// The model used by the agent.
    pub model: String,
    /// The number of messages in this session.
    pub message_count: usize,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
    /// When the session was last updated.
    pub updated_at: DateTime<Utc>,
}

/// Thread-safe session store with SQLite persistence.
///
/// Manages active sessions with database storage for durability across
/// server restarts.
#[derive(Clone)]
pub struct SessionStore {
    pool: SqlitePool,
}

impl SessionStore {
    /// Create a new session store with SQLite backend.
    ///
    /// Initializes the database schema if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created or initialized.
    pub async fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let db_path = db_path.as_ref();

        // Create parent directory if it doesn't exist
        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        info!(path = %db_path.display(), "Initializing session database");

        // Connect to database
        let options = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .context("Failed to connect to session database")?;

        // Initialize schema
        Self::init_schema(&pool).await?;

        info!("Session database initialized");

        Ok(Self { pool })
    }

    /// Initialize database schema.
    async fn init_schema(pool: &SqlitePool) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                model TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await?;

        // Create indexes for common queries
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC)",
        )
        .execute(pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_sessions_agent_name ON sessions(agent_name)")
            .execute(pool)
            .await?;

        Ok(())
    }

    /// Create a new session for an agent.
    pub async fn create_session(&self, agent_name: String, model: String) -> Result<Session> {
        let session = Session::new(agent_name, model);

        debug!(session_id = %session.id, "Creating new session");

        let messages_json = serde_json::to_string(&session.messages)?;

        sqlx::query(
            r#"
            INSERT INTO sessions (id, agent_name, model, messages, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&session.id)
        .bind(&session.agent_name)
        .bind(&session.model)
        .bind(&messages_json)
        .bind(session.created_at.to_rfc3339())
        .bind(session.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(session)
    }

    /// Get a session by ID.
    pub async fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        debug!(session_id = %session_id, "Fetching session");

        let row = sqlx::query(
            r#"
            SELECT id, agent_name, model, messages, created_at, updated_at
            FROM sessions
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

                let created_at_str: String = row.try_get("created_at")?;
                let updated_at_str: String = row.try_get("updated_at")?;

                Ok(Some(Session {
                    id: row.try_get("id")?,
                    agent_name: row.try_get("agent_name")?,
                    model: row.try_get("model")?,
                    messages,
                    created_at: DateTime::parse_from_rfc3339(&created_at_str)?.into(),
                    updated_at: DateTime::parse_from_rfc3339(&updated_at_str)?.into(),
                }))
            }
            None => Ok(None),
        }
    }

    /// Update a session.
    pub async fn update_session(&self, session: &Session) -> Result<()> {
        debug!(session_id = %session.id, "Updating session");

        let messages_json = serde_json::to_string(&session.messages)?;

        sqlx::query(
            r#"
            UPDATE sessions
            SET messages = ?, updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(&messages_json)
        .bind(session.updated_at.to_rfc3339())
        .bind(&session.id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Delete a session by ID.
    pub async fn delete_session(&self, session_id: &str) -> Result<bool> {
        debug!(session_id = %session_id, "Deleting session");

        let result = sqlx::query("DELETE FROM sessions WHERE id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }

    /// List all sessions.
    pub async fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        debug!("Listing all sessions");

        let rows = sqlx::query(
            r#"
            SELECT id, agent_name, model, messages, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut summaries = Vec::new();
        for row in rows {
            let messages_json: String = row.try_get("messages")?;
            let messages: Vec<ChatMessage> = serde_json::from_str(&messages_json)?;

            let created_at_str: String = row.try_get("created_at")?;
            let updated_at_str: String = row.try_get("updated_at")?;

            summaries.push(SessionSummary {
                id: row.try_get("id")?,
                agent_name: row.try_get("agent_name")?,
                model: row.try_get("model")?,
                message_count: messages.len(),
                created_at: DateTime::parse_from_rfc3339(&created_at_str)?.into(),
                updated_at: DateTime::parse_from_rfc3339(&updated_at_str)?.into(),
            });
        }

        Ok(summaries)
    }

    /// List sessions for a specific agent.
    pub async fn list_agent_sessions(&self, agent_name: &str) -> Result<Vec<SessionSummary>> {
        debug!(agent_name = %agent_name, "Listing sessions for agent");

        let rows = sqlx::query(
            r#"
            SELECT id, agent_name, model, messages, created_at, updated_at
            FROM sessions
            WHERE agent_name = ?
            ORDER BY updated_at DESC
            "#,
        )
        .bind(agent_name)
        .fetch_all(&self.pool)
        .await?;

        let mut summaries = Vec::new();
        for row in rows {
            let messages_json: String = row.try_get("messages")?;
            let messages: Vec<ChatMessage> = serde_json::from_str(&messages_json)?;

            let created_at_str: String = row.try_get("created_at")?;
            let updated_at_str: String = row.try_get("updated_at")?;

            summaries.push(SessionSummary {
                id: row.try_get("id")?,
                agent_name: row.try_get("agent_name")?,
                model: row.try_get("model")?,
                message_count: messages.len(),
                created_at: DateTime::parse_from_rfc3339(&created_at_str)?.into(),
                updated_at: DateTime::parse_from_rfc3339(&updated_at_str)?.into(),
            });
        }

        Ok(summaries)
    }

    /// Get the number of active sessions.
    pub async fn session_count(&self) -> Result<usize> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM sessions")
            .fetch_one(&self.pool)
            .await?;

        let count: i64 = row.try_get("count")?;
        Ok(count as usize)
    }

    /// Clear all sessions.
    pub async fn clear(&self) -> Result<()> {
        debug!("Clearing all sessions");
        sqlx::query("DELETE FROM sessions")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Cleanup old sessions (older than the specified duration).
    pub async fn cleanup_old_sessions(&self, max_age: chrono::Duration) -> Result<usize> {
        let cutoff = Utc::now() - max_age;
        debug!(cutoff = %cutoff, "Cleaning up old sessions");

        let result = sqlx::query("DELETE FROM sessions WHERE updated_at < ?")
            .bind(cutoff.to_rfc3339())
            .execute(&self.pool)
            .await?;

        let deleted = result.rows_affected() as usize;
        info!(deleted = deleted, "Cleaned up old sessions");

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_session_creation() {
        let session = Session::new("test_agent".to_string(), "gpt-5".to_string());
        assert!(!session.id.is_empty());
        assert_eq!(session.agent_name, "test_agent");
        assert_eq!(session.messages.len(), 0);
    }

    #[tokio::test]
    async fn test_session_store() {
        let temp_file = NamedTempFile::new().unwrap();
        let store = SessionStore::new(temp_file.path()).await.unwrap();

        let session = store
            .create_session("test_agent".to_string(), "gpt-5".to_string())
            .await
            .unwrap();

        assert_eq!(store.session_count().await.unwrap(), 1);

        let retrieved = store.get_session(&session.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, session.id);

        store.delete_session(&session.id).await.unwrap();
        assert_eq!(store.session_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_session_messages() {
        let mut session = Session::new("test".to_string(), "gpt-5".to_string());

        session.add_message(ChatMessage {
            role: Role::User,
            name: None,
            tool_call_id: None,
            content: Some("Hello".to_string()),
            tool_calls: None,
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: None,
            timestamp: Some(chrono::Utc::now()),
            id: None,
            provider_response_id: None,
            status: None,
        });

        assert_eq!(session.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_session_persistence() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_owned();

        // Create session in first store
        {
            let store = SessionStore::new(&db_path).await.unwrap();
            let mut session = store
                .create_session("test_agent".to_string(), "gpt-5".to_string())
                .await
                .unwrap();

            session.add_message(ChatMessage {
                role: Role::User,
                name: None,
                tool_call_id: None,
                content: Some("Test message".to_string()),
                tool_calls: None,
                reasoning: None,
                raw_content_blocks: None,
                tool_metadata: None,
                timestamp: Some(chrono::Utc::now()),
                id: None,
                provider_response_id: None,
                status: None,
            });

            store.update_session(&session).await.unwrap();
        }

        // Reopen database and verify session persisted
        {
            let store = SessionStore::new(&db_path).await.unwrap();
            let sessions = store.list_sessions().await.unwrap();
            assert_eq!(sessions.len(), 1);
            assert_eq!(sessions[0].message_count, 1);
        }
    }
}
