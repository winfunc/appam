//! Integration tests for logging, tracing, and session history.

use appam::agent::consumers::TraceConsumer;
use appam::agent::history::SessionHistory;
use appam::agent::streaming::{StreamConsumer, StreamEvent};
use appam::config::{HistoryConfig, TraceFormat};
use appam::llm::{ChatMessage, Role};
use appam::Session;
use chrono::Utc;
use tempfile::TempDir;

#[test]
fn test_trace_consumer_basic() {
    let temp_dir = TempDir::new().unwrap();
    let trace = TraceConsumer::new(temp_dir.path(), "test-session", TraceFormat::Detailed).unwrap();

    // Write some events
    trace
        .on_event(&StreamEvent::SessionStarted {
            session_id: "test-session".to_string(),
        })
        .unwrap();

    trace
        .on_event(&StreamEvent::Content {
            content: "Hello world".to_string(),
        })
        .unwrap();

    trace.on_event(&StreamEvent::Done).unwrap();

    // Verify file was created
    let trace_file = temp_dir.path().join("session-test-session.jsonl");
    assert!(trace_file.exists());

    // Verify content
    let content = std::fs::read_to_string(&trace_file).unwrap();
    assert!(content.contains("session_started"));
    assert!(content.contains("content"));
    assert!(content.contains("Hello world"));
    assert!(content.contains("done"));

    // Each line should be valid JSON
    for line in content.lines() {
        let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(parsed["timestamp"].is_string());
        assert!(parsed["elapsed_ms"].is_number());
        assert!(parsed["type"].is_string());
        assert!(parsed["data"].is_object());
    }
}

#[test]
fn test_trace_consumer_compact_vs_detailed() {
    let temp_dir = TempDir::new().unwrap();

    // Compact format should exclude reasoning
    let compact = TraceConsumer::new(temp_dir.path(), "compact", TraceFormat::Compact).unwrap();
    compact
        .on_event(&StreamEvent::Reasoning {
            content: "thinking...".to_string(),
        })
        .unwrap();

    let compact_file = temp_dir.path().join("session-compact.jsonl");
    let compact_content = std::fs::read_to_string(compact_file).unwrap();
    assert!(compact_content.is_empty()); // No reasoning in compact mode

    // Detailed format should include reasoning
    let detailed = TraceConsumer::new(temp_dir.path(), "detailed", TraceFormat::Detailed).unwrap();
    detailed
        .on_event(&StreamEvent::Reasoning {
            content: "thinking...".to_string(),
        })
        .unwrap();

    let detailed_file = temp_dir.path().join("session-detailed.jsonl");
    let detailed_content = std::fs::read_to_string(detailed_file).unwrap();
    assert!(detailed_content.contains("reasoning"));
    assert!(detailed_content.contains("thinking..."));
}

#[tokio::test]
async fn test_session_history_disabled() {
    let config = HistoryConfig::default(); // disabled by default
    let history = SessionHistory::new(config).await.unwrap();

    assert!(!history.is_enabled());
    assert_eq!(history.session_count().await.unwrap(), 0);

    // Operations should be no-ops
    let session = create_test_session("test-id");
    history.create_session(&session).await.unwrap();
    assert_eq!(history.session_count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_session_history_create_and_load() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let config = HistoryConfig {
        enabled: true,
        db_path: db_path.clone(),
        ..HistoryConfig::default()
    };

    let history = SessionHistory::new(config).await.unwrap();
    assert!(history.is_enabled());

    // Create session
    let session = create_test_session("test-123");
    history.create_session(&session).await.unwrap();

    // Verify count
    assert_eq!(history.session_count().await.unwrap(), 1);

    // Load session
    let loaded = history.load_session("test-123").await.unwrap();
    assert!(loaded.is_some());

    let loaded_session = loaded.unwrap();
    assert_eq!(loaded_session.session_id, "test-123");
    assert_eq!(loaded_session.agent_name, "test-agent");
    assert_eq!(loaded_session.messages.len(), 2);
}

#[tokio::test]
async fn test_session_history_save_and_update() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let config = HistoryConfig {
        enabled: true,
        db_path,
        ..HistoryConfig::default()
    };

    let history = SessionHistory::new(config).await.unwrap();

    // Create initial session
    let mut session = create_test_session("test-456");
    history.create_session(&session).await.unwrap();

    // Add more messages and save
    session.messages.push(ChatMessage {
        role: Role::Assistant,
        name: None,
        tool_call_id: None,
        content: Some("Response".to_string()),
        tool_calls: None,
        reasoning: None,
        raw_content_blocks: None,
        tool_metadata: None,
        timestamp: None,
        id: None,
        provider_response_id: None,
        status: None,
    });

    history.save_session(&session).await.unwrap();

    // Load and verify
    let loaded = history.load_session("test-456").await.unwrap().unwrap();
    assert_eq!(loaded.messages.len(), 3);
}

#[tokio::test]
async fn test_session_history_list() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let config = HistoryConfig {
        enabled: true,
        db_path,
        ..HistoryConfig::default()
    };

    let history = SessionHistory::new(config).await.unwrap();

    // Create multiple sessions
    for i in 0..5 {
        let session = create_test_session(&format!("test-{}", i));
        history.create_session(&session).await.unwrap();
    }

    // List all sessions
    let sessions = history.list_sessions().await.unwrap();
    assert_eq!(sessions.len(), 5);

    // Verify summaries
    for summary in &sessions {
        assert!(summary.id.starts_with("test-"));
        assert_eq!(summary.agent_name, "test-agent");
        assert_eq!(summary.message_count, 2);
    }
}

#[tokio::test]
async fn test_session_history_delete() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let config = HistoryConfig {
        enabled: true,
        db_path,
        ..HistoryConfig::default()
    };

    let history = SessionHistory::new(config).await.unwrap();

    // Create session
    let session = create_test_session("test-delete");
    history.create_session(&session).await.unwrap();
    assert_eq!(history.session_count().await.unwrap(), 1);

    // Delete session
    let deleted = history.delete_session("test-delete").await.unwrap();
    assert!(deleted);
    assert_eq!(history.session_count().await.unwrap(), 0);

    // Try to delete again
    let deleted_again = history.delete_session("test-delete").await.unwrap();
    assert!(!deleted_again);
}

#[tokio::test]
async fn test_session_history_max_sessions() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let config = HistoryConfig {
        enabled: true,
        db_path,
        max_sessions: Some(3),
        ..HistoryConfig::default()
    };

    let history = SessionHistory::new(config).await.unwrap();

    // Create 5 sessions
    for i in 0..5 {
        let session = create_test_session(&format!("test-{}", i));
        history.create_session(&session).await.unwrap();
    }

    assert_eq!(history.session_count().await.unwrap(), 5);

    // Enforce limit
    let deleted = history.enforce_max_sessions().await.unwrap();
    assert_eq!(deleted, 2);
    assert_eq!(history.session_count().await.unwrap(), 3);
}

#[test]
fn test_log_format_config() {
    use appam::config::{LogFormat, LoggingConfig};

    let mut config = LoggingConfig::default();
    assert_eq!(config.log_format, LogFormat::Both);

    config.log_format = LogFormat::Plain;
    assert_eq!(config.log_format, LogFormat::Plain);

    config.log_format = LogFormat::Json;
    assert_eq!(config.log_format, LogFormat::Json);
}

// Helper function to create a test session
fn create_test_session(id: &str) -> Session {
    Session {
        session_id: id.to_string(),
        agent_name: "test-agent".to_string(),
        model: "gpt-5".to_string(),
        messages: vec![
            ChatMessage {
                role: Role::System,
                name: None,
                tool_call_id: None,
                content: Some("System prompt".to_string()),
                tool_calls: None,
                reasoning: None,
                raw_content_blocks: None,
                tool_metadata: None,
                timestamp: None,
                id: None,
                provider_response_id: None,
                status: None,
            },
            ChatMessage {
                role: Role::User,
                name: None,
                tool_call_id: None,
                content: Some("Hello".to_string()),
                tool_calls: None,
                reasoning: None,
                raw_content_blocks: None,
                tool_metadata: None,
                timestamp: None,
                id: None,
                provider_response_id: None,
                status: None,
            },
        ],
        started_at: Some(Utc::now()),
        ended_at: None,
        usage: None,
    }
}
