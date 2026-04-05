//! Integration tests for streaming functionality.

use appam::prelude::*;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

#[allow(dead_code)]
struct TestTool;

impl Tool for TestTool {
    fn name(&self) -> &str {
        "test"
    }

    fn spec(&self) -> anyhow::Result<ToolSpec> {
        Ok(serde_json::from_value(json!({
            "type": "function",
            "function": {
                "name": "test",
                "description": "Test tool",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }))?)
    }

    fn execute(&self, _args: Value) -> anyhow::Result<Value> {
        Ok(json!({"output": "test result"}))
    }
}

#[tokio::test]
async fn test_channel_consumer() {
    let _agent = AgentBuilder::new("test")
        .system_prompt("Test prompt")
        .build()
        .unwrap();

    let (tx, mut rx) = mpsc::unbounded_channel();
    let consumer = ChannelConsumer::new(tx);

    // Emit test events
    consumer
        .on_event(&StreamEvent::Content {
            content: "test".to_string(),
        })
        .unwrap();

    consumer.on_event(&StreamEvent::Done).unwrap();

    // Verify events received
    let event1 = rx.recv().await.unwrap();
    assert!(matches!(event1, StreamEvent::Content { .. }));

    let event2 = rx.recv().await.unwrap();
    assert!(matches!(event2, StreamEvent::Done));
}

#[tokio::test]
async fn test_callback_consumer() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let consumer = CallbackConsumer::new(move |event| {
        if let StreamEvent::Content { .. } = event {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    });

    consumer
        .on_event(&StreamEvent::Content {
            content: "test".to_string(),
        })
        .unwrap();

    assert_eq!(counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_console_consumer() {
    let consumer = ConsoleConsumer::new()
        .with_colors(false)
        .with_reasoning(true)
        .with_tool_details(true);

    // Should not panic
    let result = consumer.on_event(&StreamEvent::Content {
        content: "test".to_string(),
    });

    assert!(result.is_ok());
}

#[test]
fn test_multi_consumer() {
    let counter1 = Arc::new(AtomicUsize::new(0));
    let counter2 = Arc::new(AtomicUsize::new(0));

    let c1 = counter1.clone();
    let c2 = counter2.clone();

    let consumer1 = CallbackConsumer::new(move |_| {
        c1.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });

    let consumer2 = CallbackConsumer::new(move |_| {
        c2.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });

    let multi = appam::agent::streaming::MultiConsumer::new()
        .add(Box::new(consumer1))
        .add(Box::new(consumer2));

    multi
        .on_event(&StreamEvent::Content {
            content: "test".to_string(),
        })
        .unwrap();

    assert_eq!(counter1.load(Ordering::Relaxed), 1);
    assert_eq!(counter2.load(Ordering::Relaxed), 1);
}

#[test]
fn test_stream_event_serialization() {
    let events = vec![
        StreamEvent::SessionStarted {
            session_id: "123".to_string(),
        },
        StreamEvent::Content {
            content: "Hello".to_string(),
        },
        StreamEvent::Reasoning {
            content: "Thinking...".to_string(),
        },
        StreamEvent::ToolCallStarted {
            tool_name: "test".to_string(),
            arguments: "{}".to_string(),
        },
        StreamEvent::ToolCallCompleted {
            tool_name: "test".to_string(),
            result: json!({"output": "done"}),
            success: true,
            duration_ms: 100.0,
        },
        StreamEvent::Done,
    ];

    for event in events {
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.is_empty());

        let deserialized: StreamEvent = serde_json::from_str(&json).unwrap();
        // Verify round-trip
        let json2 = serde_json::to_string(&deserialized).unwrap();
        assert_eq!(json, json2);
    }
}
