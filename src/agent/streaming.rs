//! Streaming infrastructure for real-time agent output.
//!
//! Provides event-based streaming with pluggable consumers, enabling agents to
//! stream output to console, web clients, custom handlers, or multiple destinations
//! simultaneously.
//!
//! # Architecture
//!
//! - `StreamEvent`: Structured events representing agent activity (content, tool calls, etc.)
//! - `StreamConsumer`: Trait for consuming events (console, SSE, custom, etc.)
//! - Built-in consumers: Console, Channel, Callback, Multi
//!
//! # Examples
//!
//! ```no_run
//! use appam::agent::streaming::{StreamEvent, StreamConsumer};
//!
//! struct MyConsumer;
//!
//! impl StreamConsumer for MyConsumer {
//!     fn on_event(&self, event: &StreamEvent) -> anyhow::Result<()> {
//!         match event {
//!             StreamEvent::Content { content } => println!("{}", content),
//!             _ => {}
//!         }
//!         Ok(())
//!     }
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::agent::errors::SessionFailureKind;

/// Events emitted during agent execution.
///
/// These events represent all observable agent activity: LLM output, reasoning,
/// tool execution, and completion states. Events are structured for easy
/// serialization (e.g., to JSON for SSE) and pattern matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Session started with unique identifier.
    SessionStarted {
        /// Unique session ID
        session_id: String,
    },

    /// Content chunk from the LLM.
    ///
    /// Emitted as the model generates text. Multiple content events form the
    /// complete response.
    Content {
        /// Text content chunk
        content: String,
    },

    /// Reasoning trace from extended reasoning models.
    ///
    /// Some models expose their "thinking" process. This event streams those
    /// reasoning tokens separately from the main content.
    Reasoning {
        /// Reasoning content
        content: String,
    },

    /// Tool call initiated by the model.
    ///
    /// Emitted when the model decides to invoke a tool with complete arguments.
    ToolCallStarted {
        /// Tool name
        tool_name: String,
        /// JSON-encoded arguments
        arguments: String,
    },

    /// Tool execution completed successfully.
    ///
    /// Contains the tool's result and execution metadata.
    ToolCallCompleted {
        /// Tool name
        tool_name: String,
        /// Tool result (JSON value)
        result: serde_json::Value,
        /// Whether execution succeeded
        success: bool,
        /// Execution duration in milliseconds
        duration_ms: f64,
    },

    /// Tool execution failed.
    ///
    /// Emitted when a tool raises an error during execution.
    ToolCallFailed {
        /// Tool name
        tool_name: String,
        /// Error message
        error: String,
    },

    /// Agent completed one turn of interaction.
    ///
    /// A "turn" is one complete cycle: user message → LLM response → tool calls
    /// (if any) → final response. Multi-turn conversations emit multiple events.
    TurnCompleted,

    /// Token usage update.
    ///
    /// Emitted after each LLM response with cumulative usage statistics for the session.
    /// Enables real-time cost tracking and progress display.
    UsageUpdate {
        /// Aggregated usage snapshot
        snapshot: crate::llm::usage::AggregatedUsage,
    },

    /// Stream completed successfully.
    ///
    /// Terminal event indicating the agent has finished processing and no more
    /// events will be emitted.
    Done,

    /// Error occurred during execution.
    ///
    /// Terminal event indicating the agent encountered an unrecoverable error.
    Error {
        /// Error message
        message: String,
        /// Optional structured runtime failure classification.
        #[serde(skip_serializing_if = "Option::is_none")]
        failure_kind: Option<SessionFailureKind>,
        /// Normalized provider label that observed the failure.
        #[serde(skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
        /// Model identifier active for the failed request.
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// HTTP status code when the upstream returned a non-success response.
        #[serde(skip_serializing_if = "Option::is_none")]
        http_status: Option<u16>,
        /// Raw request payload captured for the failed request.
        #[serde(skip_serializing_if = "Option::is_none")]
        request_payload: Option<String>,
        /// Raw response body or terminal stream payload captured for the failure.
        #[serde(skip_serializing_if = "Option::is_none")]
        response_payload: Option<String>,
        /// Provider-native response identifier, when available.
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_response_id: Option<String>,
    },
}

/// Consumer that receives streaming events.
///
/// Implement this trait to handle agent events in custom ways: logging, metrics,
/// UI updates, file writing, etc. The consumer receives events in real-time as
/// the agent executes.
///
/// # Thread Safety
///
/// Consumers must be `Send + Sync` as they may be called from async tasks.
///
/// # Error Handling
///
/// Returning an error stops event propagation. For non-critical failures,
/// log the error and return `Ok(())` to continue streaming.
///
/// # Examples
///
/// ```
/// use appam::agent::streaming::{StreamEvent, StreamConsumer};
///
/// struct PrintConsumer;
///
/// impl StreamConsumer for PrintConsumer {
///     fn on_event(&self, event: &StreamEvent) -> anyhow::Result<()> {
///         match event {
///             StreamEvent::Content { content } => print!("{}", content),
///             StreamEvent::Done => println!("\n[Done]"),
///             _ => {}
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait StreamConsumer: Send + Sync {
    /// Handle a streaming event.
    ///
    /// Called for each event emitted by the agent. Implementations should be
    /// fast and non-blocking; offload heavy work to background tasks.
    ///
    /// # Errors
    ///
    /// Return an error to stop event propagation. The agent execution will
    /// terminate with the error.
    fn on_event(&self, event: &StreamEvent) -> Result<()>;
}

/// Convenience wrapper for multiple consumers.
///
/// Broadcasts events to all registered consumers in order. If any consumer
/// returns an error, propagation stops and the error is returned.
///
/// # Examples
///
/// ```ignore
/// use appam::agent::streaming::MultiConsumer;
/// use appam::agent::consumers::{ConsoleConsumer, ChannelConsumer};
///
/// let multi = MultiConsumer::new()
///     .add(Box::new(ConsoleConsumer::new()))
///     .add(Box::new(ChannelConsumer::new(tx)));
/// ```
pub struct MultiConsumer {
    consumers: Vec<Box<dyn StreamConsumer>>,
}

impl MultiConsumer {
    /// Create an empty multi-consumer.
    pub fn new() -> Self {
        Self {
            consumers: Vec::new(),
        }
    }

    /// Add a consumer to the broadcast list.
    ///
    /// Consumers are invoked in the order they are added.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, consumer: Box<dyn StreamConsumer>) -> Self {
        self.consumers.push(consumer);
        self
    }
}

impl Default for MultiConsumer {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamConsumer for MultiConsumer {
    fn on_event(&self, event: &StreamEvent) -> Result<()> {
        for consumer in &self.consumers {
            consumer.on_event(event)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::usage::AggregatedUsage;

    struct TestConsumer {
        events: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl StreamConsumer for TestConsumer {
        fn on_event(&self, event: &StreamEvent) -> Result<()> {
            let mut events = self.events.lock().unwrap();
            events.push(format!("{:?}", event));
            Ok(())
        }
    }

    #[test]
    fn test_multi_consumer() {
        let events1 = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let events2 = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

        let consumer1 = TestConsumer {
            events: events1.clone(),
        };
        let consumer2 = TestConsumer {
            events: events2.clone(),
        };

        let multi = MultiConsumer::new()
            .add(Box::new(consumer1))
            .add(Box::new(consumer2));

        let event = StreamEvent::Content {
            content: "test".to_string(),
        };

        multi.on_event(&event).unwrap();

        assert_eq!(events1.lock().unwrap().len(), 1);
        assert_eq!(events2.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_multi_consumer_usage_update() {
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let consumer = TestConsumer {
            events: events.clone(),
        };

        let multi = MultiConsumer::new().add(Box::new(consumer));
        let snapshot = AggregatedUsage {
            total_input_tokens: 100,
            total_output_tokens: 50,
            total_cost_usd: 1.23,
            request_count: 2,
            ..AggregatedUsage::default()
        };

        multi
            .on_event(&StreamEvent::UsageUpdate { snapshot })
            .expect("usage update should be forwarded");

        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert!(
            captured[0].contains("UsageUpdate"),
            "Expected usage update event in captured output"
        );
    }

    #[test]
    fn test_stream_event_serialization() {
        let event = StreamEvent::Content {
            content: "Hello".to_string(),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("content"));
        assert!(json.contains("Hello"));

        let deserialized: StreamEvent = serde_json::from_str(&json).unwrap();
        match deserialized {
            StreamEvent::Content { content } => assert_eq!(content, "Hello"),
            _ => panic!("Wrong event type"),
        }
    }
}
