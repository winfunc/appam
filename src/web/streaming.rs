//! Server-Sent Events (SSE) streaming for real-time chat responses.
//!
//! Provides utilities for streaming agent responses to clients using SSE,
//! including content, reasoning traces, tool calls, and tool results.
//!
//! This module adapts the core streaming infrastructure (`agent::streaming`)
//! for web use with Axum SSE responses.

use axum::response::sse::{Event, KeepAlive};
use axum::response::Sse;
use futures::stream::Stream;
use tokio::sync::mpsc;

use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};

// Re-export core streaming types
use crate::agent::streaming::StreamConsumer;
pub use crate::agent::streaming::StreamEvent;

type SseEventStream = Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;
type SseResponse = Sse<axum::response::sse::KeepAliveStream<SseEventStream>>;

impl StreamEvent {
    /// Convert to SSE event.
    ///
    /// Serializes the event to JSON and wraps it in an SSE Event for transmission
    /// over Server-Sent Events protocol.
    pub fn to_sse_event(&self) -> Result<Event, serde_json::Error> {
        let data = serde_json::to_string(self)?;
        Ok(Event::default().data(data))
    }
}

/// Channel-based stream for SSE events.
///
/// Wraps a tokio mpsc receiver to create a stream compatible with axum SSE.
pub struct EventStream {
    rx: mpsc::UnboundedReceiver<StreamEvent>,
}

impl EventStream {
    /// Create a new event stream with a sender.
    pub fn new() -> (mpsc::UnboundedSender<StreamEvent>, Self) {
        let (tx, rx) = mpsc::unbounded_channel();
        (tx, Self { rx })
    }

    /// Convert to SSE response.
    pub fn into_sse_response(self) -> SseResponse {
        let stream: SseEventStream = Box::pin(self);
        Sse::new(stream).keep_alive(KeepAlive::default())
    }
}

impl Stream for EventStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(event)) => {
                let sse_event = event
                    .to_sse_event()
                    .unwrap_or_else(|_| Event::default().data("error"));
                Poll::Ready(Some(Ok(sse_event)))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Helper to send events to stream.
#[derive(Clone)]
pub struct EventSender {
    tx: mpsc::UnboundedSender<StreamEvent>,
}

impl EventSender {
    /// Create from unbounded sender.
    pub fn new(tx: mpsc::UnboundedSender<StreamEvent>) -> Self {
        Self { tx }
    }

    /// Send an event to the stream.
    pub fn send(&self, event: StreamEvent) -> Result<(), String> {
        self.tx
            .send(event)
            .map_err(|_| "Failed to send event".to_string())
    }

    /// Send session started event.
    pub fn session_started(&self, session_id: String) -> Result<(), String> {
        self.send(StreamEvent::SessionStarted { session_id })
    }

    /// Send reasoning trace.
    pub fn reasoning(&self, content: String) -> Result<(), String> {
        if !content.is_empty() {
            self.send(StreamEvent::Reasoning { content })
        } else {
            Ok(())
        }
    }

    /// Send content chunk.
    pub fn content(&self, content: String) -> Result<(), String> {
        if !content.is_empty() {
            self.send(StreamEvent::Content { content })
        } else {
            Ok(())
        }
    }

    /// Send tool call started.
    pub fn tool_call_started(&self, tool_name: String, arguments: String) -> Result<(), String> {
        self.send(StreamEvent::ToolCallStarted {
            tool_name,
            arguments,
        })
    }

    /// Send tool call completed.
    pub fn tool_call_completed(
        &self,
        tool_name: String,
        result: serde_json::Value,
        success: bool,
        duration_ms: f64,
    ) -> Result<(), String> {
        self.send(StreamEvent::ToolCallCompleted {
            tool_name,
            result,
            success,
            duration_ms,
        })
    }

    /// Send tool call failed.
    pub fn tool_call_failed(&self, tool_name: String, error: String) -> Result<(), String> {
        self.send(StreamEvent::ToolCallFailed { tool_name, error })
    }

    /// Send turn completed.
    pub fn turn_completed(&self) -> Result<(), String> {
        self.send(StreamEvent::TurnCompleted)
    }

    /// Send done event.
    pub fn done(&self) -> Result<(), String> {
        self.send(StreamEvent::Done)
    }

    /// Send error event.
    pub fn error(&self, message: String) -> Result<(), String> {
        self.send(StreamEvent::Error {
            message,
            failure_kind: None,
            provider: None,
            model: None,
            http_status: None,
            request_payload: None,
            response_payload: None,
            provider_response_id: None,
        })
    }
}

impl StreamConsumer for EventSender {
    fn on_event(&self, event: &StreamEvent) -> anyhow::Result<()> {
        self.send(event.clone()).map_err(|e| anyhow::anyhow!(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_event_serialization() {
        let event = StreamEvent::Content {
            content: "Hello".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("content"));
        assert!(json.contains("Hello"));
    }

    #[tokio::test]
    async fn test_event_stream() {
        let (tx, mut stream) = EventStream::new();
        let sender = EventSender::new(tx);

        sender.content("Test".to_string()).unwrap();
        sender.done().unwrap();

        // Poll stream
        use futures::StreamExt;
        let event1 = stream.next().await;
        assert!(event1.is_some());

        let event2 = stream.next().await;
        assert!(event2.is_some());
    }
}
