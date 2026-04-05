//! Channel consumer for forwarding events to mpsc channels.

use anyhow::Result;
use tokio::sync::mpsc;

use crate::agent::streaming::{StreamConsumer, StreamEvent};

/// Consumer that forwards events to a tokio mpsc channel.
///
/// Useful for bridging agent execution with async streams, web sockets,
/// SSE responses, or any other channel-based communication.
///
/// # Examples
///
/// ```ignore
/// use appam::agent::consumers::ChannelConsumer;
/// use tokio::sync::mpsc;
///
/// let (tx, mut rx) = mpsc::unbounded_channel();
/// let consumer = ChannelConsumer::new(tx);
///
/// // In another task:
/// tokio::spawn(async move {
///     while let Some(event) = rx.recv().await {
///         println!("Event: {:?}", event);
///     }
/// });
/// ```
pub struct ChannelConsumer {
    tx: mpsc::UnboundedSender<StreamEvent>,
}

impl ChannelConsumer {
    /// Create a new channel consumer.
    ///
    /// Events are sent to the provided channel. If the channel is closed
    /// (receiver dropped), subsequent sends will fail silently.
    pub fn new(tx: mpsc::UnboundedSender<StreamEvent>) -> Self {
        Self { tx }
    }
}

impl StreamConsumer for ChannelConsumer {
    fn on_event(&self, event: &StreamEvent) -> Result<()> {
        // Clone the event and send to channel
        // If send fails (receiver dropped), we silently ignore it
        // as this is not a critical error
        let _ = self.tx.send(event.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_channel_consumer() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let consumer = ChannelConsumer::new(tx);

        let event = StreamEvent::Content {
            content: "test".to_string(),
        };

        consumer.on_event(&event).unwrap();

        let received = rx.recv().await.unwrap();
        match received {
            StreamEvent::Content { content } => assert_eq!(content, "test"),
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_channel_consumer_multiple_events() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let consumer = ChannelConsumer::new(tx);

        consumer
            .on_event(&StreamEvent::SessionStarted {
                session_id: "test".to_string(),
            })
            .unwrap();

        consumer
            .on_event(&StreamEvent::Content {
                content: "hello".to_string(),
            })
            .unwrap();

        consumer.on_event(&StreamEvent::Done).unwrap();

        // Should receive all 3 events
        assert!(rx.recv().await.is_some());
        assert!(rx.recv().await.is_some());
        assert!(rx.recv().await.is_some());
    }

    #[test]
    fn test_channel_consumer_closed_receiver() {
        let (tx, rx) = mpsc::unbounded_channel();
        let consumer = ChannelConsumer::new(tx);

        // Drop receiver
        drop(rx);

        // Should not panic, just silently ignore
        let result = consumer.on_event(&StreamEvent::Done);
        assert!(result.is_ok());
    }
}
