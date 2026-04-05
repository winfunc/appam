//! Callback consumer for closure-based event handling.

use std::sync::Arc;

use anyhow::Result;

use crate::agent::streaming::{StreamConsumer, StreamEvent};

type EventCallback = dyn Fn(&StreamEvent) -> Result<()> + Send + Sync;

/// Consumer that invokes a callback function for each event.
///
/// Enables quick, inline event handling without defining a full struct.
/// The callback must be `Fn` (not `FnMut`) to ensure thread safety.
///
/// # Examples
///
/// ```
/// use appam::agent::consumers::CallbackConsumer;
/// use appam::agent::streaming::StreamEvent;
///
/// let consumer = CallbackConsumer::new(|event| {
///     match event {
///         StreamEvent::Content { content } => print!("{}", content),
///         _ => {}
///     }
///     Ok(())
/// });
/// ```
pub struct CallbackConsumer {
    callback: Arc<EventCallback>,
}

impl CallbackConsumer {
    /// Create a new callback consumer.
    ///
    /// The callback is invoked for each event. It must be thread-safe
    /// (`Send + Sync`) and immutable (`Fn`, not `FnMut`).
    ///
    /// # Examples
    ///
    /// ```
    /// use appam::agent::consumers::CallbackConsumer;
    ///
    /// let consumer = CallbackConsumer::new(|event| {
    ///     println!("Event: {:?}", event);
    ///     Ok(())
    /// });
    /// ```
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(&StreamEvent) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(callback),
        }
    }
}

impl StreamConsumer for CallbackConsumer {
    fn on_event(&self, event: &StreamEvent) -> Result<()> {
        (self.callback)(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_callback_consumer() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let consumer = CallbackConsumer::new(move |_event| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });

        consumer
            .on_event(&StreamEvent::Content {
                content: "test".to_string(),
            })
            .unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callback_consumer_error_propagation() {
        let consumer = CallbackConsumer::new(|_event| Err(anyhow::anyhow!("Test error")));

        let result = consumer.on_event(&StreamEvent::Done);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Test error"));
    }

    #[test]
    fn test_callback_consumer_pattern_matching() {
        let content_only = Arc::new(std::sync::Mutex::new(String::new()));
        let content_clone = content_only.clone();

        let consumer = CallbackConsumer::new(move |event| {
            if let StreamEvent::Content { content } = event {
                let mut text = content_clone.lock().unwrap();
                text.push_str(content);
            }
            Ok(())
        });

        consumer
            .on_event(&StreamEvent::Content {
                content: "hello ".to_string(),
            })
            .unwrap();

        consumer.on_event(&StreamEvent::Done).unwrap();

        consumer
            .on_event(&StreamEvent::Content {
                content: "world".to_string(),
            })
            .unwrap();

        let text = content_only.lock().unwrap();
        assert_eq!(*text, "hello world");
    }
}
