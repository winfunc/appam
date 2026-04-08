//! Closure-based streaming helpers for [`RuntimeAgent`].
//!
//! The types in this module offer the most ergonomic streaming API in the crate:
//! attach closures for the events you care about, then run the agent. This is
//! ideal for CLIs, small services, demos, and tests that want streaming without
//! a bespoke [`StreamConsumer`] type.
//!
//! # Examples
//!
//! ```no_run
//! use appam::prelude::*;
//! use anyhow::Result;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let agent = Agent::quick(
//!         "anthropic/claude-sonnet-4-5",
//!         "You are helpful.",
//!         vec![],
//!     )?;
//!
//!     agent
//!         .stream("Hello")
//!         .on_content(|text| print!("{}", text))
//!         .on_tool_call(|name, args| println!("Tool: {}", name))
//!         .run()
//!         .await?;
//!
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::runtime_agent::RuntimeAgent;
use super::streaming::{StreamConsumer, StreamEvent};
use super::{Agent, Session};

/// Type alias for boxed future
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
type ContentHandler = dyn Fn(&str) + Send + Sync;
type ToolCallHandler = dyn Fn(&str, &str) + Send + Sync;
type ToolResultHandler = dyn Fn(&str, &Value) + Send + Sync;
type DoneHandler = dyn Fn() + Send + Sync;
type AsyncToolCallHandler = dyn Fn(String, String) -> BoxFuture<'static, Result<()>> + Send + Sync;
type AsyncToolResultHandler = dyn Fn(String, Value) -> BoxFuture<'static, Result<()>> + Send + Sync;

/// Builder for closure-driven handling of streamed agent events.
///
/// A `StreamBuilder` is created by [`RuntimeAgent::stream`] and consumed by
/// [`StreamBuilder::run`]. Each registration method stores one closure for one
/// event category; later registrations replace earlier ones.
///
/// Synchronous handlers run inline on the agent's streaming path. Keep them
/// lightweight so they do not stall provider reads. The `*_async` handlers are
/// spawned onto Tokio tasks for side work that can happen out of band.
///
/// # Examples
///
/// ```no_run
/// # use appam::prelude::*;
/// # use anyhow::Result;
/// # #[tokio::main]
/// # async fn main() -> Result<()> {
/// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
/// agent
///     .stream("Tell me about Rust")
///     .on_content(|text| {
///         print!("{}", text);
///         std::io::Write::flush(&mut std::io::stdout()).ok();
///     })
///     .on_tool_call(|name, args| {
///         println!("\n🔧 Calling: {}", name);
///     })
///     .on_reasoning(|thinking| {
///         print!("\x1b[36m{}\x1b[0m", thinking);  // Cyan
///     })
///     .on_done(|| {
///         println!("\n✓ Complete");
///     })
///     .run()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct StreamBuilder<'a> {
    agent: &'a RuntimeAgent,
    message: String,

    // Sync closure handlers
    on_content: Option<Arc<ContentHandler>>,
    on_reasoning: Option<Arc<ContentHandler>>,
    on_tool_call: Option<Arc<ToolCallHandler>>,
    on_tool_result: Option<Arc<ToolResultHandler>>,
    on_tool_failed: Option<Arc<ToolCallHandler>>,
    on_error: Option<Arc<ContentHandler>>,
    on_done: Option<Arc<DoneHandler>>,
    on_session_started: Option<Arc<ContentHandler>>,

    // Async closure handlers (spawn tasks)
    on_tool_call_async: Option<Arc<AsyncToolCallHandler>>,
    on_tool_result_async: Option<Arc<AsyncToolResultHandler>>,
}

impl<'a> StreamBuilder<'a> {
    /// Create a new streaming builder for one user message.
    ///
    /// Most callers should use [`RuntimeAgent::stream`] instead of invoking
    /// this constructor directly.
    pub fn new(agent: &'a RuntimeAgent, message: impl Into<String>) -> Self {
        Self {
            agent,
            message: message.into(),
            on_content: None,
            on_reasoning: None,
            on_tool_call: None,
            on_tool_result: None,
            on_tool_failed: None,
            on_error: None,
            on_done: None,
            on_session_started: None,
            on_tool_call_async: None,
            on_tool_result_async: None,
        }
    }

    /// Handle text content chunks emitted by the model.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Hello")
    ///     .on_content(|text| {
    ///         print!("{}", text);
    ///         std::io::Write::flush(&mut std::io::stdout()).ok();
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_content<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.on_content = Some(Arc::new(f));
        self
    }

    /// Handle reasoning or thinking text emitted by models that expose it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("anthropic/claude-sonnet-4-5", "Think carefully.", vec![])?;
    /// agent
    ///     .stream("Analyze this problem")
    ///     .on_reasoning(|thinking| {
    ///         print!("\x1b[36m{}\x1b[0m", thinking);  // Cyan colored thinking
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_reasoning<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.on_reasoning = Some(Arc::new(f));
        self
    }

    /// Handle finalized tool-call requests.
    ///
    /// Called when the model decides to invoke a tool with complete arguments.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Use the calculator")
    ///     .on_tool_call(|name, args| {
    ///         println!("🔧 Calling: {}", name);
    ///         println!("   Arguments: {}", args);
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_tool_call<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str) + Send + Sync + 'static,
    {
        self.on_tool_call = Some(Arc::new(f));
        self
    }

    /// Handle successful tool execution results.
    ///
    /// Called when a tool completes successfully with its result.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Calculate something")
    ///     .on_tool_result(|name, result| {
    ///         println!("✓ {} returned: {:?}", name, result);
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_tool_result<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Value) + Send + Sync + 'static,
    {
        self.on_tool_result = Some(Arc::new(f));
        self
    }

    /// Handle tool execution failures.
    ///
    /// Called when a tool raises an error during execution.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Do something")
    ///     .on_tool_failed(|name, error| {
    ///         eprintln!("✗ {} failed: {}", name, error);
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_tool_failed<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str) + Send + Sync + 'static,
    {
        self.on_tool_failed = Some(Arc::new(f));
        self
    }

    /// Handle errors during agent execution
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Hello")
    ///     .on_error(|error| {
    ///         eprintln!("❌ Error: {}", error);
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.on_error = Some(Arc::new(f));
        self
    }

    /// Handle stream completion
    ///
    /// Called when the agent finishes processing and no more events will be emitted.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Hello")
    ///     .on_done(|| {
    ///         println!("\n✓ Stream completed successfully");
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_done<F>(mut self, f: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.on_done = Some(Arc::new(f));
        self
    }

    /// Handle session start
    ///
    /// Called when a new session is created with its unique ID.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Hello")
    ///     .on_session_started(|session_id| {
    ///         println!("🆔 Session: {}", session_id);
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_session_started<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.on_session_started = Some(Arc::new(f));
        self
    }

    /// Handle tool calls with async operations
    ///
    /// Use this when you need to perform async operations like database writes or
    /// API calls in response to tool invocations.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # use std::sync::Arc;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Do something")
    ///     .on_tool_call_async(|name, args| async move {
    ///         // Simulate async database write
    ///         tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    ///         println!("Logged tool call: {} with {}", name, args);
    ///         Ok(())
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_tool_call_async<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(String, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        self.on_tool_call_async = Some(Arc::new(move |name, args| Box::pin(f(name, args))));
        self
    }

    /// Handle tool results with async operations
    ///
    /// Use this when you need to perform async operations like database writes or
    /// API calls with tool results.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Do something")
    ///     .on_tool_result_async(|name, result| async move {
    ///         // Simulate async database write
    ///         tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    ///         println!("Saved result from {}: {:?}", name, result);
    ///         Ok(())
    ///     })
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn on_tool_result_async<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(String, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        self.on_tool_result_async = Some(Arc::new(move |name, result| Box::pin(f(name, result))));
        self
    }

    /// Execute the streaming with configured handlers
    ///
    /// This consumes the builder and runs the agent with the configured event handlers.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// let session = agent
    ///     .stream("Hello")
    ///     .on_content(|text| print!("{}", text))
    ///     .run()
    ///     .await?;
    ///
    /// println!("Session ID: {}", session.session_id);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run(self) -> Result<Session> {
        // Create a consumer that dispatches to closures
        let consumer = ClosureConsumer {
            on_content: self.on_content,
            on_reasoning: self.on_reasoning,
            on_tool_call: self.on_tool_call,
            on_tool_result: self.on_tool_result,
            on_tool_failed: self.on_tool_failed,
            on_error: self.on_error,
            on_done: self.on_done,
            on_session_started: self.on_session_started,
            on_tool_call_async: self.on_tool_call_async,
            on_tool_result_async: self.on_tool_result_async,
        };

        self.agent
            .run_streaming(&self.message, Box::new(consumer))
            .await
    }
}

/// Internal consumer that wraps closures and dispatches events
struct ClosureConsumer {
    on_content: Option<Arc<ContentHandler>>,
    on_reasoning: Option<Arc<ContentHandler>>,
    on_tool_call: Option<Arc<ToolCallHandler>>,
    on_tool_result: Option<Arc<ToolResultHandler>>,
    on_tool_failed: Option<Arc<ToolCallHandler>>,
    on_error: Option<Arc<ContentHandler>>,
    on_done: Option<Arc<DoneHandler>>,
    on_session_started: Option<Arc<ContentHandler>>,
    on_tool_call_async: Option<Arc<AsyncToolCallHandler>>,
    on_tool_result_async: Option<Arc<AsyncToolResultHandler>>,
}

impl StreamConsumer for ClosureConsumer {
    fn on_event(&self, event: &StreamEvent) -> Result<()> {
        match event {
            StreamEvent::SessionStarted { session_id } => {
                if let Some(ref f) = self.on_session_started {
                    f(session_id);
                }
            }
            StreamEvent::Content { content } => {
                if let Some(ref f) = self.on_content {
                    f(content);
                }
            }
            StreamEvent::Reasoning { content } => {
                if let Some(ref f) = self.on_reasoning {
                    f(content);
                }
            }
            StreamEvent::ToolCallStarted {
                tool_name,
                arguments,
            } => {
                // Sync handler
                if let Some(ref f) = self.on_tool_call {
                    f(tool_name, arguments);
                }
                // Async handler (spawn task)
                if let Some(ref f) = self.on_tool_call_async {
                    let fut = f(tool_name.clone(), arguments.clone());
                    tokio::spawn(async move {
                        if let Err(e) = fut.await {
                            eprintln!("Error in async tool call handler: {}", e);
                        }
                    });
                }
            }
            StreamEvent::ToolCallCompleted {
                tool_name,
                result,
                success,
                ..
            } => {
                if *success {
                    // Sync handler
                    if let Some(ref f) = self.on_tool_result {
                        f(tool_name, result);
                    }
                    // Async handler (spawn task)
                    if let Some(ref f) = self.on_tool_result_async {
                        let fut = f(tool_name.clone(), result.clone());
                        tokio::spawn(async move {
                            if let Err(e) = fut.await {
                                eprintln!("Error in async tool result handler: {}", e);
                            }
                        });
                    }
                }
            }
            StreamEvent::ToolCallFailed { tool_name, error } => {
                if let Some(ref f) = self.on_tool_failed {
                    f(tool_name, error);
                }
            }
            StreamEvent::Error { message, .. } => {
                if let Some(ref f) = self.on_error {
                    f(message);
                }
            }
            StreamEvent::Done => {
                if let Some(ref f) = self.on_done {
                    f();
                }
            }
            // Ignore other events (TurnCompleted, UsageUpdate)
            _ => {}
        }
        Ok(())
    }
}

/// Convenience entry point for closure-based streaming.
impl RuntimeAgent {
    /// Create a [`StreamBuilder`] for one user message.
    ///
    /// Use this when you want inline closures rather than a dedicated
    /// [`StreamConsumer`] implementation.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let agent = Agent::quick("openai/gpt-4o", "You are helpful.", vec![])?;
    /// agent
    ///     .stream("Tell me about Rust")
    ///     .on_content(|text| print!("{}", text))
    ///     .on_done(|| println!("\nDone!"))
    ///     .run()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn stream(&self, message: impl Into<String>) -> StreamBuilder<'_> {
        StreamBuilder::new(self, message)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_stream_builder_creation() {
        // Just verify the builder can be created
        // Actual streaming tests require a full agent setup
    }
}
