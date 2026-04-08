//! Built-in [`StreamConsumer`](crate::agent::streaming::StreamConsumer) implementations.
//!
//! These consumers cover the most common sinks used by Appam applications:
//!
//! - [`ConsoleConsumer`] for human-readable terminal output
//! - [`ChannelConsumer`] for async handoff into other tasks
//! - [`CallbackConsumer`] for inline integration code
//! - [`TraceConsumer`] and [`SqliteTraceConsumer`] for persisted trace capture
//!
//! # Examples
//!
//! ```no_run
//! use appam::agent::consumers::ConsoleConsumer;
//!
//! let consumer = ConsoleConsumer::new()
//!     .with_colors(true)
//!     .with_reasoning(true);
//! ```

pub mod callback;
pub mod channel;
pub mod console;
pub mod sqlite_trace;
pub mod trace;

pub use callback::CallbackConsumer;
pub use channel::ChannelConsumer;
pub use console::ConsoleConsumer;
pub use sqlite_trace::SqliteTraceConsumer;
pub use trace::TraceConsumer;
