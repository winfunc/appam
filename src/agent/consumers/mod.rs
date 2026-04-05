//! Built-in stream consumers for common use cases.
//!
//! Provides ready-to-use implementations of the `StreamConsumer` trait:
//! - `ConsoleConsumer`: Pretty-printed console output
//! - `ChannelConsumer`: Forward events to an mpsc channel
//! - `CallbackConsumer`: Execute a closure for each event
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
