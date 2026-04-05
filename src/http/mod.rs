//! Shared HTTP utilities.
//!
//! This module exposes helpers for constructing and reusing HTTP clients across
//! the project. Centralizing the client lifecycle prevents us from spawning
//! redundant connection pools when thousands of requests are issued in parallel.
//! By reusing clients we dramatically reduce the number of open sockets, which
//! in turn avoids hitting the operating system's file descriptor limits during
//! large scans.

pub mod client_pool;
