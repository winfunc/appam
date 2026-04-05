//! Prompt caching utilities for Anthropic API.
//!
//! Provides helpers for applying cache control breakpoints to tools,
//! system prompts, and messages.
//!
//! # Cache Behavior
//!
//! - Minimum cacheable: 1024 tokens (Sonnet/Opus), 4096 tokens (Haiku 4.5)
//! - TTL: 5 minutes (default) or 1 hour (2x cost)
//! - Automatic prefix matching up to ~20 content blocks
//! - Up to 4 cache breakpoints per request
//!
//! # Pricing
//!
//! - Cache writes (5m): 1.25x base input token price
//! - Cache writes (1h): 2x base input token price
//! - Cache reads: 0.1x base input token price

use super::types::CacheControl;

/// Create cache control for 5-minute TTL.
pub fn cache_control_5m() -> CacheControl {
    CacheControl::ephemeral_5m()
}

/// Create cache control for 1-hour TTL.
pub fn cache_control_1h() -> CacheControl {
    CacheControl::ephemeral_1h()
}
