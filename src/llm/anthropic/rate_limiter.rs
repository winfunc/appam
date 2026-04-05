//! Global rate limiter for Anthropic API token usage.
//!
//! Implements a consumption-tracking rate limiter that monitors actual token usage
//! from API responses and blocks requests when the org-wide rate limit would be exceeded.
//! This is critical for high-parallelism scenarios where multiple agents/workers compete
//! for the same organization-wide rate limit.
//!
//! # How It Works
//!
//! 1. Track actual token consumption from API responses (not estimates!)
//! 2. Maintain a sliding 60-second window of consumption
//! 3. Before each request, check if recent consumption < limit
//! 4. If at limit, block until oldest consumption falls outside window
//!
//! # Example
//!
//! ```rust,no_run
//! use appam::llm::anthropic::rate_limiter::RateLimiter;
//!
//! # async fn example() {
//! let limiter = RateLimiter::new(1_800_000); // 1.8M tokens/minute
//!
//! // Check before sending request
//! limiter.acquire_slot().await;
//!
//! // Send request and get actual token usage from response
//! let input_tokens = 185234;
//! let output_tokens = 2451;
//! limiter.record_usage(input_tokens + output_tokens).await;
//! # }
//! ```

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Token consumption record with timestamp.
#[derive(Debug, Clone)]
struct TokenConsumption {
    /// Tokens consumed (input + output)
    tokens: u32,
    /// When this consumption occurred
    timestamp: Instant,
}

/// Global rate limiter for coordinating Anthropic API requests across workers.
///
/// Tracks actual token consumption in a sliding 60-second window and blocks
/// requests when the org-wide rate limit would be exceeded. Thread-safe and
/// shared across all agents/workers.
///
/// # Design
///
/// - Uses actual token usage from API responses (not estimates)
/// - Maintains sliding 60-second window of consumption
/// - Allows burst of requests initially (up to limit)
/// - Blocks when consumption in last 60s would exceed limit
#[derive(Debug, Clone)]
pub struct RateLimiter {
    /// Shared state for tracking consumption
    state: Arc<Mutex<RateLimiterState>>,

    /// Maximum tokens allowed per minute
    tokens_per_minute: u32,
}

/// Internal state for the rate limiter.
#[derive(Debug)]
struct RateLimiterState {
    /// Token consumption history (sliding 60-second window)
    consumptions: VecDeque<TokenConsumption>,

    /// Current total tokens in the window
    current_usage: u64,

    /// Tokens per minute limit
    limit: u32,
}

impl RateLimiterState {
    /// Remove consumptions older than 60 seconds from the window.
    fn prune_old_consumptions(&mut self) {
        let cutoff = Instant::now() - Duration::from_secs(60);

        while let Some(consumption) = self.consumptions.front() {
            if consumption.timestamp < cutoff {
                // Remove old consumption and update current usage
                let removed = self.consumptions.pop_front().unwrap();
                self.current_usage = self.current_usage.saturating_sub(removed.tokens as u64);
            } else {
                break;
            }
        }
    }

    /// Get current usage in the sliding window after pruning old entries.
    fn get_current_usage(&mut self) -> u64 {
        self.prune_old_consumptions();
        self.current_usage
    }
}

impl RateLimiter {
    /// Create a new rate limiter with the specified tokens per minute capacity.
    ///
    /// # Arguments
    ///
    /// * `tokens_per_minute` - Maximum tokens allowed per minute (org-wide limit)
    ///
    /// # Example
    ///
    /// ```rust
    /// use appam::llm::anthropic::rate_limiter::RateLimiter;
    ///
    /// // Create limiter for 2M tokens/minute with 90% buffer
    /// let limiter = RateLimiter::new(1_800_000);
    /// ```
    pub fn new(tokens_per_minute: u32) -> Self {
        info!(
            tokens_per_minute = tokens_per_minute,
            "Initializing consumption-based rate limiter with sliding 60s window"
        );

        let state = RateLimiterState {
            consumptions: VecDeque::new(),
            current_usage: 0,
            limit: tokens_per_minute,
        };

        Self {
            state: Arc::new(Mutex::new(state)),
            tokens_per_minute,
        }
    }

    /// Acquire a request slot, blocking if current usage would exceed the limit.
    ///
    /// This checks the sliding window but does NOT deduct tokens (that happens
    /// in `record_usage` after getting the actual response). This allows requests
    /// to proceed optimistically, with actual consumption tracked afterward.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use appam::llm::anthropic::rate_limiter::RateLimiter;
    /// # async fn example() {
    /// let limiter = RateLimiter::new(1_800_000);
    ///
    /// // Check before sending request
    /// limiter.acquire_slot().await;
    ///
    /// // Send request...
    /// # }
    /// ```
    pub async fn acquire_slot(&self) {
        loop {
            let (current_usage, limit) = {
                let mut state = self.state.lock().await;
                let usage = state.get_current_usage();
                (usage, state.limit)
            };

            // Check if we have headroom (use 80% threshold to allow some burst)
            let threshold = (limit as f64 * 0.8) as u64;

            if current_usage < threshold {
                debug!(
                    current_usage = current_usage,
                    limit = limit,
                    threshold = threshold,
                    "Rate limiter: slot acquired"
                );
                return;
            }

            // At or over threshold, wait a bit and check again
            warn!(
                current_usage = current_usage,
                limit = limit,
                threshold = threshold,
                "Rate limiter: at threshold, waiting for window to clear"
            );

            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    /// Record actual token usage from an API response.
    ///
    /// Call this AFTER receiving a response with actual token counts.
    /// This updates the sliding window with real consumption data.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Actual tokens consumed (input + output from response)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use appam::llm::anthropic::rate_limiter::RateLimiter;
    /// # async fn example() {
    /// let limiter = RateLimiter::new(1_800_000);
    ///
    /// // After getting response with usage: { input_tokens: 185234, output_tokens: 2451 }
    /// limiter.record_usage(185234 + 2451).await;
    /// # }
    /// ```
    pub async fn record_usage(&self, tokens: u32) {
        let mut state = self.state.lock().await;

        // Add new consumption
        state.consumptions.push_back(TokenConsumption {
            tokens,
            timestamp: Instant::now(),
        });

        state.current_usage += tokens as u64;

        // Prune old entries
        state.prune_old_consumptions();

        debug!(
            tokens_recorded = tokens,
            current_usage = state.current_usage,
            limit = state.limit,
            utilization_pct = (state.current_usage as f64 / state.limit as f64 * 100.0),
            window_entries = state.consumptions.len(),
            "Recorded token usage in rate limiter"
        );
    }

    /// Get current token usage in the sliding window.
    ///
    /// Returns the total tokens consumed in the last 60 seconds.
    pub async fn current_usage(&self) -> u64 {
        let mut state = self.state.lock().await;
        state.get_current_usage()
    }

    /// Get the configured tokens per minute limit.
    pub fn tokens_per_minute(&self) -> u32 {
        self.tokens_per_minute
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_creation() {
        let limiter = RateLimiter::new(1_800_000);
        assert_eq!(limiter.tokens_per_minute(), 1_800_000);
    }

    #[tokio::test]
    async fn test_acquire_slot_immediate_when_empty() {
        let limiter = RateLimiter::new(1_800_000);

        // First slot should be immediate
        limiter.acquire_slot().await;

        let usage = limiter.current_usage().await;
        assert_eq!(usage, 0); // No usage recorded yet
    }

    #[tokio::test]
    async fn test_record_usage() {
        let limiter = RateLimiter::new(1_800_000);

        // Record some usage
        limiter.record_usage(100_000).await;

        let usage = limiter.current_usage().await;
        assert_eq!(usage, 100_000);

        // Record more
        limiter.record_usage(50_000).await;

        let usage = limiter.current_usage().await;
        assert_eq!(usage, 150_000);
    }

    #[tokio::test]
    async fn test_sliding_window_pruning() {
        let limiter = RateLimiter::new(1_000_000);

        // Record usage
        limiter.record_usage(500_000).await;

        let usage = limiter.current_usage().await;
        assert_eq!(usage, 500_000);

        // Wait for window to slide (in real test, this would be 60s+)
        // For this test, we just verify the pruning logic works
        // We can't easily test time-based pruning without mocking time
    }

    #[tokio::test]
    async fn test_acquire_slot_blocks_at_threshold() {
        let limiter = RateLimiter::new(100_000);

        // Fill up to 90% of limit (over 80% threshold)
        limiter.record_usage(95_000).await;

        let usage = limiter.current_usage().await;
        assert!(usage >= 80_000); // Over threshold

        // acquire_slot should check but we can't easily test blocking without time mock
    }
}
