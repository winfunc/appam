//! Usage tracking and aggregation for LLM token consumption.
//!
//! Provides thread-safe tracking of cumulative token usage and costs across
//! multiple concurrent LLM requests. Used by agents to monitor real-time usage
//! during batch processing operations.
//!
//! # Provider Support
//!
//! Usage tracking is fully implemented for all supported providers:
//! - **Anthropic**: All token types including cache creation/read
//! - **OpenAI**: All token types including reasoning and cached tokens
//! - **OpenRouter**: Full usage tracking with detailed token counts, cached tokens,
//!   reasoning tokens, and cost information (when `usage: {include: true}` is enabled)

use super::pricing::calculate_cost;
use super::UnifiedUsage;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Aggregated usage statistics across multiple LLM requests.
///
/// Tracks cumulative token consumption and calculated costs for a batch of
/// requests. All token counts are 64-bit to handle large batch operations.
///
/// # Thread Safety
///
/// This type is designed to be wrapped in `Arc<Mutex<>>` for thread-safe
/// aggregation across concurrent tasks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedUsage {
    /// Total input tokens consumed
    pub total_input_tokens: u64,

    /// Total output tokens generated
    pub total_output_tokens: u64,

    /// Total tokens written to cache
    pub total_cache_creation_tokens: u64,

    /// Total tokens read from cache
    pub total_cache_read_tokens: u64,

    /// Total reasoning tokens (for models with extended thinking)
    pub total_reasoning_tokens: u64,

    /// Total cost in USD
    pub total_cost_usd: f64,

    /// Number of LLM requests made
    pub request_count: u64,
}

impl AggregatedUsage {
    /// Create a new empty usage tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add usage from a single LLM request.
    ///
    /// Updates all cumulative counters and adds the calculated cost.
    ///
    /// # Arguments
    ///
    /// * `usage` - Token usage from LLM response
    /// * `provider` - Provider name (e.g., "anthropic", "openai")
    /// * `model` - Model identifier
    pub fn add_usage(&mut self, usage: &UnifiedUsage, provider: &str, model: &str) {
        self.total_input_tokens += usage.input_tokens as u64;
        self.total_output_tokens += usage.output_tokens as u64;

        if let Some(cache_creation) = usage.cache_creation_input_tokens {
            self.total_cache_creation_tokens += cache_creation as u64;
        }

        if let Some(cache_read) = usage.cache_read_input_tokens {
            self.total_cache_read_tokens += cache_read as u64;
        }

        if let Some(reasoning) = usage.reasoning_tokens {
            self.total_reasoning_tokens += reasoning as u64;
        }

        // Calculate and add cost
        let cost = calculate_cost(usage, provider, model);
        self.total_cost_usd += cost;

        self.request_count += 1;
    }

    /// Get total tokens (input + output).
    pub fn total_tokens(&self) -> u64 {
        self.total_input_tokens + self.total_output_tokens
    }

    /// Get total tokens including reasoning.
    pub fn total_tokens_with_reasoning(&self) -> u64 {
        self.total_tokens() + self.total_reasoning_tokens
    }

    /// Format usage for display in progress bars.
    ///
    /// Returns a compact string showing total tokens and cost.
    ///
    /// # Examples
    ///
    /// ```
    /// use appam::llm::usage::AggregatedUsage;
    ///
    /// let mut usage = AggregatedUsage::new();
    /// // ... add some usage ...
    /// println!("{}", usage.format_display());
    /// // Output: "123K tokens | $0.45"
    /// ```
    pub fn format_display(&self) -> String {
        let total_tokens = self.total_tokens_with_reasoning();
        if total_tokens >= 1_000_000 {
            format!(
                "{:.1}M tokens | ${:.2}",
                total_tokens as f64 / 1_000_000.0,
                self.total_cost_usd
            )
        } else if total_tokens >= 1_000 {
            format!(
                "{}K tokens | ${:.2}",
                total_tokens / 1_000,
                self.total_cost_usd
            )
        } else {
            format!("{} tokens | ${:.4}", total_tokens, self.total_cost_usd)
        }
    }

    /// Format detailed usage breakdown.
    ///
    /// Returns a multi-line string with detailed token and cost information.
    pub fn format_detailed(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!("Total Requests:    {}", self.request_count));
        lines.push(format!("Input Tokens:      {}", self.total_input_tokens));
        lines.push(format!("Output Tokens:     {}", self.total_output_tokens));

        if self.total_reasoning_tokens > 0 {
            lines.push(format!(
                "Reasoning Tokens:  {}",
                self.total_reasoning_tokens
            ));
        }

        if self.total_cache_creation_tokens > 0 {
            lines.push(format!(
                "Cache Write:       {}",
                self.total_cache_creation_tokens
            ));
        }

        if self.total_cache_read_tokens > 0 {
            lines.push(format!(
                "Cache Read:        {}",
                self.total_cache_read_tokens
            ));
        }

        lines.push(format!(
            "Total Tokens:      {}",
            self.total_tokens_with_reasoning()
        ));
        lines.push(format!("Total Cost:        ${:.4}", self.total_cost_usd));

        lines.join("\n")
    }
}

/// Thread-safe usage tracker for concurrent operations.
///
/// Wraps `AggregatedUsage` in `Arc<Mutex<>>` to enable safe updates from
/// multiple parallel tasks. Cloning this type is cheap (only clones the Arc).
///
/// # Examples
///
/// ```
/// use appam::llm::usage::UsageTracker;
/// use appam::llm::UnifiedUsage;
///
/// let tracker = UsageTracker::new();
///
/// // In a parallel task
/// let usage = UnifiedUsage {
///     input_tokens: 1000,
///     output_tokens: 500,
///     cache_creation_input_tokens: None,
///     cache_read_input_tokens: None,
///     reasoning_tokens: None,
/// };
///
/// tracker.add_usage(&usage, "anthropic", "claude-sonnet-4-20250514");
///
/// // Get snapshot for display
/// let snapshot = tracker.get_snapshot();
/// println!("{}", snapshot.format_display());
/// ```
#[derive(Debug, Clone)]
pub struct UsageTracker {
    /// Inner aggregated usage (public for advanced use cases)
    pub inner: Arc<Mutex<AggregatedUsage>>,
}

impl UsageTracker {
    /// Create a new usage tracker.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AggregatedUsage::new())),
        }
    }

    /// Add usage from a single LLM request.
    ///
    /// Thread-safe operation that acquires a lock internally.
    ///
    /// # Arguments
    ///
    /// * `usage` - Token usage from LLM response
    /// * `provider` - Provider name (e.g., "anthropic", "openai")
    /// * `model` - Model identifier
    pub fn add_usage(&self, usage: &UnifiedUsage, provider: &str, model: &str) {
        let mut inner = self.inner.lock().unwrap();
        inner.add_usage(usage, provider, model);
    }

    /// Merge another aggregated usage into this tracker.
    ///
    /// Used to combine usage from multiple sessions or parallel operations.
    /// Does not recalculate costs - directly adds the pre-calculated values.
    ///
    /// # Arguments
    ///
    /// * `other` - Aggregated usage to merge
    pub fn merge_aggregated(&self, other: &AggregatedUsage) {
        let mut inner = self.inner.lock().unwrap();
        inner.total_input_tokens += other.total_input_tokens;
        inner.total_output_tokens += other.total_output_tokens;
        inner.total_cache_creation_tokens += other.total_cache_creation_tokens;
        inner.total_cache_read_tokens += other.total_cache_read_tokens;
        inner.total_reasoning_tokens += other.total_reasoning_tokens;
        inner.total_cost_usd += other.total_cost_usd;
        inner.request_count += other.request_count;
    }

    /// Get a snapshot of current usage.
    ///
    /// Returns a clone of the current aggregated usage. This is a relatively
    /// cheap operation as it only clones counters.
    pub fn get_snapshot(&self) -> AggregatedUsage {
        let inner = self.inner.lock().unwrap();
        inner.clone()
    }

    /// Format current usage for display.
    ///
    /// Convenience method that gets a snapshot and formats it.
    pub fn format_display(&self) -> String {
        let snapshot = self.get_snapshot();
        snapshot.format_display()
    }

    /// Format detailed usage breakdown.
    pub fn format_detailed(&self) -> String {
        let snapshot = self.get_snapshot();
        snapshot.format_detailed()
    }
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregated_usage_accumulation() {
        let mut usage = AggregatedUsage::new();

        let u1 = UnifiedUsage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_creation_input_tokens: Some(100),
            cache_read_input_tokens: Some(900),
            reasoning_tokens: None,
        };

        usage.add_usage(&u1, "anthropic", "claude-haiku-4-20250514");

        assert_eq!(usage.total_input_tokens, 1000);
        assert_eq!(usage.total_output_tokens, 500);
        assert_eq!(usage.total_cache_creation_tokens, 100);
        assert_eq!(usage.total_cache_read_tokens, 900);
        assert_eq!(usage.request_count, 1);
        assert!(usage.total_cost_usd > 0.0);

        // Add another usage
        let u2 = UnifiedUsage {
            input_tokens: 2000,
            output_tokens: 1000,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: Some(500),
            reasoning_tokens: None,
        };

        usage.add_usage(&u2, "anthropic", "claude-haiku-4-20250514");

        assert_eq!(usage.total_input_tokens, 3000);
        assert_eq!(usage.total_output_tokens, 1500);
        assert_eq!(usage.total_cache_creation_tokens, 100);
        assert_eq!(usage.total_cache_read_tokens, 1400);
        assert_eq!(usage.request_count, 2);
    }

    #[test]
    fn test_usage_tracker_thread_safe() {
        let tracker = UsageTracker::new();

        let usage = UnifiedUsage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            reasoning_tokens: None,
        };

        tracker.add_usage(&usage, "anthropic", "claude-haiku-4-20250514");

        let snapshot = tracker.get_snapshot();
        assert_eq!(snapshot.total_input_tokens, 1000);
        assert_eq!(snapshot.total_output_tokens, 500);
        assert_eq!(snapshot.request_count, 1);
    }

    #[test]
    fn test_format_display() {
        let mut usage = AggregatedUsage::new();

        let u = UnifiedUsage {
            input_tokens: 123_456,
            output_tokens: 67_890,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            reasoning_tokens: None,
        };

        usage.add_usage(&u, "anthropic", "claude-haiku-4-20250514");

        let display = usage.format_display();
        assert!(display.contains("K tokens"));
        assert!(display.contains("$"));
    }

    #[test]
    fn test_format_display_millions() {
        let mut usage = AggregatedUsage::new();
        usage.total_input_tokens = 1_500_000;
        usage.total_output_tokens = 500_000;
        usage.total_cost_usd = 15.5;

        let display = usage.format_display();
        assert!(display.contains("M tokens"));
        assert!(display.contains("$15.50"));
    }
}
