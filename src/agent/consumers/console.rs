//! Console output consumer with formatting and colors.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use crate::agent::streaming::{StreamConsumer, StreamEvent};

/// Console consumer with configurable formatting.
///
/// Outputs events to stdout with optional colors, reasoning display, and
/// detailed tool information. This is the default consumer for CLI applications.
///
/// # Examples
///
/// ```
/// use appam::agent::consumers::ConsoleConsumer;
///
/// let consumer = ConsoleConsumer::new()
///     .with_colors(true)
///     .with_reasoning(true)
///     .with_tool_details(true);
/// ```
pub struct ConsoleConsumer {
    /// Show reasoning traces
    show_reasoning: bool,
    /// Show detailed tool information
    show_tool_details: bool,
    /// Use ANSI color codes
    colored: bool,
    /// Track if we just emitted reasoning (for newline separation)
    last_was_reasoning: AtomicBool,
}

impl ConsoleConsumer {
    /// Create a new console consumer with default settings.
    ///
    /// Defaults: colors enabled, reasoning shown, tool details shown.
    pub fn new() -> Self {
        Self {
            show_reasoning: true,
            show_tool_details: true,
            colored: true,
            last_was_reasoning: AtomicBool::new(false),
        }
    }

    /// Enable or disable reasoning display.
    ///
    /// When enabled, shows model thinking/reasoning traces in cyan.
    pub fn with_reasoning(mut self, show: bool) -> Self {
        self.show_reasoning = show;
        self
    }

    /// Enable or disable detailed tool information.
    ///
    /// When enabled, shows tool names, durations, and status with icons.
    pub fn with_tool_details(mut self, show: bool) -> Self {
        self.show_tool_details = show;
        self
    }

    /// Enable or disable ANSI colors.
    ///
    /// When disabled, output is plain text without color codes.
    pub fn with_colors(mut self, enabled: bool) -> Self {
        self.colored = enabled;
        self
    }

    /// Apply color code if colors are enabled.
    fn color(&self, code: &str, text: &str) -> String {
        if self.colored {
            format!("{}{}\x1b[0m", code, text)
        } else {
            text.to_string()
        }
    }
}

impl Default for ConsoleConsumer {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamConsumer for ConsoleConsumer {
    fn on_event(&self, event: &StreamEvent) -> Result<()> {
        match event {
            StreamEvent::SessionStarted { .. } => {
                // Silent - no output for session start
                self.last_was_reasoning.store(false, Ordering::Relaxed);
            }

            StreamEvent::Content { content } => {
                // Add newline if transitioning from reasoning to content
                if self.last_was_reasoning.load(Ordering::Relaxed) {
                    println!(); // Separate reasoning from content
                    self.last_was_reasoning.store(false, Ordering::Relaxed);
                }
                print!("{}", content);
                io::stdout().flush()?;
            }

            StreamEvent::Reasoning { content } => {
                if self.show_reasoning {
                    print!("{}", self.color("\x1b[36m", content)); // Cyan
                    io::stdout().flush()?;
                    self.last_was_reasoning.store(true, Ordering::Relaxed);
                }
            }

            StreamEvent::ToolCallStarted {
                tool_name,
                arguments,
            } => {
                self.last_was_reasoning.store(false, Ordering::Relaxed);
                if self.show_tool_details {
                    println!();
                    println!(
                        "{}",
                        self.color("\x1b[33m", &format!("╭─ Tool Call: {}", tool_name))
                    );
                    if !arguments.is_empty() {
                        println!(
                            "{}",
                            self.color("\x1b[33m", &format!("├─ Arguments: {}", arguments))
                        );
                    }
                    println!("{}", self.color("\x1b[33m", "╰─ Executing..."));
                    io::stdout().flush()?;
                }
            }

            StreamEvent::ToolCallCompleted {
                tool_name,
                result,
                success,
                duration_ms,
            } => {
                if self.show_tool_details {
                    if *success {
                        println!(
                            "{}",
                            self.color(
                                "\x1b[32m",
                                &format!("✓ {} completed (took {:.2}ms)", tool_name, duration_ms)
                            )
                        );
                    } else {
                        println!(
                            "{}",
                            self.color(
                                "\x1b[31m",
                                &format!("✗ {} failed (took {:.2}ms)", tool_name, duration_ms)
                            )
                        );
                    }

                    // Show result if it has an "output" field
                    if let Some(output) = result.get("output") {
                        if let Some(output_str) = output.as_str() {
                            println!("{}", self.color("\x1b[34m", &format!("📤 {}", output_str)));
                        }
                    }

                    println!(); // Blank line after tool execution
                    io::stdout().flush()?;
                }
            }

            StreamEvent::ToolCallFailed { tool_name, error } => {
                if self.show_tool_details {
                    println!(
                        "{}",
                        self.color("\x1b[31m", &format!("✗ {} failed: {}", tool_name, error))
                    );
                    println!();
                    io::stdout().flush()?;
                }
            }

            StreamEvent::TurnCompleted => {
                // Silent - no output for turn completion
            }

            StreamEvent::UsageUpdate { .. } => {
                // Silent - usage is tracked but not displayed in console
            }

            StreamEvent::Done => {
                println!(); // Final newline
                io::stdout().flush()?;
            }

            StreamEvent::Error { message, .. } => {
                eprintln!("{}", self.color("\x1b[31m", &format!("Error: {}", message)));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_console_consumer_creation() {
        let consumer = ConsoleConsumer::new();
        assert!(consumer.colored);
        assert!(consumer.show_reasoning); // Default changed to true for Responses API
        assert!(consumer.show_tool_details);
        assert!(!consumer.last_was_reasoning.load(Ordering::Relaxed));
    }

    #[test]
    fn test_console_consumer_configuration() {
        let consumer = ConsoleConsumer::new()
            .with_colors(false)
            .with_reasoning(true)
            .with_tool_details(false);

        assert!(!consumer.colored);
        assert!(consumer.show_reasoning);
        assert!(!consumer.show_tool_details);
    }

    #[test]
    fn test_color_formatting() {
        let consumer = ConsoleConsumer::new().with_colors(true);
        let colored = consumer.color("\x1b[31m", "test");
        assert!(colored.contains("\x1b[31m"));
        assert!(colored.contains("test"));

        let consumer_no_color = ConsoleConsumer::new().with_colors(false);
        let plain = consumer_no_color.color("\x1b[31m", "test");
        assert_eq!(plain, "test");
    }
}
