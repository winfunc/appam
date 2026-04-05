//! Anthropic Messages API streaming event types.
//!
//! Handles Server-Sent Events (SSE) from the Anthropic streaming API.
//!
//! # Event Flow
//!
//! 1. `message_start`: Response begins, empty content
//! 2. `content_block_start`: New content block (text/tool_use/thinking)
//! 3. `content_block_delta`: Incremental updates (text_delta/input_json_delta/thinking_delta)
//! 4. `content_block_stop`: Block finalized
//! 5. `message_delta`: Top-level updates (stop_reason, usage)
//! 6. `message_stop`: Response complete
//!
//! # Special Events
//!
//! - `ping`: Keepalive (no action needed)
//! - `error`: API error (stop processing)
//!
//! # Content Block Types
//!
//! - **text**: Plain text with `text_delta` events
//! - **tool_use**: Tool call with `input_json_delta` events (partial JSON)
//! - **thinking**: Reasoning with `thinking_delta` and `signature_delta` events
//! - **redacted_thinking**: Encrypted thinking (no deltas)

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use super::types::{ContentBlock, Usage};

/// Streaming event from Anthropic SSE.
///
/// Each SSE event has a `type` field that determines its structure.
/// Events are sent as:
/// ```text
/// event: message_start
/// data: {"type": "message_start", "message": {...}}
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Message started.
    MessageStart {
        /// Initial message with empty content
        message: MessageStart,
    },

    /// Content block started.
    ContentBlockStart {
        /// Block index in content array
        index: usize,
        /// Content block details
        content_block: ContentBlockStart,
    },

    /// Content block delta.
    ContentBlockDelta {
        /// Block index
        index: usize,
        /// Delta update
        delta: Delta,
    },

    /// Content block stopped.
    ContentBlockStop {
        /// Block index
        index: usize,
    },

    /// Message delta (top-level updates).
    MessageDelta {
        /// Delta updates
        delta: MessageDeltaData,
        /// Usage updates (cumulative)
        usage: Usage,
    },

    /// Message stopped.
    MessageStop,

    /// Ping (keepalive).
    Ping,

    /// Error.
    Error {
        /// Error details
        error: ErrorData,
    },
}

/// Message start data.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageStart {
    /// Unique identifier for the message
    pub id: String,
    /// Object type discriminator
    #[serde(rename = "type")]
    pub object_type: String,
    /// Role of the message sender (e.g., "assistant")
    pub role: String,
    /// Initial content blocks (may be empty at start)
    pub content: Vec<JsonValue>,
    /// Model used for generating the response
    pub model: String,
    /// Reason the model stopped generating (if known at start)
    pub stop_reason: Option<String>,
    /// Stop sequence that triggered completion (if any)
    pub stop_sequence: Option<String>,
    /// Token usage statistics
    pub usage: Usage,
}

/// Content block start data.
///
/// Indicates the type of block that's starting (text, tool_use, thinking).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockStart {
    /// Text block starting.
    Text {
        /// Initial text (empty)
        text: String,
    },

    /// Tool use block starting.
    ToolUse {
        /// Tool call ID
        id: String,
        /// Tool name
        name: String,
        /// Initial input (empty object)
        input: JsonValue,
    },

    /// Thinking block starting.
    Thinking {
        /// Initial thinking (empty)
        thinking: String,
    },
}

/// Delta update for content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    /// Text content delta.
    TextDelta {
        /// Text chunk
        text: String,
    },

    /// Tool input JSON delta.
    ///
    /// Provides PARTIAL JSON string. Must accumulate until `content_block_stop`
    /// and then parse as complete JSON.
    InputJsonDelta {
        /// Partial JSON string
        partial_json: String,
    },

    /// Thinking content delta.
    ThinkingDelta {
        /// Thinking chunk
        thinking: String,
    },

    /// Signature delta (for thinking blocks).
    ///
    /// Sent just before `content_block_stop` for thinking blocks.
    /// Used to verify thinking authenticity.
    SignatureDelta {
        /// Cryptographic signature
        signature: String,
    },
}

/// Message-level delta.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageDeltaData {
    /// Stop reason (when final)
    pub stop_reason: Option<String>,
    /// Stop sequence (if applicable)
    pub stop_sequence: Option<String>,
}

/// Error data in error events.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorData {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message
    pub message: String,
}

impl ErrorData {
    /// Check if this error is retryable.
    ///
    /// Returns true for transient errors that may succeed on retry:
    /// - `rate_limit_error` (429)
    /// - `overloaded_error` (529)
    /// - `api_error` (500) - Internal server errors
    ///
    /// All other errors are considered non-retryable.
    ///
    /// Note: HTTP status codes 502, 503, and 504 are also retried even if they
    /// don't have a structured error response. See `AnthropicClient::is_status_code_retryable`.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.error_type.as_str(),
            "rate_limit_error" | "overloaded_error" | "api_error"
        )
    }
}

/// Accumulated content block during streaming.
///
/// Used to track the state of a content block as deltas arrive.
#[derive(Debug, Clone, Default)]
pub struct AccumulatedBlock {
    /// Block type
    pub block_type: Option<String>,
    /// Accumulated text (for text blocks)
    pub text: String,
    /// Tool call ID (for tool_use blocks)
    pub tool_id: Option<String>,
    /// Tool name (for tool_use blocks)
    pub tool_name: Option<String>,
    /// Accumulated partial JSON (for tool_use blocks)
    pub partial_json: String,
    /// Initial tool input provided at block start (for parameterless tools)
    pub initial_input: Option<JsonValue>,
    /// Accumulated thinking (for thinking blocks)
    pub thinking: String,
    /// Signature (for thinking blocks)
    pub signature: Option<String>,
}

impl AccumulatedBlock {
    /// Create a new block from content_block_start event.
    pub fn from_start(start: &ContentBlockStart) -> Self {
        let mut block = Self::default();

        match start {
            ContentBlockStart::Text { .. } => {
                block.block_type = Some("text".to_string());
            }
            ContentBlockStart::ToolUse { id, name, input } => {
                block.block_type = Some("tool_use".to_string());
                block.tool_id = Some(id.clone());
                block.tool_name = Some(name.clone());
                if !input.is_null() {
                    block.initial_input = Some(input.clone());
                }
            }
            ContentBlockStart::Thinking { .. } => {
                block.block_type = Some("thinking".to_string());
            }
        }

        block
    }

    /// Apply a delta update to this block.
    pub fn apply_delta(&mut self, delta: &Delta) {
        match delta {
            Delta::TextDelta { text } => {
                self.text.push_str(text);
            }
            Delta::InputJsonDelta { partial_json } => {
                self.partial_json.push_str(partial_json);
            }
            Delta::ThinkingDelta { thinking } => {
                self.thinking.push_str(thinking);
            }
            Delta::SignatureDelta { signature } => {
                self.signature = Some(signature.clone());
            }
        }
    }

    /// Convert to a finalized ContentBlock.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulated data is incomplete or invalid
    /// (e.g., tool_use block with unparseable JSON).
    pub fn to_content_block(&self) -> Result<ContentBlock, serde_json::Error> {
        match self.block_type.as_deref() {
            Some("text") => Ok(ContentBlock::Text {
                text: self.text.clone(),
                cache_control: None,
            }),
            Some("tool_use") => {
                let input: JsonValue = if self.partial_json.trim().is_empty() {
                    self.initial_input.clone()
                } else {
                    serde_json::from_str(&self.partial_json).ok()
                }
                .or_else(|| self.initial_input.clone())
                .unwrap_or_else(|| JsonValue::Object(Default::default()));
                Ok(ContentBlock::ToolUse {
                    id: self.tool_id.clone().unwrap_or_default(),
                    name: self.tool_name.clone().unwrap_or_default(),
                    input,
                    cache_control: None,
                })
            }
            Some("thinking") => Ok(ContentBlock::Thinking {
                thinking: self.thinking.clone(),
                signature: self.signature.clone().unwrap_or_default(),
            }),
            _ => Ok(ContentBlock::Text {
                text: String::new(),
                cache_control: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_accumulated_block_text() {
        let start = ContentBlockStart::Text {
            text: String::new(),
        };
        let mut block = AccumulatedBlock::from_start(&start);

        block.apply_delta(&Delta::TextDelta {
            text: "Hello".to_string(),
        });
        block.apply_delta(&Delta::TextDelta {
            text: " world".to_string(),
        });

        assert_eq!(block.text, "Hello world");
        let content = block.to_content_block().unwrap();
        match content {
            ContentBlock::Text { text, .. } => assert_eq!(text, "Hello world"),
            _ => panic!("Expected text block"),
        }
    }

    #[test]
    fn test_accumulated_block_tool_use() {
        let start = ContentBlockStart::ToolUse {
            id: "call_123".to_string(),
            name: "get_weather".to_string(),
            input: json!({}),
        };
        let mut block = AccumulatedBlock::from_start(&start);

        block.apply_delta(&Delta::InputJsonDelta {
            partial_json: r#"{"location":"#.to_string(),
        });
        block.apply_delta(&Delta::InputJsonDelta {
            partial_json: r#" "Paris"}"#.to_string(),
        });

        let content = block.to_content_block().unwrap();
        match content {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "get_weather");
                assert_eq!(input["location"], "Paris");
            }
            _ => panic!("Expected tool_use block"),
        }
    }

    #[test]
    fn test_accumulated_block_tool_use_empty_input() {
        let start = ContentBlockStart::ToolUse {
            id: "call_456".to_string(),
            name: "read_notes".to_string(),
            input: json!({}),
        };
        let block = AccumulatedBlock::from_start(&start);

        let content = block.to_content_block().unwrap();
        match content {
            ContentBlock::ToolUse { input, .. } => {
                assert_eq!(input, json!({}));
            }
            _ => panic!("Expected tool_use block"),
        }
    }

    #[test]
    fn test_accumulated_block_tool_use_truncated_partial_fallback() {
        let start = ContentBlockStart::ToolUse {
            id: "call_789".to_string(),
            name: "read_todo_list".to_string(),
            input: json!({}),
        };
        let mut block = AccumulatedBlock::from_start(&start);

        // Simulate truncated partial JSON that would previously trigger EOF errors.
        block.partial_json = r#"{"note": "unfinished"#.to_string();

        let content = block.to_content_block().unwrap();
        match content {
            ContentBlock::ToolUse { input, .. } => {
                // We should gracefully fall back to the initial empty input.
                assert_eq!(input, json!({}));
            }
            _ => panic!("Expected tool_use block"),
        }
    }

    #[test]
    fn test_accumulated_block_thinking() {
        let start = ContentBlockStart::Thinking {
            thinking: String::new(),
        };
        let mut block = AccumulatedBlock::from_start(&start);

        block.apply_delta(&Delta::ThinkingDelta {
            thinking: "Let me think...".to_string(),
        });
        block.apply_delta(&Delta::SignatureDelta {
            signature: "sig123".to_string(),
        });

        let content = block.to_content_block().unwrap();
        match content {
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "Let me think...");
                assert_eq!(signature, "sig123");
            }
            _ => panic!("Expected thinking block"),
        }
    }
}
