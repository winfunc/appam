//! Unified message format for provider-agnostic LLM interactions.
//!
//! This module defines canonical data structures that abstract over the differences
//! between OpenRouter's Responses API and Anthropic's Messages API. The agent
//! runtime operates solely on these unified types, enabling seamless provider
//! switching without code changes.
//!
//! # Design Principles
//!
//! - **Provider Agnostic**: Types represent capabilities common to both APIs
//! - **Lossless Conversion**: All provider-specific features map to unified types
//! - **Streaming Compatible**: Supports incremental updates and partial data
//! - **Extensible**: Easy to add new providers or content types
//!
//! # Mapping Strategy
//!
//! ## Anthropic → Unified
//! - `thinking` content block → `UnifiedContentBlock::Thinking`
//! - `tool_use` content block → `UnifiedContentBlock::ToolUse`
//! - `tool_result` content block → `UnifiedContentBlock::ToolResult`
//! - `text` content block → `UnifiedContentBlock::Text`
//! - `image` content block → `UnifiedContentBlock::Image`
//! - `document` content block → `UnifiedContentBlock::Document`
//!
//! ## OpenRouter → Unified
//! - `reasoning` output item → `UnifiedContentBlock::Thinking`
//! - `function_call` output item → `UnifiedContentBlock::ToolUse`
//! - `function_call_output` input item → `UnifiedContentBlock::ToolResult`
//! - `message` output item → `UnifiedContentBlock::Text`

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Role in a conversation.
///
/// Defines who is sending a message in the conversation flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnifiedRole {
    /// System instructions defining agent behavior
    System,
    /// User input/query
    User,
    /// Assistant (LLM) response
    Assistant,
}

/// Unified message in a conversation.
///
/// Represents a single message from any participant (system, user, or assistant)
/// with support for multimodal content, tool calls, and reasoning traces.
///
/// # Structure
///
/// Messages contain an array of content blocks rather than simple text strings.
/// This enables:
/// - Mixed text and images in a single message
/// - Tool use and results embedded in conversation
/// - Reasoning traces alongside responses
///
/// # Examples
///
/// Simple text message:
/// ```ignore
/// UnifiedMessage {
///     role: UnifiedRole::User,
///     content: vec![UnifiedContentBlock::Text {
///         text: "Hello!".to_string(),
///     }],
/// }
/// ```
///
/// Message with tool use:
/// ```ignore
/// UnifiedMessage {
///     role: UnifiedRole::Assistant,
///     content: vec![
///         UnifiedContentBlock::Text { text: "Let me check that.".to_string() },
///         UnifiedContentBlock::ToolUse {
///             id: "call_123".to_string(),
///             name: "get_weather".to_string(),
///             input: json!({"location": "Paris"}),
///         },
///     ],
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedMessage {
    /// Message role
    pub role: UnifiedRole,
    /// Content blocks (text, images, tool calls, etc.)
    pub content: Vec<UnifiedContentBlock>,
    /// Optional message ID (for conversation continuity)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Optional timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// Reasoning text (simple string, aggregated)
    /// Automatically preserved by appam for multi-turn conversations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Structured reasoning details (for preservation across tool calls)
    /// Automatically populated and preserved by appam
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Vec<crate::llm::openrouter::types::ReasoningDetail>>,
}

impl UnifiedMessage {
    /// Create a simple text message.
    pub fn text(role: UnifiedRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![UnifiedContentBlock::Text { text: text.into() }],
            id: None,
            timestamp: Some(chrono::Utc::now()),
            reasoning: None,
            reasoning_details: None,
        }
    }

    /// Create a user text message (convenience).
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(UnifiedRole::User, text)
    }

    /// Create a system message (convenience).
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(UnifiedRole::System, text)
    }

    /// Create an assistant text message (convenience).
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(UnifiedRole::Assistant, text)
    }

    /// Extract all text content from this message.
    ///
    /// Concatenates text from all `Text` content blocks, ignoring other block types.
    pub fn extract_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                UnifiedContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extract all tool calls from this message.
    pub fn extract_tool_calls(&self) -> Vec<UnifiedToolCall> {
        self.content
            .iter()
            .filter_map(|block| match block {
                UnifiedContentBlock::ToolUse { id, name, input } => Some(UnifiedToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    raw_input_json: Some(input.to_string()),
                }),
                _ => None,
            })
            .collect()
    }

    /// Check if this message contains any tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|block| matches!(block, UnifiedContentBlock::ToolUse { .. }))
    }

    /// Extract reasoning/thinking content.
    ///
    /// Concatenates all reasoning blocks in the order they appear, separated by
    /// newline characters. Returns `None` when the message contains no
    /// reasoning blocks.
    pub fn extract_reasoning(&self) -> Option<String> {
        let reasoning_blocks: Vec<&str> = self
            .content
            .iter()
            .filter_map(|block| match block {
                UnifiedContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                _ => None,
            })
            .collect();

        if reasoning_blocks.is_empty() {
            None
        } else {
            Some(reasoning_blocks.join("\n"))
        }
    }
}

/// Content block within a message.
///
/// Messages contain arrays of typed content blocks, enabling multimodal
/// interactions, tool calling, and reasoning traces.
///
/// # Block Types
///
/// - **Text**: Standard text content
/// - **Image**: Image data (base64 or URL)
/// - **Document**: Document/PDF data
/// - **ToolUse**: Model's request to invoke a tool
/// - **ToolResult**: Result of tool execution
/// - **Thinking**: Reasoning trace from extended thinking
///
/// # Provider Mapping
///
/// Different providers use different formats, but all map to these unified blocks:
///
/// **Anthropic:**
/// - `{"type": "text", "text": "..."}` → `Text`
/// - `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}` → `ToolUse`
/// - `{"type": "tool_result", "tool_use_id": "...", "content": "..."}` → `ToolResult`
/// - `{"type": "thinking", "thinking": "...", "signature": "..."}` → `Thinking`
///
/// **OpenRouter:**
/// - `message` output item → `Text`
/// - `function_call` output item → `ToolUse`
/// - `function_call_output` input item → `ToolResult`
/// - `reasoning` output item → `Thinking`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UnifiedContentBlock {
    /// Plain text content.
    Text {
        /// Text content
        text: String,
    },

    /// Image content (base64-encoded or URL).
    ///
    /// Supports JPEG, PNG, GIF, and WebP formats.
    Image {
        /// Image source (base64 data or URL)
        source: ImageSource,
        /// Optional detail level for vision models
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },

    /// Document content (PDF or plain text).
    Document {
        /// Document source
        source: DocumentSource,
        /// Optional title for the document
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },

    /// Tool use request from the model.
    ///
    /// When the LLM decides to invoke a tool, it generates this block with
    /// the tool name and arguments. The caller should execute the tool and
    /// return results via `ToolResult`.
    ToolUse {
        /// Unique identifier for this tool call
        id: String,
        /// Tool name to invoke
        name: String,
        /// Tool arguments (JSON object)
        input: JsonValue,
    },

    /// Tool execution result.
    ///
    /// Contains the output from executing a tool. Must reference the
    /// corresponding `ToolUse` block's ID.
    ToolResult {
        /// ID of the tool call this is a result for
        tool_use_id: String,
        /// Tool execution result (can be string or structured JSON)
        content: JsonValue,
        /// Whether the tool execution resulted in an error
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },

    /// Reasoning/thinking trace.
    ///
    /// Extended thinking or reasoning process from models that expose their
    /// internal reasoning. This content is separate from the main response.
    ///
    /// # Provider Differences
    ///
    /// - **Anthropic**: Thinking blocks include cryptographic signatures for verification
    /// - **OpenRouter**: Reasoning may be encrypted, with summary provided
    Thinking {
        /// Reasoning content (may be summarized)
        thinking: String,
        /// Optional signature for verification (Anthropic-specific)
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        /// Optional provider-specific encrypted continuation payload.
        ///
        /// OpenAI's Responses API can emit `reasoning.encrypted_content` so
        /// clients can replay reasoning items on later stateless turns. Appam
        /// preserves that opaque value here and only forwards it back to the
        /// originating provider.
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        /// Whether this thinking was redacted/encrypted
        #[serde(default)]
        redacted: bool,
    },
}

/// Image source specification.
///
/// Images can be provided as base64-encoded data or URLs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image data.
    Base64 {
        /// Media type (e.g., "image/jpeg", "image/png")
        media_type: String,
        /// Base64-encoded image data
        data: String,
    },
    /// Image URL.
    Url {
        /// Image URL
        url: String,
    },
}

/// Document source specification.
///
/// Documents can be PDFs (base64 or URL) or plain text.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Base64-encoded PDF.
    Base64Pdf {
        /// Media type (always "application/pdf")
        media_type: String,
        /// Base64-encoded PDF data
        data: String,
    },
    /// PDF URL.
    UrlPdf {
        /// PDF URL
        url: String,
    },
    /// Plain text document.
    Text {
        /// Media type (always "text/plain")
        media_type: String,
        /// Text content
        data: String,
    },
}

/// Unified tool specification.
///
/// Defines a tool/function that the LLM can invoke. Compatible with both
/// Anthropic's `input_schema` format and OpenRouter's `parameters` format.
///
/// # JSON Schema
///
/// The `parameters` field uses JSON Schema to define the expected input structure.
/// Both providers support the standard JSON Schema specification.
///
/// # Examples
///
/// ```ignore
/// use serde_json::json;
///
/// let weather_tool = UnifiedTool {
///     name: "get_weather".to_string(),
///     description: "Get current weather for a location".to_string(),
///     parameters: json!({
///         "type": "object",
///         "properties": {
///             "location": {
///                 "type": "string",
///                 "description": "City and state, e.g. San Francisco, CA"
///             },
///             "unit": {
///                 "type": "string",
///                 "enum": ["celsius", "fahrenheit"]
///             }
///         },
///         "required": ["location"]
///     }),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedTool {
    /// Tool name (must match `^[a-zA-Z0-9_-]{1,64}$` for Anthropic)
    pub name: String,
    /// Detailed description of what the tool does
    pub description: String,
    /// JSON Schema defining the tool's input parameters
    pub parameters: JsonValue,
}

/// Tool call requested by the LLM.
///
/// When the model decides to use a tool, it generates a tool call with
/// the function name and arguments as a JSON object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedToolCall {
    /// Unique identifier for this tool call
    pub id: String,
    /// Tool name to invoke
    pub name: String,
    /// Tool input arguments (JSON object)
    pub input: JsonValue,
    /// Raw JSON string as provided by the provider (may be partial during streaming)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_input_json: Option<String>,
}

impl UnifiedToolCall {
    /// Parse the input arguments as a specific type.
    ///
    /// # Errors
    ///
    /// Returns an error if the arguments cannot be deserialized into type `T`.
    pub fn parse_input<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_value(self.input.clone())
    }
}

/// Stop reason for response completion.
///
/// Indicates why the model stopped generating content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural completion (end of turn)
    EndTurn,
    /// Tool use requested
    ToolUse,
    /// Maximum token limit reached
    MaxTokens,
    /// Custom stop sequence encountered
    StopSequence,
    /// Model paused (server tool execution, can continue)
    PauseTurn,
    /// Content filtered/refused
    Refusal,
}

/// Token usage statistics.
///
/// Tracks input and output token consumption, including caching metrics
/// for providers that support prompt caching (Anthropic).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UnifiedUsage {
    /// Number of input tokens consumed
    pub input_tokens: u32,
    /// Number of output tokens generated
    pub output_tokens: u32,
    /// Number of tokens written to cache (Anthropic-specific)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// Number of tokens read from cache (Anthropic-specific)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    /// Number of reasoning tokens (may differ from visible output for summarized thinking)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

impl UnifiedUsage {
    /// Calculate total tokens consumed (input + output).
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// Calculate cache-adjusted input tokens.
    ///
    /// For Anthropic with caching: total input = regular input + cache creation - cache reads
    /// For others: just input_tokens
    pub fn effective_input_tokens(&self) -> u32 {
        let cache_creation = self.cache_creation_input_tokens.unwrap_or(0) as i64;
        let cache_read = self.cache_read_input_tokens.unwrap_or(0) as i64;
        let base = self.input_tokens as i64 + cache_creation - cache_read;
        if base <= 0 {
            0
        } else {
            base as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_unified_message_text() {
        let msg = UnifiedMessage::user("Hello, world!");
        assert_eq!(msg.role, UnifiedRole::User);
        assert_eq!(msg.content.len(), 1);
        assert_eq!(msg.extract_text(), "Hello, world!");
    }

    #[test]
    fn test_unified_message_tool_calls() {
        let msg = UnifiedMessage {
            role: UnifiedRole::Assistant,
            content: vec![
                UnifiedContentBlock::Text {
                    text: "Checking weather...".to_string(),
                },
                UnifiedContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({"location": "Paris"}),
                },
            ],
            id: None,
            timestamp: None,
            reasoning: None,
            reasoning_details: None,
        };

        assert!(msg.has_tool_calls());
        let calls = msg.extract_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn test_unified_tool_serialization() {
        let tool = UnifiedTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }),
        };

        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: UnifiedTool = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test_tool");
    }

    #[test]
    fn test_usage_calculations() {
        let usage = UnifiedUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: Some(200),
            cache_read_input_tokens: Some(150),
            reasoning_tokens: Some(30),
        };

        assert_eq!(usage.total_tokens(), 150);
        assert_eq!(usage.effective_input_tokens(), 150); // 100 + 200 - 150
    }

    #[test]
    fn test_extract_reasoning_multiple_blocks() {
        let msg = UnifiedMessage {
            role: UnifiedRole::Assistant,
            content: vec![
                UnifiedContentBlock::Thinking {
                    thinking: "Step 1".to_string(),
                    signature: None,
                    encrypted_content: None,
                    redacted: false,
                },
                UnifiedContentBlock::Thinking {
                    thinking: "Step 2".to_string(),
                    signature: None,
                    encrypted_content: None,
                    redacted: false,
                },
            ],
            id: None,
            timestamp: None,
            reasoning: None,
            reasoning_details: None,
        };

        assert_eq!(msg.extract_reasoning().unwrap(), "Step 1\nStep 2");
    }
}
