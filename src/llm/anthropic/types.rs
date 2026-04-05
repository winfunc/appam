//! Anthropic Messages API type definitions.
//!
//! Complete type system for the Anthropic Messages API, including requests,
//! responses, content blocks, tools, and streaming events.
//!
//! # API Version
//!
//! These types correspond to `anthropic-version: 2023-06-01`.
//!
//! # Key Differences from OpenRouter
//!
//! - System prompt is a top-level parameter, not a message role
//! - Content is an array of typed blocks, not simple strings
//! - Tool definitions use `input_schema` instead of `parameters`
//! - Streaming uses different event types
//! - Tool use is embedded in message content, not separate output items

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Messages API request.
///
/// Top-level request structure for creating a message with Claude.
///
/// # Required Fields
///
/// - `model`: Model identifier
/// - `max_tokens`: Maximum tokens to generate (strict limit)
/// - `messages`: Conversation history
///
/// # Important
///
/// - System prompt goes in `system` field, NOT as a message
/// - Messages must alternate user/assistant (consecutive turns auto-combined)
/// - `max_tokens` is enforced strictly (no auto-adjustment)
/// - Top-level `cache_control` is Anthropic's automatic prompt-caching helper
///   for the direct Anthropic and Azure Anthropic transports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRequest {
    /// Model identifier (e.g., "claude-sonnet-4-5")
    pub model: String,

    /// Maximum tokens to generate (strict limit)
    pub max_tokens: u32,

    /// Conversation messages
    ///
    /// Must alternate between user and assistant roles. Consecutive messages
    /// with the same role are automatically combined by the API.
    pub messages: Vec<Message>,

    /// Top-level prompt cache control.
    ///
    /// This mirrors Anthropic's request-level `cache_control` field. When
    /// present, Anthropic automatically applies the cache breakpoint to the
    /// last cacheable block in the request instead of requiring the client to
    /// manually inject `cache_control` into individual content blocks.
    ///
    /// The field is intentionally optional because requests without prompt
    /// caching should omit it entirely. AWS Bedrock uses block-level
    /// `cache_control` checkpoints instead of this top-level helper.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,

    /// System prompt (optional)
    ///
    /// Provides context and instructions. Can be a string or array of text blocks.
    /// This is NOT a message role - it's a top-level parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    /// Tool definitions (optional)
    ///
    /// Array of client tools and/or server tools (web search, bash, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Tool choice strategy (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Enable streaming (optional, default: false)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Temperature (0.0-1.0, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p sampling (0.0-1.0, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Custom stop sequences (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Extended thinking configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingParam>,

    /// Request metadata (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MetadataParam>,
}

/// System prompt parameter.
///
/// Can be either a simple string or an array of text blocks (for caching).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Simple string system prompt
    String(String),
    /// Array of system text blocks (for cache control)
    Blocks(Vec<SystemBlock>),
}

/// System text block.
///
/// Used when applying cache control to system prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBlock {
    /// Block type (always "text")
    #[serde(rename = "type")]
    pub block_type: String,
    /// Text content
    pub text: String,
    /// Optional cache control
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Cache control for prompt caching.
///
/// Marks content for caching to reduce latency and costs.
/// Can be applied to tools, system blocks, or message content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Cache type (always "ephemeral")
    #[serde(rename = "type")]
    pub cache_type: String,
    /// Time-to-live ("5m" or "1h")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl CacheControl {
    /// Create a cache control with 5-minute TTL.
    pub fn ephemeral_5m() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
            ttl: Some("5m".to_string()),
        }
    }

    /// Create a cache control with 1-hour TTL.
    pub fn ephemeral_1h() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
            ttl: Some("1h".to_string()),
        }
    }
}

/// Message in the conversation.
///
/// Contains a role and array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message role
    pub role: MessageRole,
    /// Content blocks
    pub content: Vec<ContentBlock>,
}

/// Message role.
///
/// Only user and assistant are allowed in messages array.
/// System is a top-level parameter, not a message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// User message
    User,
    /// Assistant message
    Assistant,
}

/// Content block in a message.
///
/// Messages contain arrays of typed content blocks.
///
/// # User Message Blocks
///
/// - `text`: Plain text
/// - `image`: Image (base64 or URL)
/// - `document`: Document/PDF
/// - `tool_result`: Result of tool execution
///
/// # Assistant Message Blocks
///
/// - `text`: Plain text response
/// - `tool_use`: Tool invocation request
/// - `thinking`: Reasoning trace (extended thinking)
/// - `redacted_thinking`: Encrypted reasoning
///
/// # Critical Ordering
///
/// In user messages with tool results, `tool_result` blocks MUST come
/// before any `text` blocks, or the API returns a 400 error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content.
    Text {
        /// Text content
        text: String,
        /// Optional cache control
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Image content.
    Image {
        /// Image source (base64 or URL)
        source: ImageSource,
        /// Optional cache control
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Document content (PDF or plain text).
    Document {
        /// Document source
        source: DocumentSource,
        /// Optional document title
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Optional cache control
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Tool use request (assistant only).
    ToolUse {
        /// Unique tool call ID
        id: String,
        /// Tool name
        name: String,
        /// Tool input (JSON object)
        input: JsonValue,
        /// Optional cache control
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Tool execution result (user only).
    ///
    /// MUST come before any text blocks in user messages.
    ToolResult {
        /// ID of the tool call this is a result for
        tool_use_id: String,
        /// Result content (string or array of content blocks)
        content: ToolResultContent,
        /// Whether this is an error result
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        /// Optional cache control
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    /// Thinking content (assistant only, from extended thinking).
    Thinking {
        /// Reasoning content (may be summarized in Claude 4)
        thinking: String,
        /// Cryptographic signature for verification
        signature: String,
    },

    /// Redacted thinking (assistant only).
    ///
    /// Thinking content flagged by safety systems is encrypted.
    /// Can be passed back to API for continuity but not human-readable.
    RedactedThinking {
        /// Encrypted thinking data
        data: String,
    },
}

/// Tool result content.
///
/// Tool results can be simple strings or arrays of content blocks (text, images).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// Simple string result
    String(String),
    /// Array of content blocks (text, image, document)
    Blocks(Vec<ToolResultBlock>),
}

/// Content block allowed in tool results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    /// Text content
    Text {
        /// The text content of the tool result
        text: String,
    },
    /// Image content
    Image {
        /// Image source specification (base64 or URL)
        source: ImageSource,
    },
    /// Document content
    Document {
        /// Document source specification
        source: DocumentSource,
    },
}

/// Image source specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image.
    Base64 {
        /// Media type (image/jpeg, image/png, image/gif, image/webp)
        media_type: String,
        /// Base64-encoded data
        data: String,
    },
    /// Image URL.
    Url {
        /// Image URL
        url: String,
    },
}

/// Document source specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Base64-encoded PDF.
    Base64 {
        /// Media type (application/pdf)
        media_type: String,
        /// Base64-encoded data
        data: String,
    },
    /// PDF URL.
    Url {
        /// PDF URL
        url: String,
    },
    /// Plain text document.
    Text {
        /// Media type (text/plain)
        media_type: String,
        /// Text content
        data: String,
    },
}

/// Tool definition.
///
/// Defines a tool that Claude can invoke.
///
/// # Tool Types
///
/// - **Client tools**: Custom tools (type omitted or "custom")
/// - **Server tools**: Anthropic-hosted tools (specific type strings)
///
/// Server tools:
/// - `web_search_20250305`: Web search
/// - `web_fetch_20250910`: Web page fetching
/// - `bash_20250124`: Bash command execution
/// - `code_execution_20250825`: Python code execution
/// - `text_editor_20250728`: File editing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool type (optional for client tools, required for server tools)
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,

    /// Tool name
    pub name: String,

    /// Tool description (strongly recommended)
    ///
    /// Should be detailed (3-4+ sentences) explaining:
    /// - What the tool does
    /// - When to use it
    /// - What each parameter means
    /// - Important caveats or limitations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Input schema (JSON Schema)
    ///
    /// Defines the shape of the `input` that Claude will generate.
    /// Only for client tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<JsonValue>,

    /// Optional cache control
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,

    // Server tool specific fields
    /// Max uses for server tools (web search, web fetch)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,

    /// Allowed domains for web search/fetch
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,

    /// Blocked domains for web search/fetch
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
}

impl Tool {
    /// Create a client tool definition.
    pub fn client(name: String, description: String, input_schema: JsonValue) -> Self {
        Self {
            tool_type: None,
            name,
            description: Some(description),
            input_schema: Some(input_schema),
            cache_control: None,
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    /// Create a web search server tool.
    pub fn web_search(max_uses: Option<u32>) -> Self {
        Self {
            tool_type: Some("web_search_20250305".to_string()),
            name: "web_search".to_string(),
            description: None,
            input_schema: None,
            cache_control: None,
            max_uses,
            allowed_domains: None,
            blocked_domains: None,
        }
    }

    /// Create a web fetch server tool.
    pub fn web_fetch() -> Self {
        Self {
            tool_type: Some("web_fetch_20250910".to_string()),
            name: "web_fetch".to_string(),
            description: None,
            input_schema: None,
            cache_control: None,
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        }
    }
}

/// Tool choice strategy.
///
/// Controls how Claude uses the provided tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolChoice {
    /// Claude decides whether to use tools (default)
    Auto {
        /// If true, prevents Claude from using multiple tools in parallel
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Claude must use at least one tool
    Any {
        /// If true, prevents Claude from using multiple tools in parallel
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Force Claude to use a specific tool
    Tool {
        /// Tool name to force
        name: String,
        /// If true, prevents Claude from using multiple tools in parallel
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    /// Claude cannot use tools
    None,
}

/// Extended thinking parameter.
///
/// Enables Claude's reasoning process with configurable token budget.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ThinkingParam {
    /// Thinking enabled
    Enabled {
        /// Token budget for reasoning (min: 1024, must be < max_tokens)
        budget_tokens: u32,
    },
    /// Thinking disabled
    Disabled,
}

/// Request metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataParam {
    /// External user identifier (UUID/hash, no PII)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// Messages API response.
///
/// Returned from the API for non-streaming requests, or accumulated
/// from streaming events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageResponse {
    /// Unique message ID
    pub id: String,

    /// Object type (always "message")
    #[serde(rename = "type")]
    pub object_type: String,

    /// Role (always "assistant")
    pub role: String,

    /// Content blocks
    pub content: Vec<ContentBlock>,

    /// Model that generated the response
    pub model: String,

    /// Stop reason
    pub stop_reason: String,

    /// Stop sequence (if stop_reason is "stop_sequence")
    pub stop_sequence: Option<String>,

    /// Token usage
    pub usage: Usage,
}

/// Token usage statistics.
///
/// Fields default to 0 when absent because different events include
/// different subsets of usage data:
/// - `message_start`: full breakdown (input, output, cache)
/// - `message_delta`: only `output_tokens` (cumulative)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Input tokens consumed
    #[serde(default)]
    pub input_tokens: u32,

    /// Output tokens generated
    #[serde(default)]
    pub output_tokens: u32,

    /// Tokens written to cache (if caching enabled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,

    /// Tokens read from cache (if caching enabled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

/// Error response from the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error type (always "error")
    #[serde(rename = "type")]
    pub error_type: String,

    /// Error details
    pub error: ErrorDetail,
}

/// Error detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,

    /// Human-readable error message
    pub message: String,

    /// Retry-after duration in seconds (for rate limit errors)
    ///
    /// When present, indicates how long to wait before retrying.
    /// This is typically provided by the API in rate limit responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u32>,
}

impl ErrorDetail {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_request_serialization() {
        let req = MessageRequest {
            model: "claude-sonnet-4-5".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: MessageRole::User,
                content: vec![ContentBlock::Text {
                    text: "Hello!".to_string(),
                    cache_control: None,
                }],
            }],
            cache_control: Some(CacheControl::ephemeral_5m()),
            system: Some(SystemPrompt::String("You are helpful.".to_string())),
            tools: None,
            tool_choice: None,
            stream: Some(true),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            metadata: None,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-sonnet-4-5");
        assert_eq!(json["max_tokens"], 1024);
        assert!(json["stream"].as_bool().unwrap());
        assert_eq!(json["cache_control"]["type"], "ephemeral");
        assert_eq!(json["cache_control"]["ttl"], "5m");
    }

    #[test]
    fn test_tool_creation() {
        let tool = Tool::client(
            "get_weather".to_string(),
            "Get weather".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        );

        assert_eq!(tool.name, "get_weather");
        assert!(tool.input_schema.is_some());
    }

    #[test]
    fn test_cache_control() {
        let cc = CacheControl::ephemeral_5m();
        assert_eq!(cc.cache_type, "ephemeral");
        assert_eq!(cc.ttl.as_ref().unwrap(), "5m");

        let cc_1h = CacheControl::ephemeral_1h();
        assert_eq!(cc_1h.ttl.as_ref().unwrap(), "1h");
    }

    #[test]
    fn test_tool_choice_serialization() {
        let auto = ToolChoice::Auto {
            disable_parallel_tool_use: Some(true),
        };
        let json = serde_json::to_value(&auto).unwrap();
        assert_eq!(json["type"], "auto");
        assert!(json["disable_parallel_tool_use"].as_bool().unwrap());

        let tool = ToolChoice::Tool {
            name: "get_weather".to_string(),
            disable_parallel_tool_use: None,
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "tool");
        assert_eq!(json["name"], "get_weather");
    }

    #[test]
    fn test_error_detail_is_retryable() {
        // Retryable errors
        let rate_limit_error = ErrorDetail {
            error_type: "rate_limit_error".to_string(),
            message: "Rate limit exceeded".to_string(),
            retry_after: Some(60),
        };
        assert!(rate_limit_error.is_retryable());

        let overloaded_error = ErrorDetail {
            error_type: "overloaded_error".to_string(),
            message: "Service overloaded".to_string(),
            retry_after: None,
        };
        assert!(overloaded_error.is_retryable());

        let api_error = ErrorDetail {
            error_type: "api_error".to_string(),
            message: "Internal server error".to_string(),
            retry_after: None,
        };
        assert!(api_error.is_retryable());

        // Non-retryable errors
        let invalid_request = ErrorDetail {
            error_type: "invalid_request_error".to_string(),
            message: "Invalid request".to_string(),
            retry_after: None,
        };
        assert!(!invalid_request.is_retryable());

        let auth_error = ErrorDetail {
            error_type: "authentication_error".to_string(),
            message: "Invalid API key".to_string(),
            retry_after: None,
        };
        assert!(!auth_error.is_retryable());

        let not_found = ErrorDetail {
            error_type: "not_found_error".to_string(),
            message: "Not found".to_string(),
            retry_after: None,
        };
        assert!(!not_found.is_retryable());
    }

    #[test]
    fn test_error_response_serialization() {
        let error_json = json!({
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded",
                "retry_after": 30
            }
        });

        let error_response: ErrorResponse = serde_json::from_value(error_json).unwrap();
        assert_eq!(error_response.error.error_type, "rate_limit_error");
        assert_eq!(error_response.error.message, "Rate limit exceeded");
        assert_eq!(error_response.error.retry_after, Some(30));
        assert!(error_response.error.is_retryable());
    }

    #[test]
    fn test_usage_with_cache_tokens() {
        // Test usage parsing with all optional cache fields populated
        let usage_json = json!({
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 800
        });

        let usage: Usage = serde_json::from_value(usage_json).unwrap();
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.output_tokens, 500);
        assert_eq!(usage.cache_creation_input_tokens, Some(200));
        assert_eq!(usage.cache_read_input_tokens, Some(800));
    }

    #[test]
    fn test_usage_without_cache_tokens() {
        // Test usage parsing when cache fields are absent
        let usage_json = json!({
            "input_tokens": 1000,
            "output_tokens": 500
        });

        let usage: Usage = serde_json::from_value(usage_json).unwrap();
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.output_tokens, 500);
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.cache_read_input_tokens, None);
    }

    #[test]
    fn test_unified_usage_conversion_from_anthropic() {
        // Test conversion from Anthropic Usage to UnifiedUsage
        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_creation_input_tokens: Some(200),
            cache_read_input_tokens: Some(800),
        };

        // Simulate the conversion logic in client.rs (MessageDelta handler)
        let unified = crate::llm::unified::UnifiedUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            reasoning_tokens: None, // Anthropic doesn't separate reasoning tokens
        };

        assert_eq!(unified.input_tokens, 1000);
        assert_eq!(unified.output_tokens, 500);
        assert_eq!(unified.cache_creation_input_tokens, Some(200));
        assert_eq!(unified.cache_read_input_tokens, Some(800));
        assert_eq!(unified.reasoning_tokens, None);
    }
}
