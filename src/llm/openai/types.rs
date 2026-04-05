//! Type definitions for OpenAI Responses API.
//!
//! Complete type definitions matching the OpenAI Responses API contract,
//! including request parameters, response structures, streaming events,
//! and all content/output item types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Response API request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCreateParams {
    /// Model identifier
    pub model: String,

    /// Input (can be string or structured input items)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<ResponseInput>,

    /// System/developer instructions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Maximum output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,

    /// Tool definitions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Tool choice strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Enable parallel tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Maximum tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<i32>,

    /// Temperature (0.0-2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p sampling (0.0-1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Enable streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Stream options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Text output configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfig>,

    /// Reasoning configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,

    /// Service tier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Conversation ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<Conversation>,

    /// Previous response ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Background processing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    /// Store for later retrieval
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// Include fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,

    /// Truncation strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,

    /// Top logprobs (0-20)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<i32>,

    /// Metadata (max 16 key-value pairs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,

    /// Prompt cache key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,

    /// Safety identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

/// Input types for Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    /// Simple string input
    Simple(String),
    /// Structured input items
    Structured(Vec<InputItem>),
}

/// Input item types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    /// Message input
    Message {
        /// Message role
        role: MessageRole,
        /// Message content
        content: MessageContent,
        /// Optional status
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        /// Optional ID
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    /// Function call output (tool result)
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        /// Call ID matching function_call
        call_id: String,
        /// Function output
        output: String,
        /// Optional item ID
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Optional status
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    /// Function tool call (from previous response)
    #[serde(rename = "function_call")]
    FunctionToolCall {
        /// Item ID
        id: String,
        /// Call ID
        call_id: String,
        /// Function name
        name: String,
        /// Function arguments (JSON string)
        arguments: String,
        /// Optional status
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    /// Reasoning item
    Reasoning {
        /// Item ID
        id: String,
        /// Reasoning content
        #[serde(default)]
        content: Vec<ReasoningContent>,
        /// Reasoning summary
        #[serde(default)]
        summary: Vec<ReasoningContent>,
        /// Encrypted content (for stateless caching)
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// User message
    User,
    /// Assistant message
    Assistant,
    /// System message
    System,
    /// Developer message
    Developer,
}

/// Message content types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text
    Text(String),
    /// Content parts (multimodal)
    Parts(Vec<ContentPart>),
}

/// Content part types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text input
    #[serde(rename = "input_text")]
    InputText {
        /// Text content
        text: String,
    },
    /// Image input
    #[serde(rename = "input_image")]
    InputImage {
        /// Image URL or base64 data
        image_url: String,
        /// Detail level
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// File input
    #[serde(rename = "input_file")]
    InputFile {
        /// File ID
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// File URL
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
    },
    /// Text output
    #[serde(rename = "output_text")]
    OutputText {
        /// Text content
        text: String,
    },
}

/// Response structure from OpenAI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Response ID
    pub id: String,
    /// Creation timestamp (Unix time)
    pub created_at: f64,
    /// Object type (always "response")
    pub object: String,
    /// Model used
    pub model: String,
    /// Response status
    pub status: ResponseStatus,
    /// Output items
    pub output: Vec<OutputItem>,

    /// Instructions (echoed back)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Tools (echoed back)
    #[serde(default)]
    pub tools: Vec<Tool>,

    /// Tool choice (echoed back)
    #[serde(default)]
    pub tool_choice: ToolChoice,

    /// Parallel tool calls enabled
    #[serde(default)]
    pub parallel_tool_calls: bool,

    /// Temperature (echoed back)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p (echoed back)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Usage statistics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// Error information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,

    /// Incomplete details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,

    /// Conversation reference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<Conversation>,

    /// Previous response ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
}

/// Response status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Response completed successfully
    Completed,
    /// Response failed
    Failed,
    /// Response in progress
    InProgress,
    /// Response cancelled
    Cancelled,
    /// Response queued
    Queued,
    /// Response incomplete
    Incomplete,
}

/// Output item types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    /// Assistant message
    Message {
        /// Item ID
        id: String,
        /// Role (always assistant)
        role: String,
        /// Status
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        /// Content parts
        content: Vec<OutputContent>,
    },
    /// Function call
    #[serde(rename = "function_call")]
    FunctionCall {
        /// Item ID
        id: String,
        /// Call ID
        call_id: String,
        /// Function name
        name: String,
        /// Function arguments (JSON string)
        arguments: String,
        /// Status
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    /// Reasoning output
    Reasoning {
        /// Item ID
        id: String,
        /// Reasoning content
        #[serde(default)]
        content: Vec<ReasoningContent>,
        /// Reasoning summary
        #[serde(default)]
        summary: Vec<ReasoningContent>,
        /// Encrypted content
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

/// Output content types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContent {
    /// Text output
    #[serde(rename = "output_text")]
    OutputText {
        /// Text content
        text: String,
        /// Annotations (citations, file references)
        #[serde(default)]
        annotations: Vec<Annotation>,
        /// Logprobs
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<Logprob>>,
    },
    /// Refusal
    #[serde(rename = "output_refusal")]
    OutputRefusal {
        /// Refusal message
        refusal: String,
    },
}

/// Reasoning content types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningContent {
    /// Reasoning text
    #[serde(rename = "reasoning_text")]
    ReasoningText {
        /// Reasoning content
        text: String,
    },
    /// Summary text
    #[serde(rename = "summary_text")]
    SummaryText {
        /// Summary content
        text: String,
    },
}

/// Annotation types (citations, file references).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// File citation
    #[serde(rename = "file_citation")]
    FileCitation {
        /// File ID
        file_id: String,
        /// Filename
        filename: String,
        /// Citation index
        index: usize,
    },
    /// URL citation
    #[serde(rename = "url_citation")]
    UrlCitation {
        /// URL
        url: String,
        /// Title
        title: String,
        /// Start index in text
        start_index: usize,
        /// End index in text
        end_index: usize,
    },
}

/// Log probability information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprob {
    /// Token
    pub token: String,
    /// Token bytes
    pub bytes: Vec<u8>,
    /// Log probability
    pub logprob: f32,
    /// Top logprobs
    #[serde(default)]
    pub top_logprobs: Vec<TopLogprob>,
}

/// Top log probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogprob {
    /// Token
    pub token: String,
    /// Token bytes
    pub bytes: Vec<u8>,
    /// Log probability
    pub logprob: f32,
}

/// Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    /// Function tool
    Function {
        /// Function name
        name: String,
        /// Function description
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        /// Function parameters (JSON Schema)
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<serde_json::Value>,
        /// Strict mode
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// Tool choice options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String choice ("auto", "none", "required")
    String(String),
    /// Specific tool choice
    Specific(ToolChoiceSpecific),
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::String("auto".to_string())
    }
}

/// Specific tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoiceSpecific {
    /// Force specific function
    Function {
        /// Function to call
        function: FunctionChoice,
    },
}

/// Function choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionChoice {
    /// Function name
    pub name: String,
}

/// Stream options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Include obfuscation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}

/// Response text verbosity level.
///
/// Controls the verbosity of the model's response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TextVerbosity {
    /// Low verbosity (concise responses)
    Low,
    /// Medium verbosity (balanced responses, default)
    #[default]
    Medium,
    /// High verbosity (detailed responses)
    High,
}

/// Response text format configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTextFormat {
    /// Plain text
    Text,
    /// JSON object
    #[serde(rename = "json_object")]
    JsonObject,
    /// JSON schema
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// Schema name
        name: String,
        /// Schema description
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        /// JSON Schema
        schema: serde_json::Value,
        /// Strict mode
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// Response text configuration.
///
/// Configures the format and verbosity of text output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTextConfig {
    /// Text format (text, json_object, json_schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ResponseTextFormat>,

    /// Verbosity level (low, medium, high)
    ///
    /// Controls how verbose the model's response will be.
    /// Lower values result in more concise responses, while higher values
    /// result in more detailed responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<TextVerbosity>,
}

/// Reasoning configuration for the Responses API.
///
/// Controls reasoning effort and summary verbosity for o-series and gpt-5 models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reasoning {
    /// Reasoning effort level ("low", "medium", "high")
    ///
    /// Controls how many reasoning tokens the model generates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,

    /// Summary verbosity ("auto", "concise", "detailed")
    ///
    /// Controls the level of detail in the reasoning summary.
    /// This is a string value, not an object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Conversation reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Conversation {
    /// Conversation ID
    Id {
        /// Conversation ID
        id: String,
    },
    /// Simple ID string
    Simple(String),
}

/// Usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens
    pub input_tokens: i32,
    /// Input token details
    #[serde(default)]
    pub input_tokens_details: InputTokensDetails,
    /// Output tokens
    pub output_tokens: i32,
    /// Output token details
    #[serde(default)]
    pub output_tokens_details: OutputTokensDetails,
    /// Total tokens
    #[serde(default)]
    pub total_tokens: i32,
}

/// Input token details.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Cached tokens
    #[serde(default)]
    pub cached_tokens: i32,
}

/// Output token details.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Reasoning tokens
    #[serde(default)]
    pub reasoning_tokens: i32,
}

/// Response error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    /// Error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Error message
    pub message: String,
}

impl ResponseError {
    /// Returns true if this error should trigger a retry.
    ///
    /// Retryable error codes:
    /// - `internal_server_error` - Transient server issues (500)
    /// - `server_error` - General server errors
    /// - `timeout` - Request timeout
    /// - None/missing code with certain message patterns
    ///
    /// # Examples
    ///
    /// ```
    /// # use appam::llm::openai::types::ResponseError;
    /// let error = ResponseError {
    ///     code: Some("internal_server_error".to_string()),
    ///     message: "Internal server error".to_string(),
    /// };
    /// assert!(error.is_retryable());
    ///
    /// let error = ResponseError {
    ///     code: Some("invalid_request".to_string()),
    ///     message: "Invalid request".to_string(),
    /// };
    /// assert!(!error.is_retryable());
    /// ```
    pub fn is_retryable(&self) -> bool {
        if let Some(ref code) = self.code {
            matches!(
                code.as_str(),
                "internal_server_error" | "server_error" | "timeout"
            )
        } else {
            // If no code, check message for retryable patterns
            let msg = self.message.to_ascii_lowercase();
            msg.contains("internal server error")
                || msg.contains("server error")
                || msg.contains("timeout")
                || msg.contains("temporarily unavailable")
        }
    }
}

/// Incomplete details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteDetails {
    /// Incomplete reason
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_error_is_retryable_with_code() {
        let error = ResponseError {
            code: Some("internal_server_error".to_string()),
            message: "Internal server error".to_string(),
        };
        assert!(error.is_retryable());

        let error = ResponseError {
            code: Some("server_error".to_string()),
            message: "Server error".to_string(),
        };
        assert!(error.is_retryable());

        let error = ResponseError {
            code: Some("timeout".to_string()),
            message: "Request timeout".to_string(),
        };
        assert!(error.is_retryable());
    }

    #[test]
    fn test_response_error_is_retryable_without_code() {
        let error = ResponseError {
            code: None,
            message: "Internal server error occurred".to_string(),
        };
        assert!(error.is_retryable());

        let error = ResponseError {
            code: None,
            message: "Server error: timeout".to_string(),
        };
        assert!(error.is_retryable());

        let error = ResponseError {
            code: None,
            message: "Service temporarily unavailable".to_string(),
        };
        assert!(error.is_retryable());
    }

    #[test]
    fn test_response_error_not_retryable() {
        let error = ResponseError {
            code: Some("invalid_request".to_string()),
            message: "Invalid request".to_string(),
        };
        assert!(!error.is_retryable());

        let error = ResponseError {
            code: Some("authentication_error".to_string()),
            message: "Invalid API key".to_string(),
        };
        assert!(!error.is_retryable());

        let error = ResponseError {
            code: None,
            message: "Bad request format".to_string(),
        };
        assert!(!error.is_retryable());
    }

    #[test]
    fn test_usage_defaults_when_details_missing() {
        let payload = r#"{
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0}
        }"#;

        let usage: Usage = serde_json::from_str(payload).expect("usage payload should deserialize");
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
    }

    #[test]
    fn test_usage_with_all_fields_populated() {
        let payload = r#"{
            "input_tokens": 1500,
            "output_tokens": 750,
            "input_tokens_details": {
                "cached_tokens": 500
            },
            "output_tokens_details": {
                "reasoning_tokens": 250
            },
            "total_tokens": 2250
        }"#;

        let usage: Usage = serde_json::from_str(payload).expect("usage payload should deserialize");
        assert_eq!(usage.input_tokens, 1500);
        assert_eq!(usage.output_tokens, 750);
        assert_eq!(usage.input_tokens_details.cached_tokens, 500);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 250);
        assert_eq!(usage.total_tokens, 2250);
    }

    #[test]
    fn test_unified_usage_conversion_from_openai() {
        // Test conversion from OpenAI Usage to UnifiedUsage
        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 500,
            input_tokens_details: InputTokensDetails { cached_tokens: 200 },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 100,
            },
            total_tokens: 1500,
        };

        // Simulate the conversion logic in client.rs
        let input_tokens = usage.input_tokens.max(0) as u32;
        let output_tokens = usage.output_tokens.max(0) as u32;
        let cache_read_tokens = usage.input_tokens_details.cached_tokens.max(0) as u32;
        let reasoning_tokens = usage.output_tokens_details.reasoning_tokens.max(0) as u32;

        let unified = crate::llm::unified::UnifiedUsage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: (cache_read_tokens > 0).then_some(cache_read_tokens),
            reasoning_tokens: (reasoning_tokens > 0).then_some(reasoning_tokens),
        };

        assert_eq!(unified.input_tokens, 1000);
        assert_eq!(unified.output_tokens, 500);
        assert_eq!(unified.cache_read_input_tokens, Some(200));
        assert_eq!(unified.reasoning_tokens, Some(100));
        assert_eq!(unified.cache_creation_input_tokens, None);
    }

    #[test]
    fn test_usage_with_zeros() {
        // Test the actual payload structure from the user's example
        let payload = r#"{
            "input_tokens": 0,
            "output_tokens": 0,
            "output_tokens_details": {
                "reasoning_tokens": 0
            },
            "total_tokens": 0
        }"#;

        let usage: Usage = serde_json::from_str(payload).expect("usage payload should deserialize");
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);

        // Verify conversion to UnifiedUsage
        let input_tokens = usage.input_tokens.max(0) as u32;
        let output_tokens = usage.output_tokens.max(0) as u32;
        let cache_read_tokens = usage.input_tokens_details.cached_tokens.max(0) as u32;
        let reasoning_tokens = usage.output_tokens_details.reasoning_tokens.max(0) as u32;

        let unified = crate::llm::unified::UnifiedUsage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: (cache_read_tokens > 0).then_some(cache_read_tokens),
            reasoning_tokens: (reasoning_tokens > 0).then_some(reasoning_tokens),
        };

        // All fields should be zero or None when there's no usage
        assert_eq!(unified.input_tokens, 0);
        assert_eq!(unified.output_tokens, 0);
        assert_eq!(unified.cache_read_input_tokens, None);
        assert_eq!(unified.reasoning_tokens, None);
    }
}
