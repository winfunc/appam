//! Shared types for OpenRouter APIs (Completions and Responses).
//!
//! Contains request/response structures, reasoning details, error types,
//! and tool calling definitions used across both APIs.

use serde::{Deserialize, Serialize};

use super::config::ProviderPreferences;

/// Reasoning detail format identifier.
///
/// Indicates which provider/format the reasoning detail uses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ReasoningFormat {
    /// Format is not specified
    Unknown,
    /// OpenAI responses format version 1
    #[serde(rename = "openai-responses-v1")]
    OpenaiResponsesV1,
    /// xAI responses format version 1
    #[serde(rename = "xai-responses-v1")]
    XaiResponsesV1,
    /// Anthropic Claude format version 1 (default)
    #[serde(rename = "anthropic-claude-v1")]
    #[default]
    AnthropicClaudeV1,
}
/// Structured reasoning detail from Completions API.
///
/// Reasoning details provide transparent insight into the model's reasoning process.
/// They come in three types: summary, encrypted, and text.
///
/// These details are automatically accumulated during streaming and preserved
/// across multi-turn conversations by the appam library.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningDetail {
    /// High-level summary of the reasoning process
    #[serde(rename = "reasoning.summary")]
    Summary {
        /// Summary text
        summary: String,
        /// Unique identifier
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Format identifier
        #[serde(default)]
        format: ReasoningFormat,
        /// Sequential index
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
    },
    /// Encrypted reasoning data (redacted/protected)
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        /// Base64-encoded encrypted data
        data: String,
        /// Unique identifier
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Format identifier
        #[serde(default)]
        format: ReasoningFormat,
        /// Sequential index
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
    },
    /// Raw text reasoning with optional signature
    #[serde(rename = "reasoning.text")]
    Text {
        /// Reasoning text
        text: String,
        /// Optional signature for verification
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        /// Unique identifier
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Format identifier
        #[serde(default)]
        format: ReasoningFormat,
        /// Sequential index
        #[serde(skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
    },
}

impl ReasoningDetail {
    /// Extract text content from reasoning detail for callback.
    ///
    /// This method is used internally to convert structured reasoning
    /// into simple strings for the `on_reasoning` callback.
    ///
    /// # Returns
    ///
    /// - Summary: Returns the summary text
    /// - Encrypted: Returns `"[REDACTED]"`
    /// - Text: Returns the reasoning text
    pub fn extract_text(&self) -> &str {
        match self {
            Self::Summary { summary, .. } => summary,
            Self::Encrypted { .. } => "[REDACTED]",
            Self::Text { text, .. } => text,
        }
    }
}

/// Tool specification for function calling (Responses API format).
///
/// This flat format is used by the OpenRouter Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub type_field: String,

    /// Function name
    pub name: String,

    /// Function description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for function parameters
    pub parameters: serde_json::Value,

    /// Strict mode (for structured outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Tool specification for Chat Completions API (OpenAI-compatible format).
///
/// This nested format is required by the Chat Completions API, where
/// the function details are wrapped in a `function` object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionsTool {
    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub type_field: String,

    /// Function specification
    pub function: ChatCompletionsToolFunction,
}

/// Function specification inside a ChatCompletionsTool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionsToolFunction {
    /// Function name
    pub name: String,

    /// Function description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for function parameters
    pub parameters: serde_json::Value,

    /// Strict mode (for structured outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl From<ToolSpec> for ChatCompletionsTool {
    fn from(spec: ToolSpec) -> Self {
        Self {
            type_field: spec.type_field,
            function: ChatCompletionsToolFunction {
                name: spec.name,
                description: spec.description,
                parameters: spec.parameters,
                strict: spec.strict,
            },
        }
    }
}

/// Tool call from assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique tool call identifier
    pub id: String,

    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub type_field: String,

    /// Function details
    pub function: ToolCallFunction,
}

/// Function details in a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Function name
    pub name: String,

    /// JSON-encoded function arguments
    pub arguments: String,
}

/// Tool choice configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String mode: "auto", "none", "required"
    String(String),

    /// Specific function to call
    Specific {
        /// Tool type
        #[serde(rename = "type")]
        type_field: String,

        /// Function specification
        function: ToolChoiceFunction,
    },
}

/// Function specification for tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    /// Function name to call
    pub name: String,
}

/// Error response from OpenRouter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error code
    pub code: i32,

    /// Error message
    pub message: String,

    /// Additional error metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Token usage statistics from OpenRouter API.
///
/// Contains detailed token counts including cached tokens, reasoning tokens,
/// and cost information. This structure matches OpenRouter's usage response
/// format when `usage: {include: true}` is enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of prompt/input tokens
    pub prompt_tokens: u32,

    /// Number of completion/output tokens
    pub completion_tokens: u32,

    /// Total tokens (prompt + completion)
    pub total_tokens: u32,

    /// Detailed prompt token breakdown
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,

    /// Detailed completion token breakdown
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,

    /// Cost in credits (OpenRouter-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,

    /// Detailed cost breakdown
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_details: Option<CostDetails>,

    /// Legacy cached tokens field (deprecated, use prompt_tokens_details)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

/// Detailed breakdown of prompt/input tokens.
///
/// Provides information about token caching for input tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    /// Number of tokens read from cache (reduces cost)
    #[serde(default)]
    pub cached_tokens: u32,

    /// Number of audio input tokens (for multimodal models)
    #[serde(default)]
    pub audio_tokens: u32,
}

/// Detailed breakdown of completion/output tokens.
///
/// Provides information about reasoning tokens for extended thinking models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    /// Number of reasoning tokens (for models with extended thinking)
    #[serde(default)]
    pub reasoning_tokens: u32,
}

/// Cost breakdown for OpenRouter requests.
///
/// Provides upstream inference costs for transparency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostDetails {
    /// Actual cost charged by upstream AI provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_inference_cost: Option<f64>,

    /// Upstream cost for input tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_inference_input_cost: Option<f64>,

    /// Upstream cost for output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_inference_output_cost: Option<f64>,
}

impl Usage {
    /// Convert OpenRouter usage to UnifiedUsage format.
    ///
    /// Maps OpenRouter's detailed usage structure to the unified format
    /// used by the appam library for cross-provider compatibility.
    pub fn to_unified(&self) -> crate::llm::unified::UnifiedUsage {
        crate::llm::unified::UnifiedUsage {
            input_tokens: self.prompt_tokens,
            output_tokens: self.completion_tokens,
            cache_read_input_tokens: self.prompt_tokens_details.as_ref().map(|d| d.cached_tokens),
            cache_creation_input_tokens: None, // OpenRouter doesn't provide cache write info
            reasoning_tokens: self
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens),
        }
    }
}

// ============================================================================
// Completions API Types
// ============================================================================

/// Chat completion request for Completions API.
///
/// This is the request body sent to `/api/v1/chat/completions`.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionRequest {
    /// Model identifier (e.g., "anthropic/claude-sonnet-4-5")
    pub model: String,

    /// Conversation messages
    pub messages: Vec<CompletionMessage>,

    /// Tool specifications (in nested function format for Chat Completions API)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionsTool>>,

    /// Tool choice configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Enable parallel tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    // Sampling parameters
    /// Temperature (0.0-2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Top-p sampling (0.0-1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Repetition penalty (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Minimum probability threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,

    /// Top-a threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f32>,

    /// Random seed for deterministic sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Reasoning configuration (NEW)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<serde_json::Value>,

    // OpenRouter-specific
    /// Provider routing preferences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,

    /// Prompt transforms
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transforms: Option<Vec<String>>,

    /// Fallback models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,

    /// Routing strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,

    /// Enable streaming
    pub stream: bool,

    /// Usage tracking configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            model: "openai/gpt-5".to_string(),
            messages: Vec::new(),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: Some(false),
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            min_p: None,
            top_a: None,
            seed: None,
            stop: None,
            reasoning: None,
            provider: None,
            transforms: None,
            models: None,
            route: None,
            stream: true,
            usage: None,
        }
    }
}

/// Completion message in Completions API format.
///
/// This is the message format used in conversation history.
/// The appam library automatically preserves `reasoning_details` when
/// building multi-turn conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionMessage {
    /// Message role: "system", "user", "assistant", "tool"
    pub role: String,

    /// Message content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Optional name (for function calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls (for assistant messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Tool call ID (for tool result messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    // Reasoning preservation (AUTOMATIC)
    /// Simple reasoning string (aggregated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// Structured reasoning details (for preservation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Vec<ReasoningDetail>>,
}

/// Streaming chunk from Completions API.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionChunk {
    /// Response ID
    pub id: String,

    /// Completion choices
    pub choices: Vec<CompletionChoice>,

    /// Unix timestamp
    pub created: i64,

    /// Model used
    pub model: String,

    /// Object type ("chat.completion.chunk")
    pub object: String,

    /// Token usage (present at end of stream)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Completion choice in streaming response.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionChoice {
    /// Choice index
    pub index: usize,

    /// Delta (incremental content)
    pub delta: CompletionDelta,

    /// Finish reason (present when complete)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,

    /// Provider's native finish reason
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_finish_reason: Option<String>,

    /// Error (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorResponse>,
}

/// Delta (incremental content) in streaming response.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionDelta {
    /// Message role (present at start)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Content chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool call deltas (partial)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,

    // Reasoning streaming
    /// Simple reasoning string (legacy)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// Structured reasoning details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Vec<ReasoningDetail>>,
}

/// Tool call delta in streaming response.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallDelta {
    /// Tool call index
    pub index: usize,

    /// Tool call ID (present at start)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Tool type (present at start)
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_field: Option<String>,

    /// Function delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Function call delta in streaming response.
#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name (present at start)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Arguments chunk (streamed incrementally)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Tool call accumulator for streaming.
///
/// Used internally to build complete tool calls from deltas.
#[derive(Debug, Clone)]
pub struct ToolCallBuilder {
    /// Tool call ID
    pub id: String,
    /// Function name
    pub name: String,
    /// Accumulated JSON arguments
    pub arguments: String,
}

impl ToolCallBuilder {
    /// Create a new tool call builder.
    pub fn new(id: String) -> Self {
        Self {
            id,
            name: String::new(),
            arguments: String::new(),
        }
    }

    /// Finalize into a complete tool call.
    pub fn finalize(self) -> ToolCall {
        ToolCall {
            id: self.id,
            type_field: "function".to_string(),
            function: ToolCallFunction {
                name: self.name,
                arguments: self.arguments,
            },
        }
    }
}
