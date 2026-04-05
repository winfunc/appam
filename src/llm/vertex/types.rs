//! Vertex API request/response types.
//!
//! These structs intentionally model the subset of Vertex payloads required by
//! appam's agent runtime: multi-turn chat content, function calling, streaming,
//! and usage accounting.

use serde::{Deserialize, Serialize};

/// Request payload for `generateContent` and `streamGenerateContent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexGenerateContentRequest {
    /// Conversation history and current turn.
    pub contents: Vec<VertexContent>,

    /// Optional system instruction content.
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<VertexContent>,

    /// Tool/function declarations available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<VertexTool>>,

    /// Tool-calling controls.
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<VertexToolConfig>,

    /// Sampling and token controls.
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<VertexGenerationConfig>,
}

/// Conversation message for Vertex API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexContent {
    /// Role: typically `user` or `model`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Typed message parts.
    pub parts: Vec<VertexPart>,
}

/// Typed part inside a Vertex content message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexPart {
    /// Text payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Function-call request from the model.
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    pub function_call: Option<VertexFunctionCall>,

    /// Function-response payload returned by the caller.
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    pub function_response: Option<VertexFunctionResponse>,

    /// Thought signature associated with this part.
    #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,

    /// Whether this part is thought/reasoning content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
}

/// Function-call payload produced by Vertex.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexFunctionCall {
    /// Declared function name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Fully assembled arguments (non-streaming mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,

    /// Incremental argument updates (streaming mode).
    #[serde(rename = "partialArgs", skip_serializing_if = "Option::is_none")]
    pub partial_args: Option<Vec<VertexPartialArg>>,

    /// Indicates if additional argument fragments are expected.
    #[serde(rename = "willContinue", skip_serializing_if = "Option::is_none")]
    pub will_continue: Option<bool>,
}

/// Incremental argument fragment emitted by Vertex.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexPartialArg {
    /// JSON path location where this value applies.
    #[serde(rename = "jsonPath", skip_serializing_if = "Option::is_none")]
    pub json_path: Option<String>,

    /// String fragment.
    #[serde(rename = "stringValue", skip_serializing_if = "Option::is_none")]
    pub string_value: Option<String>,

    /// Numeric value.
    #[serde(rename = "numberValue", skip_serializing_if = "Option::is_none")]
    pub number_value: Option<f64>,

    /// Boolean value.
    #[serde(rename = "boolValue", skip_serializing_if = "Option::is_none")]
    pub bool_value: Option<bool>,

    /// Null marker.
    #[serde(rename = "nullValue", skip_serializing_if = "Option::is_none")]
    pub null_value: Option<serde_json::Value>,

    /// Structured value payload.
    #[serde(rename = "structValue", skip_serializing_if = "Option::is_none")]
    pub struct_value: Option<serde_json::Value>,

    /// List value payload.
    #[serde(rename = "listValue", skip_serializing_if = "Option::is_none")]
    pub list_value: Option<serde_json::Value>,

    /// Optional continuation hint for this fragment.
    #[serde(rename = "willContinue", skip_serializing_if = "Option::is_none")]
    pub will_continue: Option<bool>,
}

/// Function-response payload sent back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexFunctionResponse {
    /// Function name corresponding to a prior function call.
    pub name: String,

    /// Structured function result.
    pub response: serde_json::Value,
}

/// Tool declaration wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexTool {
    /// Function declarations available to the model.
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<VertexFunctionDeclaration>,
}

/// Function declaration schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexFunctionDeclaration {
    /// Function name.
    pub name: String,

    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON schema for parameters.
    pub parameters: serde_json::Value,
}

/// Tool configuration payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexToolConfig {
    /// Function-calling configuration.
    #[serde(rename = "functionCallingConfig")]
    pub function_calling_config: VertexFunctionCallingConfig,
}

/// Function-calling behavior controls.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexFunctionCallingConfig {
    /// Calling mode (`AUTO`, `ANY`, `NONE`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,

    /// Optional allow-list for callable functions.
    #[serde(
        rename = "allowedFunctionNames",
        skip_serializing_if = "Option::is_none"
    )]
    pub allowed_function_names: Option<Vec<String>>,

    /// Whether function args should stream incrementally.
    #[serde(
        rename = "streamFunctionCallArguments",
        skip_serializing_if = "Option::is_none"
    )]
    pub stream_function_call_arguments: Option<bool>,
}

/// Generation configuration subset used by appam.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexGenerationConfig {
    /// Temperature sampling parameter.
    #[serde(rename = "temperature", skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p sampling parameter.
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling parameter.
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Maximum output tokens.
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Optional thinking config.
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<crate::llm::vertex::config::VertexThinkingConfig>,
}

/// Vertex generation response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexGenerateContentResponse {
    /// Candidate responses.
    #[serde(default)]
    pub candidates: Vec<VertexCandidate>,

    /// Usage metadata.
    #[serde(rename = "usageMetadata", skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<VertexUsageMetadata>,
}

/// Candidate response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexCandidate {
    /// Candidate content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<VertexContent>,

    /// Finish reason.
    #[serde(rename = "finishReason", skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Usage metadata in Vertex responses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexUsageMetadata {
    /// Prompt token count.
    #[serde(rename = "promptTokenCount", skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<u32>,

    /// Candidate token count.
    #[serde(
        rename = "candidatesTokenCount",
        skip_serializing_if = "Option::is_none"
    )]
    pub candidates_token_count: Option<u32>,

    /// Thought token count (if available).
    #[serde(rename = "thoughtsTokenCount", skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<u32>,
}

/// Error payload returned by Vertex.
#[derive(Debug, Clone, Deserialize)]
pub struct VertexErrorResponse {
    /// Structured error object.
    pub error: VertexError,
}

/// Structured error details.
#[derive(Debug, Clone, Deserialize)]
pub struct VertexError {
    /// Error message.
    pub message: String,
    /// Optional status string.
    #[serde(default)]
    pub status: Option<String>,
}
