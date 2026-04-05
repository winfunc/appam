//! LLM client types and abstractions for multiple providers.
//!
//! This module provides a unified interface for different LLM providers:
//! - **OpenRouter**: Responses API with reasoning and tool calling
//! - **Anthropic**: Messages API with extended thinking and prompt caching
//! - **OpenAI**: Responses API with reasoning, structured outputs, and service tiers
//! - **OpenAI Codex**: ChatGPT subscription-backed Codex Responses API
//! - **Vertex**: Gemini generateContent API with streaming and function calling
//!
//! # Provider Abstraction
//!
//! The `LlmClient` trait defines a common interface that all providers implement,
//! enabling seamless switching between providers with identical semantics.
//!
//! # Unified Format
//!
//! Internal types (`UnifiedMessage`, `UnifiedTool`, etc.) abstract over provider
//! differences. The agent runtime operates solely on these unified types.

pub mod anthropic;
pub mod openai;
pub mod openai_codex;
pub mod openrouter;
pub mod pricing;
pub mod provider;
pub mod unified;
pub mod usage;
pub mod vertex;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Role in a message for the Responses API.
///
/// The Responses API distinguishes between input roles (user, system, developer, assistant)
/// and output roles (assistant only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instructions defining agent behavior
    System,
    /// User input/query
    User,
    /// Assistant response (used in both input history and output)
    Assistant,
    /// Developer instructions (alternative to system)
    Developer,
    /// Tool execution result (legacy - converted to function_call_output)
    #[serde(rename = "tool")]
    Tool,
}

/// Status of a message or output item.
///
/// Used to track the completion state of assistant messages and output items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageStatus {
    /// Item is being processed
    #[serde(rename = "in_progress")]
    InProgress,
    /// Item completed successfully
    Completed,
    /// Item completed but truncated or incomplete
    Incomplete,
    /// Item failed to complete
    Failed,
}

/// Input content types for the Responses API.
///
/// Input messages contain an array of content items, each with a specific type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    /// Text input
    InputText {
        /// The text content to send
        text: String,
    },
    /// Image input (URL or base64)
    InputImage {
        /// URL of the image or base64-encoded data
        image_url: Option<String>,
        /// Level of detail for image analysis (e.g., "auto", "low", "high")
        detail: Option<String>,
    },
    /// File input
    InputFile {
        /// Unique identifier for the file
        file_id: Option<String>,
        /// Base64-encoded file data
        file_data: Option<String>,
        /// Name of the file
        filename: Option<String>,
        /// URL where the file can be accessed
        file_url: Option<String>,
    },
}

/// Output content types from the Responses API.
///
/// Output messages contain typed content with optional annotations (citations, file paths).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContent {
    /// Text output with annotations
    OutputText {
        /// The text content of the response
        text: String,
        /// Annotations for citations and file references within the text
        #[serde(default)]
        annotations: Vec<Annotation>,
    },
    /// Refusal message
    Refusal {
        /// The refusal message explaining why the model declined to respond
        refusal: String,
    },
}

/// Annotations for output content (citations, file references).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// URL citation with text range
    UrlCitation {
        /// Starting character index in the text where the citation begins
        start_index: usize,
        /// Ending character index in the text where the citation ends
        end_index: usize,
        /// URL being cited
        url: String,
        /// Title of the cited resource
        title: String,
    },
    /// File citation with index
    FileCitation {
        /// Unique identifier for the cited file
        file_id: String,
        /// Name of the cited file
        filename: String,
        /// Citation index for ordering multiple citations
        index: usize,
    },
    /// File path reference
    FilePath {
        /// Unique identifier for the referenced file
        file_id: String,
        /// Reference index for ordering multiple file paths
        index: usize,
    },
}

/// Input message for Responses API requests.
///
/// Messages sent to the API must use this format, with a role and content array.
/// Assistant messages in conversation history must include `id` and `status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMessage {
    /// Message type discriminator (always "message")
    #[serde(rename = "type")]
    pub type_field: String,
    /// Message role
    pub role: Role,
    /// Content array
    pub content: Vec<InputContent>,
    /// Message ID (required for assistant messages in history)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Message status (required for assistant messages in history)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
}

impl InputMessage {
    /// Create a simple user text message.
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            type_field: "message".to_string(),
            role: Role::User,
            content: vec![InputContent::InputText { text: text.into() }],
            id: None,
            status: None,
        }
    }

    /// Create a system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            type_field: "message".to_string(),
            role: Role::System,
            content: vec![InputContent::InputText { text: text.into() }],
            id: None,
            status: None,
        }
    }

    /// Create an assistant message with ID and status (for conversation history).
    pub fn assistant_completed(id: String, text: impl Into<String>) -> Self {
        Self {
            type_field: "message".to_string(),
            role: Role::Assistant,
            content: vec![InputContent::InputText { text: text.into() }],
            id: Some(id),
            status: Some(MessageStatus::Completed),
        }
    }
}

/// Output message from the Responses API.
///
/// The API returns messages with an ID, status, role, and content array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMessage {
    /// Output item type (always "message")
    #[serde(rename = "type")]
    pub type_field: String,
    /// Message ID
    pub id: String,
    /// Message role (always assistant for outputs)
    pub role: Role,
    /// Message status
    pub status: MessageStatus,
    /// Content parts
    pub content: Vec<OutputContent>,
}

/// Reasoning content for reasoning items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningContent {
    /// Plain reasoning text
    ReasoningText {
        /// The reasoning content showing the model's thinking process
        text: String,
    },
    /// Summary text
    SummaryText {
        /// A summary of the reasoning process
        text: String,
    },
}

/// Reasoning item from the Responses API.
///
/// When reasoning is enabled, the API returns separate reasoning items that
/// contain the model's internal thinking process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningItem {
    /// Type discriminator (always "reasoning")
    #[serde(rename = "type")]
    pub type_field: String,
    /// Reasoning item ID
    pub id: String,
    /// Reasoning content (may be encrypted)
    #[serde(default)]
    pub content: Vec<ReasoningContent>,
    /// Summary array
    #[serde(default)]
    pub summary: Vec<ReasoningContent>,
    /// Encrypted reasoning content (some models encrypt the reasoning chain)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    /// Signature for reasoning verification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// Function call item from the Responses API.
///
/// Function calls are separate output items (not embedded in messages).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallItem {
    /// Type discriminator (always "function_call")
    #[serde(rename = "type")]
    pub type_field: String,
    /// Function call item ID
    pub id: String,
    /// Call ID for pairing with function_call_output
    pub call_id: String,
    /// Function name to invoke
    pub name: String,
    /// JSON-encoded function arguments
    pub arguments: String,
    /// Call status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
}

/// Function call output for conversation history.
///
/// Used to send tool execution results back to the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallOutput {
    /// Type discriminator (always "function_call_output")
    #[serde(rename = "type")]
    pub type_field: String,
    /// Output item ID
    pub id: String,
    /// Call ID matching the function_call
    pub call_id: String,
    /// JSON-encoded function result
    pub output: String,
    /// Execution status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
}

/// Unified output item enum for all response types.
///
/// The Responses API output array contains a mix of messages, reasoning,
/// function calls, and other items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    /// Assistant message
    Message(OutputMessage),
    /// Reasoning item
    Reasoning(ReasoningItem),
    /// Function call
    FunctionCall(FunctionCallItem),
    /// Web search call
    WebSearchCall {
        /// Unique identifier for the web search call
        id: String,
        /// Status of the web search operation
        status: String,
    },
    /// File search call
    FileSearchCall {
        /// Unique identifier for the file search call
        id: String,
        /// Search queries to execute
        queries: Vec<String>,
        /// Status of the file search operation
        status: String,
    },
    /// Image generation call
    ImageGenerationCall {
        /// Unique identifier for the image generation call
        id: String,
        /// Generated image result (if completed)
        result: Option<String>,
        /// Status of the image generation operation
        status: String,
    },
}

/// Input item enum for conversation history.
///
/// Input can include messages, previous outputs, reasoning items, function calls,
/// and function call outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    /// Input message
    Message(InputMessage),
    /// Previous output message (for history)
    OutputMessage(OutputMessage),
    /// Previous reasoning item (for history)
    ReasoningItem(ReasoningItem),
    /// Function call (for history)
    FunctionCall(FunctionCallItem),
    /// Function call output
    FunctionCallOutput(FunctionCallOutput),
}

/// Internal chat message structure for agent use.
///
/// This is the simplified, unified message format used internally by agents.
/// It gets converted to/from the Responses API format as needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role
    pub role: Role,
    /// Optional name identifier (for tool calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool call ID (for tool response messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Message content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls requested by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Reasoning/thinking trace (legacy text-only format)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Raw content blocks from API response
    ///
    /// Preserves complete blocks including thinking signatures, multimodal content,
    /// and other structured data that can't be represented in legacy text fields.
    /// When present, this takes precedence over individual fields during reconstruction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_content_blocks: Option<Vec<crate::llm::unified::UnifiedContentBlock>>,
    /// Tool execution metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_metadata: Option<ToolExecutionMetadata>,
    /// Timestamp when this message was created
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// Message ID (for Responses API compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider-native response ID for APIs that support cross-turn continuation.
    ///
    /// OpenAI's Responses API exposes a top-level response ID that can be passed
    /// back as `previous_response_id` on the next turn. This field persists that
    /// identifier without overloading the per-message `id` field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_response_id: Option<String>,
    /// Message status (for Responses API compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
}

impl ChatMessage {
    /// Generate a unique message ID.
    pub fn generate_id() -> String {
        format!("msg_{}", Uuid::new_v4().simple())
    }

    /// Convert internal ChatMessage to Responses API InputItem format.
    ///
    /// This handles conversion of the unified internal format to the structured
    /// API format required by the Responses API.
    pub fn to_input_items(&self) -> Vec<InputItem> {
        let mut items = Vec::new();

        match self.role {
            Role::User | Role::System | Role::Developer => {
                // User/system messages: simple input message
                if let Some(ref content) = self.content {
                    items.push(InputItem::Message(InputMessage {
                        type_field: "message".to_string(),
                        role: self.role,
                        content: vec![InputContent::InputText {
                            text: content.clone(),
                        }],
                        id: None,
                        status: None,
                    }));
                }
            }
            Role::Assistant => {
                // Assistant messages: may have content and/or tool calls
                // Must include ID and status for conversation history
                let msg_id = self.id.clone().unwrap_or_else(Self::generate_id);
                let status = self.status.unwrap_or(MessageStatus::Completed);

                // Add reasoning item if present
                if let Some(ref reasoning) = self.reasoning {
                    items.push(InputItem::ReasoningItem(ReasoningItem {
                        type_field: "reasoning".to_string(),
                        id: format!("reasoning_{}", Uuid::new_v4().simple()),
                        content: vec![ReasoningContent::ReasoningText {
                            text: reasoning.clone(),
                        }],
                        summary: vec![],
                        encrypted_content: None,
                        signature: None,
                    }));
                }

                // Add message if content present
                if let Some(ref content) = self.content {
                    items.push(InputItem::Message(InputMessage {
                        type_field: "message".to_string(),
                        role: Role::Assistant,
                        content: vec![InputContent::InputText {
                            text: content.clone(),
                        }],
                        id: Some(msg_id.clone()),
                        status: Some(status),
                    }));
                }

                // Add function calls if present
                if let Some(ref tool_calls) = self.tool_calls {
                    for tc in tool_calls {
                        items.push(InputItem::FunctionCall(FunctionCallItem {
                            type_field: "function_call".to_string(),
                            id: format!("fc_{}", Uuid::new_v4().simple()),
                            call_id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                            status: Some(MessageStatus::Completed),
                        }));
                    }
                }
            }
            Role::Tool => {
                // Tool messages: convert to function_call_output
                if let Some(ref content) = self.content {
                    items.push(InputItem::FunctionCallOutput(FunctionCallOutput {
                        type_field: "function_call_output".to_string(),
                        id: format!("fc_output_{}", Uuid::new_v4().simple()),
                        call_id: self.tool_call_id.clone().unwrap_or_default(),
                        output: content.clone(),
                        status: Some(MessageStatus::Completed),
                    }));
                }
            }
        }

        items
    }

    /// Create ChatMessage from Responses API output items.
    ///
    /// Reconstructs internal message format from API output items.
    /// Groups reasoning, messages, and function calls into unified structure.
    pub fn from_output_items(items: &[ResponsesOutputItem]) -> Vec<Self> {
        let mut messages = Vec::new();
        let mut current_reasoning: Option<String> = None;

        for item in items {
            match item {
                ResponsesOutputItem::Message(OutputMessage {
                    id,
                    role,
                    status,
                    content,
                    ..
                }) => {
                    // Extract text from content array
                    let text = content
                        .iter()
                        .filter_map(|c| match c {
                            OutputContent::OutputText { text, .. } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    messages.push(ChatMessage {
                        role: *role,
                        name: None,
                        tool_call_id: None,
                        content: Some(text),
                        tool_calls: None,
                        reasoning: current_reasoning.take(),
                        raw_content_blocks: None,
                        tool_metadata: None,
                        timestamp: Some(chrono::Utc::now()),
                        id: Some(id.clone()),
                        provider_response_id: None,
                        status: Some(*status),
                    });
                }
                ResponsesOutputItem::Reasoning(ReasoningItem {
                    content, summary, ..
                }) => {
                    // Accumulate reasoning for next message
                    let reasoning_texts: Vec<String> = content
                        .iter()
                        .filter_map(|c| match c {
                            ReasoningContent::ReasoningText { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect();

                    let summary_texts: Vec<String> = summary
                        .iter()
                        .filter_map(|c| match c {
                            ReasoningContent::SummaryText { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect();

                    let mut full_reasoning = String::new();
                    if !summary_texts.is_empty() {
                        full_reasoning.push_str("Summary:\n");
                        full_reasoning.push_str(&summary_texts.join("\n"));
                        full_reasoning.push_str("\n\n");
                    }
                    if !reasoning_texts.is_empty() {
                        full_reasoning.push_str(&reasoning_texts.join("\n"));
                    }

                    current_reasoning = Some(full_reasoning);
                }
                ResponsesOutputItem::FunctionCall(FunctionCallItem {
                    call_id,
                    name,
                    arguments,
                    id,
                    ..
                }) => {
                    // Create tool call message
                    messages.push(ChatMessage {
                        role: Role::Assistant,
                        name: None,
                        tool_call_id: None,
                        content: None,
                        tool_calls: Some(vec![ToolCall {
                            id: call_id.clone(),
                            type_field: "function".to_string(),
                            function: ToolCallFunction {
                                name: name.clone(),
                                arguments: arguments.clone(),
                            },
                        }]),
                        reasoning: current_reasoning.take(),
                        raw_content_blocks: None,
                        tool_metadata: None,
                        timestamp: Some(chrono::Utc::now()),
                        id: Some(id.clone()),
                        provider_response_id: None,
                        status: Some(MessageStatus::Completed),
                    });
                }
                _ => {
                    // Handle other item types as needed
                }
            }
        }

        messages
    }
}

/// Tool/function specification compatible with Responses API.
///
/// Defines the schema for a tool that can be invoked by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Type field, always "function"
    #[serde(rename = "type")]
    pub type_field: String,
    /// Function name
    pub name: String,
    /// Function description
    pub description: String,
    /// JSON Schema for parameters
    pub parameters: serde_json::Value,
    /// Strict mode (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Tool call requested by the model.
///
/// When the LLM decides to use a tool, it emits a ToolCall with the function
/// name and JSON-encoded arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: String,
    /// Type field, always "function"
    #[serde(rename = "type")]
    pub type_field: String,
    /// Function call details
    pub function: ToolCallFunction,
}

/// Function call details within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Function name to invoke
    pub name: String,
    /// JSON-encoded function arguments
    pub arguments: String,
}

/// A single streamed delta chunk (legacy compatibility).
///
/// During streaming responses, the LLM sends incremental updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    /// Incremental content chunk
    #[serde(default)]
    pub content: Option<String>,
    /// Tool call deltas
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    /// Finish reason
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Reasoning content
    #[serde(default)]
    pub reasoning: Option<String>,
}

/// Incremental tool call update during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Index in the tool calls array
    #[serde(default)]
    pub index: Option<usize>,
    /// Tool call ID
    pub id: Option<String>,
    /// Function delta
    pub function: Option<ToolCallFunctionDelta>,
}

/// Function delta within a tool call delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunctionDelta {
    /// Function name
    pub name: Option<String>,
    /// Incremental arguments chunk
    pub arguments: Option<String>,
}

/// Tool execution metadata.
///
/// Stores information about tool execution for audit trails and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionMetadata {
    /// Whether the tool execution succeeded
    pub success: bool,
    /// Execution duration in milliseconds
    pub duration_ms: f64,
    /// Tool name that was executed
    pub tool_name: String,
    /// Arguments passed to the tool
    pub arguments: String,
}

// Re-exports for unified interface
pub use provider::{DynamicLlmClient, LlmClient, LlmProvider, ProviderFailureCapture};
pub use unified::{
    DocumentSource, ImageSource, StopReason, UnifiedContentBlock, UnifiedMessage, UnifiedRole,
    UnifiedTool, UnifiedToolCall, UnifiedUsage,
};
