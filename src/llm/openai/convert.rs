//! Conversion between unified messages and OpenAI Responses API format.
//!
//! Provides bidirectional conversion functions to translate between the
//! appam unified message format and OpenAI's specific request/response structures.

use serde_json::json;
use uuid::Uuid;

use super::types::*;
use crate::llm::unified::{
    UnifiedContentBlock, UnifiedMessage, UnifiedRole, UnifiedTool, UnifiedToolCall,
};

/// Convert unified messages to OpenAI input format.
///
/// Transforms Appam's unified message representation into OpenAI's structured
/// Responses API input items.
///
/// # Continuation behavior
///
/// When `previous_response_id` is present, OpenAI expects callers to send only
/// the new items that happened after the anchored assistant turn, not a second
/// copy of the full transcript. Appam therefore trims the replay window to the
/// post-anchor delta in the runtime's message history.
///
/// System instructions are intentionally excluded from this replay window. The
/// caller should pass them separately via the top-level `instructions` request
/// field when needed, which keeps tool-call continuations ordered correctly for
/// the Responses API validator.
///
/// If no assistant anchor can be inferred, the full transcript is sent as a
/// defensive fallback.
///
/// # Arguments
///
/// * `messages` - Unified conversation history in chronological order
/// * `previous_response_id` - Optional OpenAI response ID used for continuation
///
/// # Returns
///
/// A Responses API `input` payload ready for JSON serialization.
pub fn from_unified_messages(
    messages: &[UnifiedMessage],
    previous_response_id: Option<&str>,
) -> ResponseInput {
    let selected_messages = select_messages_for_openai_input(messages, previous_response_id);
    let input_items: Vec<InputItem> = selected_messages
        .iter()
        .flat_map(|msg| message_to_input_items(msg))
        .collect();

    ResponseInput::Structured(input_items)
}

/// Extract system messages into a single OpenAI `instructions` payload.
///
/// The Responses API treats system/developer guidance as top-level
/// instructions. Consolidating them outside the `input` item stream avoids
/// interleaving system messages with tool-result continuations.
pub fn extract_instructions(messages: &[UnifiedMessage]) -> Option<String> {
    let instructions = messages
        .iter()
        .filter_map(|message| {
            matches!(message.role, UnifiedRole::System)
                .then(|| message.extract_text())
                .filter(|text| !text.trim().is_empty())
        })
        .collect::<Vec<_>>();

    if instructions.is_empty() {
        None
    } else {
        Some(instructions.join("\n\n"))
    }
}

/// Select the subset of messages that should be serialized into the next
/// OpenAI Responses API request.
fn select_messages_for_openai_input<'a>(
    messages: &'a [UnifiedMessage],
    previous_response_id: Option<&str>,
) -> Vec<&'a UnifiedMessage> {
    if previous_response_id.is_none() {
        return messages.iter().collect();
    }

    let Some(last_assistant_idx) = messages
        .iter()
        .rposition(|msg| msg.role == UnifiedRole::Assistant)
    else {
        return messages.iter().collect();
    };

    messages
        .iter()
        .enumerate()
        .filter_map(|(idx, msg)| (idx > last_assistant_idx).then_some(msg))
        .collect()
}

/// Convert a single unified message to OpenAI input items.
fn message_to_input_items(msg: &UnifiedMessage) -> Vec<InputItem> {
    let mut items = Vec::new();

    let role = match msg.role {
        UnifiedRole::User => MessageRole::User,
        UnifiedRole::Assistant => MessageRole::Assistant,
        UnifiedRole::System => MessageRole::System,
    };

    // Extract content blocks by type
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();
    let mut reasoning_items = Vec::new();

    for block in &msg.content {
        match block {
            UnifiedContentBlock::Text { text } => {
                // OpenAI Responses API requires different content types:
                // - User/System messages use input_text
                // - Assistant messages use output_text
                if msg.role == UnifiedRole::Assistant {
                    text_parts.push(ContentPart::OutputText { text: text.clone() });
                } else {
                    text_parts.push(ContentPart::InputText { text: text.clone() });
                }
            }
            UnifiedContentBlock::Image { source, detail } => {
                // Convert image source to URL
                let image_url = match source {
                    crate::llm::unified::ImageSource::Base64 { media_type, data } => {
                        format!("data:{};base64,{}", media_type, data)
                    }
                    crate::llm::unified::ImageSource::Url { url } => url.clone(),
                };
                text_parts.push(ContentPart::InputImage {
                    image_url,
                    detail: detail.clone(),
                });
            }
            UnifiedContentBlock::ToolUse { id, name, input } => {
                tool_calls.push((id.clone(), name.clone(), input.clone()));
            }
            UnifiedContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                tool_results.push((tool_use_id.clone(), content.clone()));
            }
            UnifiedContentBlock::Thinking {
                thinking,
                encrypted_content,
                ..
            } => {
                if msg.role == UnifiedRole::Assistant {
                    let content = if !thinking.is_empty() {
                        vec![ReasoningContent::ReasoningText {
                            text: thinking.clone(),
                        }]
                    } else {
                        Vec::new()
                    };

                    reasoning_items.push(InputItem::Reasoning {
                        id: format!("rs_{}", Uuid::new_v4().simple()),
                        content,
                        summary: Vec::new(),
                        encrypted_content: encrypted_content.clone(),
                    });
                }
            }
            _ => {}
        }
    }

    // Add message if there's text content
    if !text_parts.is_empty() {
        items.push(InputItem::Message {
            role,
            content: MessageContent::Parts(text_parts),
            status: if msg.role == UnifiedRole::Assistant {
                Some("completed".to_string())
            } else {
                None
            },
            id: msg.id.clone(),
        });
    }

    // Add function calls
    for (id, name, input) in tool_calls {
        items.push(InputItem::FunctionToolCall {
            id: format!("fc_{}", Uuid::new_v4().simple()),
            call_id: id,
            name,
            arguments: serde_json::to_string(&input).unwrap_or_default(),
            status: Some("completed".to_string()),
        });
    }

    // Add function call outputs
    for (call_id, content) in tool_results {
        let output = serde_json::to_string(&content).unwrap_or_else(|_| {
            // If content is not valid JSON, treat as plain text
            match content {
                serde_json::Value::String(s) => s,
                _ => content.to_string(),
            }
        });

        items.push(InputItem::FunctionCallOutput {
            call_id,
            output,
            id: None,
            status: Some("completed".to_string()),
        });
    }

    items.extend(reasoning_items);

    items
}

/// Convert unified tools to OpenAI format.
///
/// Transforms appam's unified tool specifications into OpenAI's
/// function tool format.
///
/// OpenAI requires that all parameter schemas include `"additionalProperties": false`
/// at the root level. This function ensures this requirement is met by injecting
/// the field if it's not already present.
///
/// Note: Strict mode is disabled by default for broader compatibility. When strict mode
/// is enabled, OpenAI requires all properties to be listed in the `required` array.
pub fn from_unified_tools(tools: &[UnifiedTool]) -> Vec<Tool> {
    use tracing::debug;

    tools
        .iter()
        .map(|tool| {
            // Extract the actual parameters schema
            // Handle case where tool.parameters might be a full tool spec or just the schema
            let params_obj = if let Some(obj) = tool.parameters.as_object() {
                // Check if this is a full tool spec (has "parameters" field)
                if let Some(inner_params) = obj.get("parameters") {
                    // This is a full spec like {"type": "function", "name": "...", "parameters": {...}}
                    // Extract the inner parameters object
                    debug!(
                        tool_name = %tool.name,
                    "Extracting inner parameters from full tool spec"
                    );
                    inner_params.as_object().cloned()
                } else {
                    // This is already just the parameters schema
                    Some(obj.clone())
                }
            } else {
                None
            };

            // Ensure parameters schema has additionalProperties: false for OpenAI
            let parameters = if let Some(mut params) = params_obj {
                // Add additionalProperties: false if not already present
                if !params.contains_key("additionalProperties") {
                    debug!(
                    tool_name = %tool.name,
                        "Adding additionalProperties: false to tool schema for OpenAI compatibility"
                    );
                    params.insert(
                        "additionalProperties".to_string(),
                        serde_json::Value::Bool(false),
                    );
                }
                Some(serde_json::Value::Object(params))
            } else {
                Some(tool.parameters.clone())
            };

            Tool::Function {
                name: tool.name.clone(),
                description: Some(tool.description.clone()),
                parameters,
                // Don't specify strict mode - let OpenAI use default behavior
                // This allows tools to work without strict schema validation
                strict: None,
            }
        })
        .collect()
}

/// Convert OpenAI output items to unified tool calls.
///
/// Extracts function calls from OpenAI's output format and converts
/// them to appam's unified tool call representation.
pub fn to_unified_tool_calls(output_items: &[OutputItem]) -> Vec<UnifiedToolCall> {
    output_items
        .iter()
        .filter_map(|item| {
            if let OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } = item
            {
                let input = serde_json::from_str(arguments).unwrap_or(json!({}));
                Some(UnifiedToolCall {
                    id: call_id.clone(),
                    name: name.clone(),
                    input,
                    raw_input_json: Some(arguments.clone()),
                })
            } else {
                None
            }
        })
        .collect()
}

/// Convert OpenAI output items to unified content blocks.
///
/// Transforms OpenAI's response output into appam's unified content block format.
pub fn to_unified_content_blocks(output_items: &[OutputItem]) -> Vec<UnifiedContentBlock> {
    let mut blocks = Vec::new();

    for item in output_items {
        match item {
            OutputItem::Message { content, .. } => {
                for content_item in content {
                    match content_item {
                        OutputContent::OutputText { text, .. } => {
                            blocks.push(UnifiedContentBlock::Text { text: text.clone() });
                        }
                        OutputContent::OutputRefusal { refusal } => {
                            // Treat refusal as text for now
                            blocks.push(UnifiedContentBlock::Text {
                                text: format!("[REFUSAL] {}", refusal),
                            });
                        }
                    }
                }
            }
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let input = serde_json::from_str(arguments).unwrap_or(json!({}));
                blocks.push(UnifiedContentBlock::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input,
                });
            }
            OutputItem::Reasoning {
                content,
                summary,
                encrypted_content,
                ..
            } => {
                // Combine reasoning content and summary
                let mut reasoning_parts = Vec::new();

                for part in content {
                    if let ReasoningContent::ReasoningText { text } = part {
                        reasoning_parts.push(text.clone());
                    }
                }

                for part in summary {
                    if let ReasoningContent::SummaryText { text } = part {
                        reasoning_parts.push(format!("Summary: {}", text));
                    }
                }

                if !reasoning_parts.is_empty() {
                    blocks.push(UnifiedContentBlock::Thinking {
                        thinking: reasoning_parts.join("\n"),
                        signature: None,
                        encrypted_content: encrypted_content.clone(),
                        redacted: false,
                    });
                }
            }
        }
    }

    blocks
}

/// Convert OpenAI response to unified message.
///
/// Transforms a complete OpenAI response into appam's unified message format.
pub fn response_to_unified_message(response: &Response) -> UnifiedMessage {
    let content_blocks = to_unified_content_blocks(&response.output);

    UnifiedMessage {
        role: UnifiedRole::Assistant,
        content: content_blocks,
        id: Some(response.id.clone()),
        timestamp: Some(chrono::DateTime::from_timestamp(response.created_at as i64, 0).unwrap()),
        reasoning: None, // Reasoning is embedded in content blocks
        reasoning_details: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_from_unified_messages_simple_text() {
        let msg = UnifiedMessage::user("Hello, world!");
        let input = from_unified_messages(&[msg], None);

        match input {
            ResponseInput::Structured(items) => {
                assert_eq!(items.len(), 1);
                match &items[0] {
                    InputItem::Message { role, content, .. } => {
                        assert!(matches!(role, MessageRole::User));
                        match content {
                            MessageContent::Parts(parts) => {
                                assert_eq!(parts.len(), 1);
                            }
                            _ => panic!("Expected Parts variant"),
                        }
                    }
                    _ => panic!("Expected Message variant"),
                }
            }
            _ => panic!("Expected Structured variant"),
        }
    }

    #[test]
    fn test_from_unified_tools() {
        let tool = UnifiedTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }),
        };

        let tools = from_unified_tools(&[tool]);
        assert_eq!(tools.len(), 1);

        match &tools[0] {
            Tool::Function { name, strict, .. } => {
                assert_eq!(name, "test_tool");
                assert_eq!(*strict, None);
            }
        }
    }

    #[test]
    fn test_to_unified_tool_calls() {
        let output_items = vec![OutputItem::FunctionCall {
            id: "item_1".to_string(),
            call_id: "call_1".to_string(),
            name: "test_function".to_string(),
            arguments: r#"{"arg": "value"}"#.to_string(),
            status: Some("completed".to_string()),
        }];

        let tool_calls = to_unified_tool_calls(&output_items);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].name, "test_function");
    }

    #[test]
    fn test_from_unified_messages_replays_only_post_anchor_delta() {
        let input = from_unified_messages(
            &[
                UnifiedMessage::system("Keep answers short."),
                UnifiedMessage::assistant("Earlier answer"),
                UnifiedMessage::user("New follow-up"),
            ],
            Some("resp_prev"),
        );

        match input {
            ResponseInput::Structured(items) => {
                assert_eq!(items.len(), 1);
                assert!(matches!(
                    &items[0],
                    InputItem::Message {
                        role: MessageRole::User,
                        ..
                    }
                ));
            }
            _ => panic!("Expected Structured variant"),
        }
    }

    #[test]
    fn test_extract_instructions_collects_system_messages() {
        let messages = vec![
            UnifiedMessage::system("System A"),
            UnifiedMessage::user("Hello"),
            UnifiedMessage::system("System B"),
        ];

        assert_eq!(
            extract_instructions(&messages).as_deref(),
            Some("System A\n\nSystem B")
        );
    }

    #[test]
    fn test_from_unified_messages_preserves_reasoning_replay_items() {
        let input = from_unified_messages(
            &[UnifiedMessage {
                role: UnifiedRole::Assistant,
                content: vec![UnifiedContentBlock::Thinking {
                    thinking: "Step 1".to_string(),
                    signature: None,
                    encrypted_content: Some("enc_reasoning_blob".to_string()),
                    redacted: false,
                }],
                id: Some("msg_1".to_string()),
                timestamp: None,
                reasoning: None,
                reasoning_details: None,
            }],
            None,
        );

        match input {
            ResponseInput::Structured(items) => {
                assert_eq!(items.len(), 1);
                match &items[0] {
                    InputItem::Reasoning {
                        content,
                        encrypted_content,
                        ..
                    } => {
                        assert_eq!(encrypted_content.as_deref(), Some("enc_reasoning_blob"));
                        assert!(matches!(
                            content.as_slice(),
                            [ReasoningContent::ReasoningText { text }] if text == "Step 1"
                        ));
                    }
                    _ => panic!("Expected Reasoning item"),
                }
            }
            _ => panic!("Expected Structured variant"),
        }
    }
}
