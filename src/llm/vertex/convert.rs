//! Conversion helpers between appam unified types and Vertex payloads.

use std::collections::HashMap;

use serde_json::json;

use super::types::{
    VertexContent, VertexFunctionCall, VertexFunctionDeclaration, VertexFunctionResponse,
    VertexPart, VertexTool,
};
use crate::llm::unified::{UnifiedContentBlock, UnifiedMessage, UnifiedRole, UnifiedTool};

/// Converted conversation payload split into system instruction and chat contents.
#[derive(Debug, Clone, Default)]
pub struct VertexConversation {
    /// Optional system instruction payload.
    pub system_instruction: Option<VertexContent>,
    /// User/model contents for `contents` field.
    pub contents: Vec<VertexContent>,
}

/// Convert unified messages into Vertex conversation content.
///
/// This conversion preserves tool-call IDs through an internal map and encodes
/// signature metadata as empty `Thinking` blocks so follow-up turns can
/// reconstruct `thoughtSignature` values required by Vertex.
pub fn from_unified_messages(messages: &[UnifiedMessage]) -> VertexConversation {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut tool_name_by_call_id: HashMap<String, String> = HashMap::new();

    let mut message_index = 0usize;
    while message_index < messages.len() {
        let message = &messages[message_index];

        if message.role == UnifiedRole::System {
            for block in &message.content {
                if let UnifiedContentBlock::Text { text } = block {
                    if !text.trim().is_empty() {
                        system_parts.push(VertexPart {
                            text: Some(text.clone()),
                            ..Default::default()
                        });
                    }
                }
            }
            message_index += 1;
            continue;
        }

        if message_has_only_tool_results(message) {
            let mut parts = Vec::new();

            // Vertex requires the tool-result turn to contain a functionResponse
            // part for each functionCall part emitted by the immediately
            // preceding model turn, so we batch consecutive tool-result messages.
            while message_index < messages.len() {
                let candidate = &messages[message_index];
                if !message_has_only_tool_results(candidate) {
                    break;
                }

                for block in &candidate.content {
                    if let UnifiedContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } = block
                    {
                        let name = tool_name_by_call_id
                            .get(tool_use_id)
                            .cloned()
                            .unwrap_or_else(|| "tool_result".to_string());

                        let response = if content.is_object() {
                            content.clone()
                        } else {
                            json!({ "result": content })
                        };

                        parts.push(VertexPart {
                            function_response: Some(VertexFunctionResponse { name, response }),
                            ..Default::default()
                        });
                    }
                }

                message_index += 1;
            }

            if !parts.is_empty() {
                contents.push(VertexContent {
                    role: Some("user".to_string()),
                    parts,
                });
            }

            continue;
        }

        let role = match message.role {
            UnifiedRole::User => "user",
            UnifiedRole::Assistant => "model",
            UnifiedRole::System => "user",
        };

        let mut parts = Vec::new();
        let mut idx = 0usize;
        while idx < message.content.len() {
            let block = &message.content[idx];
            let signature = message
                .content
                .get(idx + 1)
                .and_then(signature_metadata)
                .cloned();

            let mut consumed_signature = false;

            match block {
                UnifiedContentBlock::Text { text } => {
                    if !text.is_empty() {
                        parts.push(VertexPart {
                            text: Some(text.clone()),
                            thought_signature: signature,
                            ..Default::default()
                        });
                        consumed_signature = parts
                            .last()
                            .and_then(|p| p.thought_signature.as_ref())
                            .is_some();
                    }
                }
                UnifiedContentBlock::ToolUse { id, name, input } => {
                    tool_name_by_call_id.insert(id.clone(), name.clone());
                    parts.push(VertexPart {
                        function_call: Some(VertexFunctionCall {
                            name: Some(name.clone()),
                            args: Some(input.clone()),
                            ..Default::default()
                        }),
                        thought_signature: signature,
                        ..Default::default()
                    });
                    consumed_signature = parts
                        .last()
                        .and_then(|p| p.thought_signature.as_ref())
                        .is_some();
                }
                UnifiedContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => {
                    let name = tool_name_by_call_id
                        .get(tool_use_id)
                        .cloned()
                        .unwrap_or_else(|| "tool_result".to_string());

                    let response = if content.is_object() {
                        content.clone()
                    } else {
                        json!({ "result": content })
                    };

                    parts.push(VertexPart {
                        function_response: Some(VertexFunctionResponse { name, response }),
                        ..Default::default()
                    });
                }
                UnifiedContentBlock::Thinking {
                    thinking,
                    signature,
                    ..
                } => {
                    if thinking.is_empty() && signature.is_some() {
                        // Signature-only metadata blocks are consumed by lookahead.
                    } else {
                        parts.push(VertexPart {
                            text: Some(thinking.clone()),
                            thought: Some(true),
                            thought_signature: signature.clone(),
                            ..Default::default()
                        });
                    }
                }
                _ => {}
            }

            idx += if consumed_signature { 2 } else { 1 };
        }

        if !parts.is_empty() {
            contents.push(VertexContent {
                role: Some(role.to_string()),
                parts,
            });
        }

        message_index += 1;
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(VertexContent {
            role: None,
            parts: system_parts,
        })
    };

    VertexConversation {
        system_instruction,
        contents,
    }
}

/// Convert unified tools into Vertex function declarations.
pub fn from_unified_tools(tools: &[UnifiedTool]) -> Vec<VertexTool> {
    if tools.is_empty() {
        return Vec::new();
    }

    let declarations: Vec<VertexFunctionDeclaration> = tools
        .iter()
        .map(|tool| VertexFunctionDeclaration {
            name: tool.name.clone(),
            description: Some(tool.description.clone()),
            parameters: extract_parameters_schema(&tool.parameters),
        })
        .collect();

    vec![VertexTool {
        function_declarations: declarations,
    }]
}

fn extract_parameters_schema(raw: &serde_json::Value) -> serde_json::Value {
    raw.as_object()
        .and_then(|obj| obj.get("parameters"))
        .cloned()
        .unwrap_or_else(|| raw.clone())
}

fn signature_metadata(block: &UnifiedContentBlock) -> Option<&String> {
    match block {
        UnifiedContentBlock::Thinking {
            thinking,
            signature: Some(signature),
            redacted,
            ..
        } if thinking.is_empty() && !redacted => Some(signature),
        _ => None,
    }
}

fn message_has_only_tool_results(message: &UnifiedMessage) -> bool {
    !message.content.is_empty()
        && message.role == UnifiedRole::User
        && message
            .content
            .iter()
            .all(|block| matches!(block, UnifiedContentBlock::ToolResult { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::unified::{UnifiedMessage, UnifiedRole};

    #[test]
    fn test_from_unified_messages_preserves_tool_signature_metadata() {
        let messages = vec![UnifiedMessage {
            role: UnifiedRole::Assistant,
            content: vec![
                UnifiedContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "search_docs".to_string(),
                    input: serde_json::json!({"query": "vertex"}),
                },
                UnifiedContentBlock::Thinking {
                    thinking: String::new(),
                    signature: Some("sig-tool-1".to_string()),
                    encrypted_content: None,
                    redacted: false,
                },
            ],
            id: None,
            timestamp: None,
            reasoning: None,
            reasoning_details: None,
        }];

        let conversation = from_unified_messages(&messages);
        assert_eq!(conversation.contents.len(), 1);
        assert_eq!(conversation.contents[0].role.as_deref(), Some("model"));

        let part = &conversation.contents[0].parts[0];
        assert_eq!(
            part.function_call
                .as_ref()
                .and_then(|fc| fc.name.as_deref()),
            Some("search_docs")
        );
        assert_eq!(part.thought_signature.as_deref(), Some("sig-tool-1"));
    }

    #[test]
    fn test_from_unified_messages_maps_tool_result_to_function_response() {
        let messages = vec![
            UnifiedMessage {
                role: UnifiedRole::Assistant,
                content: vec![UnifiedContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "search_docs".to_string(),
                    input: serde_json::json!({"query": "vertex"}),
                }],
                id: None,
                timestamp: None,
                reasoning: None,
                reasoning_details: None,
            },
            UnifiedMessage {
                role: UnifiedRole::User,
                content: vec![UnifiedContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: serde_json::json!("ok"),
                    is_error: Some(false),
                }],
                id: None,
                timestamp: None,
                reasoning: None,
                reasoning_details: None,
            },
        ];

        let conversation = from_unified_messages(&messages);
        assert_eq!(conversation.contents.len(), 2);

        let response = conversation.contents[1].parts[0]
            .function_response
            .as_ref()
            .expect("expected functionResponse in user tool-result message");
        assert_eq!(response.name, "search_docs");
        assert_eq!(response.response, serde_json::json!({"result": "ok"}));
    }

    #[test]
    fn test_from_unified_messages_batches_consecutive_tool_results() {
        let messages = vec![
            UnifiedMessage {
                role: UnifiedRole::Assistant,
                content: vec![
                    UnifiedContentBlock::ToolUse {
                        id: "call_1".to_string(),
                        name: "mkdir".to_string(),
                        input: serde_json::json!({"path": "poem_generator"}),
                    },
                    UnifiedContentBlock::ToolUse {
                        id: "call_2".to_string(),
                        name: "write_file".to_string(),
                        input: serde_json::json!({"file_path": "poem_generator/generator.py"}),
                    },
                ],
                id: None,
                timestamp: None,
                reasoning: None,
                reasoning_details: None,
            },
            UnifiedMessage {
                role: UnifiedRole::User,
                content: vec![UnifiedContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: serde_json::json!({"success": true}),
                    is_error: Some(false),
                }],
                id: None,
                timestamp: None,
                reasoning: None,
                reasoning_details: None,
            },
            UnifiedMessage {
                role: UnifiedRole::User,
                content: vec![UnifiedContentBlock::ToolResult {
                    tool_use_id: "call_2".to_string(),
                    content: serde_json::json!({"success": true}),
                    is_error: Some(false),
                }],
                id: None,
                timestamp: None,
                reasoning: None,
                reasoning_details: None,
            },
        ];

        let conversation = from_unified_messages(&messages);
        assert_eq!(conversation.contents.len(), 2);
        assert_eq!(conversation.contents[1].role.as_deref(), Some("user"));
        assert_eq!(conversation.contents[1].parts.len(), 2);

        let first = conversation.contents[1].parts[0]
            .function_response
            .as_ref()
            .expect("expected first functionResponse");
        let second = conversation.contents[1].parts[1]
            .function_response
            .as_ref()
            .expect("expected second functionResponse");

        assert_eq!(first.name, "mkdir");
        assert_eq!(second.name, "write_file");
    }
}
