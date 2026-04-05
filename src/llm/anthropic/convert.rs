//! Conversion between Anthropic types and unified types.
//!
//! Provides bidirectional conversion:
//! - `UnifiedMessage` → Anthropic `Message` (for requests)
//! - Anthropic `ContentBlock` → `UnifiedContentBlock` (from responses)
//! - `UnifiedTool` → Anthropic `Tool` (for requests)
//!
//! # Critical Rules
//!
//! 1. **System Messages**: Anthropic uses top-level `system` parameter, not message role
//! 2. **Tool Result Ordering**: tool_result blocks MUST come before text blocks
//! 3. **Thinking Preservation**: thinking blocks must be included during tool use

use anyhow::Result;
use serde_json::{json, Value as JsonValue};

use super::config::{AnthropicConfig, ToolChoiceConfig};
use super::types::{
    ContentBlock, DocumentSource as AnthropicDocumentSource, ImageSource as AnthropicImageSource,
    Message, MessageRole, SystemPrompt, Tool, ToolResultBlock, ToolResultContent,
};
use crate::llm::unified::{
    DocumentSource, ImageSource, UnifiedContentBlock, UnifiedMessage, UnifiedRole, UnifiedTool,
};

/// Merge consecutive messages of the same role.
///
/// Anthropic's API has specific requirements:
/// 1. All tool results for a set of tool_use blocks must be in a single user message
/// 2. Assistant messages with text and tool_use can be split by the runtime but should be merged
///
/// This function:
/// - Merges consecutive User messages that contain only ToolResult blocks
/// - Merges consecutive Assistant messages (happens when runtime splits text and tool calls)
///
/// # Parameters
///
/// - `messages`: Vector of Anthropic Message objects
///
/// # Returns
///
/// Vector with consecutive messages of the same role merged
///
/// # Examples
///
/// Input:
/// ```text
/// [Assistant: [Text], Assistant: [ToolUse], User: [ToolResult], User: [ToolResult]]
/// ```
///
/// Output:
/// ```text
/// [Assistant: [Text, ToolUse], User: [ToolResult, ToolResult]]
/// ```
fn merge_consecutive_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        let msg = &messages[i];
        let mut merged_content = msg.content.clone();
        let role = msg.role;

        // Look ahead for consecutive messages with the same role
        let mut j = i + 1;

        match role {
            MessageRole::User => {
                // For User messages, only merge if they contain ONLY ToolResult blocks
                if is_only_tool_results(&merged_content) {
                    while j < messages.len() {
                        let next_msg = &messages[j];
                        if matches!(next_msg.role, MessageRole::User)
                            && is_only_tool_results(&next_msg.content)
                        {
                            merged_content.extend(next_msg.content.clone());
                            j += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            MessageRole::Assistant => {
                // For Assistant messages, always merge consecutive ones
                // This handles cases where runtime splits text response and tool calls
                while j < messages.len() {
                    let next_msg = &messages[j];
                    if matches!(next_msg.role, MessageRole::Assistant) {
                        merged_content.extend(next_msg.content.clone());
                        j += 1;
                    } else {
                        break;
                    }
                }
            }
        }

        // Add merged message
        result.push(Message {
            role,
            content: merged_content,
        });

        // Skip all merged messages
        i = j;
    }

    result
}

/// Check if a content block array contains only ToolResult blocks.
///
/// Returns true if all blocks are ToolResult, false if any other block type exists.
fn is_only_tool_results(content: &[ContentBlock]) -> bool {
    !content.is_empty()
        && content
            .iter()
            .all(|block| matches!(block, ContentBlock::ToolResult { .. }))
}

/// Convert unified messages to Anthropic Messages API format.
///
/// # Returns
///
/// - System prompt (extracted from system role messages)
/// - Conversation messages (user/assistant only)
///
/// # Processing Rules
///
/// 1. System messages are extracted and combined into system prompt
/// 2. Remaining messages are converted to user/assistant format
/// 3. Content blocks are properly ordered (tool_result before text)
/// 4. Thinking blocks are preserved if present
/// 5. Automatic prompt caching is applied at the top-level request field, not
///    by injecting cache markers into converted blocks
///
/// # Errors
///
/// Returns an error if messages cannot be converted (e.g., invalid roles).
pub fn from_unified_messages(
    messages: &[UnifiedMessage],
    _config: &AnthropicConfig,
) -> Result<(Option<SystemPrompt>, Vec<Message>)> {
    let mut system_texts: Vec<String> = Vec::new();
    let mut conversation: Vec<Message> = Vec::new();

    for msg in messages {
        match msg.role {
            UnifiedRole::System => {
                // Extract system message text
                for block in &msg.content {
                    if let UnifiedContentBlock::Text { text } = block {
                        system_texts.push(text.clone());
                    }
                }
            }
            UnifiedRole::User => {
                // Convert to Anthropic user message
                // CRITICAL: Ensure tool_result blocks come first!
                let mut tool_result_blocks = Vec::new();
                let mut other_blocks = Vec::new();

                for block in &msg.content {
                    let anthropic_block = from_unified_content_block(block)?;
                    match anthropic_block {
                        ContentBlock::ToolResult { .. } => {
                            tool_result_blocks.push(anthropic_block);
                        }
                        _ => {
                            other_blocks.push(anthropic_block);
                        }
                    }
                }

                // Tool results MUST come first
                let mut content = tool_result_blocks;
                content.extend(other_blocks);

                if !content.is_empty() {
                    conversation.push(Message {
                        role: MessageRole::User,
                        content,
                    });
                }
            }
            UnifiedRole::Assistant => {
                // Convert to Anthropic assistant message
                // Note: Keep ALL blocks including thinking (signatures now preserved via raw_content_blocks)
                let content = msg
                    .content
                    .iter()
                    .map(from_unified_content_block)
                    .collect::<Result<Vec<_>>>()?;

                if !content.is_empty() {
                    conversation.push(Message {
                        role: MessageRole::Assistant,
                        content,
                    });
                }
            }
        }
    }

    // Post-process: merge consecutive messages where needed
    // 1. Merge consecutive Assistant messages (text + tool_use can be split by runtime)
    // 2. Merge consecutive User messages with only tool results
    conversation = merge_consecutive_messages(conversation);

    // Build system prompt
    let system_prompt = if !system_texts.is_empty() {
        let combined = system_texts.join("\n\n");

        Some(SystemPrompt::String(combined))
    } else {
        None
    };

    Ok((system_prompt, conversation))
}

/// Convert unified content block to Anthropic content block.
fn from_unified_content_block(block: &UnifiedContentBlock) -> Result<ContentBlock> {
    match block {
        UnifiedContentBlock::Text { text } => Ok(ContentBlock::Text {
            text: text.clone(),
            cache_control: None,
        }),

        UnifiedContentBlock::Image { source, detail: _ } => {
            let anthropic_source = match source {
                ImageSource::Base64 { media_type, data } => AnthropicImageSource::Base64 {
                    media_type: media_type.clone(),
                    data: data.clone(),
                },
                ImageSource::Url { url } => AnthropicImageSource::Url { url: url.clone() },
            };

            Ok(ContentBlock::Image {
                source: anthropic_source,
                cache_control: None,
            })
        }

        UnifiedContentBlock::Document { source, title } => {
            let anthropic_source = match source {
                DocumentSource::Base64Pdf { media_type, data } => AnthropicDocumentSource::Base64 {
                    media_type: media_type.clone(),
                    data: data.clone(),
                },
                DocumentSource::UrlPdf { url } => AnthropicDocumentSource::Url { url: url.clone() },
                DocumentSource::Text { media_type, data } => AnthropicDocumentSource::Text {
                    media_type: media_type.clone(),
                    data: data.clone(),
                },
            };

            Ok(ContentBlock::Document {
                source: anthropic_source,
                title: title.clone(),
                cache_control: None,
            })
        }

        UnifiedContentBlock::ToolUse { id, name, input } => Ok(ContentBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
            cache_control: None,
        }),

        UnifiedContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            // Convert content to appropriate format
            let tool_result_content = if let Some(str_content) = content.as_str() {
                ToolResultContent::String(str_content.to_string())
            } else {
                // For structured content, convert to string
                ToolResultContent::String(content.to_string())
            };

            Ok(ContentBlock::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: tool_result_content,
                is_error: *is_error,
                cache_control: None,
            })
        }

        UnifiedContentBlock::Thinking {
            thinking,
            signature,
            redacted,
            ..
        } => {
            if *redacted {
                // Redacted thinking
                Ok(ContentBlock::RedactedThinking {
                    data: thinking.clone(),
                })
            } else {
                // Normal thinking - preserve exact signature from API
                // Signature is a cryptographic hash that must be preserved exactly
                Ok(ContentBlock::Thinking {
                    thinking: thinking.clone(),
                    signature: signature.clone().unwrap_or_default(),
                })
            }
        }
    }
}

/// Convert Anthropic content block to unified content block.
pub fn to_unified_content_block(block: &ContentBlock) -> UnifiedContentBlock {
    match block {
        ContentBlock::Text { text, .. } => UnifiedContentBlock::Text { text: text.clone() },

        ContentBlock::Image { source, .. } => {
            let unified_source = match source {
                AnthropicImageSource::Base64 { media_type, data } => ImageSource::Base64 {
                    media_type: media_type.clone(),
                    data: data.clone(),
                },
                AnthropicImageSource::Url { url } => ImageSource::Url { url: url.clone() },
            };

            UnifiedContentBlock::Image {
                source: unified_source,
                detail: None,
            }
        }

        ContentBlock::Document { source, title, .. } => {
            let unified_source = match source {
                AnthropicDocumentSource::Base64 { media_type, data } => DocumentSource::Base64Pdf {
                    media_type: media_type.clone(),
                    data: data.clone(),
                },
                AnthropicDocumentSource::Url { url } => DocumentSource::UrlPdf { url: url.clone() },
                AnthropicDocumentSource::Text { media_type, data } => DocumentSource::Text {
                    media_type: media_type.clone(),
                    data: data.clone(),
                },
            };

            UnifiedContentBlock::Document {
                source: unified_source,
                title: title.clone(),
            }
        }

        ContentBlock::ToolUse {
            id, name, input, ..
        } => UnifiedContentBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },

        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            ..
        } => {
            let unified_content = match content {
                ToolResultContent::String(s) => json!(s),
                ToolResultContent::Blocks(blocks) => {
                    // Extract text from blocks
                    let text = blocks
                        .iter()
                        .filter_map(|b| match b {
                            ToolResultBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    json!(text)
                }
            };

            UnifiedContentBlock::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: unified_content,
                is_error: *is_error,
            }
        }

        ContentBlock::Thinking {
            thinking,
            signature,
        } => UnifiedContentBlock::Thinking {
            thinking: thinking.clone(),
            signature: Some(signature.clone()),
            encrypted_content: None,
            redacted: false,
        },

        ContentBlock::RedactedThinking { data } => UnifiedContentBlock::Thinking {
            thinking: data.clone(),
            signature: None,
            encrypted_content: None,
            redacted: true,
        },
    }
}

/// Convert unified tools to Anthropic tool definitions.
///
/// Appam intentionally does not inject automatic cache markers into the final
/// tool definition. Anthropic's top-level `cache_control` request field is the
/// reference implementation for automatic prompt caching and lets the API pick
/// the correct final cacheable block across `system`, `tools`, and `messages`.
pub fn from_unified_tools(tools: &[UnifiedTool], _config: &AnthropicConfig) -> Result<Vec<Tool>> {
    let mut anthropic_tools = Vec::new();

    for tool in tools {
        anthropic_tools.push(Tool {
            tool_type: None, // Client tool
            name: tool.name.clone(),
            description: Some(tool.description.clone()),
            input_schema: Some(tool.parameters.clone()),
            cache_control: None,
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
        });
    }

    Ok(anthropic_tools)
}

/// Convert tool choice config to JSON.
pub fn tool_choice_to_json(choice: &ToolChoiceConfig) -> Result<JsonValue> {
    let value = match choice {
        ToolChoiceConfig::Auto {
            disable_parallel_tool_use,
        } => {
            json!({
                "type": "auto",
                "disable_parallel_tool_use": disable_parallel_tool_use
            })
        }
        ToolChoiceConfig::Any {
            disable_parallel_tool_use,
        } => {
            json!({
                "type": "any",
                "disable_parallel_tool_use": disable_parallel_tool_use
            })
        }
        ToolChoiceConfig::Tool {
            name,
            disable_parallel_tool_use,
        } => {
            json!({
                "type": "tool",
                "name": name,
                "disable_parallel_tool_use": disable_parallel_tool_use
            })
        }
        ToolChoiceConfig::None => {
            json!({"type": "none"})
        }
    };

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::anthropic::config::{CacheTTL, CachingConfig};

    #[test]
    fn test_system_message_extraction() {
        let messages = vec![
            UnifiedMessage::system("You are a helpful assistant."),
            UnifiedMessage::user("Hello!"),
        ];

        let config = AnthropicConfig::default();
        let (system, conversation) = from_unified_messages(&messages, &config).unwrap();

        assert!(system.is_some());
        match system.unwrap() {
            SystemPrompt::String(text) => assert_eq!(text, "You are a helpful assistant."),
            _ => panic!("Expected string system prompt"),
        }
        assert_eq!(conversation.len(), 1);
        assert!(matches!(conversation[0].role, MessageRole::User));
    }

    #[test]
    fn test_system_message_extraction_with_caching_keeps_string_prompt() {
        let messages = vec![
            UnifiedMessage::system("You are a cached assistant."),
            UnifiedMessage::user("Hello!"),
        ];

        let config = AnthropicConfig {
            caching: Some(CachingConfig {
                enabled: true,
                ttl: CacheTTL::OneHour,
            }),
            ..Default::default()
        };

        let (system, _) = from_unified_messages(&messages, &config).unwrap();

        match system.unwrap() {
            SystemPrompt::String(text) => assert_eq!(text, "You are a cached assistant."),
            SystemPrompt::Blocks(_) => {
                panic!("automatic caching should use top-level cache_control, not system blocks")
            }
        }
    }

    #[test]
    fn test_tool_result_ordering() {
        let messages = vec![UnifiedMessage {
            role: UnifiedRole::User,
            content: vec![
                UnifiedContentBlock::Text {
                    text: "Additional context".to_string(),
                },
                UnifiedContentBlock::ToolResult {
                    tool_use_id: "call_1".to_string(),
                    content: json!("42"),
                    is_error: Some(false),
                },
            ],
            id: None,
            timestamp: None,
            reasoning: None,
            reasoning_details: None,
        }];

        let config = AnthropicConfig::default();
        let (_, conversation) = from_unified_messages(&messages, &config).unwrap();

        // Tool result should be first
        assert_eq!(conversation.len(), 1);
        match &conversation[0].content[0] {
            ContentBlock::ToolResult { .. } => (),
            _ => panic!("Tool result should be first!"),
        }
    }

    #[test]
    fn test_tool_conversion() {
        let unified_tool = UnifiedTool {
            name: "get_weather".to_string(),
            description: "Get weather for a location".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        };

        let config = AnthropicConfig::default();
        let anthropic_tools = from_unified_tools(&[unified_tool], &config).unwrap();

        assert_eq!(anthropic_tools.len(), 1);
        assert_eq!(anthropic_tools[0].name, "get_weather");
        assert!(anthropic_tools[0].input_schema.is_some());
    }

    #[test]
    fn test_tool_conversion_with_caching_does_not_inject_block_markers() {
        let tools = vec![
            UnifiedTool {
                name: "tool1".to_string(),
                description: "First tool".to_string(),
                parameters: json!({"type": "object"}),
            },
            UnifiedTool {
                name: "tool2".to_string(),
                description: "Second tool".to_string(),
                parameters: json!({"type": "object"}),
            },
        ];

        let config = AnthropicConfig {
            caching: Some(super::super::config::CachingConfig {
                enabled: true,
                ttl: super::super::config::CacheTTL::FiveMinutes,
            }),
            ..Default::default()
        };

        let anthropic_tools = from_unified_tools(&tools, &config).unwrap();

        assert!(anthropic_tools[0].cache_control.is_none());
        assert!(anthropic_tools[1].cache_control.is_none());
    }

    #[test]
    fn test_thinking_block_conversion() {
        let unified_block = UnifiedContentBlock::Thinking {
            thinking: "Let me think...".to_string(),
            signature: Some("sig123".to_string()),
            encrypted_content: None,
            redacted: false,
        };

        let anthropic_block = from_unified_content_block(&unified_block).unwrap();
        match anthropic_block {
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

    #[test]
    fn test_redacted_thinking_conversion() {
        let unified_block = UnifiedContentBlock::Thinking {
            thinking: "encrypted_data_here".to_string(),
            signature: None,
            encrypted_content: None,
            redacted: true,
        };

        let anthropic_block = from_unified_content_block(&unified_block).unwrap();
        match anthropic_block {
            ContentBlock::RedactedThinking { data } => {
                assert_eq!(data, "encrypted_data_here");
            }
            _ => panic!("Expected redacted thinking block"),
        }
    }
}
