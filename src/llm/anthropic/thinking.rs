//! Extended thinking utilities for Claude.
//!
//! Handles thinking block preservation, signature verification, and
//! thinking block management during tool use.
//!
//! # Critical Rules
//!
//! 1. During tool use, thinking blocks MUST be included in assistant messages
//! 2. Thinking blocks should preserve signatures exactly as received
//! 3. Outside of tool use, thinking blocks are auto-stripped by the API
//!
//! # Interleaved Thinking
//!
//! Claude 4 models support interleaved thinking (beta), allowing reasoning
//! between tool calls. Requires `interleaved-thinking-2025-05-14` beta header.

use super::types::ContentBlock;

/// Check if a content block is a thinking block (thinking or redacted_thinking).
pub fn is_thinking_block(block: &ContentBlock) -> bool {
    matches!(
        block,
        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. }
    )
}

/// Check if a content block is a tool use block.
pub fn is_tool_use_block(block: &ContentBlock) -> bool {
    matches!(block, ContentBlock::ToolUse { .. })
}

/// Extract thinking blocks from a content array.
///
/// Used to preserve thinking blocks during tool use conversations.
pub fn extract_thinking_blocks(content: &[ContentBlock]) -> Vec<ContentBlock> {
    content
        .iter()
        .filter(|block| is_thinking_block(block))
        .cloned()
        .collect()
}

/// Extract tool use blocks from a content array.
pub fn extract_tool_use_blocks(content: &[ContentBlock]) -> Vec<ContentBlock> {
    content
        .iter()
        .filter(|block| is_tool_use_block(block))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_is_thinking_block() {
        let thinking = ContentBlock::Thinking {
            thinking: "test".to_string(),
            signature: "sig".to_string(),
        };
        assert!(is_thinking_block(&thinking));

        let text = ContentBlock::Text {
            text: "test".to_string(),
            cache_control: None,
        };
        assert!(!is_thinking_block(&text));
    }

    #[test]
    fn test_extract_thinking_blocks() {
        let content = vec![
            ContentBlock::Text {
                text: "Hello".to_string(),
                cache_control: None,
            },
            ContentBlock::Thinking {
                thinking: "Let me think".to_string(),
                signature: "sig1".to_string(),
            },
            ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "tool".to_string(),
                input: json!({}),
                cache_control: None,
            },
        ];

        let thinking = extract_thinking_blocks(&content);
        assert_eq!(thinking.len(), 1);
    }
}
