//! Tests for session continuation functionality
//!
//! Verifies that agents properly inject continuation messages when sessions
//! end without calling required completion tools.

use appam::agent::AgentBuilder;
use appam::llm::{ChatMessage, Role, ToolSpec};
use appam::tools::Tool;
use serde_json::json;
use std::sync::{Arc, Mutex};

/// Mock tool that tracks calls (for future test expansion)
#[derive(Clone)]
#[allow(dead_code)]
struct MockCompletionTool {
    name: String,
    calls: Arc<Mutex<Vec<serde_json::Value>>>,
}

#[allow(dead_code)]
impl MockCompletionTool {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

impl Tool for MockCompletionTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec(&self) -> anyhow::Result<ToolSpec> {
        Ok(serde_json::from_value(json!({
            "type": "function",
            "name": self.name,
            "description": "Mock completion tool for testing",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to return"
                    }
                },
                "required": ["message"]
            }
        }))?)
    }

    fn execute(&self, args: serde_json::Value) -> anyhow::Result<serde_json::Value> {
        self.calls.lock().unwrap().push(args.clone());
        Ok(json!({
            "success": true,
            "message": args.get("message").and_then(|v| v.as_str()).unwrap_or("completed")
        }))
    }
}

#[test]
fn test_continuation_config_fields() {
    // Test that continuation configuration is properly set
    let tool1 = Arc::new(MockCompletionTool::new("tool1"));
    let tool2 = Arc::new(MockCompletionTool::new("tool2"));

    let agent = AgentBuilder::new("test-agent")
        .system_prompt("Test prompt")
        .require_completion_tools(vec![tool1 as Arc<dyn Tool>, tool2 as Arc<dyn Tool>])
        .continuation_message("Custom continuation message")
        .max_continuations(3)
        .build()
        .unwrap();

    assert_eq!(
        agent.required_completion_tools(),
        Some(&vec!["tool1".to_string(), "tool2".to_string()])
    );
    assert_eq!(agent.max_continuations(), 3);
    assert_eq!(
        agent.continuation_message(),
        Some("Custom continuation message")
    );
}

#[test]
fn test_no_continuation_when_disabled() {
    // When required_completion_tools is not set, continuation should not happen
    let agent = AgentBuilder::new("test-agent")
        .system_prompt("Test prompt")
        .build()
        .unwrap();

    assert_eq!(agent.required_completion_tools(), None);
    assert_eq!(agent.max_continuations(), 2); // Default value
}

#[test]
fn test_default_max_continuations() {
    // Test that default max_continuations is 2
    let tool1 = Arc::new(MockCompletionTool::new("tool1"));

    let agent = AgentBuilder::new("test-agent")
        .system_prompt("Test prompt")
        .require_completion_tools(vec![tool1 as Arc<dyn Tool>])
        .build()
        .unwrap();

    assert_eq!(agent.max_continuations(), 2);
    assert_eq!(agent.continuation_message(), None); // No custom message
}

/// Test helper to check if messages contain a continuation attempt
fn has_continuation_message(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|msg| {
        msg.role == Role::User
            && msg
                .content
                .as_ref()
                .map(|c| c.contains("Continue analysis"))
                .unwrap_or(false)
    })
}

/// Test helper to count continuation messages
fn count_continuation_messages(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .filter(|msg| {
            msg.role == Role::User
                && msg
                    .content
                    .as_ref()
                    .map(|c| c.contains("Continue analysis"))
                    .unwrap_or(false)
        })
        .count()
}

/// Test helper to check if a required tool was called
fn has_tool_call(messages: &[ChatMessage], tool_name: &str) -> bool {
    messages.iter().any(|msg| {
        if let Some(tool_calls) = &msg.tool_calls {
            tool_calls.iter().any(|tc| tc.function.name == tool_name)
        } else {
            false
        }
    })
}

// Note: Full integration tests that actually run agents would require:
// 1. A test LLM endpoint or mock LLM client
// 2. Async test infrastructure
// 3. More complex setup
//
// The tests above verify the configuration and helper functions.
// The actual continuation logic in runtime.rs is tested via the hunter
// agent integration tests in the main example-resource project.

#[test]
fn test_continuation_message_detection() {
    // Test that our helper correctly identifies continuation messages
    let messages = vec![
        ChatMessage {
            role: Role::User,
            content: Some("Initial prompt".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: None,
            timestamp: None,
            id: None,
            provider_response_id: None,
            status: None,
        },
        ChatMessage {
            role: Role::User,
            content: Some("Continue analysis. You should call `no_vulnerabilities_found` or `store_vulnerability_finding` to complete this analysis session.".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: None,
            timestamp: None,
            id: None,
            provider_response_id: None,
            status: None,
        },
    ];

    assert!(has_continuation_message(&messages));
    assert_eq!(count_continuation_messages(&messages), 1);
}

#[test]
fn test_tool_call_detection() {
    // Test that our helper correctly identifies tool calls
    use appam::llm::{ToolCall, ToolCallFunction};

    let messages = vec![ChatMessage {
        role: Role::Assistant,
        content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Some(vec![ToolCall {
            id: "call_1".to_string(),
            type_field: "function".to_string(),
            function: ToolCallFunction {
                name: "no_vulnerabilities_found".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        reasoning: None,
        raw_content_blocks: None,
        tool_metadata: None,
        timestamp: None,
        id: None,
        provider_response_id: None,
        status: None,
    }];

    assert!(has_tool_call(&messages, "no_vulnerabilities_found"));
    assert!(!has_tool_call(&messages, "store_vulnerability_finding"));
}

#[test]
fn test_multiple_continuation_messages() {
    // Test counting multiple continuation attempts
    let messages = vec![
        ChatMessage {
            role: Role::User,
            content: Some("Continue analysis. You should call `no_vulnerabilities_found` or `store_vulnerability_finding` to complete this analysis session.".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: None,
            timestamp: None,
            id: None,
            provider_response_id: None,
            status: None,
        },
        ChatMessage {
            role: Role::Assistant,
            content: Some("I'm still analyzing...".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: None,
            timestamp: None,
            id: None,
            provider_response_id: None,
            status: None,
        },
        ChatMessage {
            role: Role::User,
            content: Some("Continue analysis. You should call `no_vulnerabilities_found` or `store_vulnerability_finding` to complete this analysis session.".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: None,
            timestamp: None,
            id: None,
            provider_response_id: None,
            status: None,
        },
    ];

    assert_eq!(count_continuation_messages(&messages), 2);
}
