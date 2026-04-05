//! Integration tests for provider switching functionality.
//!
//! Tests that the same code works correctly with both OpenRouter and Anthropic providers.

use appam::config::AppConfig;
use appam::llm::provider::{DynamicLlmClient, LlmProvider};
use appam::llm::{UnifiedMessage, UnifiedTool};
use serde_json::json;

#[test]
fn test_dynamic_client_creation_openrouter_completions() {
    let mut config = AppConfig {
        provider: LlmProvider::OpenRouterCompletions,
        ..AppConfig::default()
    };
    config.openrouter.api_key = Some("test-key".to_string());

    let result = DynamicLlmClient::from_config(&config);
    assert!(result.is_ok());

    let client = result.unwrap();
    assert_eq!(client.provider(), LlmProvider::OpenRouterCompletions);
    assert_eq!(client.provider_name(), "openrouter-completions");
}

#[test]
fn test_dynamic_client_creation_openrouter_responses() {
    let mut config = AppConfig {
        provider: LlmProvider::OpenRouterResponses,
        ..AppConfig::default()
    };
    config.openrouter.api_key = Some("test-key".to_string());

    let result = DynamicLlmClient::from_config(&config);
    assert!(result.is_ok());

    let client = result.unwrap();
    assert_eq!(client.provider(), LlmProvider::OpenRouterResponses);
    assert_eq!(client.provider_name(), "openrouter-responses");
}

#[test]
fn test_dynamic_client_creation_anthropic() {
    let mut config = AppConfig {
        provider: LlmProvider::Anthropic,
        ..AppConfig::default()
    };
    config.anthropic.api_key = Some("test-key".to_string());

    let result = DynamicLlmClient::from_config(&config);
    assert!(result.is_ok());

    let client = result.unwrap();
    assert_eq!(client.provider(), LlmProvider::Anthropic);
    assert_eq!(client.provider_name(), "anthropic");
}

#[test]
fn test_dynamic_client_creation_azure_anthropic() {
    let mut config = AppConfig {
        provider: LlmProvider::AzureAnthropic {
            base_url: "https://example-resource.services.ai.azure.com/anthropic".to_string(),
            auth_method: appam::llm::anthropic::AzureAnthropicAuthMethod::XApiKey,
        },
        ..AppConfig::default()
    };
    config.anthropic.api_key = Some("test-key".to_string());

    let result = DynamicLlmClient::from_config(&config);
    assert!(result.is_ok());

    let client = result.unwrap();
    assert_eq!(client.provider(), config.provider);
    assert_eq!(client.provider_name(), "azure-anthropic");
}

#[test]
fn test_dynamic_client_creation_openai_with_prefixed_gpt54_model() {
    let mut config = AppConfig {
        provider: LlmProvider::OpenAI,
        ..AppConfig::default()
    };
    config.openai.api_key = Some("test-key".to_string());
    config.openai.model = "openai/gpt-5.4".to_string();

    let result = DynamicLlmClient::from_config(&config);
    assert!(result.is_ok());

    let client = result.unwrap();
    assert_eq!(client.provider(), LlmProvider::OpenAI);
    assert_eq!(client.provider_name(), "openai");
}

#[test]
fn test_dynamic_client_creation_openai_codex() {
    let mut config = AppConfig {
        provider: LlmProvider::OpenAICodex,
        ..AppConfig::default()
    };
    config.openai_codex.access_token = Some("test-openai-codex-token".to_string());
    config.openai_codex.model = "gpt-5.4".to_string();

    let result = DynamicLlmClient::from_config(&config);
    assert!(result.is_ok());

    let client = result.unwrap();
    assert_eq!(client.provider(), LlmProvider::OpenAICodex);
    assert_eq!(client.provider_name(), "openai-codex");
}

#[test]
fn test_dynamic_client_creation_vertex() {
    let mut config = AppConfig {
        provider: LlmProvider::Vertex,
        ..AppConfig::default()
    };
    config.vertex.api_key = Some("test-key".to_string());

    let result = DynamicLlmClient::from_config(&config);
    assert!(result.is_ok());

    let client = result.unwrap();
    assert_eq!(client.provider(), LlmProvider::Vertex);
    assert_eq!(client.provider_name(), "vertex");
}

#[test]
fn test_unified_message_creation() {
    let msg = UnifiedMessage::user("Hello, world!");
    assert_eq!(msg.role, appam::llm::UnifiedRole::User);
    assert_eq!(msg.extract_text(), "Hello, world!");
}

#[test]
fn test_unified_tool_creation() {
    let tool = UnifiedTool {
        name: "get_weather".to_string(),
        description: "Get weather for a location".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state"
                }
            },
            "required": ["location"]
        }),
    };

    assert_eq!(tool.name, "get_weather");
    assert!(tool.parameters.is_object());
}

#[test]
fn test_provider_enum_serialization() {
    let anthropic = LlmProvider::Anthropic;
    let json = serde_json::to_string(&anthropic).unwrap();
    assert_eq!(json, r#""anthropic""#);

    let completions = LlmProvider::OpenRouterCompletions;
    let json = serde_json::to_string(&completions).unwrap();
    assert_eq!(json, r#""openroutercompletions""#);

    let responses = LlmProvider::OpenRouterResponses;
    let json = serde_json::to_string(&responses).unwrap();
    assert_eq!(json, r#""openrouterresponses""#);
}

#[test]
fn test_provider_enum_parsing() {
    let anthropic: LlmProvider = "anthropic".parse().unwrap();
    assert_eq!(anthropic, LlmProvider::Anthropic);

    // Test multiple formats for Completions (default)
    let completions: LlmProvider = "openrouter".parse().unwrap();
    assert_eq!(completions, LlmProvider::OpenRouterCompletions);

    let completions2: LlmProvider = "openrouter-completions".parse().unwrap();
    assert_eq!(completions2, LlmProvider::OpenRouterCompletions);

    // Test Responses API
    let responses: LlmProvider = "openrouter-responses".parse().unwrap();
    assert_eq!(responses, LlmProvider::OpenRouterResponses);

    let vertex: LlmProvider = "vertex".parse().unwrap();
    assert_eq!(vertex, LlmProvider::Vertex);

    let invalid = "invalid".parse::<LlmProvider>();
    assert!(invalid.is_err());
}

#[test]
fn test_provider_display() {
    assert_eq!(LlmProvider::Anthropic.to_string(), "anthropic");
    assert_eq!(
        LlmProvider::OpenRouterCompletions.to_string(),
        "openrouter-completions"
    );
    assert_eq!(
        LlmProvider::OpenRouterResponses.to_string(),
        "openrouter-responses"
    );
    assert_eq!(LlmProvider::Vertex.to_string(), "vertex");
}

#[test]
fn test_message_content_extraction() {
    use appam::llm::UnifiedContentBlock;

    let msg = UnifiedMessage {
        role: appam::llm::UnifiedRole::Assistant,
        content: vec![
            UnifiedContentBlock::Text {
                text: "Hello".to_string(),
            },
            UnifiedContentBlock::Text {
                text: "World".to_string(),
            },
        ],
        id: None,
        timestamp: None,
        reasoning: None,
        reasoning_details: None,
    };

    let text = msg.extract_text();
    assert_eq!(text, "Hello\nWorld");
}

#[test]
fn test_message_tool_call_extraction() {
    use appam::llm::UnifiedContentBlock;

    let msg = UnifiedMessage {
        role: appam::llm::UnifiedRole::Assistant,
        content: vec![
            UnifiedContentBlock::Text {
                text: "Let me check".to_string(),
            },
            UnifiedContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                input: json!({"location": "Paris"}),
            },
        ],
        id: None,
        timestamp: None,
        reasoning: None,
        reasoning_details: None,
    };

    assert!(msg.has_tool_calls());
    let calls = msg.extract_tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
}

#[test]
fn test_usage_calculations() {
    use appam::llm::UnifiedUsage;

    let usage = UnifiedUsage {
        input_tokens: 100,
        output_tokens: 50,
        cache_creation_input_tokens: Some(200),
        cache_read_input_tokens: Some(150),
        reasoning_tokens: Some(30),
    };

    assert_eq!(usage.total_tokens(), 150);
    assert_eq!(usage.effective_input_tokens(), 150); // 100 + 200 - 150
}
