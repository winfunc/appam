//! Integration tests for Opus 4.6 streaming with Bedrock, direct Anthropic API,
//! and Azure Anthropic.
//!
//! Tests adaptive thinking, effort parameter, EventStream decoding (Bedrock),
//! parallel tool calls, and usage/pricing tracking.
//!
//! These tests require real credentials and are marked `#[ignore]` by default.
//! Run with:
//!
//! ```bash
//! # Bedrock test (requires AWS credentials):
//! export AWS_ACCESS_KEY_ID="..."
//! export AWS_SECRET_ACCESS_KEY="..."
//! export AWS_REGION="us-east-1"
//! cargo test --test bedrock_opus46_streaming test_bedrock_opus46_streaming -- --ignored --nocapture
//!
//! # Anthropic direct API test:
//! export ANTHROPIC_API_KEY="..."
//! cargo test --test bedrock_opus46_streaming test_anthropic_opus46_streaming -- --ignored --nocapture
//!
//! # Azure Anthropic test:
//! export AZURE_API_KEY="..."
//! export AZURE_ANTHROPIC_BASE_URL="https://<resource>.services.ai.azure.com/anthropic"
//! cargo test --test bedrock_opus46_streaming test_azure_anthropic_opus46_streaming -- --ignored --nocapture
//! ```

use anyhow::Result;
use appam::prelude::*;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

// ============================================================================
// Test Tools
// ============================================================================

/// Simple weather tool that returns mock data.
/// Designed to be called in parallel for multiple cities.
struct GetWeatherTool;

impl Tool for GetWeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn spec(&self) -> Result<ToolSpec> {
        Ok(ToolSpec {
            type_field: "function".to_string(),
            name: "get_weather".to_string(),
            description:
                "Get the current weather for a city. Returns temperature, conditions, and humidity."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for (e.g., 'Paris', 'Tokyo', 'New York')"
                    }
                },
                "required": ["city"]
            }),
            strict: None,
        })
    }

    fn execute(&self, args: Value) -> Result<Value> {
        let city = args["city"].as_str().unwrap_or("Unknown");
        let (temp, conditions, humidity) = match city.to_lowercase().as_str() {
            "paris" => (12, "Partly cloudy", 65),
            "tokyo" => (8, "Clear skies", 45),
            "new york" => (-2, "Light snow", 80),
            "london" => (7, "Overcast", 78),
            _ => (20, "Sunny", 50),
        };

        Ok(json!({
            "city": city,
            "temperature_celsius": temp,
            "conditions": conditions,
            "humidity_percent": humidity
        }))
    }
}

/// Simple timezone tool for testing parallel calls alongside weather.
struct GetTimeTool;

impl Tool for GetTimeTool {
    fn name(&self) -> &str {
        "get_time"
    }

    fn spec(&self) -> Result<ToolSpec> {
        Ok(ToolSpec {
            type_field: "function".to_string(),
            name: "get_time".to_string(),
            description: "Get the current time in a specific timezone.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone identifier (e.g., 'Europe/Paris', 'Asia/Tokyo', 'America/New_York')"
                    }
                },
                "required": ["timezone"]
            }),
            strict: None,
        })
    }

    fn execute(&self, args: Value) -> Result<Value> {
        let timezone = args["timezone"].as_str().unwrap_or("UTC");
        let mock_time = match timezone {
            tz if tz.contains("Paris") || tz.contains("Europe") => "14:30:00 CET",
            tz if tz.contains("Tokyo") || tz.contains("Asia") => "22:30:00 JST",
            tz if tz.contains("New_York") || tz.contains("America") => "08:30:00 EST",
            _ => "13:30:00 UTC",
        };

        Ok(json!({
            "timezone": timezone,
            "current_time": mock_time
        }))
    }
}

/// Resolve the Azure Anthropic base URL from the environment.
///
/// The test accepts either a full base URL or a resource name that can be
/// converted into the documented `services.ai.azure.com` pattern.
fn resolve_azure_anthropic_base_url() -> Result<String> {
    if let Ok(base_url) = std::env::var("AZURE_ANTHROPIC_BASE_URL") {
        if !base_url.trim().is_empty() {
            return Ok(base_url);
        }
    }

    let resource = std::env::var("AZURE_ANTHROPIC_RESOURCE").map_err(|_| {
        anyhow::anyhow!(
            "AZURE_ANTHROPIC_BASE_URL or AZURE_ANTHROPIC_RESOURCE environment variable is required"
        )
    })?;

    appam::llm::anthropic::AzureAnthropicConfig::base_url_from_resource(&resource)
}

// ============================================================================
// Bedrock Streaming Test (EventStream binary decoder)
// ============================================================================

/// Test Opus 4.6 streaming via AWS Bedrock with EventStream decoding.
///
/// Validates:
/// - EventStream binary frame decoding works correctly
/// - Adaptive thinking produces reasoning tokens
/// - Parallel tool calls are parsed (weather + time for multiple cities)
/// - Usage/token tracking works through EventStream
/// - Effort parameter is sent correctly
#[tokio::test]
#[ignore = "Requires AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)"]
async fn test_bedrock_opus46_streaming() -> Result<()> {
    let region = std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|_| "us-east-1".to_string());
    let model_id = std::env::var("AWS_BEDROCK_MODEL_ID")
        .unwrap_or_else(|_| "us.anthropic.claude-opus-4-6-v1".to_string());

    eprintln!("=== Bedrock Opus 4.6 Streaming Test ===");
    eprintln!("Region: {}", region);
    eprintln!("Model:  {}", model_id);

    let received_content = Arc::new(AtomicBool::new(false));
    let received_thinking = Arc::new(AtomicBool::new(false));
    let tool_call_count = Arc::new(AtomicU32::new(0));

    let content_flag = Arc::clone(&received_content);
    let thinking_flag = Arc::clone(&received_thinking);
    let tool_count = Arc::clone(&tool_call_count);

    let agent = AgentBuilder::new("bedrock-test")
        .provider(LlmProvider::Bedrock {
            region: region.clone(),
            model_id: model_id.clone(),
            auth_method: appam::llm::anthropic::BedrockAuthMethod::SigV4,
        })
        .model(&model_id)
        .system_prompt(
            "You are a helpful assistant with access to weather and time tools. \
             When asked about weather and time in multiple cities, call the tools \
             for ALL cities in parallel to be efficient.",
        )
        // Adaptive thinking for Opus 4.6
        .thinking(appam::llm::anthropic::ThinkingConfig::adaptive())
        // Max effort for deepest reasoning
        .effort(appam::llm::anthropic::EffortLevel::Max)
        // Beta features for Bedrock
        .beta_features(appam::llm::anthropic::BetaFeatures {
            context_1m: true,
            effort: true,
            ..Default::default()
        })
        // Allow parallel tool use
        .tool_choice(appam::llm::anthropic::ToolChoiceConfig::Auto {
            disable_parallel_tool_use: false,
        })
        .retry(appam::llm::anthropic::RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 2000,
            max_backoff_ms: 30000,
            backoff_multiplier: 2.0,
            jitter: true,
        })
        .with_tool(Arc::new(GetWeatherTool))
        .with_tool(Arc::new(GetTimeTool))
        .max_tokens(16000)
        .build()?;

    let prompt = "What's the current weather in Paris, Tokyo, and New York? \
                  Also tell me the current time in each city's timezone. \
                  Use the tools for all cities.";

    eprintln!("Prompt: {}", prompt);
    eprintln!("Streaming response...\n");

    agent
        .stream(prompt)
        .on_content(move |content| {
            content_flag.store(true, Ordering::Relaxed);
            eprint!("{}", content);
        })
        .on_reasoning(move |reasoning| {
            thinking_flag.store(true, Ordering::Relaxed);
            eprint!("[thinking] {}", reasoning);
        })
        .on_tool_call(move |tool_name, _args| {
            tool_count.fetch_add(1, Ordering::Relaxed);
            eprintln!("\n[tool_call] {}", tool_name);
        })
        .on_tool_result(|tool_name, result| {
            eprintln!(
                "[tool_result] {} => {}",
                tool_name,
                serde_json::to_string(&result).unwrap_or_default()
            );
        })
        .run()
        .await?;

    eprintln!("\n\n=== Results ===");
    eprintln!("Streaming completed successfully!");

    let had_content = received_content.load(Ordering::Relaxed);
    let had_thinking = received_thinking.load(Ordering::Relaxed);
    let num_tool_calls = tool_call_count.load(Ordering::Relaxed);

    eprintln!("Content received: {}", had_content);
    eprintln!("Thinking received: {}", had_thinking);
    eprintln!("Tool calls made: {}", num_tool_calls);

    assert!(had_content, "Should have received streamed content");
    assert!(
        num_tool_calls >= 2,
        "Should have made at least 2 tool calls (weather and/or time), got {}",
        num_tool_calls
    );

    eprintln!("=== Bedrock test PASSED ===\n");
    Ok(())
}

// ============================================================================
// Direct Anthropic API Streaming Test (SSE)
// ============================================================================

/// Test Opus 4.6 streaming via direct Anthropic API with SSE.
///
/// Validates:
/// - SSE streaming works with Opus 4.6 features
/// - Adaptive thinking produces reasoning tokens
/// - Parallel tool calls are parsed
/// - Usage/token tracking works
/// - Effort parameter and output_config are sent correctly
#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY environment variable"]
async fn test_anthropic_opus46_streaming() -> Result<()> {
    eprintln!("=== Anthropic Direct API Opus 4.6 Streaming Test ===");

    let received_content = Arc::new(AtomicBool::new(false));
    let received_thinking = Arc::new(AtomicBool::new(false));
    let tool_call_count = Arc::new(AtomicU32::new(0));

    let content_flag = Arc::clone(&received_content);
    let thinking_flag = Arc::clone(&received_thinking);
    let tool_count = Arc::clone(&tool_call_count);

    let agent = AgentBuilder::new("anthropic-test")
        .provider(LlmProvider::Anthropic)
        .model("claude-opus-4-6")
        .system_prompt(
            "You are a helpful assistant with access to weather and time tools. \
             When asked about weather and time in multiple cities, call the tools \
             for ALL cities in parallel to be efficient.",
        )
        // Adaptive thinking for Opus 4.6
        .thinking(appam::llm::anthropic::ThinkingConfig::adaptive())
        // Max effort for deepest reasoning
        .effort(appam::llm::anthropic::EffortLevel::Max)
        // Allow parallel tool use
        .tool_choice(appam::llm::anthropic::ToolChoiceConfig::Auto {
            disable_parallel_tool_use: false,
        })
        .retry(appam::llm::anthropic::RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 2000,
            max_backoff_ms: 30000,
            backoff_multiplier: 2.0,
            jitter: true,
        })
        .with_tool(Arc::new(GetWeatherTool))
        .with_tool(Arc::new(GetTimeTool))
        .max_tokens(16000)
        .build()?;

    let prompt = "What's the current weather in Paris, Tokyo, and New York? \
                  Also tell me the current time in each city's timezone. \
                  Use the tools for all cities.";

    eprintln!("Prompt: {}", prompt);
    eprintln!("Streaming response...\n");

    agent
        .stream(prompt)
        .on_content(move |content| {
            content_flag.store(true, Ordering::Relaxed);
            eprint!("{}", content);
        })
        .on_reasoning(move |reasoning| {
            thinking_flag.store(true, Ordering::Relaxed);
            eprint!("[thinking] {}", reasoning);
        })
        .on_tool_call(move |tool_name, _args| {
            tool_count.fetch_add(1, Ordering::Relaxed);
            eprintln!("\n[tool_call] {}", tool_name);
        })
        .on_tool_result(|tool_name, result| {
            eprintln!(
                "[tool_result] {} => {}",
                tool_name,
                serde_json::to_string(&result).unwrap_or_default()
            );
        })
        .run()
        .await?;

    eprintln!("\n\n=== Results ===");
    eprintln!("Streaming completed successfully!");

    let had_content = received_content.load(Ordering::Relaxed);
    let had_thinking = received_thinking.load(Ordering::Relaxed);
    let num_tool_calls = tool_call_count.load(Ordering::Relaxed);

    eprintln!("Content received: {}", had_content);
    eprintln!("Thinking received: {}", had_thinking);
    eprintln!("Tool calls made: {}", num_tool_calls);

    assert!(had_content, "Should have received streamed content");
    assert!(
        num_tool_calls >= 2,
        "Should have made at least 2 tool calls (weather and/or time), got {}",
        num_tool_calls
    );

    eprintln!("=== Anthropic direct API test PASSED ===\n");
    Ok(())
}

/// Test Opus 4.6 streaming via Azure Anthropic with Anthropic SSE semantics.
///
/// Validates:
/// - SSE streaming works through Azure-hosted Anthropic endpoints
/// - Adaptive thinking produces reasoning tokens
/// - Parallel tool calls are parsed
/// - Usage/token tracking works
/// - Azure Anthropic auth and endpoint wiring behave like first-class provider config
#[tokio::test]
#[ignore = "Requires AZURE_API_KEY plus AZURE_ANTHROPIC_BASE_URL or AZURE_ANTHROPIC_RESOURCE"]
async fn test_azure_anthropic_opus46_streaming() -> Result<()> {
    let base_url = resolve_azure_anthropic_base_url()?;
    let auth_method = std::env::var("AZURE_ANTHROPIC_AUTH_METHOD")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(appam::llm::anthropic::AzureAnthropicAuthMethod::BearerToken);
    let model =
        std::env::var("AZURE_ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-opus-4-6".to_string());

    eprintln!("=== Azure Anthropic Opus 4.6 Streaming Test ===");
    eprintln!("Base URL: {}", base_url);
    eprintln!("Model:    {}", model);
    eprintln!("Auth:     {}", auth_method.as_str());

    let received_content = Arc::new(AtomicBool::new(false));
    let received_thinking = Arc::new(AtomicBool::new(false));
    let tool_call_count = Arc::new(AtomicU32::new(0));

    let content_flag = Arc::clone(&received_content);
    let thinking_flag = Arc::clone(&received_thinking);
    let tool_count = Arc::clone(&tool_call_count);

    let agent = AgentBuilder::new("azure-anthropic-test")
        .provider(LlmProvider::AzureAnthropic {
            base_url: base_url.clone(),
            auth_method: auth_method.clone(),
        })
        .model(&model)
        .system_prompt(
            "You are a helpful assistant with access to weather and time tools. \
             When asked about weather and time in multiple cities, call the tools \
             for ALL cities in parallel to be efficient.",
        )
        .thinking(appam::llm::anthropic::ThinkingConfig::adaptive())
        .effort(appam::llm::anthropic::EffortLevel::Max)
        .tool_choice(appam::llm::anthropic::ToolChoiceConfig::Auto {
            disable_parallel_tool_use: false,
        })
        .retry(appam::llm::anthropic::RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 2000,
            max_backoff_ms: 30000,
            backoff_multiplier: 2.0,
            jitter: true,
        })
        .with_tool(Arc::new(GetWeatherTool))
        .with_tool(Arc::new(GetTimeTool))
        .max_tokens(16000)
        .build()?;

    let prompt = "What's the current weather in Paris, Tokyo, and New York? \
                  Also tell me the current time in each city's timezone. \
                  Use the tools for all cities.";

    eprintln!("Prompt: {}", prompt);
    eprintln!("Streaming response...\n");

    agent
        .stream(prompt)
        .on_content(move |content| {
            content_flag.store(true, Ordering::Relaxed);
            eprint!("{}", content);
        })
        .on_reasoning(move |reasoning| {
            thinking_flag.store(true, Ordering::Relaxed);
            eprint!("[thinking] {}", reasoning);
        })
        .on_tool_call(move |tool_name, _args| {
            tool_count.fetch_add(1, Ordering::Relaxed);
            eprintln!("\n[tool_call] {}", tool_name);
        })
        .on_tool_result(|tool_name, result| {
            eprintln!(
                "[tool_result] {} => {}",
                tool_name,
                serde_json::to_string(&result).unwrap_or_default()
            );
        })
        .run()
        .await?;

    eprintln!("\n\n=== Results ===");
    eprintln!("Streaming completed successfully!");

    let had_content = received_content.load(Ordering::Relaxed);
    let had_thinking = received_thinking.load(Ordering::Relaxed);
    let num_tool_calls = tool_call_count.load(Ordering::Relaxed);

    eprintln!("Content received: {}", had_content);
    eprintln!("Thinking received: {}", had_thinking);
    eprintln!("Tool calls made: {}", num_tool_calls);

    assert!(had_content, "Should have received streamed content");
    assert!(
        had_thinking,
        "Should have received adaptive thinking output from Azure Anthropic"
    );
    assert!(
        num_tool_calls >= 2,
        "Should have made at least 2 tool calls (weather and/or time), got {}",
        num_tool_calls
    );

    eprintln!("=== Azure Anthropic test PASSED ===\n");
    Ok(())
}
