//! Integration tests for the #[tool] macro.

use appam::prelude::*;
use serde_json::json;
use std::sync::Arc;

// Test basic tool macro usage
#[tool(description = "Echoes back the input message")]
fn echo_tool(message: String) -> anyhow::Result<String> {
    Ok(format!("Echo: {}", message))
}

// Test tool with multiple parameters
#[tool(
    name = "calculator",
    description = "Performs basic arithmetic operations"
)]
fn calc_tool(a: f64, b: f64, operation: String) -> anyhow::Result<f64> {
    match operation.as_str() {
        "add" => Ok(a + b),
        "subtract" => Ok(a - b),
        "multiply" => Ok(a * b),
        "divide" => {
            if b == 0.0 {
                anyhow::bail!("Division by zero");
            }
            Ok(a / b)
        }
        _ => anyhow::bail!("Unknown operation: {}", operation),
    }
}

// Test tool with custom name
#[tool(name = "uppercase", description = "Converts text to uppercase")]
fn to_upper(text: String) -> anyhow::Result<String> {
    Ok(text.to_uppercase())
}

#[derive(Default)]
struct CounterState {
    count: u64,
}

#[tool(description = "Increment the session counter")]
async fn bump_counter(
    ctx: ToolContext,
    counter: SessionState<CounterState>,
    #[arg(description = "Amount to add")] amount: u64,
) -> anyhow::Result<String> {
    let total = counter.update(|state| {
        state.count += amount;
        state.count
    })?;
    Ok(format!("{}:{}", ctx.session_id(), total))
}

#[derive(Clone)]
struct PrefixState {
    prefix: String,
}

#[tool(description = "Prefix the input with app-managed state")]
async fn prefix_message(
    state: State<PrefixState>,
    #[arg(description = "Message to prefix")] message: String,
) -> anyhow::Result<String> {
    Ok(format!("{}{}", state.prefix, message))
}

#[derive(Deserialize, Schema)]
struct AppendInput {
    #[description = "Value to append"]
    value: String,
}

#[tool(description = "Append a value into session state")]
async fn append_value(
    items: SessionState<Vec<String>>,
    input: AppendInput,
) -> anyhow::Result<usize> {
    items.update(|values| {
        values.push(input.value);
        values.len()
    })
}

#[test]
fn test_echo_tool_macro() {
    let tool = echo_tool();

    assert_eq!(tool.name(), "echo_tool");

    let spec = tool.spec().unwrap();
    assert_eq!(spec.name, "echo_tool");
    assert_eq!(spec.description, "Echoes back the input message");

    let result = tool.execute(json!({"message": "hello"})).unwrap();
    assert_eq!(result["output"], "Echo: hello");
}

#[test]
fn test_calculator_tool_macro() {
    let tool = calc_tool();

    assert_eq!(tool.name(), "calculator");

    let spec = tool.spec().unwrap();
    assert_eq!(spec.name, "calculator");

    // Test addition
    let result = tool
        .execute(json!({
            "a": 10.0,
            "b": 5.0,
            "operation": "add"
        }))
        .unwrap();
    assert_eq!(result["output"], 15.0);
}

#[test]
fn test_uppercase_tool_macro() {
    let tool = to_upper();

    assert_eq!(tool.name(), "uppercase");

    let result = tool.execute(json!({"text": "hello"})).unwrap();
    assert_eq!(result["output"], "HELLO");
}

#[test]
fn test_tool_macro_in_agent() {
    let agent = AgentBuilder::new("test-agent")
        .system_prompt("You are a test assistant.")
        .with_tools(vec![
            Arc::new(echo_tool()),
            Arc::new(calc_tool()),
            Arc::new(to_upper()),
        ])
        .build()
        .unwrap();

    let tools = agent.available_tools().unwrap();
    assert_eq!(tools.len(), 3);

    // Verify tool names
    let tool_names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"echo_tool"));
    assert!(tool_names.contains(&"calculator"));
    assert!(tool_names.contains(&"uppercase"));
}

#[test]
fn test_tool_macro_execution() {
    let registry = ToolRegistry::new();
    registry.register(Arc::new(echo_tool()));
    registry.register(Arc::new(calc_tool()));

    // Test echo
    let result = registry
        .execute("echo_tool", json!({"message": "test"}))
        .unwrap();
    assert_eq!(result["output"], "Echo: test");

    // Test calculator
    let result = registry
        .execute(
            "calculator",
            json!({
                "a": 20.0,
                "b": 4.0,
                "operation": "multiply"
            }),
        )
        .unwrap();
    assert_eq!(result["output"], 80.0);
}

#[test]
fn test_tool_macro_error_handling() {
    let tool = calc_tool();

    // Test division by zero
    let result = tool.execute(json!({
        "a": 10.0,
        "b": 0.0,
        "operation": "divide"
    }));

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Division by zero"));
}

#[tokio::test]
async fn test_async_tool_macro_injected_params_are_excluded_from_schema() {
    let tool = bump_counter();
    let spec = tool.spec().unwrap();
    let properties = spec
        .parameters
        .get("properties")
        .and_then(|value| value.as_object())
        .unwrap();

    assert!(properties.contains_key("amount"));
    assert!(!properties.contains_key("ctx"));
    assert!(!properties.contains_key("counter"));
}

#[tokio::test]
async fn test_async_tool_macro_session_state_execution() {
    let registry = ToolRegistry::new();
    registry.session_state_with::<CounterState, _>(CounterState::default);
    registry.register_async(Arc::new(bump_counter()));

    let first = registry
        .execute_with_context(
            ToolContext::new("session-1", "agent", "call-1"),
            "bump_counter",
            json!({"amount": 2}),
        )
        .await
        .unwrap();
    let second = registry
        .execute_with_context(
            ToolContext::new("session-1", "agent", "call-2"),
            "bump_counter",
            json!({"amount": 3}),
        )
        .await
        .unwrap();

    assert_eq!(first["output"], "session-1:2");
    assert_eq!(second["output"], "session-1:5");
}

#[tokio::test]
async fn test_async_tool_macro_app_state_execution() {
    let registry = ToolRegistry::new();
    registry.manage(PrefixState {
        prefix: "hello-".to_string(),
    });
    registry.register_async(Arc::new(prefix_message()));

    let result = registry
        .execute_with_context(
            ToolContext::new("session-1", "agent", "call-1"),
            "prefix_message",
            json!({"message": "world"}),
        )
        .await
        .unwrap();

    assert_eq!(result["output"], "hello-world");
}

#[tokio::test]
async fn test_async_tool_macro_typed_input_with_injected_state() {
    let registry = ToolRegistry::new();
    registry.session_state_with::<Vec<String>, _>(Vec::new);
    registry.register_async(Arc::new(append_value()));

    let result = registry
        .execute_with_context(
            ToolContext::new("session-typed", "agent", "call-1"),
            "append_value",
            json!({"value": "alpha"}),
        )
        .await
        .unwrap();

    assert_eq!(result["output"], 1);
}
