---
name: appam-agent-implementation
description: Implement, extend, debug, or review Rust AI agents built with Appam. Use when working with `appam::prelude::*`, `Agent::quick`, `AgentBuilder`, `TomlAgent`, Appam tool macros, streaming consumers, session persistence, trace output, provider configuration, or Appam-specific runtime behavior and caveats.
---

# Appam Agent Implementation

Treat this file as the source-backed operating manual for building agents with Appam.

## Keep the mental model straight

Appam is a Rust agent runtime with four core layers:

- Construction: `Agent::quick(...)`, `Agent::new(...)`, `AgentBuilder`, and `TomlAgent`.
- Tools: Rust `Tool` implementations, `#[tool]` macro-generated tools, or TOML-declared Python tools.
- Runtime loop: provider selection, streaming, tool execution, continuation handling, tracing, and history.
- Provider adapters: OpenAI, OpenAI Codex, Anthropic, OpenRouter, Vertex, Azure OpenAI, Azure Anthropic, and Bedrock.

The important design point is that most user code never talks to a provider client directly. It builds an agent, streams a prompt, and lets Appam run the tool-calling loop.

## Use the right source of truth

If documentation and code disagree, trust them in this order:

1. `src/agent/runtime.rs` and `src/agent/runtime_agent.rs`
2. `src/agent/builder.rs`, `src/agent/toml_agent.rs`, `src/agent/quick.rs`
3. `src/tools/*.rs` and `appam_macros/src/lib.rs`
4. `tests/*.rs`
5. `examples/coding-agent-*.rs`
6. `README.md`
7. `docs/content/docs/**/*.mdx`

Read these files first for repo work:

- `README.md`
- `src/lib.rs`
- `src/agent/mod.rs`
- `src/agent/builder.rs`
- `src/agent/quick.rs`
- `src/agent/runtime.rs`
- `src/agent/runtime_agent.rs`
- `src/agent/toml_agent.rs`
- `src/agent/streaming.rs`
- `src/agent/streaming_builder.rs`
- `src/agent/history.rs`
- `src/agent/errors.rs`
- `src/tools/mod.rs`
- `src/tools/registry.rs`
- `src/tools/loader.rs`
- `src/tools/register.rs`
- `appam_macros/src/lib.rs`
- `src/llm/provider.rs`
- `src/llm/unified.rs`
- `src/llm/openai/config.rs`
- `src/llm/openai_codex/config.rs`
- `src/llm/anthropic/config.rs`
- `src/llm/openrouter/config.rs`
- `src/llm/vertex/config.rs`
- `examples/coding-agent-anthropic.rs`
- `examples/coding-agent-openai-responses.rs`
- `examples/coding-agent-openai-codex.rs`
- `examples/coding-agent-openrouter-responses.rs`
- `examples/coding-agent-vertex.rs`
- `tests/agent_builder.rs`
- `tests/tool_macro.rs`
- `tests/streaming.rs`
- `tests/logging_tracing_history.rs`
- `tests/continuation_tests.rs`
- `tests/hybrid_config.rs`
- `tests/provider_switching.rs`

## Understand the naming trap in the prelude

`use appam::prelude::*;` is the right default import, but it exports two different “Agent” concepts:

- `Agent` in the prelude is the quick-constructor helper namespace from `src/agent/quick.rs`.
- `AgentTrait` in the prelude is the actual trait.

If you need to implement the trait manually, import `appam::agent::Agent` or use `AgentTrait`. Do not confuse that trait with `Agent::quick(...)`.

## Choose the construction style deliberately

Use `Agent::quick(...)` when:

- You need the smallest working agent.
- You can express the system prompt inline.
- You do not need unusual provider tuning beyond model, prompt, and tools.

Use `Agent::new(name, model)` when:

- You want builder ergonomics but still want automatic provider detection from the model string.
- You want to write `.prompt(...)`, `.tool(...)`, and `.build()?`.

Use `AgentBuilder` when:

- You need production-oriented configuration.
- You need Anthropic thinking, caching, retries, beta features, rate limiting, or tool choice.
- You need OpenAI reasoning or service-tier controls.
- You need traces, history, continuation requirements, or explicit provider selection.

Use `TomlAgent::from_file(...)` when:

- The system prompt and tool declarations should live on disk.
- Non-Rust contributors need to edit agent behavior.
- You are okay with the current TOML tool reality: Python tools are the practical dynamic path.

Use `RuntimeAgent` directly only when you are already assembling your own `ToolRegistry` and want a concrete in-memory agent type.

## Remember the runtime configuration reality

This is one of the most important Appam-specific facts:

- The default runtime path does not automatically read `appam.toml`.
- `Agent::quick(...)`, `AgentBuilder::build()`, `RuntimeAgent::run()`, and `TomlAgent::run()` start from `load_config_from_env()`, then apply agent/runtime overrides.
- `load_global_config(...)` exists, but the standard run path does not call it for you.

Practical consequence:

- Use environment variables for provider credentials and other global defaults.
- Use builder methods or TOML agent fields for agent-specific settings.
- Do not assume dropping an `appam.toml` file into the repo changes runtime behavior unless your own code loads it explicitly.

## Start from the prelude

For most code, begin with:

```rust
use appam::prelude::*;
```

That gives you:

- `Agent`, `AgentBuilder`, `TomlAgent`, `RuntimeAgent`
- `Tool`, `ToolRegistry`, `ClosureTool`, `ToolRegistryExt`
- `tool`, `Schema`
- `StreamBuilder`, `StreamEvent`, `StreamConsumer`
- `ConsoleConsumer`, `ChannelConsumer`, `CallbackConsumer`, `TraceConsumer`
- `SessionHistory`, `SessionSummary`
- `Result`, `Context`, `json`, `Value`, `Arc`, and serde derives

If you need lower-level types like `SqliteTraceConsumer`, import them explicitly.

## Build the smallest correct agent

Use this as the canonical minimal pattern:

```rust
use appam::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::quick(
        "anthropic/claude-sonnet-4-5",
        "You are a concise Rust assistant.",
        vec![],
    )?;

    agent
        .stream("Explain ownership in three sentences.")
        .on_content(|text| print!("{text}"))
        .run()
        .await?;

    println!();
    Ok(())
}
```

Current `Agent::quick(...)` behavior worth remembering:

- It returns a `RuntimeAgent`.
- It auto-detects the provider from the model string.
- It sets `temperature(0.7)`, `max_tokens(4096)`, and `top_p(0.9)`.
- Unknown model prefixes fall back to `LlmProvider::OpenRouterResponses`.

## Prefer `AgentBuilder` for real agent implementations

Use this shape for most serious work:

```rust
use appam::prelude::*;

#[derive(Deserialize, Schema)]
struct AddInput {
    #[description = "First number"]
    a: f64,
    #[description = "Second number"]
    b: f64,
}

#[derive(Serialize)]
struct AddOutput {
    sum: f64,
}

#[tool(description = "Add two numbers together")]
fn add(input: AddInput) -> Result<AddOutput> {
    Ok(AddOutput { sum: input.a + input.b })
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::new("calculator")
        .provider(LlmProvider::OpenAI)
        .model("gpt-5.5")
        .system_prompt("You are a careful calculator. Use the add tool for exact arithmetic.")
        .tool(add())
        .build()?;

    agent
        .stream("What is 42 + 58?")
        .on_content(|text| print!("{text}"))
        .on_tool_call(|name, args| println!("\ncalling {name}: {args}"))
        .on_tool_result(|name, result| println!("\n{name} -> {result}"))
        .run()
        .await?;

    Ok(())
}
```

Prefer `.tool(my_tool())` for single concrete tools. Prefer `.with_tools(vec![...])` if you already have `Vec<Arc<dyn Tool>>`.

## Implement tools the Appam way

Prefer the `#[tool]` macro for Rust tools.

### Use structured inputs for nontrivial tools

Use:

- `#[derive(Deserialize, Schema)]` on the input struct
- `#[description = "..."]` on each field
- `#[tool(description = "...")]` on the function

This produces the cleanest JSON Schema for the model and is the dominant pattern across the docs, examples, and tests.

### Use inline parameters only for simple tools

This is valid:

```rust
#[tool(description = "Multiply two numbers")]
fn multiply(
    #[arg(description = "First number")] a: f64,
    #[arg(description = "Second number")] b: f64,
) -> Result<f64> {
    Ok(a * b)
}
```

Use `#[arg(default = ...)]` only when the parameter should be optional.

### Know what the macro actually does

`#[tool]` currently expects:

- A synchronous function body
- A return type of `Result<T>`
- `T` to be serializable

The macro generates:

- A concrete tool type named from the function in PascalCase
- A constructor function with the original function name
- A `Tool` implementation
- JSON Schema generation
- JSON argument parsing

Successful macro-generated tool outputs are wrapped as:

```json
{ "output": ... }
```

### Use the correct manual `ToolSpec` shape

If you implement `Tool` manually, the spec shape is:

```rust
ToolSpec {
    type_field: "function".to_string(),
    name: "read_file".to_string(),
    description: "Read a file".to_string(),
    parameters: json!({
        "type": "object",
        "properties": {
            "file_path": { "type": "string", "description": "Absolute path to the file" }
        },
        "required": ["file_path"]
    }),
    strict: None,
}
```

Do not use a nested OpenAI-style `{ "function": { ... } }` wrapper when constructing `appam::llm::ToolSpec`. Appam’s `ToolSpec` is the direct flattened struct above.

### Validate tool inputs aggressively

Treat tool arguments as untrusted model output.

Always:

- Validate required fields.
- Validate path expectations.
- Bound resource usage.
- Return structured error context.
- Avoid shell injection, path traversal, and silent fallthrough.

Examples in `examples/coding-agent-*.rs` are intentionally simple. For production code, tighten them.

## Use closure tools when that is genuinely simpler

`ClosureTool` and `ToolRegistryExt::register_fn(...)` are good for:

- Fast prototypes
- Tests
- Small inline helpers

They are not the best default for larger tools that benefit from typed inputs and compile-time structure.

## Use TOML agents with the current limitations in mind

The current TOML agent shape is:

```toml
[agent]
name = "assistant"
model = "openai/gpt-5.5"
system_prompt = "prompt.txt"
description = "A helpful assistant"

[[tools]]
name = "echo"
schema = "tools/echo.json"
implementation = { type = "python", script = "tools/echo.py" }
```

Remember these implementation realities:

- Paths are resolved relative to the TOML file’s directory.
- `TomlAgent::from_file(...)` validates the prompt path and tool entries immediately.
- `TomlAgent::model()` falls back to `"openai/gpt-5"` if the TOML file omits a model.
- Python tools only work when Appam is built with `--features python`.
- TOML-declared Rust module loading is currently disabled in `src/tools/loader.rs`.

If you need TOML-defined prompts plus Rust tools, use the hybrid path:

```rust
let agent = TomlAgent::from_file("agents/assistant/agent.toml")?
    .with_additional_tool(Arc::new(my_tool()));
```

That hybrid pattern is covered in `tests/hybrid_config.rs`.

## Stream the agent instead of waiting for a final blob

Appam is streaming-first.

Use `.stream(prompt)` and attach handlers:

- `on_session_started`
- `on_content`
- `on_reasoning`
- `on_tool_call`
- `on_tool_result`
- `on_tool_failed`
- `on_error`
- `on_done`
- `on_tool_call_async`
- `on_tool_result_async`

This is the canonical runtime shape for CLI, TUI, and app integrations.

Use `StreamConsumer` implementations when you need reusable sinks or multiple destinations.

Built-in consumers:

- `ConsoleConsumer`: default for `run()`
- `ChannelConsumer`: forward `StreamEvent`s into a Tokio mpsc channel
- `CallbackConsumer`: wrap a closure
- `TraceConsumer`: write JSONL traces

### Handle `StreamEvent::Error` as a rich event

The current `StreamEvent::Error` variant is richer than older docs imply. It can include:

- `message`
- `failure_kind`
- `provider`
- `model`
- `http_status`
- `request_payload`
- `response_payload`
- `provider_response_id`

If you build custom consumers, log or persist those fields.

## Enable tracing and history explicitly

History and traces are opt-in.

For a runtime-built agent:

```rust
let agent = AgentBuilder::new("stateful-agent")
    .provider(LlmProvider::Anthropic)
    .model("claude-sonnet-4-5")
    .system_prompt("You are helpful.")
    .enable_history()
    .history_db_path("data/sessions.db")
    .auto_save_sessions(true)
    .enable_traces()
    .trace_format(TraceFormat::Detailed)
    .build()?;
```

What happens when enabled:

- History uses `SessionHistory` over SQLite.
- The default DB path is `data/sessions.db`.
- `continue_session(...)` only works when the agent’s runtime points at the same history database.
- `enable_traces()` causes the runtime to attach `TraceConsumer`.
- Trace output writes `session-<id>.jsonl`.
- The runtime also writes `session-<id>.json` at the end of the session when traces are enabled.
- `TraceFormat::Compact` omits reasoning entries.

## Use continuation enforcement when the agent must finish by calling a tool

Appam supports required completion tools via:

- `require_completion_tools(...)`
- `max_continuations(...)`
- `continuation_message(...)`

Use this when the agent is not allowed to “just answer” and must call a finalizing tool.

Source-backed behavior:

- The runtime injects a continuation user message if the session ends without a required tool call.
- Default continuation budget is `2`.
- The continuation counter is based on exact message matching against the configured continuation message.
- Exhaustion produces a structured `SessionFailureKind::RequiredCompletionToolMissing`.

See `src/agent/runtime.rs` and `tests/continuation_tests.rs`.

## Know the provider-specific controls

### Anthropic

Use:

- `.provider(LlmProvider::Anthropic)`
- `.thinking(...)`
- `.caching(...)`
- `.tool_choice(...)`
- `.effort(...)`
- `.beta_features(...)`
- `.retry(...)`
- `.rate_limiter(...)`

This is the richest builder surface in Appam. `examples/coding-agent-anthropic.rs` is the best source example.

### OpenAI

Use:

- `.provider(LlmProvider::OpenAI)`
- `.openai_reasoning(...)`
- `.openai_text_verbosity(...)`
- `.openai_service_tier(...)`
- `.openai_pricing_model(...)`

OpenAI defaults live in `src/llm/openai/config.rs`.

### OpenAI Codex

Use:

- `.provider(LlmProvider::OpenAICodex)`
- `.openai_codex_access_token(...)`
- `.openai_reasoning(...)`

Auth precedence is:

1. `OpenAICodexConfig.access_token`
2. `OPENAI_CODEX_ACCESS_TOKEN`
3. cached OAuth credentials in `OPENAI_CODEX_AUTH_FILE` or the default auth file

Use the interactive login helpers only for trusted local developer flows. The example is `examples/coding-agent-openai-codex.rs`.

### OpenRouter

Choose the API flavor explicitly:

- `LlmProvider::OpenRouterCompletions`
- `LlmProvider::OpenRouterResponses`

Use:

- `.openrouter_api_key(...)`
- `.openrouter_reasoning(...)`
- `.openrouter_provider_routing(...)`
- `.openrouter_transforms(...)`
- `.openrouter_models(...)`

`Agent::quick("openrouter/...")` resolves to `OpenRouterResponses`.

### Vertex

Use:

- `.provider(LlmProvider::Vertex)`
- `.vertex_api_key(...)`

Current caveat:

- `AgentBuilder` does not expose every `VertexConfig` field.
- Advanced Vertex settings like `thinking`, `function_calling_mode`, `allowed_function_names`, and `include_thoughts` are currently configured through env vars or lower-level config/client work, not rich builder methods.

See `src/llm/vertex/config.rs` and `examples/coding-agent-vertex.rs`.

### Azure OpenAI

Use explicit provider selection:

```rust
.provider(LlmProvider::AzureOpenAI {
    resource_name: "my-resource".to_string(),
    api_version: "2025-04-01-preview".to_string(),
})
```

Do not expect quick provider detection to infer Azure.

### Azure Anthropic

Use explicit provider selection:

```rust
.provider(LlmProvider::AzureAnthropic {
    base_url: "https://my-resource.services.ai.azure.com/anthropic".to_string(),
    auth_method: appam::llm::anthropic::AzureAnthropicAuthMethod::XApiKey,
})
```

Treat `.model(...)` as the Azure deployment/model name for that endpoint.

### Bedrock

Use explicit provider selection:

```rust
.provider(LlmProvider::Bedrock {
    region: "us-east-1".to_string(),
    model_id: "us.anthropic.claude-sonnet-4-5-20250514-v1:0".to_string(),
    auth_method: appam::llm::anthropic::BedrockAuthMethod::SigV4,
})
```

Important:

- SigV4 is the streaming-friendly default.
- Bearer token auth is the non-streaming path.

## Set the right environment variables

Use environment variables as the safest standard runtime path.

Minimum credentials by provider:

- OpenAI: `OPENAI_API_KEY`
- OpenAI Codex: `OPENAI_CODEX_ACCESS_TOKEN` or cached auth file
- Anthropic: `ANTHROPIC_API_KEY`
- OpenRouter: `OPENROUTER_API_KEY`
- Vertex API key mode: `GOOGLE_VERTEX_API_KEY` or `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- Vertex bearer mode: `GOOGLE_VERTEX_ACCESS_TOKEN`
- Azure OpenAI: `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_RESOURCE`
- Azure Anthropic: `AZURE_API_KEY` or `AZURE_ANTHROPIC_API_KEY`, plus `AZURE_ANTHROPIC_BASE_URL` or `AZURE_ANTHROPIC_RESOURCE`
- Bedrock SigV4: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- Bedrock bearer: `AWS_BEARER_TOKEN_BEDROCK`

Useful Appam runtime env vars:

- `APPAM_PROVIDER`
- `APPAM_LOG_LEVEL`
- `APPAM_LOGS_DIR`
- `APPAM_LOG_FORMAT`
- `APPAM_TRACE_FORMAT`
- `APPAM_ENABLE_TRACES`
- `APPAM_ENABLE_LOGS`
- `APPAM_HISTORY_ENABLED`
- `APPAM_HISTORY_DB_PATH`

## Use the current CLI accurately

The CLI surface is real, but a few capabilities are intentionally limited.

Supported commands:

- `appam new <name>`
- `appam validate agents/.../agent.toml`
- `appam run agents/.../agent.toml "prompt"`
- `appam tracing --traces-dir logs`

Current caveats:

- `appam run` is single-shot.
- Omitting the prompt does not start a real interactive session yet.
- `appam serve` exists as a command, but the underlying web API is hard-disabled.

## Respect the current web API reality

The legacy Axum API is present in-tree but disabled.

Facts from `src/web/mod.rs`:

- `serve(...)` always refuses to start.
- The disablement is intentional and security-related.
- The route code is kept for redesign/reference, not for production use today.

Do not implement new user-facing features assuming `appam serve` is available unless you are explicitly reworking that subsystem.

The trace visualizer server is separate and still usable through `appam tracing` or `web::serve_tracing(...)`.

## Know the biggest sharp edges

These are the Appam-specific traps most likely to waste time:

- `appam.toml` is not auto-loaded by the normal run path.
- `appam::tools::builtin::*` is not implemented. `src/tools/builtin/mod.rs` is a placeholder.
- `ToolRegistry::with_builtins()` currently returns an empty registry.
- TOML Rust tool loading is disabled in `src/tools/loader.rs`.
- TOML Python tools require the `python` feature.
- `AgentBuilder::tools(...)` takes `Vec<Arc<dyn Tool>>`; it does not accept raw concrete tools.
- Use repeated `.tool(...)` calls if you want no-`Arc` ergonomics.
- `build_with_stream()` returns a receiver but does not automatically wire agent output into it. Prefer `run_streaming(..., Box::new(ChannelConsumer::new(tx)))` for real channel streaming.
- `RuntimeAgent::model()` and `TomlAgent::model()` fall back to `"openai/gpt-5"`.
- `AppConfig::default()` starts with `LlmProvider::OpenRouterCompletions`, but `Agent::quick(...)` falls back to `OpenRouterResponses` when provider detection fails.

## Use the examples as implementation templates

Map your task to the nearest example first:

- `examples/coding-agent-anthropic.rs`: richest builder configuration and Anthropic-specific controls
- `examples/coding-agent-openai-responses.rs`: OpenAI Responses path
- `examples/coding-agent-openai-codex.rs`: ChatGPT Codex auth flow
- `examples/coding-agent-openrouter-responses.rs`: OpenRouter Responses path
- `examples/coding-agent-openrouter-completions.rs`: OpenRouter Completions path
- `examples/coding-agent-vertex.rs`: Vertex/Gemini path
- `examples/coding-agent-azure-openai.rs`: Azure OpenAI path
- `examples/coding-agent-azure-anthropic.rs`: Azure Anthropic path
- `examples/coding-agent-bedrock.rs`: Bedrock path

One important pattern those examples reveal:

- They define their own `read_file`, `write_file`, `bash`, and `list_files` tools inline because the built-in Rust tool module is not ready.

## Debug Appam agents methodically

When an Appam agent misbehaves:

1. Verify provider credentials and model selection first.
2. Confirm which provider was actually selected.
3. Check whether history and traces were explicitly enabled.
4. Inspect `logs/session-<id>.jsonl` and `logs/session-<id>.json` if traces are on.
5. Inspect `StreamEvent::Error` payloads, not just the display string.
6. Verify tool names and schema field names match exactly.
7. Remember that tool macro outputs are wrapped under `"output"`.
8. For TOML agents, confirm every path is relative to the TOML file directory.
9. If using continuation enforcement, confirm the required tool names and continuation message are exactly what the runtime expects.

Use these tests as behavior references while debugging:

- `tests/tool_macro.rs`
- `tests/streaming.rs`
- `tests/logging_tracing_history.rs`
- `tests/continuation_tests.rs`
- `tests/hybrid_config.rs`
- `tests/provider_switching.rs`

## Follow this implementation checklist

- Import `appam::prelude::*`.
- Pick the construction path that matches the task.
- Set credentials via environment variables unless there is a strong reason not to.
- Prefer `#[tool]` plus `#[derive(Deserialize, Schema)]` for Rust tools.
- Use `AgentBuilder` for anything beyond the most minimal prototype.
- Stream responses instead of only calling `run()` blindly.
- Enable history and traces explicitly if you need continuity or observability.
- Use an example matching the provider as your template.
- Avoid the disabled TOML Rust tool path and the disabled web API path.
- If you modify framework behavior inside this repo, update the closest example, test, and doc page alongside the source.
