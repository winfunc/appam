<img align="right" src="assets/appam-logo.png" alt="Appam logo" width="180" />

<h1>Appam</h1>

<p>
  <strong>Rust agent orchestration library for tool-using, long-horizon, traceable AI systems.</strong>
</p>
<p>
  Build agents directly in Rust with typed tools, streaming events, durable sessions, JSONL traces, and one runtime that can target multiple LLM providers.
</p>
<p>
  <a href="https://crates.io/crates/appam"><img src="https://img.shields.io/crates/v/appam?style=for-the-badge" alt="Crates.io"></a>
  <a href="https://docs.rs/appam"><img src="https://img.shields.io/docsrs/appam?style=for-the-badge" alt="Docs.rs"></a>
  <a href="https://github.com/winfunc/appam/stargazers"><img src="https://img.shields.io/github/stars/winfunc/appam?style=for-the-badge" alt="GitHub stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="MIT License"></a>
</p>
<p>
  <a href="#why-appam">Why Appam</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#tool-using-agents">Tools</a> •
  <a href="#streaming-sessions-and-traces">Streaming & Sessions</a> •
  <a href="#providers">Providers</a> •
  <a href="#examples-and-docs">Examples & Docs</a>
</p>

> [!TIP]
> If Appam is useful, ⭐ Star the repo. It materially helps the project reach more systems thus better and reliable agents for all of us.

Appam is for agent systems that need more than a toy chat loop. It is designed for workloads where the hard parts are operational: multi-turn tool use, session continuation, event streaming, traceability, provider switching, and reliability under repeated runs.

> _The name `appam` is derived from the malayalam saying "അപ്പം പോലെ ചുടാം", which roughly means "as easy as baking an appam."_

## Why Appam

- **Rust-first agent construction** with `Agent::quick(...)`, `Agent::new(...)`, and `AgentBuilder`
- **Typed tool system** using the `#[tool]` macro, direct `Tool` implementations, or `ClosureTool`
- **Streaming by default** through `StreamBuilder`, `StreamConsumer`, and built-in consumers
- **Durable sessions** with SQLite-backed `SessionHistory` and `continue_session(...)`
- **Traceable runs** through built-in JSONL traces and structured stream events
- **Provider portability** across Anthropic, OpenAI, OpenAI Codex, OpenRouter, Vertex, Azure, and Bedrock
- **Production controls** for retries, continuation mechanics, reasoning, caching, rate limiting, and provider-specific tuning

## What Appam Is Good At

Appam fits best when you want to build agents like:

- Coding agents that read files, write files, and run commands
- Research or operations agents that loop through tools over many turns
- Services that need streaming output plus session persistence
- Internal automation where runs must be inspectable after the fact
- Systems that may need to switch providers without rewriting the agent runtime

If your agent is mostly "prompt in, string out", Appam still works, but its real value shows up once orchestration, tools, and observability matter.

## Installation

Add the crate and Tokio:

```bash
cargo add appam
cargo add tokio --features macros,rt-multi-thread
```

If you plan to define typed tool inputs, `serde` is useful too:

```bash
cargo add serde --features derive
```

Or add dependencies manually:

```toml
[dependencies]
appam = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
serde = { version = "1", features = ["derive"] }
```

## Provider Setup

Set credentials for the provider you want to use:

| Provider | Minimum setup |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| OpenAI Codex | `OPENAI_CODEX_ACCESS_TOKEN` or a cached login in `~/.appam/auth.json` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Vertex | `GOOGLE_VERTEX_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_VERTEX_ACCESS_TOKEN` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_RESOURCE` |
| Azure Anthropic | `AZURE_API_KEY` or `AZURE_ANTHROPIC_API_KEY`, plus `AZURE_ANTHROPIC_BASE_URL` or `AZURE_ANTHROPIC_RESOURCE` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, or `AWS_BEARER_TOKEN_BEDROCK` |

Common model override variables:

- `ANTHROPIC_MODEL`
- `OPENAI_MODEL`
- `OPENAI_CODEX_MODEL`
- `OPENROUTER_MODEL`
- `GOOGLE_VERTEX_MODEL`
- `AZURE_OPENAI_MODEL`
- `AZURE_ANTHROPIC_MODEL`
- `AWS_BEDROCK_MODEL_ID`

## Quickstart

The smallest useful Appam program is a Rust agent with streaming output:

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
        .stream("Explain ownership in Rust in three sentences.")
        .on_content(|text| print!("{text}"))
        .run()
        .await?;

    println!();
    Ok(())
}
```

`Agent::quick(...)` is the fast path:

- infers the provider from the model string
- creates a `RuntimeAgent`
- applies sensible defaults for temperature, max tokens, top-p, and retries

Examples of model strings Appam recognizes:

| Model string | Provider |
| --- | --- |
| `anthropic/claude-sonnet-4-5` | Anthropic |
| `openai/gpt-5.5` | OpenAI |
| `openai-codex/gpt-5.5` | OpenAI Codex |
| `openrouter/anthropic/claude-sonnet-4-5` | OpenRouter Responses |
| `vertex/gemini-2.5-flash` | Vertex |
| `gemini-2.5-pro` | Vertex |

## Tool-Using Agents

Appam's recommended tool path is native Rust.

The `#[tool]` macro turns normal Rust functions into runtime tools with generated JSON Schema and argument decoding:

```rust
use appam::prelude::*;

#[derive(Deserialize, Schema)]
struct AddInput {
    #[description = "First number"]
    a: i64,
    #[description = "Second number"]
    b: i64,
}

#[derive(Serialize)]
struct AddOutput {
    sum: i64,
}

#[tool(description = "Add two integers together")]
fn add(input: AddInput) -> Result<AddOutput> {
    Ok(AddOutput {
        sum: input.a + input.b,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::new("calculator", "anthropic/claude-sonnet-4-5")
        .prompt("You are a precise calculator. Always use the add tool for arithmetic.")
        .tool(add())
        .build()?;

    agent
        .stream("What is 42 + 58?")
        .on_tool_call(|name, args| println!("[tool] {name} {args}"))
        .on_tool_result(|name, result| println!("[result] {name} = {result}"))
        .on_content(|text| print!("{text}"))
        .run()
        .await?;

    println!();
    Ok(())
}
```

You can define tools in three main ways:

- `#[tool]` for the best Rust DX
- `Tool` trait implementations for full control
- `ClosureTool` for fast inline utilities

Key tooling types:

| Type | Purpose |
| --- | --- |
| `#[tool]` | Generate a `Tool` implementation from a function |
| `Schema` | Derive JSON Schema for typed input structs |
| `Tool` | Core trait for tool execution |
| `ToolRegistry` | Shared registry for tool lookup and execution |
| `ClosureTool` | Lightweight runtime tool defined from a closure |

## Agent Construction Styles

Appam gives you three Rust-first ways to construct agents:

### 1. `Agent::quick(...)`

Use this when you want the smallest amount of setup.

Best for:

- scripts
- prototypes
- simple services
- smoke tests against a provider

### 2. `Agent::new(name, model)`

This is the ergonomic middle ground. It returns an `AgentBuilder` with provider detection already applied.

```rust
let agent = Agent::new("assistant", "openai/gpt-5.5")
    .prompt("You are a helpful assistant.")
    .tool(add()) // Reuse any Tool generated with #[tool]
    .build()?;
```

### 3. `AgentBuilder`

Use this when you need explicit provider configuration and runtime control:

- reasoning or thinking settings
- prompt caching
- retry behavior
- rate limiting
- traces and session history
- Azure or Bedrock-specific provider setup

## Streaming, Sessions, and Traces

Streaming is a first-class part of the runtime, not an afterthought. Every run can emit structured events for text, reasoning, tool calls, tool results, usage updates, and completion.

### Closure-based streaming

For most applications, use `agent.stream(...)` and attach handlers:

```rust
use appam::prelude::*;

let session = agent
    .stream("Analyze this repository layout")
    .on_session_started(|id| println!("session: {id}"))
    .on_content(|text| print!("{text}"))
    .on_reasoning(|text| eprint!("{text}"))
    .on_tool_call(|name, args| println!("\n[calling {name}] {args}"))
    .on_tool_result(|name, result| println!("[done {name}] {result}"))
    .on_tool_failed(|name, error| eprintln!("[failed {name}] {error}"))
    .on_error(|error| eprintln!("error: {error}"))
    .on_done(|| println!("\ncomplete"))
    .run()
    .await?;
```

### Reusable consumers

If you need reusable pipelines, Appam also exposes `StreamConsumer` plus built-in consumers such as:

- `ConsoleConsumer`
- `ChannelConsumer`
- `CallbackConsumer`
- `TraceConsumer`

### Session persistence

If you want continuation across runs, enable history on the agent:

```rust
use appam::prelude::*;

let agent = Agent::new("researcher", "anthropic/claude-sonnet-4-5")
    .prompt("You are a research assistant.")
    .enable_history()
    .history_db_path("data/sessions.db")
    .auto_save_sessions(true)
    .build()?;

let first = agent.run("What is Rust?").await?;

let second = agent
    .continue_session(&first.session_id, "How does ownership work?")
    .await?;

println!("continued session: {}", second.session_id);
```

For direct history operations, use `SessionHistory`:

```rust
use appam::prelude::*;

let mut config = HistoryConfig::default();
config.enabled = true;
config.db_path = "data/sessions.db".into();

let history = SessionHistory::new(config).await?;
let sessions = history.list_sessions().await?;
println!("stored sessions: {}", sessions.len());
```

### Traces

Enable built-in traces on the agent when you want replayable, inspectable runs:

```rust
use appam::prelude::*;

let agent = Agent::new("audited-agent", "openai/gpt-5.5")
    .prompt("You are a careful assistant.")
    .enable_traces()
    .trace_format(TraceFormat::Detailed)
    .build()?;
```

This gives you structured event output that is much easier to inspect than plain console logs.

## Providers

Appam exposes one orchestration surface across multiple LLM providers:

| Provider | Runtime path |
| --- | --- |
| Anthropic Messages API | `LlmProvider::Anthropic` or `anthropic/...` / `claude-*` model strings |
| OpenAI Responses API | `LlmProvider::OpenAI` or `openai/...` / `gpt-*` / `o1-*` / `o3-*` model strings |
| OpenAI Codex Responses API | `LlmProvider::OpenAICodex` or `openai-codex/...` model strings |
| OpenRouter Responses API | `LlmProvider::OpenRouterResponses` or `openrouter/...` model strings |
| OpenRouter Completions API | `LlmProvider::OpenRouterCompletions` |
| Google Vertex AI | `LlmProvider::Vertex` or `vertex/...` / `gemini-*` / `google/gemini-*` model strings |
| Azure OpenAI | `LlmProvider::AzureOpenAI { .. }` |
| Azure Anthropic | `LlmProvider::AzureAnthropic { .. }` |
| AWS Bedrock | `LlmProvider::Bedrock { .. }` |

Notes:

- `Agent::quick(...)` and `Agent::new(...)` auto-detect common providers from the model string.
- Unknown model strings fall back to OpenRouter Responses.
- Azure and Bedrock are best configured explicitly through `AgentBuilder`.

## Operational Controls

Appam includes the runtime controls that usually get bolted on later:

- retries with exponential backoff
- reasoning configuration for provider families that support it
- Anthropic thinking, caching, tool choice, and rate limiting
- OpenRouter provider preferences and transform controls
- OpenAI service tier and text verbosity settings
- maximum continuations and required completion tools for long-running flows

That lets you keep the orchestration layer inside Rust instead of scattering runtime rules across wrapper scripts.

## Examples and Docs

### Start here

- [Getting started: installation](docs/content/docs/getting-started/installation.mdx)
- [Getting started: quickstart](docs/content/docs/getting-started/quickstart.mdx)
- [Getting started: first agent with tools](docs/content/docs/getting-started/first-agent.mdx)
- [Core concepts: agents](docs/content/docs/core-concepts/agents.mdx)
- [Core concepts: tools](docs/content/docs/core-concepts/tools.mdx)
- [Core concepts: streaming](docs/content/docs/core-concepts/streaming.mdx)
- [Core concepts: sessions](docs/content/docs/core-concepts/sessions.mdx)
- [Core concepts: providers](docs/content/docs/core-concepts/providers.mdx)
- [FAQ](docs/content/docs/faq.mdx)

### Working examples

- [Anthropic coding agent](examples/coding-agent-anthropic.rs)
- [OpenAI coding agent](examples/coding-agent-openai-responses.rs)
- [OpenAI Codex coding agent](examples/coding-agent-openai-codex.rs)
- [OpenRouter Responses coding agent](examples/coding-agent-openrouter-responses.rs)
- [OpenRouter Completions coding agent](examples/coding-agent-openrouter-completions.rs)
- [Vertex coding agent](examples/coding-agent-vertex.rs)
- [Azure OpenAI coding agent](examples/coding-agent-azure-openai.rs)
- [Azure Anthropic coding agent](examples/coding-agent-azure-anthropic.rs)
- [Bedrock coding agent](examples/coding-agent-bedrock.rs)

### API docs

- [docs.rs/appam](https://docs.rs/appam)

### FAQ

- [How does Appam differ with Rig? and other common questions](docs/content/docs/faq.mdx)

## Development

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
```

## License

[MIT](LICENSE)
