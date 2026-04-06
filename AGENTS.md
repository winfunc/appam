# AGENTS.md

This file applies to the entire `appam` crate, including `appam_macros/`.

## Scope and Intent

`appam` is a Rust-first agent orchestration library and CLI for long-horizon, tool-using, traceable AI systems. The highest priorities here are:

- correctness under multi-turn tool execution
- stable streaming behavior
- safe session persistence and trace capture
- provider portability without leaking provider quirks into the shared runtime
- secure handling of credentials, prompts, tool inputs, and filesystem access

Prefer small, surgical diffs. This crate has broad surface area, but most tasks should touch one subsystem and its adjacent tests, not the whole tree.

## Repository Map

- `src/agent/`
  Agent trait, builder APIs, runtime orchestration, streaming builders, continuation logic, session history, and stream consumers.
- `src/llm/`
  Unified provider abstraction plus provider-specific clients for Anthropic, OpenAI, OpenAI Codex, OpenRouter, Vertex, Azure OpenAI, Azure Anthropic, and Bedrock.
- `src/tools/`
  Tool traits, registry/state management, dynamic loading, closure tools, and optional Python tool integration.
- `src/web/`
  Experimental web API, trace visualizer, filesystem operations, SSE helpers, and session APIs.
- `src/http/`
  Shared HTTP client pooling and request infrastructure.
- `src/config/`
  Hierarchical config builders, env overrides, and runtime config resolution.
- `appam_macros/`
  Procedural macros, especially `#[tool]`.
- `examples/`
  Real user-facing examples for provider integrations and stateful agents.
- `tests/`
  Integration tests, smoke-test scripts, and provider/session coverage.

## Core Rules

### 1. Preserve public API quality

`src/lib.rs` enables `#![warn(missing_docs)]` and `#![warn(clippy::all)]`. Treat public APIs as product surface, not internal glue.

- Add extensive Rust doc comments for all new public items.
- Add or update module-level docs when introducing a new module or substantially changing its role.
- Keep examples in doc comments correct and minimally runnable.
- If public construction patterns, re-exports, model string formats, or provider environment variables change, update `README.md` and the relevant examples in the same change.

### 2. Prefer `tracing` over ad hoc output

Outside CLI UX, tests, and examples, use structured `tracing` logs rather than `println!`.

- Include stable fields such as `agent`, `provider`, `model`, `session_id`, `tool`, `tool_call_id`, `path`, or `status` where relevant.
- Never log secrets, auth headers, OAuth tokens, API keys, cookies, raw credential files, or personally identifiable information.
- Do not dump full provider payloads into logs unless they have already been explicitly sanitized for failure capture.
- When adding new provider failure instrumentation, follow the existing sanitized diagnostics pattern rather than inventing a second one.

### 3. Fail closed

Use `anyhow::Result` and `Context` pervasively.

- Validate all external input: provider payloads, tool args, TOML, env vars, HTTP params, paths, and streamed events.
- Prefer explicit, user-facing errors over panics.
- Avoid `unwrap` and `expect` outside tests, examples, or truly impossible invariants.
- When uncertain about safety or state correctness, reject the operation instead of guessing.

### 4. Keep provider-specific complexity contained

The shared runtime should remain provider-agnostic.

- Put wire-format, auth, retry, SSE parsing, and provider quirks inside the provider modules under `src/llm/`.
- Keep normalized semantics in the unified/provider abstraction layer.
- Do not let one provider's special case silently change behavior for all providers without targeted tests.
- Preserve model-string inference and provider-switching behavior unless the task explicitly changes it.

### 5. Respect existing architecture

- Reuse `AgentBuilder`, `RuntimeAgent`, `ToolRegistry`, `StreamBuilder`, `SessionHistory`, shared HTTP clients, and existing config builders instead of creating parallel abstractions.
- Keep changes minimal and local.
- Do not introduce new dependencies unless necessary and justified by the task.
- If package metadata changes involve versions, keep `appam` and `appam_macros` aligned.

## Subsystem-Specific Guidance

### Agent Runtime, Streaming, and History

These files define the crate's behavioral core. Be conservative.

- Preserve stream event ordering and consumer semantics.
- Continuation logic must remain bounded. Never introduce paths that can loop indefinitely or replay tool calls unpredictably.
- Session persistence must remain backward-compatible for existing SQLite data unless the task explicitly requests schema changes.
- If you change history or trace schemas, provide safe upgrade behavior and add focused tests.
- Session and trace writes should stay durable and debuggable, but must not leak secrets in stored artifacts.

Relevant files:

- `src/agent/runtime.rs`
- `src/agent/runtime_agent.rs`
- `src/agent/streaming.rs`
- `src/agent/streaming_builder.rs`
- `src/agent/history.rs`
- `src/agent/consumers/`

### Tools, Registry State, and Macros

Tool calls are untrusted input originating from an LLM. Treat them accordingly.

- Validate tool arguments defensively even if the schema claims they are well-formed.
- Default tool concurrency is intentionally conservative. Only mark a tool `ParallelSafe` when you can defend that it has no ordering or shared-state hazards.
- `ToolContext`, `State<T>`, and `SessionState<T>` are runtime-only injected values and must not leak into tool JSON schemas.
- Managed state access should remain fail-closed with explicit errors when state is unavailable or unregistered.
- Reuse `ToolRegistry` and existing registration paths instead of duplicating tool execution plumbing.

If you touch `appam_macros` or tool-schema generation:

- verify schema output
- verify async tool injection behavior
- verify state injection behavior
- verify runtime execution through the registry

Relevant files:

- `src/tools/mod.rs`
- `src/tools/registry.rs`
- `src/tools/register.rs`
- `src/tools/loader.rs`
- `src/tools/python.rs`
- `appam_macros/src/lib.rs`
- `tests/tool_macro.rs`

### Provider Clients

Provider work is easy to get subtly wrong. Maintain parity carefully.

- Keep auth handling provider-specific and never log secrets.
- Preserve retry/backoff, stream parsing, and request construction behavior unless the task explicitly changes them.
- Keep normalized outputs stable for the runtime and consumers.
- When changing provider env var handling or model defaults, update docs and examples.
- For OpenAI Codex auth, treat `~/.appam/auth.json` and related cached auth as sensitive local credentials. Never print, commit, or expose its contents.
- For Bedrock and Azure variants, be careful not to break auth-mode-specific behavior when editing shared branches.

Relevant files:

- `src/llm/provider.rs`
- `src/llm/unified.rs`
- `src/llm/anthropic/`
- `src/llm/openai/`
- `src/llm/openai_codex/`
- `src/llm/openrouter/`
- `src/llm/vertex/`

### Web, Trace Visualizer, and Filesystem Operations

This area is security-sensitive.

- The legacy web API is intentionally disabled in `src/web/mod.rs`. Do not enable it, relax the guard, or weaken trust boundaries unless the task explicitly requires it and you also add the necessary auth/security design.
- Path handling must defend against traversal, symlink surprises, and accidental writes outside the intended agent/traces directories.
- File writes should be deliberate and easy to reason about.
- Route handlers must return clean errors and avoid leaking internal filesystem layout or secrets.

Relevant files:

- `src/web/mod.rs`
- `src/web/routes.rs`
- `src/web/fs_operations.rs`
- `src/web/session.rs`
- `src/web/trace_routes.rs`
- `src/web/trace_parser.rs`

### Config and User-Facing Behavior

Configuration precedence matters here:

1. defaults
2. global config
3. per-agent TOML
4. environment variables

Do not accidentally invert that ordering.

- Preserve backward-compatible config parsing where possible.
- If new config fields are added, thread them through builders, env loading, docs, and tests together.
- Keep environment variable handling explicit and documented.

## Verification

Run the smallest relevant set of commands that proves the change. Always format first if you edited Rust code.

Common baseline:

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Targeted verification by area:

- Runtime/builder/config:

```bash
cargo test --test agent_builder
cargo test --test continuation_tests
cargo test --test hybrid_config
cargo test --test provider_switching
```

- Streaming/history/logging/traces:

```bash
cargo test --test streaming
cargo test --test logging_tracing_history
```

- Tool registry and proc macros:

```bash
cargo test --test tool_macro
```

- Provider-specific behavior:

```bash
cargo test --test bedrock_opus46_streaming
```

- Python integration changes:

```bash
cargo test --all-features
```

Examples and live smoke tests:

- `cargo test` already builds examples, which is usually enough for compile coverage.
- `tests/test-run-examples.sh` and `tests/test-run-vertex-model-matrix.sh` exercise real providers and require credentials. Use them only when the change affects live provider behavior or example correctness.

Examples:

```bash
bash tests/test-run-examples.sh --example coding-agent-openai-responses
bash tests/test-run-vertex-model-matrix.sh --models "gemini-2.5-flash,gemini-2.5-pro"
```

If you change only documentation or this `AGENTS.md`, no code test run is required.

## Practical Editing Checklist

Before changing code, identify which layer owns the behavior:

- provider adapter
- runtime/session orchestration
- tool registry or macro generation
- config loading
- web/filesystem surface
- docs/examples/public API

Before finalizing a change, make sure you did not:

- add generic abstractions where an existing one already fits
- leak provider-specific behavior into shared runtime code
- log sensitive inputs or credentials
- re-enable the disabled web API
- break schema compatibility for tools or SQLite-backed state without tests
- change public behavior without updating docs and examples

## Defaults for This Crate

- Rust edition: 2021
- Minimum Rust version: 1.89
- Preferred style: idiomatic, explicit, readable Rust with strong docs and structured logs
- Preferred runtime behavior: deterministic, fail-closed, and observable
