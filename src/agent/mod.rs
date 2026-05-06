//! Core agent abstractions, runtime entry points, and session metadata.
//!
//! This module defines the public agent contract used by the rest of the crate.
//! Most applications interact with Appam through one of three agent forms:
//!
//! - [`RuntimeAgent`] for programmatic Rust construction
//! - [`TomlAgent`] for loading agents from on-disk configuration
//! - [`AgentBuilder`] for fluent assembly of a runtime-backed agent
//!
//! The shared [`Agent`] trait keeps the runtime provider agnostic. Implementors
//! supply prompts, tool schemas, tool execution, and optional continuation
//! policy while the runtime module handles streaming, multi-turn tool loops,
//! persistence, and trace capture.

pub mod builder;
pub mod config;
pub mod consumers;
pub mod errors;
pub mod history;
pub mod quick;
pub mod runtime;
pub mod runtime_agent;
pub mod streaming;
pub mod streaming_builder;
pub mod toml_agent;

use anyhow::Result;
use async_trait::async_trait;

use crate::llm::{ChatMessage, Role, ToolSpec};
use crate::tools::{ToolConcurrency, ToolContext};

/// Common interface implemented by all Appam agent types.
///
/// An `Agent` supplies the pieces the runtime cannot infer on its own:
///
/// - a stable name for logging, history, and trace output
/// - a system prompt or prompt-loading strategy
/// - a set of tool schemas exposed to the model
/// - tool execution for provider-emitted tool calls
/// - optional continuation policy when a session ends prematurely
///
/// The runtime intentionally assumes tool arguments are untrusted model output.
/// Implementations should therefore validate inputs, fail closed on missing
/// state, and avoid side effects that depend on undocumented provider behavior.
///
/// # Examples
///
/// ```no_run
/// use appam::agent::{Agent, toml_agent::TomlAgent};
/// use anyhow::Result;
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let agent = TomlAgent::from_file("agents/assistant.toml")?;
///     agent.run("Hello, how can you help me?").await?;
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait Agent: Send + Sync {
    /// Return the agent's unique name.
    fn name(&self) -> &str;

    /// Return the provider override for this agent.
    ///
    /// If `Some(provider)`, this agent will use the specified provider regardless
    /// of the global configuration. If `None`, the global provider config is used.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::Agent;
    /// # use appam::llm::LlmProvider;
    /// # struct MyAgent;
    /// # impl Agent for MyAgent {
    /// #     fn name(&self) -> &str { "my-agent" }
    /// #     fn system_prompt(&self) -> anyhow::Result<String> { Ok("prompt".to_string()) }
    /// #     fn available_tools(&self) -> anyhow::Result<Vec<appam::llm::ToolSpec>> { Ok(vec![]) }
    ///     fn provider(&self) -> Option<LlmProvider> {
    ///         Some(LlmProvider::Anthropic)  // Force Anthropic for this agent
    ///     }
    /// # }
    /// ```
    fn provider(&self) -> Option<crate::llm::LlmProvider> {
        None // Default: use global config
    }

    /// Apply agent-specific configuration overrides to global config.
    ///
    /// This hook exists so agent implementations can inject configuration that
    /// was determined at construction time, such as provider selection, model
    /// overrides, retry settings, or history/tracing preferences.
    ///
    /// The runtime applies this after loading global configuration and before
    /// constructing the provider client, so implementations should treat it as
    /// the last agent-controlled layer in the configuration precedence chain.
    fn apply_config_overrides(&self, _cfg: &mut crate::config::AppConfig) {
        // Default: no overrides
    }

    /// Return the list of tools required for session completion.
    ///
    /// If `Some`, the runtime will automatically inject a continuation message
    /// when the session ends without calling any of these tools. This is useful
    /// for agents whose contract requires a concrete side effect before the run
    /// may be considered complete.
    ///
    /// Default implementation returns `None`, meaning the runtime accepts the
    /// model's first completed answer without any additional tool requirements.
    fn required_completion_tools(&self) -> Option<&Vec<String>> {
        None
    }

    /// Return the maximum number of continuation attempts.
    ///
    /// This bounds the runtime's recovery behavior when the model stops before
    /// calling required completion tools. A low number avoids infinite loops
    /// while still giving the model a chance to recover from an early stop.
    fn max_continuations(&self) -> usize {
        2
    }

    /// Return the custom continuation message, if any.
    ///
    /// If `Some`, this message will be injected when the session ends without
    /// calling required tools. Use this to explain the missing side effect in
    /// domain terms rather than relying on the runtime's generic fallback.
    fn continuation_message(&self) -> Option<&str> {
        None
    }

    /// Return the full system prompt for this agent.
    ///
    /// The system prompt defines the agent's personality, capabilities,
    /// instructions, and constraints. It is sent as the first message in
    /// every conversation.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt cannot be loaded (e.g., file not found).
    fn system_prompt(&self) -> Result<String>;

    /// Return the set of tool specifications available to this agent.
    ///
    /// Tools are exposed to the LLM via their JSON schemas. The LLM can
    /// decide to invoke tools based on user queries and the system prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if tool specifications cannot be loaded.
    fn available_tools(&self) -> Result<Vec<ToolSpec>>;

    /// Resolve a tool by name and execute it.
    ///
    /// Default implementation returns an error. Agents should override this
    /// to provide tool resolution logic. New integrations should prefer
    /// [`Agent::execute_tool_with_context`] so tools receive runtime metadata
    /// and fail-closed access to managed state.
    fn execute_tool(&self, name: &str, _args: serde_json::Value) -> Result<serde_json::Value> {
        Err(anyhow::anyhow!("Tool not found: {}", name))
    }

    /// Resolve a tool by name and execute it with runtime metadata.
    ///
    /// The default implementation preserves backward compatibility by
    /// delegating to the legacy synchronous `execute_tool(...)` path and
    /// ignoring the provided context. Agents backed by a `ToolRegistry`
    /// should override this to call the registry's async/context-aware
    /// execution entrypoint.
    async fn execute_tool_with_context(
        &self,
        name: &str,
        _ctx: ToolContext,
        args: serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.execute_tool(name, args)
    }

    /// Return the concurrency policy for a specific tool.
    ///
    /// Legacy agents default every tool to serial execution. Registry-backed
    /// agents should override this to surface per-tool policies from the
    /// underlying tool registry. Declaring a tool parallel-safe only affects
    /// runtime scheduling; tool implementations remain responsible for their
    /// own synchronization and external side-effect safety.
    fn tool_concurrency(&self, _name: &str) -> ToolConcurrency {
        ToolConcurrency::SerialOnly
    }

    /// Whether Appam should request provider-side parallel tool batching.
    ///
    /// This only affects provider request wiring. Runtime execution still
    /// additionally requires `max_concurrent_tool_executions() > 1` and that
    /// every returned tool in the batch be marked `ParallelSafe`. The default
    /// stays `false` because some providers emit subtly different tool-call
    /// semantics when batching is enabled.
    fn provider_parallel_tool_calls(&self) -> bool {
        false
    }

    /// Maximum number of concurrent tool executions allowed for one batch.
    ///
    /// The runtime clamps execution to this limit after the provider has
    /// already decided which tool calls belong in a batch.
    fn max_concurrent_tool_executions(&self) -> usize {
        1
    }

    /// Run the agent with a user prompt.
    ///
    /// Orchestrates the full conversation loop:
    /// 1. Builds initial messages (system + user)
    /// 2. Streams LLM response with tool calling
    /// 3. Executes requested tools
    /// 4. Continues until LLM stops requesting tools
    /// 5. Returns session metadata
    ///
    /// Output is streamed to console with default formatting.
    ///
    /// The default implementation is provided by `runtime::default_run`.
    /// Agents can override this for custom orchestration.
    ///
    /// # Errors
    ///
    /// Returns an error if the LLM request fails, tool execution fails,
    /// or session logging fails.
    async fn run(&self, user_prompt: &str) -> Result<Session> {
        runtime::default_run(self, user_prompt).await
    }

    /// Run the agent with a custom stream consumer.
    ///
    /// Like `run()`, but streams events to the provided consumer instead of
    /// the console. Use this for web streaming, logging, metrics, or custom
    /// output handling.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use appam::agent::consumers::ChannelConsumer;
    /// # async fn example() -> anyhow::Result<()> {
    /// # let agent = AgentBuilder::new("test").system_prompt("test").build()?;
    /// let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    /// let consumer = ChannelConsumer::new(tx);
    ///
    /// agent.run_streaming("Hello!", Box::new(consumer)).await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn run_streaming(
        &self,
        user_prompt: &str,
        consumer: Box<dyn streaming::StreamConsumer>,
    ) -> Result<Session> {
        runtime::default_run_streaming(self, user_prompt, consumer).await
    }

    /// Run the agent with multiple stream consumers.
    ///
    /// Events are broadcast to all consumers. If any consumer returns an error,
    /// execution stops and the error is returned.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use appam::agent::consumers::*;
    /// # async fn example() -> anyhow::Result<()> {
    /// # let agent = AgentBuilder::new("test").system_prompt("test").build()?;
    /// # let tx = tokio::sync::mpsc::unbounded_channel().0;
    /// agent.run_with_consumers("Hello!", vec![
    ///     Box::new(ConsoleConsumer::new()),
    ///     Box::new(ChannelConsumer::new(tx)),
    /// ]).await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn run_with_consumers(
        &self,
        user_prompt: &str,
        consumers: Vec<Box<dyn streaming::StreamConsumer>>,
    ) -> Result<Session> {
        let multi = streaming::MultiConsumer::new();
        let multi = consumers.into_iter().fold(multi, |m, c| m.add(c));
        self.run_streaming(user_prompt, Box::new(multi)).await
    }

    /// Build the initial message list for a conversation.
    ///
    /// By default, creates a system message and a user message. Override
    /// for custom message initialization (e.g., few-shot examples, context).
    fn initial_messages(&self, user_prompt: &str) -> Result<Vec<ChatMessage>> {
        let system = self.system_prompt()?;
        let now = chrono::Utc::now();
        Ok(vec![
            ChatMessage {
                role: Role::System,
                name: None,
                tool_call_id: None,
                content: Some(system),
                tool_calls: None,
                reasoning: None,
                raw_content_blocks: None,
                tool_metadata: None,
                timestamp: Some(now),
                id: None,
                provider_response_id: None,
                status: None,
            },
            ChatMessage {
                role: Role::User,
                name: None,
                tool_call_id: None,
                content: Some(user_prompt.to_string()),
                tool_calls: None,
                reasoning: None,
                raw_content_blocks: None,
                tool_metadata: None,
                timestamp: Some(now),
                id: None,
                provider_response_id: None,
                status: None,
            },
        ])
    }

    /// Continue an existing session with a new user prompt.
    ///
    /// Loads the session from history and continues the conversation,
    /// preserving all previous messages and context. Streams output to
    /// console with default formatting.
    ///
    /// # Requirements
    ///
    /// - Session history must be enabled in configuration
    /// - The session ID must exist in the database
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::{Agent, AgentBuilder};
    /// # use anyhow::Result;
    /// # async fn example() -> Result<()> {
    /// # let agent = AgentBuilder::new("test").system_prompt("test").build()?;
    /// // First conversation
    /// let session = agent.run("Hello!").await?;
    ///
    /// // Continue later
    /// agent.continue_session(&session.session_id, "How are you?").await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Session history is not enabled
    /// - Session ID does not exist
    /// - LLM request fails
    /// - Tool execution fails
    async fn continue_session(&self, session_id: &str, user_prompt: &str) -> Result<Session> {
        runtime::continue_session_run(self, session_id, user_prompt).await
    }

    /// Continue an existing session with custom streaming.
    ///
    /// Like `continue_session()`, but streams events to the provided consumer
    /// instead of the console.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::prelude::*;
    /// # use appam::agent::consumers::ChannelConsumer;
    /// # async fn example() -> anyhow::Result<()> {
    /// # let agent = AgentBuilder::new("test").system_prompt("test").build()?;
    /// let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    /// let consumer = ChannelConsumer::new(tx);
    ///
    /// agent.continue_session_streaming("session-123", "Continue...", Box::new(consumer)).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Session history is not enabled
    /// - Session ID does not exist
    /// - LLM request fails
    /// - Tool execution fails
    /// - Consumer returns an error
    async fn continue_session_streaming(
        &self,
        session_id: &str,
        user_prompt: &str,
        consumer: Box<dyn streaming::StreamConsumer>,
    ) -> Result<Session> {
        runtime::continue_session_streaming(self, session_id, user_prompt, consumer).await
    }
}

/// Session metadata.
///
/// Contains the conversation history and metadata for a single agent interaction.
/// Sessions are logged for debugging, evaluation, and compliance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Session {
    /// Unique session identifier
    pub session_id: String,
    /// Agent name
    pub agent_name: String,
    /// Model used
    pub model: String,
    /// Full conversation history
    pub messages: Vec<ChatMessage>,
    /// Session start time
    #[serde(default)]
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Session end time
    #[serde(default)]
    pub ended_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Token usage and cost tracking
    #[serde(default)]
    pub usage: Option<crate::llm::usage::AggregatedUsage>,
}

pub use builder::{AgentBuilder, ReasoningProvider};
pub use runtime::{continue_session_streaming_with_messages, continue_session_with_messages};
pub use runtime_agent::RuntimeAgent;
pub use toml_agent::TomlAgent;
