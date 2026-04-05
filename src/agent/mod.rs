//! Agent system and runtime.
//!
//! Defines the core `Agent` trait and provides implementations for loading agents
//! from TOML configurations, orchestrating tool-calling conversations, and managing
//! session state.

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

/// Core trait for AI agents.
///
/// An agent defines:
/// - A system prompt that establishes behavior and capabilities
/// - A set of available tools that the LLM can invoke
/// - A runtime that orchestrates the conversation loop
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
    /// Allows agents to override global settings (provider, model, Anthropic features, etc.)
    /// Default implementation does nothing.
    fn apply_config_overrides(&self, _cfg: &mut crate::config::AppConfig) {
        // Default: no overrides
    }

    /// Return the list of tools required for session completion.
    ///
    /// If `Some`, the runtime will automatically inject a continuation message
    /// when the session ends without calling any of these tools.
    /// Default implementation returns `None` (no continuation).
    fn required_completion_tools(&self) -> Option<&Vec<String>> {
        None
    }

    /// Return the maximum number of continuation attempts.
    ///
    /// Limits how many times the runtime will inject continuation messages
    /// before giving up. Default is 2.
    fn max_continuations(&self) -> usize {
        2
    }

    /// Return the custom continuation message, if any.
    ///
    /// If `Some`, this message will be injected when the session ends without
    /// calling required tools. If `None`, a default message is used.
    /// Default implementation returns `None`.
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
    /// to provide tool resolution logic.
    fn execute_tool(&self, name: &str, _args: serde_json::Value) -> Result<serde_json::Value> {
        Err(anyhow::anyhow!("Tool not found: {}", name))
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
pub use runtime_agent::RuntimeAgent;
pub use toml_agent::TomlAgent;
