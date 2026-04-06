//! Runtime agent implementation for programmatically-built agents.
//!
//! Provides a concrete `Agent` implementation that can be built entirely in Rust
//! without TOML configuration files.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use super::Agent;
use crate::llm::ToolSpec;
use crate::tools::{AsyncTool, ToolConcurrency, ToolContext, ToolRegistry};

/// Agent built programmatically at runtime.
///
/// This agent implementation is designed to be constructed using the `AgentBuilder`
/// or directly in Rust code, without requiring TOML configuration files.
///
/// # Examples
///
/// ```no_run
/// use appam::agent::RuntimeAgent;
/// use appam::tools::ToolRegistry;
/// use std::sync::Arc;
///
/// let registry = Arc::new(ToolRegistry::new());
/// // Register tools...
///
/// let agent = RuntimeAgent::new(
///     "my-agent",
///     "You are a helpful assistant.",
///     registry,
/// ).with_model("openai/gpt-5");
/// ```
#[derive(Debug)]
pub struct RuntimeAgent {
    /// Agent name (unique identifier)
    name: String,
    /// System prompt defining agent behavior
    system_prompt: String,
    /// Optional provider override
    provider: Option<crate::llm::LlmProvider>,
    /// Optional model override
    model: Option<String>,
    /// Tool registry with available tools
    registry: Arc<ToolRegistry>,

    // API key overrides (highest priority)
    anthropic_api_key: Option<String>,
    openrouter_api_key: Option<String>,
    openai_api_key: Option<String>,
    openai_codex_access_token: Option<String>,
    vertex_api_key: Option<String>,

    // OpenAI-specific overrides
    openai_service_tier: Option<crate::llm::openai::ServiceTier>,
    openai_text_verbosity: Option<crate::llm::openai::TextVerbosity>,
    openai_pricing_model: Option<String>,

    // Anthropic-specific overrides
    anthropic_pricing_model: Option<String>,
    thinking: Option<crate::llm::anthropic::ThinkingConfig>,
    caching: Option<crate::llm::anthropic::CachingConfig>,
    tool_choice: Option<crate::llm::anthropic::ToolChoiceConfig>,
    effort: Option<crate::llm::anthropic::EffortLevel>,
    beta_features: Option<crate::llm::anthropic::BetaFeatures>,
    retry: Option<crate::llm::anthropic::RetryConfig>,
    rate_limiter: Option<crate::llm::anthropic::RateLimiterConfig>,

    // Provider-specific reasoning configuration
    reasoning: Option<crate::agent::builder::ReasoningProvider>,

    // OpenRouter-specific overrides
    provider_preferences: Option<crate::llm::openrouter::config::ProviderPreferences>,
    openrouter_transforms: Option<Vec<String>>,
    openrouter_models: Option<Vec<String>>,

    // Shared LLM parameters
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,

    // Logging configuration overrides
    logs_dir: Option<std::path::PathBuf>,
    log_level: Option<String>,
    log_format: Option<crate::config::LogFormat>,
    enable_traces: Option<bool>,
    trace_format: Option<crate::config::TraceFormat>,

    // History configuration overrides
    history_enabled: Option<bool>,
    history_db_path: Option<std::path::PathBuf>,
    history_auto_save: Option<bool>,

    // Session continuation configuration
    /// If set, agent will auto-continue when session ends without calling any of these tools
    required_completion_tools: Option<Vec<String>>,
    /// Maximum continuation attempts (default: 2)
    max_continuations: usize,
    /// Custom continuation message (if None, uses default)
    continuation_message: Option<String>,
    /// Whether provider-side parallel tool batching should be enabled.
    provider_parallel_tool_calls: bool,
    /// Maximum number of concurrent tool executions per batch.
    max_concurrent_tool_executions: usize,
}

impl RuntimeAgent {
    /// Create a new runtime agent.
    ///
    /// # Parameters
    ///
    /// - `name`: Unique agent identifier
    /// - `system_prompt`: System prompt defining agent behavior and capabilities
    /// - `registry`: Tool registry with available tools
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use appam::agent::RuntimeAgent;
    /// use appam::tools::ToolRegistry;
    /// use std::sync::Arc;
    ///
    /// let registry = Arc::new(ToolRegistry::new());
    /// let agent = RuntimeAgent::new(
    ///     "assistant",
    ///     "You are a helpful AI assistant.",
    ///     registry,
    /// );
    /// ```
    pub fn new(
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        registry: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            name: name.into(),
            system_prompt: system_prompt.into(),
            provider: None,
            model: None,
            registry,
            anthropic_api_key: None,
            openrouter_api_key: None,
            openai_api_key: None,
            openai_codex_access_token: None,
            vertex_api_key: None,
            openai_service_tier: None,
            openai_text_verbosity: None,
            openai_pricing_model: None,
            anthropic_pricing_model: None,
            thinking: None,
            caching: None,
            tool_choice: None,
            effort: None,
            beta_features: None,
            retry: None,
            rate_limiter: None,
            reasoning: None,
            provider_preferences: None,
            openrouter_transforms: None,
            openrouter_models: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            logs_dir: None,
            log_level: None,
            log_format: None,
            enable_traces: None,
            trace_format: None,
            history_enabled: None,
            history_db_path: None,
            history_auto_save: None,
            required_completion_tools: None,
            max_continuations: 2,
            continuation_message: None,
            provider_parallel_tool_calls: false,
            max_concurrent_tool_executions: 1,
        }
    }

    /// Create a new runtime agent with full configuration.
    ///
    /// Used internally by AgentBuilder to construct agents with all settings.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_config(
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        registry: Arc<ToolRegistry>,
        provider: Option<crate::llm::LlmProvider>,
        model: Option<String>,
        anthropic_api_key: Option<String>,
        openrouter_api_key: Option<String>,
        openai_api_key: Option<String>,
        openai_codex_access_token: Option<String>,
        vertex_api_key: Option<String>,
        openai_service_tier: Option<crate::llm::openai::ServiceTier>,
        openai_text_verbosity: Option<crate::llm::openai::TextVerbosity>,
        openai_pricing_model: Option<String>,
        anthropic_pricing_model: Option<String>,
        thinking: Option<crate::llm::anthropic::ThinkingConfig>,
        caching: Option<crate::llm::anthropic::CachingConfig>,
        tool_choice: Option<crate::llm::anthropic::ToolChoiceConfig>,
        effort: Option<crate::llm::anthropic::EffortLevel>,
        beta_features: Option<crate::llm::anthropic::BetaFeatures>,
        retry: Option<crate::llm::anthropic::RetryConfig>,
        rate_limiter: Option<crate::llm::anthropic::RateLimiterConfig>,
        reasoning: Option<crate::agent::builder::ReasoningProvider>,
        provider_preferences: Option<crate::llm::openrouter::config::ProviderPreferences>,
        openrouter_transforms: Option<Vec<String>>,
        openrouter_models: Option<Vec<String>>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        stop_sequences: Option<Vec<String>>,
        logs_dir: Option<std::path::PathBuf>,
        log_level: Option<String>,
        log_format: Option<crate::config::LogFormat>,
        enable_traces: Option<bool>,
        trace_format: Option<crate::config::TraceFormat>,
        history_enabled: Option<bool>,
        history_db_path: Option<std::path::PathBuf>,
        history_auto_save: Option<bool>,
        required_completion_tools: Option<Vec<String>>,
        max_continuations: usize,
        continuation_message: Option<String>,
        provider_parallel_tool_calls: bool,
        max_concurrent_tool_executions: usize,
    ) -> Self {
        Self {
            name: name.into(),
            system_prompt: system_prompt.into(),
            provider,
            model,
            registry,
            anthropic_api_key,
            openrouter_api_key,
            openai_api_key,
            openai_codex_access_token,
            vertex_api_key,
            openai_service_tier,
            openai_text_verbosity,
            openai_pricing_model,
            anthropic_pricing_model,
            thinking,
            caching,
            tool_choice,
            effort,
            beta_features,
            retry,
            rate_limiter,
            reasoning,
            provider_preferences,
            openrouter_transforms,
            openrouter_models,
            max_tokens,
            temperature,
            top_p,
            top_k,
            stop_sequences,
            logs_dir,
            log_level,
            log_format,
            enable_traces,
            trace_format,
            history_enabled,
            history_db_path,
            history_auto_save,
            required_completion_tools,
            max_continuations,
            continuation_message,
            provider_parallel_tool_calls,
            max_concurrent_tool_executions: max_concurrent_tool_executions.max(1),
        }
    }

    /// Set the provider for this agent.
    ///
    /// Overrides the global provider configuration for this specific agent.
    pub fn with_provider(mut self, provider: crate::llm::LlmProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model for this agent.
    ///
    /// This overrides any model specified in the global configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::RuntimeAgent;
    /// # use appam::tools::ToolRegistry;
    /// # use std::sync::Arc;
    /// # let registry = Arc::new(ToolRegistry::new());
    /// let agent = RuntimeAgent::new("agent", "prompt", registry)
    ///     .with_model("anthropic/claude-3.5-sonnet");
    /// ```
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Get the provider override for this agent.
    ///
    /// Returns None if no provider override is set (uses global config).
    pub fn provider(&self) -> Option<crate::llm::LlmProvider> {
        self.provider.clone()
    }

    /// Get the model to use for this agent.
    ///
    /// Returns the configured model or a default if not specified.
    pub fn model(&self) -> String {
        self.model
            .clone()
            .unwrap_or_else(|| "openai/gpt-5".to_string())
    }

    /// Get a reference to the tool registry.
    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }

    /// Update the system prompt.
    ///
    /// Replaces the existing system prompt with a new one.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// Add a tool to this agent's registry.
    ///
    /// This is a convenience method for registering tools after creation.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::RuntimeAgent;
    /// # use appam::tools::{ToolRegistry, Tool};
    /// # use std::sync::Arc;
    /// # let registry = Arc::new(ToolRegistry::new());
    /// # struct MyTool;
    /// # impl Tool for MyTool {
    /// #     fn name(&self) -> &str { "my_tool" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// let mut agent = RuntimeAgent::new("agent", "prompt", registry);
    /// agent.add_tool(Arc::new(MyTool));
    /// ```
    pub fn add_tool(&mut self, tool: Arc<dyn crate::tools::Tool>) {
        self.registry.register(tool);
    }

    /// Add an async/context-aware tool to this agent's registry.
    pub fn add_async_tool(&mut self, tool: Arc<dyn AsyncTool>) {
        self.registry.register_async(tool);
    }

    /// Get the list of required completion tools.
    ///
    /// Returns `None` if session continuation is not enabled (default behavior).
    /// When set, the agent will automatically continue the conversation if the
    /// session ends without calling any of the specified tools.
    pub fn required_completion_tools(&self) -> Option<&Vec<String>> {
        self.required_completion_tools.as_ref()
    }

    /// Get the maximum number of continuation attempts.
    ///
    /// Returns the maximum number of times the agent will inject a continuation
    /// message when required tools are not called. Default is 2.
    pub fn max_continuations(&self) -> usize {
        self.max_continuations
    }

    /// Get the custom continuation message, if set.
    ///
    /// Returns the user-provided continuation message or None to use the default.
    pub fn continuation_message(&self) -> Option<&str> {
        self.continuation_message.as_deref()
    }
}

#[async_trait]
impl Agent for RuntimeAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn provider(&self) -> Option<crate::llm::LlmProvider> {
        self.provider.clone()
    }

    fn required_completion_tools(&self) -> Option<&Vec<String>> {
        self.required_completion_tools.as_ref()
    }

    fn max_continuations(&self) -> usize {
        self.max_continuations
    }

    fn continuation_message(&self) -> Option<&str> {
        self.continuation_message.as_deref()
    }

    fn apply_config_overrides(&self, cfg: &mut crate::config::AppConfig) {
        // Apply API key overrides (highest priority)
        if let Some(ref api_key) = self.anthropic_api_key {
            cfg.anthropic.api_key = Some(api_key.clone());
        }
        if let Some(ref api_key) = self.openrouter_api_key {
            cfg.openrouter.api_key = Some(api_key.clone());
        }
        if let Some(ref api_key) = self.openai_api_key {
            cfg.openai.api_key = Some(api_key.clone());
        }
        if let Some(ref access_token) = self.openai_codex_access_token {
            cfg.openai_codex.access_token = Some(access_token.clone());
        }
        if let Some(ref api_key) = self.vertex_api_key {
            cfg.vertex.api_key = Some(api_key.clone());
        }

        // Apply OpenAI-specific overrides
        if let Some(service_tier) = self.openai_service_tier {
            cfg.openai.service_tier = Some(service_tier);
        }
        if let Some(text_verbosity) = self.openai_text_verbosity {
            cfg.openai.text_verbosity = Some(text_verbosity);
            cfg.openai_codex.text_verbosity = Some(text_verbosity);
        }
        if let Some(ref pricing_model) = self.openai_pricing_model {
            cfg.openai.pricing_model = Some(pricing_model.clone());
            cfg.openai_codex.pricing_model = Some(pricing_model.clone());
        }
        if let Some(ref pricing_model) = self.anthropic_pricing_model {
            cfg.anthropic.pricing_model = Some(pricing_model.clone());
        }

        // Apply model override (to all providers)
        if let Some(ref model) = self.model {
            cfg.anthropic.model = model.clone();
            cfg.openrouter.model = model.clone();
            cfg.openai.model = model.clone();
            cfg.openai_codex.model = model.clone();
            cfg.vertex.model = model.clone();
        }

        // Apply Anthropic-specific overrides
        if let Some(thinking) = &self.thinking {
            cfg.anthropic.thinking = Some(thinking.clone());
        }
        if let Some(caching) = &self.caching {
            cfg.anthropic.caching = Some(caching.clone());
        }
        if let Some(tool_choice) = &self.tool_choice {
            cfg.anthropic.tool_choice = Some(tool_choice.clone());
        }
        if let Some(effort) = &self.effort {
            cfg.anthropic.effort = Some(*effort);
        }
        if let Some(beta_features) = &self.beta_features {
            cfg.anthropic.beta_features = beta_features.clone();
        }
        if let Some(retry) = &self.retry {
            cfg.anthropic.retry = Some(retry.clone());
        }
        if let Some(rate_limiter) = &self.rate_limiter {
            cfg.anthropic.rate_limiter = Some(rate_limiter.clone());
        }
        if let Some(max_tokens) = self.max_tokens {
            cfg.anthropic.max_tokens = max_tokens;
        }
        if let Some(temperature) = self.temperature {
            cfg.anthropic.temperature = Some(temperature);
        }
        if let Some(top_p) = self.top_p {
            cfg.anthropic.top_p = Some(top_p);
        }
        if let Some(top_k) = self.top_k {
            cfg.anthropic.top_k = Some(top_k);
        }
        if let Some(ref stop_sequences) = self.stop_sequences {
            cfg.anthropic.stop_sequences = stop_sequences.clone();
        }

        // Apply provider-specific reasoning configuration
        if let Some(ref reasoning) = self.reasoning {
            match reasoning {
                crate::agent::builder::ReasoningProvider::OpenAI(config) => {
                    cfg.openai.reasoning = Some(config.clone());
                    cfg.openai_codex.reasoning = Some(config.clone());
                }
                crate::agent::builder::ReasoningProvider::OpenRouter(config) => {
                    cfg.openrouter.reasoning = Some(config.clone());
                }
            }
        }

        // Apply OpenRouter-specific overrides
        if let Some(provider_preferences) = &self.provider_preferences {
            cfg.openrouter.provider_preferences = Some(provider_preferences.clone());
        }
        if let Some(ref transforms) = self.openrouter_transforms {
            cfg.openrouter.transforms = Some(transforms.clone());
        }
        if let Some(ref models) = self.openrouter_models {
            cfg.openrouter.models = Some(models.clone());
        }
        if let Some(max_tokens) = self.max_tokens {
            cfg.openrouter.max_output_tokens = Some(max_tokens);
            cfg.openai.max_output_tokens = Some(max_tokens as i32);
            cfg.openai_codex.max_output_tokens = Some(max_tokens as i32);
            cfg.vertex.max_output_tokens = Some(max_tokens);
        }
        if let Some(temperature) = self.temperature {
            cfg.openrouter.temperature = Some(temperature);
            cfg.openai.temperature = Some(temperature);
            cfg.openai_codex.temperature = Some(temperature);
            cfg.vertex.temperature = Some(temperature);
        }
        if let Some(top_p) = self.top_p {
            cfg.openrouter.top_p = Some(top_p);
            cfg.openai.top_p = Some(top_p);
            cfg.openai_codex.top_p = Some(top_p);
            cfg.vertex.top_p = Some(top_p);
        }
        if let Some(top_k) = self.top_k {
            cfg.vertex.top_k = Some(top_k);
        }

        // Apply logging configuration overrides
        if let Some(ref logs_dir) = self.logs_dir {
            cfg.logging.logs_dir = logs_dir.clone();
        }
        if let Some(ref log_level) = self.log_level {
            cfg.logging.level = log_level.clone();
        }
        if let Some(log_format) = self.log_format {
            cfg.logging.log_format = log_format;
        }
        if let Some(enable_traces) = self.enable_traces {
            cfg.logging.enable_traces = enable_traces;
        }
        if let Some(trace_format) = self.trace_format {
            cfg.logging.trace_format = trace_format;
        }

        // Apply history configuration overrides
        if let Some(history_enabled) = self.history_enabled {
            cfg.history.enabled = history_enabled;
        }
        if let Some(ref history_db_path) = self.history_db_path {
            cfg.history.db_path = history_db_path.clone();
        }
        if let Some(history_auto_save) = self.history_auto_save {
            cfg.history.auto_save = history_auto_save;
        }
    }

    fn system_prompt(&self) -> Result<String> {
        Ok(self.system_prompt.clone())
    }

    fn available_tools(&self) -> Result<Vec<ToolSpec>> {
        self.registry.specs()
    }

    fn execute_tool(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        self.registry.execute(name, args)
    }

    async fn execute_tool_with_context(
        &self,
        name: &str,
        ctx: ToolContext,
        args: serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.registry.execute_with_context(ctx, name, args).await
    }

    fn tool_concurrency(&self, name: &str) -> ToolConcurrency {
        self.registry
            .concurrency(name)
            .unwrap_or(ToolConcurrency::SerialOnly)
    }

    fn provider_parallel_tool_calls(&self) -> bool {
        self.provider_parallel_tool_calls
    }

    fn max_concurrent_tool_executions(&self) -> usize {
        self.max_concurrent_tool_executions
    }

    // Uses default run implementation from runtime module
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolSpec;
    use crate::tools::Tool;
    use serde_json::json;

    struct MockTool {
        name: String,
    }

    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn spec(&self) -> Result<ToolSpec> {
            Ok(serde_json::from_value(json!({
                "type": "function",
                "name": self.name,
                "description": "Mock tool",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }))?)
        }

        fn execute(&self, _args: serde_json::Value) -> Result<serde_json::Value> {
            Ok(json!({"success": true}))
        }
    }

    #[test]
    fn test_runtime_agent_creation() {
        let registry = Arc::new(ToolRegistry::new());
        let agent = RuntimeAgent::new("test-agent", "You are a test assistant.", registry);

        assert_eq!(agent.name(), "test-agent");
        assert_eq!(agent.system_prompt().unwrap(), "You are a test assistant.");
        assert_eq!(agent.model(), "openai/gpt-5");
    }

    #[test]
    fn test_runtime_agent_with_model() {
        let registry = Arc::new(ToolRegistry::new());
        let agent = RuntimeAgent::new("test-agent", "Prompt", registry)
            .with_model("anthropic/claude-3.5-sonnet");

        assert_eq!(agent.model(), "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn test_runtime_agent_with_tools() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(MockTool {
            name: "test_tool".to_string(),
        }));

        let agent = RuntimeAgent::new("test-agent", "Prompt", registry);
        let tools = agent.available_tools().unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "test_tool");
    }

    #[test]
    fn test_add_tool_after_creation() {
        let registry = Arc::new(ToolRegistry::new());
        let mut agent = RuntimeAgent::new("test-agent", "Prompt", registry);

        agent.add_tool(Arc::new(MockTool {
            name: "new_tool".to_string(),
        }));

        let tools = agent.available_tools().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "new_tool");
    }

    #[test]
    fn test_execute_tool() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(MockTool {
            name: "test_tool".to_string(),
        }));

        let agent = RuntimeAgent::new("test-agent", "Prompt", registry);
        let result = agent.execute_tool("test_tool", json!({})).unwrap();

        assert_eq!(result, json!({"success": true}));
    }

    #[test]
    fn test_set_system_prompt() {
        let registry = Arc::new(ToolRegistry::new());
        let mut agent = RuntimeAgent::new("test-agent", "Original prompt", registry);

        agent.set_system_prompt("New prompt");
        assert_eq!(agent.system_prompt().unwrap(), "New prompt");
    }
}
