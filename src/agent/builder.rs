//! Agent builder for fluent programmatic agent creation.
//!
//! Provides a builder pattern for constructing agents entirely in Rust code
//! with excellent ergonomics and compile-time validation.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};

use super::runtime_agent::RuntimeAgent;
use crate::tools::{Tool, ToolRegistry};

/// Provider-specific reasoning configuration.
///
/// Wraps reasoning configurations for different LLM providers to enable
/// type-safe, provider-specific reasoning settings in the AgentBuilder.
///
/// # Examples
///
/// ```no_run
/// # use appam::agent::{AgentBuilder, ReasoningProvider};
/// # use appam::llm::LlmProvider;
/// # use appam::llm::openai::{ReasoningConfig, ReasoningEffort};
/// # use anyhow::Result;
/// # fn main() -> Result<()> {
/// // OpenAI-specific reasoning
/// let agent = AgentBuilder::new("agent")
///     .provider(LlmProvider::OpenAI)
///     .reasoning(ReasoningProvider::OpenAI(
///         ReasoningConfig::high_effort()
///     ))
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub enum ReasoningProvider {
    /// OpenAI Responses API reasoning configuration
    OpenAI(crate::llm::openai::ReasoningConfig),
    /// OpenRouter (Completions or Responses) reasoning configuration
    OpenRouter(crate::llm::openrouter::config::ReasoningConfig),
}

/// Fluent builder for creating agents programmatically.
///
/// Provides a chainable API for constructing agents with type-safe configuration,
/// tool registration, and system prompt management.
///
/// # Examples
///
/// ```no_run
/// use appam::agent::AgentBuilder;
/// use anyhow::Result;
///
/// # fn main() -> Result<()> {
/// let agent = AgentBuilder::new("my-agent")
///     .model("anthropic/claude-3.5-sonnet")
///     .system_prompt("You are a helpful AI assistant.")
///     .build()?;
/// # Ok(())
/// # }
/// ```
///
/// With tools:
///
/// ```no_run
/// # use appam::agent::AgentBuilder;
/// # use anyhow::Result;
/// # use std::sync::Arc;
/// # use appam::tools::{Tool, ToolRegistry};
/// # struct MyTool;
/// # impl Tool for MyTool {
/// #     fn name(&self) -> &str { "my_tool" }
/// #     fn spec(&self) -> Result<appam::llm::ToolSpec> { todo!() }
/// #     fn execute(&self, _: serde_json::Value) -> Result<serde_json::Value> { todo!() }
/// # }
/// # fn main() -> Result<()> {
/// let agent = AgentBuilder::new("my-agent")
///     .system_prompt("You are a helpful assistant.")
///     .with_tool(Arc::new(MyTool))
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct AgentBuilder {
    name: String,
    provider: Option<crate::llm::LlmProvider>,
    model: Option<String>,
    system_prompt: Option<String>,
    system_prompt_file: Option<std::path::PathBuf>,
    registry: Option<Arc<ToolRegistry>>,
    tools: Vec<Arc<dyn Tool>>,

    // API keys (override config file and env vars)
    anthropic_api_key: Option<String>,
    openrouter_api_key: Option<String>,
    vertex_api_key: Option<String>,

    // Anthropic-specific configuration
    anthropic_pricing_model: Option<String>,
    thinking: Option<crate::llm::anthropic::ThinkingConfig>,
    caching: Option<crate::llm::anthropic::CachingConfig>,
    tool_choice: Option<crate::llm::anthropic::ToolChoiceConfig>,
    effort: Option<crate::llm::anthropic::EffortLevel>,
    beta_features: Option<crate::llm::anthropic::BetaFeatures>,
    retry: Option<crate::llm::anthropic::RetryConfig>,
    rate_limiter: Option<crate::llm::anthropic::RateLimiterConfig>,

    // Provider-specific reasoning configuration
    reasoning: Option<ReasoningProvider>,

    // OpenRouter-specific configuration
    provider_preferences: Option<crate::llm::openrouter::config::ProviderPreferences>,
    openrouter_transforms: Option<Vec<String>>,
    openrouter_models: Option<Vec<String>>,

    // OpenAI-specific configuration
    openai_api_key: Option<String>,
    openai_codex_access_token: Option<String>,
    openai_service_tier: Option<crate::llm::openai::ServiceTier>,
    openai_text_verbosity: Option<crate::llm::openai::TextVerbosity>,
    openai_pricing_model: Option<String>,

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
    required_completion_tools: Vec<Arc<dyn Tool>>,
    max_continuations: usize,
    continuation_message: Option<String>,
}

impl AgentBuilder {
    /// Create a new agent builder with the given name.
    ///
    /// The name should be a unique identifier for the agent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use appam::agent::AgentBuilder;
    ///
    /// let builder = AgentBuilder::new("my-agent");
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            provider: None,
            model: None,
            system_prompt: None,
            system_prompt_file: None,
            registry: None,
            tools: Vec::new(),
            anthropic_api_key: None,
            openrouter_api_key: None,
            vertex_api_key: None,
            openai_api_key: None,
            openai_codex_access_token: None,
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
            required_completion_tools: Vec::new(),
            max_continuations: 2,
            continuation_message: None,
        }
    }

    /// Set the LLM provider to use.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::agent::{Agent, AgentBuilder};
    /// # use appam::llm::LlmProvider;
    /// let builder = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .model("claude-sonnet-4-5");
    /// ```
    pub fn provider(mut self, provider: crate::llm::LlmProvider) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the LLM model to use.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::agent::AgentBuilder;
    /// let builder = AgentBuilder::new("agent")
    ///     .model("anthropic/claude-3.5-sonnet");
    /// ```
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set Anthropic API key programmatically.
    ///
    /// This overrides the API key from:
    /// - `appam.toml` config file
    /// - `ANTHROPIC_API_KEY` environment variable
    ///
    /// # Configuration Priority
    ///
    /// Programmatic configuration (highest priority) > Environment variables > Config file > Defaults
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .anthropic_api_key("sk-ant-...")
    ///     .model("claude-sonnet-4-5")
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn anthropic_api_key(mut self, key: impl Into<String>) -> Self {
        self.anthropic_api_key = Some(key.into());
        self
    }

    /// Set OpenRouter API key programmatically.
    ///
    /// This overrides the API key from:
    /// - `appam.toml` config file
    /// - `OPENROUTER_API_KEY` environment variable
    ///
    /// # Configuration Priority
    ///
    /// Programmatic configuration (highest priority) > Environment variables > Config file > Defaults
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenRouterCompletions)
    ///     .openrouter_api_key("sk-or-v1-...")
    ///     .model("openai/gpt-5")
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn openrouter_api_key(mut self, key: impl Into<String>) -> Self {
        self.openrouter_api_key = Some(key.into());
        self
    }

    /// Enable Anthropic extended thinking with token budget.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::anthropic::ThinkingConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .model("claude-sonnet-4-5")
    ///     .thinking(ThinkingConfig::enabled(10000))
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn thinking(mut self, config: crate::llm::anthropic::ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    /// Enable Anthropic prompt caching.
    ///
    /// This maps to Anthropic's top-level `cache_control` request field on the
    /// direct Anthropic and Azure Anthropic transports. Anthropic then applies
    /// the breakpoint to the last cacheable block in the request.
    ///
    /// On AWS Bedrock, Appam instead injects block-level `cache_control`
    /// checkpoints into supported Anthropic fields because Bedrock's InvokeModel
    /// integration uses explicit checkpoints rather than Anthropic's top-level
    /// helper.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::anthropic::CachingConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .caching(CachingConfig {
    ///         enabled: true,
    ///         ttl: appam::llm::anthropic::CacheTTL::OneHour,
    ///     })
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn caching(mut self, config: crate::llm::anthropic::CachingConfig) -> Self {
        self.caching = Some(config);
        self
    }

    /// Set Anthropic tool choice strategy.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::anthropic::ToolChoiceConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .tool_choice(ToolChoiceConfig::Auto { disable_parallel_tool_use: false })
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn tool_choice(mut self, config: crate::llm::anthropic::ToolChoiceConfig) -> Self {
        self.tool_choice = Some(config);
        self
    }

    /// Set the effort level for Claude's token spending.
    ///
    /// Controls how eagerly Claude spends tokens when responding. Affects text
    /// responses, tool calls, and extended thinking. Serialized as
    /// `output_config.effort` in the request body.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::anthropic::EffortLevel;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .model("claude-opus-4-6")
    ///     .effort(EffortLevel::Max)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn effort(mut self, level: crate::llm::anthropic::EffortLevel) -> Self {
        self.effort = Some(level);
        self
    }

    /// Enable Anthropic beta features.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::anthropic::BetaFeatures;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .beta_features(BetaFeatures {
    ///         interleaved_thinking: true,
    ///         context_1m: true,
    ///         ..Default::default()
    ///     })
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn beta_features(mut self, features: crate::llm::anthropic::BetaFeatures) -> Self {
        self.beta_features = Some(features);
        self
    }

    /// Configure retry behavior for rate limit and overload errors.
    ///
    /// By default, retries are enabled with exponential backoff. Use this method
    /// to customize retry behavior for high-parallelism scenarios or disable retries.
    ///
    /// # Examples
    ///
    /// Custom retry configuration for high parallelism (20+ parallel workers):
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::anthropic::RetryConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .retry(RetryConfig {
    ///         max_retries: 8,              // More attempts
    ///         initial_backoff_ms: 5000,    // Start with 5s
    ///         max_backoff_ms: 120000,      // Cap at 2min
    ///         backoff_multiplier: 2.0,
    ///         jitter: true,
    ///     })
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// Disable retries:
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .disable_retry()
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn retry(mut self, config: crate::llm::anthropic::RetryConfig) -> Self {
        self.retry = Some(config);
        self
    }

    /// Disable automatic retry for rate limit errors.
    ///
    /// By default, retries are enabled. Use this to disable them completely.
    pub fn disable_retry(mut self) -> Self {
        // Setting to None will prevent retry from being applied
        self.retry = Some(crate::llm::anthropic::RetryConfig {
            max_retries: 0,
            initial_backoff_ms: 0,
            max_backoff_ms: 0,
            backoff_multiplier: 1.0,
            jitter: false,
        });
        self
    }

    /// Configure global rate limiting for Anthropic API requests.
    ///
    /// Rate limiting prevents rate limit errors by coordinating requests across
    /// all parallel workers. Highly recommended for high-parallelism scenarios
    /// (15+ workers) where multiple agents compete for the same org-wide rate limit.
    ///
    /// # Examples
    ///
    /// Enable rate limiting for high parallelism:
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::anthropic::RateLimiterConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .rate_limiter(RateLimiterConfig {
    ///         enabled: true,
    ///         tokens_per_minute: 1_800_000,  // 90% of 2M limit
    ///         ..Default::default()
    ///     })
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// Quick enable with defaults:
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::anthropic::RateLimiterConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::Anthropic)
    ///     .enable_rate_limiter()
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn rate_limiter(mut self, config: crate::llm::anthropic::RateLimiterConfig) -> Self {
        self.rate_limiter = Some(config);
        self
    }

    /// Enable global rate limiting with default settings.
    ///
    /// Shorthand for `.rate_limiter(RateLimiterConfig { enabled: true, ..Default::default() })`.
    /// Uses conservative defaults: 1.8M tokens/min, 250K tokens/request.
    pub fn enable_rate_limiter(mut self) -> Self {
        self.rate_limiter = Some(crate::llm::anthropic::RateLimiterConfig {
            enabled: true,
            ..Default::default()
        });
        self
    }

    // ====================================================================
    // Reasoning configuration methods (provider-agnostic)
    // ====================================================================

    /// Set provider-specific reasoning configuration.
    ///
    /// Accepts either OpenAI or OpenRouter reasoning configurations wrapped in
    /// the `ReasoningProvider` enum.
    ///
    /// # Examples
    ///
    /// OpenAI reasoning:
    /// ```no_run
    /// # use appam::agent::{AgentBuilder, ReasoningProvider};
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::openai::ReasoningConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenAI)
    ///     .reasoning(ReasoningProvider::OpenAI(
    ///         ReasoningConfig::high_effort()
    ///     ))
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// OpenRouter reasoning:
    /// ```no_run
    /// # use appam::agent::{AgentBuilder, ReasoningProvider};
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::openrouter::ReasoningConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenRouterResponses)
    ///     .reasoning(ReasoningProvider::OpenRouter(
    ///         ReasoningConfig::high_effort(32000)
    ///     ))
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn reasoning(mut self, config: ReasoningProvider) -> Self {
        self.reasoning = Some(config);
        self
    }

    /// Set OpenAI-specific reasoning configuration (convenience method).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::openai::ReasoningConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenAI)
    ///     .openai_reasoning(ReasoningConfig::high_effort())
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn openai_reasoning(mut self, config: crate::llm::openai::ReasoningConfig) -> Self {
        self.reasoning = Some(ReasoningProvider::OpenAI(config));
        self
    }

    /// Set OpenAI text verbosity level.
    ///
    /// Controls the verbosity of the model's text responses.
    /// Lower values result in more concise responses, while higher values result in more detailed responses.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::openai::TextVerbosity;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenAI)
    ///     .openai_text_verbosity(TextVerbosity::High)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn openai_text_verbosity(mut self, verbosity: crate::llm::openai::TextVerbosity) -> Self {
        self.openai_text_verbosity = Some(verbosity);
        self
    }

    /// Set the canonical OpenAI model identifier used for pricing/accounting.
    pub fn openai_pricing_model(mut self, model: impl Into<String>) -> Self {
        self.openai_pricing_model = Some(model.into());
        self
    }

    /// Set the canonical Anthropic model identifier used for pricing/accounting.
    ///
    /// This is intended for transports such as Azure-hosted Anthropic where
    /// the request-facing deployment name may differ from the public model slug
    /// used by pricing tables.
    pub fn anthropic_pricing_model(mut self, model: impl Into<String>) -> Self {
        self.anthropic_pricing_model = Some(model.into());
        self
    }

    /// Set OpenRouter-specific reasoning configuration (convenience method).
    ///
    /// Works for both Completions and Responses APIs.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::openrouter::ReasoningConfig;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenRouterCompletions)
    ///     .openrouter_reasoning(ReasoningConfig::high_effort(32000))
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn openrouter_reasoning(
        mut self,
        config: crate::llm::openrouter::config::ReasoningConfig,
    ) -> Self {
        self.reasoning = Some(ReasoningProvider::OpenRouter(config));
        self
    }

    // ====================================================================
    // OpenRouter-specific configuration methods
    // ====================================================================

    /// Set OpenAI API key override.
    ///
    /// Overrides the API key from environment variables or config files.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenAI)
    ///     .openai_api_key("sk-...")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn openai_api_key(mut self, key: impl Into<String>) -> Self {
        self.openai_api_key = Some(key.into());
        self
    }

    /// Set an explicit OpenAI Codex access token override.
    ///
    /// This bypasses auth-cache lookup for the Codex provider and is intended
    /// for callers that already manage ChatGPT OAuth credentials themselves.
    pub fn openai_codex_access_token(mut self, token: impl Into<String>) -> Self {
        self.openai_codex_access_token = Some(token.into());
        self
    }

    /// Set Vertex API key override.
    ///
    /// This key is appended as the `key` query parameter for Vertex requests
    /// unless an explicit bearer token is configured.
    pub fn vertex_api_key(mut self, key: impl Into<String>) -> Self {
        self.vertex_api_key = Some(key.into());
        self
    }

    /// Set OpenAI service tier for request prioritization.
    ///
    /// Service tiers control request routing and queueing:
    /// - `Auto`: Default routing based on account settings
    /// - `Default`: Standard routing for general use
    /// - `Scale`: High-throughput workloads with increased concurrency limits
    /// - `Flex`: Cost-optimized with flexible latency
    /// - `Priority`: Lowest-latency routing for time-sensitive requests
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::openai::ServiceTier;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenAI)
    ///     .model("gpt-5-codex")
    ///     .openai_service_tier(ServiceTier::Scale)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn openai_service_tier(mut self, tier: crate::llm::openai::ServiceTier) -> Self {
        self.openai_service_tier = Some(tier);
        self
    }

    /// Set provider routing preferences (Completions API only).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// # use appam::llm::openrouter::{ProviderPreferences, DataCollection};
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenRouterCompletions)
    ///     .openrouter_provider_routing(ProviderPreferences {
    ///         order: Some(vec!["anthropic".to_string()]),
    ///         data_collection: Some(DataCollection::Deny),
    ///         zdr: Some(true),
    ///         ..Default::default()
    ///     })
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn openrouter_provider_routing(
        mut self,
        prefs: crate::llm::openrouter::config::ProviderPreferences,
    ) -> Self {
        self.provider_preferences = Some(prefs);
        self
    }

    /// Set OpenRouter prompt transforms (Completions API only).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenRouterCompletions)
    ///     .openrouter_transforms(vec!["middleware".to_string()])
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn openrouter_transforms(mut self, transforms: Vec<String>) -> Self {
        self.openrouter_transforms = Some(transforms);
        self
    }

    /// Set fallback models for OpenRouter (Completions API only).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::llm::LlmProvider;
    /// let agent = AgentBuilder::new("agent")
    ///     .provider(LlmProvider::OpenRouterCompletions)
    ///     .model("anthropic/claude-sonnet-4-5")
    ///     .openrouter_models(vec![
    ///         "anthropic/claude-sonnet-4-5".to_string(),
    ///         "openai/gpt-4o".to_string(),
    ///     ])
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn openrouter_models(mut self, models: Vec<String>) -> Self {
        self.openrouter_models = Some(models);
        self
    }

    /// Set maximum tokens to generate.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .max_tokens(8192)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature for sampling (0.0-1.0).
    ///
    /// Note: Incompatible with Anthropic extended thinking.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .temperature(0.7)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top-p nucleus sampling (0.0-1.0).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .top_p(0.9)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k sampling.
    ///
    /// Note: Incompatible with Anthropic extended thinking.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .top_k(40)
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set custom stop sequences.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .stop_sequences(vec!["END".to_string(), "STOP".to_string()])
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    // ============================================================================
    // Logging Configuration
    // ============================================================================

    /// Set the logs directory.
    ///
    /// All log files and trace files will be written to this directory.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .logs_dir("my_logs")
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn logs_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.logs_dir = Some(path.into());
        self
    }

    /// Set the log level.
    ///
    /// Valid levels: "trace", "debug", "info", "warn", "error"
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .log_level("debug")
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn log_level(mut self, level: impl Into<String>) -> Self {
        self.log_level = Some(level.into());
        self
    }

    /// Set the log file format.
    ///
    /// Determines how logs are written to disk.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::config::LogFormat;
    /// let agent = AgentBuilder::new("agent")
    ///     .log_format(LogFormat::Json)  // or LogFormat::Plain, LogFormat::Both
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn log_format(mut self, format: crate::config::LogFormat) -> Self {
        self.log_format = Some(format);
        self
    }

    /// Enable session trace files with default detailed format.
    ///
    /// Traces capture complete conversation flow including reasoning and tool calls.
    /// This enables agent traces (session-*.jsonl and session-*.json files).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .enable_traces()  // Simple and clear!
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn enable_traces(mut self) -> Self {
        self.enable_traces = Some(true);
        self
    }

    /// Disable session trace files.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .disable_traces()
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn disable_traces(mut self) -> Self {
        self.enable_traces = Some(false);
        self
    }

    /// Set the trace file detail level.
    ///
    /// - `TraceFormat::Compact`: Essential information only
    /// - `TraceFormat::Detailed`: Full details including reasoning (default)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::config::TraceFormat;
    /// let agent = AgentBuilder::new("agent")
    ///     .enable_traces()
    ///     .trace_format(TraceFormat::Compact)
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn trace_format(mut self, format: crate::config::TraceFormat) -> Self {
        self.trace_format = Some(format);
        self
    }

    // ============================================================================
    // History Configuration
    // ============================================================================

    /// Enable session history persistence with sensible defaults.
    ///
    /// Sessions will be saved to `data/sessions.db` with auto-save enabled.
    /// This is a convenience method that sets:
    /// - `history_enabled = true`
    /// - `history_db_path = "data/sessions.db"` (if not already set)
    /// - `history_auto_save = true` (if not already set)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .enable_history()  // Simple! Uses good defaults
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn enable_history(mut self) -> Self {
        self.history_enabled = Some(true);
        // Set sensible defaults if not already configured
        if self.history_db_path.is_none() {
            self.history_db_path = Some("data/sessions.db".into());
        }
        if self.history_auto_save.is_none() {
            self.history_auto_save = Some(true);
        }
        self
    }

    /// Disable session history persistence.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .disable_history()
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn disable_history(mut self) -> Self {
        self.history_enabled = Some(false);
        self
    }

    /// Set the session history database path.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .enable_history()
    ///     .history_db_path("my_data/sessions.db")
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn history_db_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.history_db_path = Some(path.into());
        self
    }

    /// Enable or disable automatic session saving.
    ///
    /// When enabled, sessions are automatically saved to the database after completion.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let agent = AgentBuilder::new("agent")
    ///     .enable_history()
    ///     .auto_save_sessions(true)
    ///     .system_prompt("You are helpful.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn auto_save_sessions(mut self, auto_save: bool) -> Self {
        self.history_auto_save = Some(auto_save);
        self
    }

    // ============================================================================
    // Session Continuation Configuration
    // ============================================================================

    /// Require specific tools to be called before session completion.
    ///
    /// When set, the agent will automatically inject a continuation message if
    /// the session ends without calling any of the specified tools. This prevents
    /// premature session termination and ensures analysis is properly completed.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use std::sync::Arc;
    /// # use appam::tools::Tool;
    /// # struct CompletionTool;
    /// # impl Tool for CompletionTool {
    /// #     fn name(&self) -> &str { "completion_tool" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// let agent = AgentBuilder::new("agent")
    ///     .system_prompt("You are a helpful assistant.")
    ///     .require_completion_tools(vec![Arc::new(CompletionTool)])
    ///     .continuation_message("Please call a completion tool to finish.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn require_completion_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.required_completion_tools = tools;
        self
    }

    /// Set the maximum number of continuation attempts.
    ///
    /// Defaults to 2 if not specified. This prevents infinite loops by limiting
    /// how many times the agent will inject continuation messages.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use std::sync::Arc;
    /// # use appam::tools::Tool;
    /// # struct CompletionTool;
    /// # impl Tool for CompletionTool {
    /// #     fn name(&self) -> &str { "completion_tool" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// let agent = AgentBuilder::new("agent")
    ///     .system_prompt("You are helpful.")
    ///     .require_completion_tools(vec![Arc::new(CompletionTool)])
    ///     .max_continuations(3)  // Allow up to 3 continuation attempts
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn max_continuations(mut self, count: usize) -> Self {
        self.max_continuations = count;
        self
    }

    /// Set a custom continuation message.
    ///
    /// This message is injected when the session ends without calling required
    /// completion tools. If not set, a default generic message is used.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use std::sync::Arc;
    /// # use appam::tools::Tool;
    /// # struct CompletionTool;
    /// # impl Tool for CompletionTool {
    /// #     fn name(&self) -> &str { "completion_tool" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// let agent = AgentBuilder::new("agent")
    ///     .system_prompt("You are helpful.")
    ///     .require_completion_tools(vec![Arc::new(CompletionTool)])
    ///     .continuation_message("Continue analysis. Call a completion tool to finish this session.")
    ///     .build()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn continuation_message(mut self, message: impl Into<String>) -> Self {
        self.continuation_message = Some(message.into());
        self
    }

    /// Set the system prompt inline.
    ///
    /// Use this for short prompts or dynamically generated prompts.
    /// For file-based prompts, use `system_prompt_file()` instead.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let builder = AgentBuilder::new("agent")
    ///     .system_prompt("You are a helpful AI assistant.");
    /// ```
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self.system_prompt_file = None; // Clear file if set
        self
    }

    /// Load the system prompt from a file.
    ///
    /// The file will be read during `build()`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// let builder = AgentBuilder::new("agent")
    ///     .system_prompt_file("prompts/assistant.txt");
    /// ```
    pub fn system_prompt_file(mut self, path: impl AsRef<Path>) -> Self {
        self.system_prompt_file = Some(path.as_ref().to_path_buf());
        self.system_prompt = None; // Clear inline if set
        self
    }

    /// Add a single tool to the agent.
    ///
    /// Tools can be added multiple times to register several tools.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use std::sync::Arc;
    /// # use appam::tools::Tool;
    /// # struct MyTool;
    /// # impl Tool for MyTool {
    /// #     fn name(&self) -> &str { "my_tool" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// let builder = AgentBuilder::new("agent")
    ///     .system_prompt("You are an assistant.")
    ///     .with_tool(Arc::new(MyTool));
    /// ```
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools at once.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use std::sync::Arc;
    /// # use appam::tools::Tool;
    /// # struct Tool1;
    /// # struct Tool2;
    /// # impl Tool for Tool1 {
    /// #     fn name(&self) -> &str { "tool1" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// # impl Tool for Tool2 {
    /// #     fn name(&self) -> &str { "tool2" }
    /// #     fn spec(&self) -> anyhow::Result<appam::llm::ToolSpec> { todo!() }
    /// #     fn execute(&self, _: serde_json::Value) -> anyhow::Result<serde_json::Value> { todo!() }
    /// # }
    /// let builder = AgentBuilder::new("agent")
    ///     .system_prompt("You are an assistant.")
    ///     .with_tools(vec![Arc::new(Tool1), Arc::new(Tool2)]);
    /// ```
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Provide a custom tool registry.
    ///
    /// If not provided, a new empty registry will be created and tools
    /// added via `with_tool()` will be registered to it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use appam::tools::ToolRegistry;
    /// # use std::sync::Arc;
    /// let registry = Arc::new(ToolRegistry::new());
    /// // Pre-register tools in the registry...
    ///
    /// let builder = AgentBuilder::new("agent")
    ///     .system_prompt("You are an assistant.")
    ///     .with_registry(registry);
    /// ```
    pub fn with_registry(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Build the agent from the configured builder.
    ///
    /// This validates the configuration and constructs a `RuntimeAgent`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No system prompt is provided (neither inline nor file)
    /// - System prompt file cannot be read
    /// - Configuration is otherwise invalid
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use appam::agent::AgentBuilder;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let agent = AgentBuilder::new("my-agent")
    ///     .model("openai/gpt-5")
    ///     .system_prompt("You are a helpful assistant.")
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<RuntimeAgent> {
        // Validate and load system prompt
        let system_prompt = if let Some(prompt) = self.system_prompt {
            prompt
        } else if let Some(path) = self.system_prompt_file {
            std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read system prompt: {}", path.display()))?
        } else {
            anyhow::bail!(
                "System prompt must be provided via system_prompt() or system_prompt_file()"
            );
        };

        // Create or use provided registry
        let registry = self
            .registry
            .unwrap_or_else(|| Arc::new(ToolRegistry::new()));

        // Register tools
        for tool in self.tools {
            registry.register(tool);
        }

        // Register required completion tools (they also need to be in the registry)
        for tool in &self.required_completion_tools {
            registry.register(Arc::clone(tool));
        }

        // Extract tool names for continuation checking
        let required_tool_names: Option<Vec<String>> = if !self.required_completion_tools.is_empty()
        {
            Some(
                self.required_completion_tools
                    .iter()
                    .map(|t| t.name().to_string())
                    .collect(),
            )
        } else {
            None
        };

        // Build RuntimeAgent with all configuration
        let agent = RuntimeAgent::with_config(
            &self.name,
            system_prompt,
            registry,
            self.provider,
            self.model,
            self.anthropic_api_key,
            self.openrouter_api_key,
            self.openai_api_key,
            self.openai_codex_access_token,
            self.vertex_api_key,
            self.openai_service_tier,
            self.openai_text_verbosity,
            self.openai_pricing_model,
            self.anthropic_pricing_model,
            self.thinking,
            self.caching,
            self.tool_choice,
            self.effort,
            self.beta_features,
            self.retry,
            self.rate_limiter,
            self.reasoning,
            self.provider_preferences,
            self.openrouter_transforms,
            self.openrouter_models,
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.top_k,
            self.stop_sequences,
            self.logs_dir,
            self.log_level,
            self.log_format,
            self.enable_traces,
            self.trace_format,
            self.history_enabled,
            self.history_db_path,
            self.history_auto_save,
            required_tool_names,
            self.max_continuations,
            self.continuation_message,
        );

        Ok(agent)
    }

    /// Build the agent with a streaming channel.
    ///
    /// Returns both the agent and a receiver for stream events. Events are sent
    /// to the receiver as the agent executes, allowing async consumption of
    /// the stream.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::agent::AgentBuilder;
    /// # use anyhow::Result;
    /// # async fn example() -> Result<()> {
    /// let (agent, mut stream) = AgentBuilder::new("my-agent")
    ///     .system_prompt("You are helpful.")
    ///     .build_with_stream()?;
    ///
    /// // Spawn task to handle events
    /// tokio::spawn(async move {
    ///     while let Some(event) = stream.recv().await {
    ///         println!("Event: {:?}", event);
    ///     }
    /// });
    ///
    /// agent.run("Hello!").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_with_stream(
        self,
    ) -> Result<(
        RuntimeAgent,
        tokio::sync::mpsc::UnboundedReceiver<super::streaming::StreamEvent>,
    )> {
        let agent = self.build()?;
        let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Note: The receiver is returned for the user to consume.
        // The agent will use ConsoleConsumer by default when calling run().
        // To stream to this channel, use run_streaming() with ChannelConsumer.

        Ok((agent, rx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::llm::ToolSpec;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

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
    fn test_builder_basic() {
        let agent = AgentBuilder::new("test-agent")
            .system_prompt("Test prompt")
            .build()
            .unwrap();

        assert_eq!(agent.name(), "test-agent");
    }

    #[test]
    fn test_builder_with_model() {
        let agent = AgentBuilder::new("test-agent")
            .model("anthropic/claude-3.5-sonnet")
            .system_prompt("Test prompt")
            .build()
            .unwrap();

        assert_eq!(agent.model(), "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn test_builder_with_tool() {
        let agent = AgentBuilder::new("test-agent")
            .system_prompt("Test prompt")
            .with_tool(Arc::new(MockTool {
                name: "test_tool".to_string(),
            }))
            .build()
            .unwrap();

        let tools = agent.available_tools().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "test_tool");
    }

    #[test]
    fn test_builder_with_multiple_tools() {
        let agent = AgentBuilder::new("test-agent")
            .system_prompt("Test prompt")
            .with_tools(vec![
                Arc::new(MockTool {
                    name: "tool1".to_string(),
                }),
                Arc::new(MockTool {
                    name: "tool2".to_string(),
                }),
            ])
            .build()
            .unwrap();

        let tools = agent.available_tools().unwrap();
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_builder_with_prompt_file() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"File-based prompt").unwrap();
        file.flush().unwrap();

        let agent = AgentBuilder::new("test-agent")
            .system_prompt_file(file.path())
            .build()
            .unwrap();

        use super::super::Agent;
        assert_eq!(agent.system_prompt().unwrap(), "File-based prompt");
    }

    #[test]
    fn test_builder_missing_prompt() {
        let result = AgentBuilder::new("test-agent").build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("System prompt"));
    }

    #[test]
    fn test_builder_with_custom_registry() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register(Arc::new(MockTool {
            name: "pre_registered".to_string(),
        }));

        let agent = AgentBuilder::new("test-agent")
            .system_prompt("Test prompt")
            .with_registry(registry.clone())
            .with_tool(Arc::new(MockTool {
                name: "added_later".to_string(),
            }))
            .build()
            .unwrap();

        let tools = agent.available_tools().unwrap();
        assert_eq!(tools.len(), 2);
    }
}
