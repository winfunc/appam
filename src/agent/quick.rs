//! Ergonomic shortcuts for building agents with minimal boilerplate.
//!
//! This module intentionally favors fast setup over exhaustive configurability.
//! It is a good fit for scripts, prototypes, smoke tests, and small services
//! that still want Appam's runtime, streaming, and tool orchestration without
//! writing a large builder chain.
//!
//! # Examples
//!
//! ```no_run
//! use appam::prelude::*;
//! use anyhow::Result;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // One-liner agent creation
//!     let agent = Agent::quick(
//!         "anthropic/claude-sonnet-4-5",
//!         "You are helpful.",
//!         vec![],
//!     )?;
//!
//!     agent.run("Hello!").await?;
//!     Ok(())
//! }
//! ```

use anyhow::Result;
use std::sync::Arc;

use super::builder::AgentBuilder;
use super::runtime_agent::RuntimeAgent;
use crate::llm::LlmProvider;
use crate::tools::{AsyncTool, Tool};

/// Convenience constructor trait implemented by [`RuntimeAgent`].
pub trait AgentQuick {
    /// Create an agent with minimal configuration.
    ///
    /// This is the fastest way to get a fully working [`RuntimeAgent`]:
    /// provide a model string, a system prompt, and any tools you want exposed
    /// to the model.
    ///
    /// The helper performs three things:
    ///
    /// 1. infers a provider from the model string
    /// 2. creates a [`crate::agent::AgentBuilder`] with common runtime defaults
    /// 3. returns the built [`RuntimeAgent`]
    ///
    /// The exact defaults are intentionally modest and may evolve with the
    /// crate. If you need precise provider tuning, switch to
    /// [`crate::agent::AgentBuilder`].
    ///
    /// # Provider Detection
    ///
    /// The provider is automatically detected from the model string:
    /// - `anthropic/*` or `claude-*` → Anthropic
    /// - `openai-codex/*` → OpenAI Codex
    /// - `openai/*`, `gpt-*`, or `o1-*` → OpenAI
    /// - `vertex/*`, `gemini-*`, or `google/gemini-*` → Vertex
    /// - `openrouter/*` → OpenRouter
    /// - Others → OpenRouter (default)
    ///
    /// # Default Behavior
    ///
    /// Today this helper sets common sampling defaults (`temperature`,
    /// `max_tokens`, and `top_p`) and leaves the rest of the runtime on the
    /// same provider defaults Appam would normally use. Anthropic-family runs
    /// also receive Appam's standard retry configuration through the builder.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// let agent = Agent::quick(
    ///     "anthropic/claude-sonnet-4-5",
    ///     "You are a helpful assistant.",
    ///     vec![],
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    fn quick(
        model: impl Into<String>,
        prompt: impl Into<String>,
        tools: Vec<Arc<dyn Tool>>,
    ) -> Result<Self>
    where
        Self: Sized;
}

impl AgentQuick for RuntimeAgent {
    fn quick(
        model: impl Into<String>,
        prompt: impl Into<String>,
        tools: Vec<Arc<dyn Tool>>,
    ) -> Result<Self> {
        let model_str = model.into();

        // Generate a simple agent name based on model
        let name = format!("agent-{}", extract_model_name(&model_str));

        // Auto-detect provider from model string
        let provider = detect_provider(&model_str);

        // Build with smart defaults
        let mut builder = AgentBuilder::new(name)
            .provider(provider)
            .model(model_str)
            .system_prompt(prompt.into())
            .with_tools(tools)
            // Smart defaults for common settings
            .temperature(0.7)
            .max_tokens(4096)
            .top_p(0.9);

        // Add retry configuration for production reliability
        builder = builder.retry(crate::llm::anthropic::RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30000,
            backoff_multiplier: 2.0,
            jitter: true,
        });

        builder.build()
    }
}

/// Namespaced helper for ergonomic agent construction.
///
/// This type exists only to provide discoverable associated functions such as
/// [`Agent::quick`] and [`Agent::new`]. It does not hold any runtime state.
pub struct Agent;

impl Agent {
    /// Create a [`RuntimeAgent`] with the quick-constructor defaults.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let agent = Agent::quick(
    ///     "anthropic/claude-sonnet-4-5",
    ///     "You are helpful.",
    ///     vec![],
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn quick(
        model: impl Into<String>,
        prompt: impl Into<String>,
        tools: Vec<Arc<dyn crate::tools::Tool>>,
    ) -> Result<RuntimeAgent> {
        RuntimeAgent::quick(model, prompt, tools)
    }

    /// Create a quick agent while automatically wrapping tools in `Arc`.
    ///
    /// This is useful when you have concrete tool values and do not want to
    /// allocate `Arc<dyn Tool>` values at the call site.
    pub fn quick_with<T: crate::tools::Tool + 'static>(
        model: impl Into<String>,
        prompt: impl Into<String>,
        tools: Vec<T>,
    ) -> Result<RuntimeAgent> {
        let wrapped_tools: Vec<Arc<dyn crate::tools::Tool>> = tools
            .into_iter()
            .map(|t| Arc::new(t) as Arc<dyn crate::tools::Tool>)
            .collect();
        RuntimeAgent::quick(model, prompt, wrapped_tools)
    }

    /// Create a builder with provider detection already applied.
    ///
    /// This is the ergonomic middle ground between [`Agent::quick`] and a
    /// fully manual [`AgentBuilder::new`](crate::agent::AgentBuilder::new)
    /// call.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let agent = Agent::new("my-agent", "openai/gpt-4o")
    ///     .prompt("You are helpful.")
    ///     .tools(vec![])
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::new_ret_no_self)]
    pub fn new(name: impl Into<String>, model: impl Into<String>) -> AgentBuilder {
        let model_str = model.into();
        let provider = detect_provider(&model_str);

        AgentBuilder::new(name).provider(provider).model(model_str)
    }
}

/// Infer an [`LlmProvider`] from a shared model string.
///
/// The helper understands both provider-prefixed forms such as
/// `"openai/gpt-5.5"` and common bare model names such as `"gpt-5.5"` or
/// `"gemini-2.5-flash"`. Unknown strings intentionally fall back to
/// OpenRouter Responses because that backend can proxy many upstream models.
///
/// # Examples
///
/// ```
/// # use appam::agent::quick::detect_provider;
/// # use appam::llm::LlmProvider;
/// assert!(matches!(detect_provider("anthropic/claude-sonnet-4-5"), LlmProvider::Anthropic));
/// assert!(matches!(detect_provider("openai/gpt-4o"), LlmProvider::OpenAI));
/// assert!(matches!(detect_provider("openai-codex/gpt-5.5"), LlmProvider::OpenAICodex));
/// assert!(matches!(detect_provider("claude-opus-4"), LlmProvider::Anthropic));
/// assert!(matches!(detect_provider("gpt-4"), LlmProvider::OpenAI));
/// assert!(matches!(detect_provider("o1-preview"), LlmProvider::OpenAI));
/// assert!(matches!(detect_provider("gemini-2.5-flash"), LlmProvider::Vertex));
/// ```
pub fn detect_provider(model: &str) -> LlmProvider {
    let model_lower = model.to_lowercase();

    // OpenRouter models
    if model_lower.starts_with("openrouter/") {
        return LlmProvider::OpenRouterResponses;
    }

    // Anthropic models
    if model_lower.starts_with("anthropic/") || model_lower.starts_with("claude-") {
        return LlmProvider::Anthropic;
    }

    // OpenAI Codex models
    if model_lower.starts_with("openai-codex/") {
        return LlmProvider::OpenAICodex;
    }

    // OpenAI models
    if model_lower.starts_with("openai/")
        || model_lower.starts_with("gpt-")
        || model_lower.starts_with("o1-")
        || model_lower.starts_with("o3-")
    {
        return LlmProvider::OpenAI;
    }

    // Vertex Gemini models
    if model_lower.starts_with("vertex/")
        || model_lower.starts_with("gemini-")
        || model_lower.starts_with("google/gemini")
    {
        return LlmProvider::Vertex;
    }

    // Default to OpenRouter (it can proxy to most providers)
    LlmProvider::OpenRouterResponses
}

/// Extract the model leaf used by [`AgentQuick::quick`] for generated names.
///
/// This helper strips a leading provider prefix when present and otherwise
/// returns the original string.
///
/// # Examples
///
/// ```
/// # use appam::agent::quick::extract_model_name;
/// assert_eq!(extract_model_name("anthropic/claude-sonnet-4-5"), "claude-sonnet-4-5");
/// assert_eq!(extract_model_name("openai/gpt-4o"), "gpt-4o");
/// assert_eq!(extract_model_name("claude-opus-4"), "claude-opus-4");
/// ```
pub fn extract_model_name(model: &str) -> String {
    // Remove provider prefix if present
    if let Some((_provider, name)) = model.split_once('/') {
        name.to_string()
    } else {
        model.to_string()
    }
}

/// Helper for adding enhanced builder methods to AgentBuilder
impl AgentBuilder {
    /// Shorthand for `system_prompt()`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let agent = Agent::new("agent", "openai/gpt-4o")
    ///     .prompt("You are helpful.")  // Cleaner than system_prompt()
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn prompt(self, prompt: impl Into<String>) -> Self {
        self.system_prompt(prompt)
    }

    /// Add multiple tools without `Arc` wrapping.
    ///
    /// The tools can be either already-wrapped `Arc<dyn Tool>` values or tool
    /// instances that will be automatically wrapped.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let agent = Agent::new("agent", "openai/gpt-4o")
    ///     .prompt("You are helpful.")
    ///     .tools(vec![])  // Pass tool instances directly
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        for tool in tools {
            self = self.with_tool(tool);
        }
        self
    }

    /// Add a single tool, automatically wrapping it in Arc if needed
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use appam::prelude::*;
    /// # use anyhow::Result;
    /// # fn main() -> Result<()> {
    /// let agent = Agent::new("agent", "openai/gpt-4o")
    ///     .prompt("You are helpful.")
    ///     .tool_dyn(Arc::new(()))  // Pass Arc<dyn Tool>
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn tool_dyn(self, tool: Arc<dyn Tool>) -> Self {
        self.with_tool(tool)
    }

    /// Add a single async/context-aware tool.
    pub fn async_tool_dyn(self, tool: Arc<dyn AsyncTool>) -> Self {
        self.with_async_tool(tool)
    }
}

/// Extension trait for adding tools to AgentBuilder without Arc wrapping
pub trait AgentBuilderToolExt {
    /// Add a single tool instance, automatically wrapping it
    fn tool<T: Tool + 'static>(self, tool: T) -> Self;
}

impl AgentBuilderToolExt for AgentBuilder {
    fn tool<T: Tool + 'static>(self, tool: T) -> Self {
        self.with_tool(Arc::new(tool))
    }
}

/// Extension trait for adding async/context-aware tools to AgentBuilder.
pub trait AgentBuilderAsyncToolExt {
    /// Add a single async tool instance, automatically wrapping it.
    fn async_tool<T: AsyncTool + 'static>(self, tool: T) -> Self;
}

impl AgentBuilderAsyncToolExt for AgentBuilder {
    fn async_tool<T: AsyncTool + 'static>(self, tool: T) -> Self {
        self.with_async_tool(Arc::new(tool))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_provider_anthropic() {
        assert!(matches!(
            detect_provider("anthropic/claude-sonnet-4-5"),
            LlmProvider::Anthropic
        ));
        assert!(matches!(
            detect_provider("claude-sonnet-4-5"),
            LlmProvider::Anthropic
        ));
        assert!(matches!(
            detect_provider("claude-opus-4"),
            LlmProvider::Anthropic
        ));
    }

    #[test]
    fn test_detect_provider_openai() {
        assert!(matches!(
            detect_provider("openai/gpt-4o"),
            LlmProvider::OpenAI
        ));
        assert!(matches!(detect_provider("gpt-4o"), LlmProvider::OpenAI));
        assert!(matches!(detect_provider("gpt-4"), LlmProvider::OpenAI));
        assert!(matches!(detect_provider("o1-preview"), LlmProvider::OpenAI));
        assert!(matches!(detect_provider("o3-mini"), LlmProvider::OpenAI));
    }

    #[test]
    fn test_detect_provider_openai_codex() {
        assert!(matches!(
            detect_provider("openai-codex/gpt-5.5"),
            LlmProvider::OpenAICodex
        ));
    }

    #[test]
    fn test_detect_provider_openrouter() {
        assert!(matches!(
            detect_provider("openrouter/anthropic/claude-sonnet-4-5"),
            LlmProvider::OpenRouterResponses
        ));
    }

    #[test]
    fn test_detect_provider_vertex() {
        assert!(matches!(
            detect_provider("gemini-2.5-flash"),
            LlmProvider::Vertex
        ));
        assert!(matches!(
            detect_provider("vertex/gemini-2.5-pro"),
            LlmProvider::Vertex
        ));
        assert!(matches!(
            detect_provider("google/gemini-2.5-flash"),
            LlmProvider::Vertex
        ));
    }

    #[test]
    fn test_detect_provider_default() {
        assert!(matches!(
            detect_provider("some-unknown-model"),
            LlmProvider::OpenRouterResponses
        ));
    }

    #[test]
    fn test_extract_model_name() {
        assert_eq!(
            extract_model_name("anthropic/claude-sonnet-4-5"),
            "claude-sonnet-4-5"
        );
        assert_eq!(extract_model_name("openai/gpt-4o"), "gpt-4o");
        assert_eq!(extract_model_name("gpt-4"), "gpt-4");
        assert_eq!(
            extract_model_name("openrouter/anthropic/claude"),
            "anthropic/claude"
        );
    }
}
