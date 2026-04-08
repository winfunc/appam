//! Tool traits, managed state, and runtime execution helpers.
//!
//! Tools are the boundary where model output becomes real side effects. The
//! types in this module therefore focus on three concerns:
//!
//! - describing tools to providers through JSON-schema-backed specs
//! - executing tools safely and predictably at runtime
//! - exposing managed app/session state through fail-closed handles
//!
//! Appam keeps the original synchronous [`Tool`] trait for simple stateless
//! tools and adds [`AsyncTool`] for context-aware or stateful implementations.

pub mod builtin;
pub mod loader;
#[cfg(feature = "python")]
pub mod python;
pub mod register;
pub mod registry;

use std::ops::Deref;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::llm::ToolSpec;

/// Synchronous tool interface for simple Rust tool implementations.
///
/// A `Tool` is intentionally small:
///
/// - [`Tool::name`] provides the stable dispatch key
/// - [`Tool::spec`] provides the provider-facing schema and description
/// - [`Tool::execute`] performs the actual side effect
///
/// The runtime treats tool arguments as untrusted model output. Implementations
/// should validate inputs, reject ambiguous requests, and avoid logging secrets
/// or other sensitive data derived from user prompts or credentials.
///
/// # Examples
///
/// ```
/// use appam::tools::Tool;
/// use appam::llm::ToolSpec;
/// use serde_json::{json, Value};
/// use anyhow::Result;
///
/// struct EchoTool;
///
/// impl Tool for EchoTool {
///     fn name(&self) -> &str {
///         "echo"
///     }
///
///     fn spec(&self) -> Result<ToolSpec> {
///         Ok(serde_json::from_value(json!({
///             "type": "function",
///             "function": {
///                 "name": "echo",
///                 "description": "Echo back the input message",
///                 "parameters": {
///                     "type": "object",
///                     "properties": {
///                         "message": {
///                             "type": "string",
///                             "description": "Message to echo"
///                         }
///                     },
///                     "required": ["message"]
///                 }
///             }
///         }))?)
///     }
///
///     fn execute(&self, args: Value) -> Result<Value> {
///         let msg = args["message"].as_str().unwrap_or("");
///         Ok(json!({ "output": msg }))
///     }
/// }
/// ```
pub trait Tool: Send + Sync {
    /// Return the unique, stable function name for this tool.
    ///
    /// This name must match the name in the tool specification and is used
    /// for routing LLM tool calls to the correct implementation.
    fn name(&self) -> &str;

    /// Return the tool specification exposed to the model.
    ///
    /// The specification includes the function signature, parameter schema,
    /// and description. This is typically loaded from a JSON file to maintain
    /// a single source of truth.
    ///
    /// # Errors
    ///
    /// Returns an error if the specification cannot be loaded or parsed.
    fn spec(&self) -> Result<ToolSpec>;

    /// Execute the tool with the given JSON arguments.
    ///
    /// Arguments are provided as a JSON value matching the schema from `spec()`.
    /// The tool should validate inputs, perform its operation, and return a
    /// JSON result.
    ///
    /// # Errors
    ///
    /// Returns an error if arguments are invalid, execution fails, or results
    /// cannot be serialized.
    ///
    /// # Security
    ///
    /// Tool implementations must validate all inputs, avoid shell injection,
    /// and respect sandbox boundaries. Never trust arguments from the LLM.
    fn execute(&self, args: Value) -> Result<Value>;
}

/// Concurrency policy for a tool implementation.
///
/// Appam defaults every tool to serial execution because many tools mutate
/// shared state, depend on ordering, or interact with external resources that
/// are not safe to run concurrently. Tool authors must explicitly opt into
/// parallel execution when they can defend that decision.
///
/// # Security considerations
///
/// Mark a tool as [`ToolConcurrency::ParallelSafe`] only when:
/// - it does not depend on relative ordering against sibling tool calls
/// - it does not mutate shared state without its own synchronization
/// - it does not rely on external side effects that would race unsafely
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolConcurrency {
    /// Execute this tool only in model order.
    SerialOnly,
    /// This tool may be executed concurrently with sibling tool calls.
    ParallelSafe,
}

/// Runtime metadata attached to one tool invocation.
///
/// `ToolContext` is created by the Appam runtime for every tool call. It gives
/// tools stable identifiers for the current session, agent, and tool call while
/// also acting as the gateway to Appam-managed state.
///
/// The state accessors are intentionally fail-closed:
/// - when a tool requests app/session state that was never registered
/// - when the context was created without a backing registry
/// - when the tool is executed outside an active session
///
/// This design ensures stateful tools fail with explicit, user-facing errors
/// instead of panicking or silently reading the wrong state.
#[derive(Clone)]
pub struct ToolContext {
    session_id: String,
    agent_name: String,
    tool_call_id: String,
    state_store: Option<Arc<registry::RegistryStateStore>>,
}

impl ToolContext {
    /// Create a standalone context without managed-state access.
    ///
    /// This constructor is primarily useful for tests that only need stable
    /// metadata and do not intend to access Appam-managed app/session state.
    ///
    /// # Parameters
    ///
    /// - `session_id`: Active session identifier
    /// - `agent_name`: Agent responsible for the tool invocation
    /// - `tool_call_id`: Provider-emitted identifier for the current tool call
    pub fn new(
        session_id: impl Into<String>,
        agent_name: impl Into<String>,
        tool_call_id: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            agent_name: agent_name.into(),
            tool_call_id: tool_call_id.into(),
            state_store: None,
        }
    }

    pub(crate) fn attach_state_store(
        mut self,
        state_store: Arc<registry::RegistryStateStore>,
    ) -> Self {
        self.state_store = Some(state_store);
        self
    }

    /// Return the active session identifier.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Return the agent name responsible for this tool invocation.
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    /// Return the provider-generated tool-call identifier.
    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    /// Resolve an app-scoped managed state value by type.
    ///
    /// # Errors
    ///
    /// Returns an error when the context has no backing registry or when the
    /// requested state type was never registered via `ToolRegistry::manage(...)`.
    pub fn app_state<T>(&self) -> Result<State<T>>
    where
        T: Send + Sync + 'static,
    {
        let store = self.require_state_store()?;
        store.get_app_state::<T>()
    }

    /// Resolve a session-scoped managed state value by type.
    ///
    /// Session state is initialized lazily on first access using the initializer
    /// that was registered for `T`.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    /// - the context has no backing registry
    /// - the requested session state type was never registered
    pub fn session_state<T>(&self) -> Result<SessionState<T>>
    where
        T: Send + Sync + 'static,
    {
        let store = self.require_state_store()?;
        store.get_session_state::<T>(&self.session_id)
    }

    fn require_state_store(&self) -> Result<&Arc<registry::RegistryStateStore>> {
        self.state_store.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Tool context for session '{}' does not have a managed-state store",
                self.session_id
            )
        })
    }
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("session_id", &self.session_id)
            .field("agent_name", &self.agent_name)
            .field("tool_call_id", &self.tool_call_id)
            .finish()
    }
}

/// Shared handle to app-scoped managed state.
///
/// App-managed state is registered once per registry using
/// `ToolRegistry::manage(...)` and shared across all tool calls handled by that
/// registry. The wrapper dereferences to the registered value, so callers can
/// use the wrapped type naturally while still preserving the ability to clone
/// the handle cheaply.
#[derive(Clone)]
pub struct State<T> {
    inner: Arc<T>,
}

impl<T> State<T> {
    pub(crate) fn from_arc(inner: Arc<T>) -> Self {
        Self { inner }
    }

    /// Return the shared `Arc<T>` backing this state handle.
    ///
    /// This is useful when the downstream API expects ownership of an `Arc`
    /// rather than a dereferenceable wrapper.
    pub fn into_inner(self) -> Arc<T> {
        self.inner
    }
}

impl<T> Deref for State<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> std::fmt::Debug for State<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("State").field(&self.inner).finish()
    }
}

/// Shared handle to lazily initialized session-scoped state.
///
/// Session-managed state is owned by the registry and keyed by `(session_id,
/// type)`. The wrapped payload stays in memory until the registry or session
/// state is explicitly cleared. Appam intentionally uses an internal lock so
/// callers can keep the payload type as plain Rust data structures instead of
/// forcing every consumer to carry `Arc<Mutex<_>>` boilerplate.
///
/// # Ergonomics
///
/// Use [`SessionState::read`] for read-only access and
/// [`SessionState::update`] for mutations. Both helpers map poisoned-lock
/// failures into `anyhow::Error` values so stateful tools fail gracefully.
#[derive(Clone)]
pub struct SessionState<T> {
    inner: Arc<RwLock<T>>,
}

impl<T> SessionState<T> {
    pub(crate) fn from_arc(inner: Arc<RwLock<T>>) -> Self {
        Self { inner }
    }

    /// Read the current session state.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal lock has been poisoned by a previous
    /// panic in tool code.
    pub fn read<R>(&self, reader: impl FnOnce(&T) -> R) -> Result<R> {
        let guard = self.inner.read().map_err(|_| {
            anyhow::anyhow!("Session state lock was poisoned while acquiring a read guard")
        })?;
        Ok(reader(&guard))
    }

    /// Mutate the current session state.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal lock has been poisoned by a previous
    /// panic in tool code.
    pub fn update<R>(&self, updater: impl FnOnce(&mut T) -> R) -> Result<R> {
        let mut guard = self.inner.write().map_err(|_| {
            anyhow::anyhow!("Session state lock was poisoned while acquiring a write guard")
        })?;
        Ok(updater(&mut guard))
    }

    /// Clone the full session payload when `T: Clone`.
    ///
    /// This is convenient for snapshot-style reads when calling code needs an
    /// owned value outside the lock scope.
    pub fn get_cloned(&self) -> Result<T>
    where
        T: Clone,
    {
        self.read(Clone::clone)
    }
}

impl<T> std::fmt::Debug for SessionState<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.inner.read() {
            Ok(guard) => f.debug_tuple("SessionState").field(&*guard).finish(),
            Err(_) => f.debug_tuple("SessionState").field(&"<poisoned>").finish(),
        }
    }
}

/// Async, context-aware tool interface for advanced Rust tools.
///
/// This trait extends Appam's original synchronous `Tool` interface with
/// runtime metadata and managed-state access while remaining additive and
/// opt-in. Legacy tools continue to implement [`Tool`] unchanged; new stateful
/// or asynchronous tools should implement `AsyncTool`.
///
/// # Concurrency
///
/// Tools are treated as [`ToolConcurrency::SerialOnly`] unless they explicitly
/// opt into [`ToolConcurrency::ParallelSafe`]. This avoids surprising races for
/// tools that mutate shared state or depend on side-effect ordering.
#[async_trait]
pub trait AsyncTool: Send + Sync {
    /// Return the unique, stable function name for this tool.
    fn name(&self) -> &str;

    /// Return the tool specification exposed to the model.
    fn spec(&self) -> Result<ToolSpec>;

    /// Return the concurrency policy for this tool.
    ///
    /// Implementations should keep the default unless concurrent execution is
    /// demonstrably safe with respect to shared state and external side effects.
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::SerialOnly
    }

    /// Execute the tool with runtime metadata and JSON arguments.
    ///
    /// The provided [`ToolContext`] contains stable identifiers for the current
    /// session and tool call plus managed state lookup helpers.
    ///
    /// # Errors
    ///
    /// Returns an error if arguments are invalid, required managed state is
    /// missing, execution fails, or the result cannot be serialized.
    async fn execute(&self, ctx: ToolContext, args: Value) -> Result<Value>;
}

pub use registry::ToolRegistry;
