//! Dynamic tool registry for runtime tool management.
//!
//! This registry keeps Appam's original synchronous tool path intact while also
//! hosting the new async/context-aware tool surface. It additionally owns the
//! managed app/session state store used by `ToolContext`.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Result};

use super::{AsyncTool, SessionState, State, Tool, ToolConcurrency, ToolContext};

type ErasedState = Arc<dyn Any + Send + Sync>;
type SessionStateFactory = Arc<dyn Fn() -> ErasedState + Send + Sync>;

#[derive(Clone)]
enum RegisteredTool {
    Legacy(Arc<dyn Tool>),
    Async(Arc<dyn AsyncTool>),
}

impl RegisteredTool {
    fn spec(&self) -> Result<crate::llm::ToolSpec> {
        match self {
            Self::Legacy(tool) => tool.spec(),
            Self::Async(tool) => tool.spec(),
        }
    }

    fn concurrency(&self) -> ToolConcurrency {
        match self {
            Self::Legacy(_) => ToolConcurrency::SerialOnly,
            Self::Async(tool) => tool.concurrency(),
        }
    }

    fn as_legacy(&self) -> Option<Arc<dyn Tool>> {
        match self {
            Self::Legacy(tool) => Some(Arc::clone(tool)),
            Self::Async(_) => None,
        }
    }

    fn execute_legacy(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        match self {
            Self::Legacy(tool) => tool.execute(args),
            Self::Async(tool) => Err(anyhow!(
                "Tool '{}' requires async/context-aware execution. Use ToolRegistry::execute_with_context(...)",
                tool.name()
            )),
        }
    }

    async fn execute_with_context(
        &self,
        ctx: ToolContext,
        args: serde_json::Value,
    ) -> Result<serde_json::Value> {
        match self {
            Self::Legacy(tool) => tool.execute(args),
            Self::Async(tool) => tool.execute(ctx, args).await,
        }
    }
}

/// Managed app/session state store owned by a [`ToolRegistry`].
///
/// App state is keyed only by type. Session state is keyed by `(session_id,
/// type)` and initialized lazily from a registered factory.
#[derive(Default)]
pub(crate) struct RegistryStateStore {
    app_state: RwLock<HashMap<TypeId, ErasedState>>,
    session_factories: RwLock<HashMap<TypeId, SessionStateFactory>>,
    session_state: RwLock<HashMap<String, HashMap<TypeId, ErasedState>>>,
}

impl RegistryStateStore {
    pub(crate) fn manage<T>(&self, state: T)
    where
        T: Send + Sync + 'static,
    {
        let mut guard = self
            .app_state
            .write()
            .expect("managed app state lock poisoned");
        guard.insert(TypeId::of::<T>(), Arc::new(state));
    }

    pub(crate) fn session_state_with<T, F>(&self, init: F)
    where
        T: Send + Sync + 'static,
        F: Fn() -> T + Send + Sync + 'static,
    {
        let mut guard = self
            .session_factories
            .write()
            .expect("managed session factory lock poisoned");
        guard.insert(
            TypeId::of::<T>(),
            Arc::new(move || Arc::new(std::sync::RwLock::new(init())) as ErasedState),
        );
    }

    pub(crate) fn get_app_state<T>(&self) -> Result<State<T>>
    where
        T: Send + Sync + 'static,
    {
        let erased = {
            let guard = self.app_state.read().map_err(|_| {
                anyhow!("Managed app state lock was poisoned while resolving state")
            })?;
            guard.get(&TypeId::of::<T>()).cloned().ok_or_else(|| {
                anyhow!(
                    "Managed app state for type '{}' was not registered",
                    std::any::type_name::<T>()
                )
            })?
        };

        let typed = erased.downcast::<T>().map_err(|_| {
            anyhow!(
                "Managed app state for type '{}' had an unexpected concrete type",
                std::any::type_name::<T>()
            )
        })?;
        Ok(State::from_arc(typed))
    }

    pub(crate) fn get_session_state<T>(&self, session_id: &str) -> Result<SessionState<T>>
    where
        T: Send + Sync + 'static,
    {
        self.get_registered_session_state::<T>(session_id)
    }

    pub(crate) fn get_registered_session_state<T>(
        &self,
        session_id: &str,
    ) -> Result<SessionState<T>>
    where
        T: Send + Sync + 'static,
    {
        let type_id = TypeId::of::<T>();

        let factory = {
            let guard = self.session_factories.read().map_err(|_| {
                anyhow!("Managed session factory lock was poisoned while resolving state")
            })?;
            guard.get(&type_id).cloned().ok_or_else(|| {
                anyhow!(
                    "Managed session state for type '{}' was not registered",
                    std::any::type_name::<T>()
                )
            })?
        };

        let erased = {
            let mut all_sessions = self.session_state.write().map_err(|_| {
                anyhow!("Managed session state lock was poisoned while resolving state")
            })?;
            let session_entry = all_sessions.entry(session_id.to_string()).or_default();
            session_entry
                .entry(type_id)
                .or_insert_with(|| factory())
                .clone()
        };

        let typed = erased.downcast::<std::sync::RwLock<T>>().map_err(|_| {
            anyhow!(
                "Managed session state for type '{}' had an unexpected concrete type",
                std::any::type_name::<T>()
            )
        })?;
        Ok(SessionState::from_arc(typed))
    }

    pub(crate) fn clear_session_state(&self, session_id: &str) {
        if let Ok(mut guard) = self.session_state.write() {
            guard.remove(session_id);
        }
    }

    pub(crate) fn clear_all_session_state(&self) {
        if let Ok(mut guard) = self.session_state.write() {
            guard.clear();
        }
    }
}

/// Thread-safe registry for sync and async tool implementations.
///
/// The registry maps tool names to executable implementations and owns the
/// managed app/session state used by context-aware tools.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Arc<RwLock<HashMap<String, RegisteredTool>>>,
    state_store: Arc<RegistryStateStore>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tool_count", &self.len())
            .finish()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create a new empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            state_store: Arc::new(RegistryStateStore::default()),
        }
    }

    /// Register a synchronous legacy tool in the registry.
    pub fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        let mut tools = self.tools.write().unwrap();
        tools.insert(name, RegisteredTool::Legacy(tool));
    }

    /// Register an async/context-aware tool in the registry.
    pub fn register_async(&self, tool: Arc<dyn AsyncTool>) {
        let name = tool.name().to_string();
        let mut tools = self.tools.write().unwrap();
        tools.insert(name, RegisteredTool::Async(tool));
    }

    /// Resolve a legacy tool by name.
    ///
    /// Async tools intentionally return `None` here because they do not expose
    /// the old synchronous `Tool` trait.
    pub fn resolve(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let tools = self.tools.read().unwrap();
        tools.get(name).and_then(RegisteredTool::as_legacy)
    }

    /// Resolve an async tool by name.
    pub fn resolve_async(&self, name: &str) -> Option<Arc<dyn AsyncTool>> {
        let tools = self.tools.read().unwrap();
        match tools.get(name) {
            Some(RegisteredTool::Async(tool)) => Some(Arc::clone(tool)),
            _ => None,
        }
    }

    /// List all registered tool names.
    pub fn list(&self) -> Vec<String> {
        let tools = self.tools.read().unwrap();
        let mut names: Vec<String> = tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Return the full list of tool specifications.
    pub fn specs(&self) -> Result<Vec<crate::llm::ToolSpec>> {
        let tools = self.tools.read().unwrap();
        let mut names: Vec<&String> = tools.keys().collect();
        names.sort();

        let mut specs = Vec::with_capacity(names.len());
        for name in names {
            if let Some(tool) = tools.get(name) {
                specs.push(tool.spec()?);
            }
        }
        Ok(specs)
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        let tools = self.tools.read().unwrap();
        tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the concurrency policy for a tool.
    pub fn concurrency(&self, name: &str) -> Option<ToolConcurrency> {
        let tools = self.tools.read().unwrap();
        tools.get(name).map(RegisteredTool::concurrency)
    }

    /// Unregister a legacy tool by name.
    ///
    /// Async tools are removed as well, but return `None` because they do not
    /// implement the legacy `Tool` trait.
    pub fn unregister(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let mut tools = self.tools.write().unwrap();
        tools.remove(name).and_then(|tool| tool.as_legacy())
    }

    /// Clear all tools from the registry.
    pub fn clear(&self) {
        let mut tools = self.tools.write().unwrap();
        tools.clear();
    }

    /// Create a registry pre-populated with built-in tools.
    pub fn with_builtins() -> Self {
        Self::new()
    }

    /// Register multiple synchronous tools at once.
    pub fn register_many(&self, tools: Vec<Arc<dyn Tool>>) {
        for tool in tools {
            self.register(tool);
        }
    }

    /// Register multiple async tools at once.
    pub fn register_many_async(&self, tools: Vec<Arc<dyn AsyncTool>>) {
        for tool in tools {
            self.register_async(tool);
        }
    }

    /// Execute a legacy tool by name with the given arguments.
    ///
    /// Async tools intentionally fail here with a clear error so callers do not
    /// accidentally bypass the runtime context required by stateful tools.
    pub fn execute(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        let tool = {
            let tools = self.tools.read().unwrap();
            tools.get(name).cloned()
        }
        .ok_or_else(|| anyhow!("Tool not found: {}", name))?;

        tool.execute_legacy(args)
    }

    /// Execute any registered tool with runtime metadata.
    pub async fn execute_with_context(
        &self,
        ctx: ToolContext,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let tool = {
            let tools = self.tools.read().unwrap();
            tools.get(name).cloned()
        }
        .ok_or_else(|| anyhow!("Tool not found: {}", name))?;

        tool.execute_with_context(ctx.attach_state_store(self.state_store()), args)
            .await
    }

    /// Register an app-scoped managed state value.
    pub fn manage<T>(&self, state: T)
    where
        T: Send + Sync + 'static,
    {
        self.state_store.manage(state);
    }

    /// Register a lazily initialized session-scoped state type.
    pub fn session_state_with<T, F>(&self, init: F)
    where
        T: Send + Sync + 'static,
        F: Fn() -> T + Send + Sync + 'static,
    {
        self.state_store.session_state_with::<T, F>(init);
    }

    /// Clear all managed state for one session.
    pub fn clear_session_state(&self, session_id: &str) {
        self.state_store.clear_session_state(session_id);
    }

    /// Clear every managed session-state entry.
    pub fn clear_all_session_state(&self) {
        self.state_store.clear_all_session_state();
    }

    /// Return a clone of the registry's managed-state store.
    pub(crate) fn state_store(&self) -> Arc<RegistryStateStore> {
        Arc::clone(&self.state_store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolSpec;
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
    fn test_registry_basic_operations() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());

        let tool = Arc::new(MockTool {
            name: "test".to_string(),
        });
        registry.register(tool.clone());

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.resolve("test").is_some());
        assert!(registry.resolve("nonexistent").is_none());

        let names = registry.list();
        assert_eq!(names, vec!["test"]);

        registry.clear();
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_execute() {
        let registry = ToolRegistry::new();
        let tool = Arc::new(MockTool {
            name: "test".to_string(),
        });
        registry.register(tool);

        let result = registry.execute("test", json!({})).unwrap();
        assert_eq!(result, json!({"success": true}));

        let error = registry.execute("nonexistent", json!({}));
        assert!(error.is_err());
    }

    #[test]
    fn test_app_state_storage() {
        let registry = ToolRegistry::new();
        registry.manage::<String>("hello".to_string());

        let state = registry
            .state_store()
            .get_app_state::<String>()
            .expect("state should resolve");
        assert_eq!(&*state, "hello");
    }

    #[test]
    fn test_session_state_isolation() {
        let registry = ToolRegistry::new();
        registry.session_state_with::<Vec<String>, _>(Vec::new);

        let session_a = registry
            .state_store()
            .get_registered_session_state::<Vec<String>>("a")
            .expect("session a");
        let session_b = registry
            .state_store()
            .get_registered_session_state::<Vec<String>>("b")
            .expect("session b");

        session_a
            .update(|items| items.push("one".to_string()))
            .expect("update");

        assert_eq!(
            session_a.get_cloned().expect("clone"),
            vec!["one".to_string()]
        );
        assert!(session_b.get_cloned().expect("clone").is_empty());
    }
}
