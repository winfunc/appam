//! Shared application state for the web server.
//!
//! Manages loaded agents and active sessions across HTTP requests.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use tracing::{info, warn};

use super::session::SessionStore;
use crate::agent::{Agent, TomlAgent};

/// Shared application state.
///
/// Contains all loaded agents and the session store. This is cloned for each
/// request but the underlying data is shared via Arc.
#[derive(Clone)]
pub struct AppState {
    /// Loaded agents by name
    agents: Arc<RwLock<HashMap<String, Arc<TomlAgent>>>>,
    /// Session storage
    pub sessions: SessionStore,
    /// Base directory for agents
    agents_dir: PathBuf,
}

impl AppState {
    /// Create new application state and load agents from directory.
    ///
    /// Scans the agents directory for TOML files and loads all valid agents.
    /// Initializes SQLite database for session persistence.
    ///
    /// # Errors
    ///
    /// Returns an error if the agents directory cannot be read or database
    /// cannot be initialized.
    pub async fn new(agents_dir: PathBuf) -> Result<Self> {
        info!(dir = %agents_dir.display(), "Loading agents from directory");

        let mut agents_map = HashMap::new();

        if agents_dir.exists() {
            // Scan for TOML files both at root level and in subdirectories
            for entry in std::fs::read_dir(&agents_dir)? {
                let entry = entry?;
                let path = entry.path();

                // Check if it's a TOML file at root level
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("toml") {
                    match TomlAgent::from_file(&path) {
                        Ok(agent) => {
                            let name = agent.name().to_string();
                            info!(agent = %name, path = %path.display(), "Loaded agent");
                            agents_map.insert(name, Arc::new(agent));
                        }
                        Err(e) => {
                            warn!(
                                path = %path.display(),
                                error = %e,
                                "Failed to load agent"
                            );
                        }
                    }
                }
                // Check subdirectories for TOML files
                else if path.is_dir() {
                    if let Ok(subdir_entries) = std::fs::read_dir(&path) {
                        for subentry in subdir_entries.flatten() {
                            let subpath = subentry.path();
                            if subpath.is_file()
                                && subpath.extension().and_then(|s| s.to_str()) == Some("toml")
                            {
                                match TomlAgent::from_file(&subpath) {
                                    Ok(agent) => {
                                        let name = agent.name().to_string();
                                        info!(agent = %name, path = %subpath.display(), "Loaded agent");
                                        agents_map.insert(name, Arc::new(agent));
                                    }
                                    Err(e) => {
                                        warn!(
                                            path = %subpath.display(),
                                            error = %e,
                                            "Failed to load agent"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            warn!(dir = %agents_dir.display(), "Agents directory does not exist");
        }

        // Initialize session store with SQLite
        let sessions = SessionStore::new("data/sessions.db").await?;

        Ok(Self {
            agents: Arc::new(RwLock::new(agents_map)),
            sessions,
            agents_dir,
        })
    }

    /// Get an agent by name.
    pub fn get_agent(&self, name: &str) -> Option<Arc<TomlAgent>> {
        let agents = self.agents.read().unwrap();
        agents.get(name).cloned()
    }

    /// List all loaded agent names.
    pub fn list_agents(&self) -> Vec<String> {
        let agents = self.agents.read().unwrap();
        let mut names: Vec<String> = agents.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get agent information.
    pub fn get_agent_info(&self, name: &str) -> Option<AgentInfo> {
        let agents = self.agents.read().unwrap();
        agents.get(name).map(|agent| {
            let config = agent.config();
            AgentInfo {
                name: agent.name().to_string(),
                model: agent.model(),
                description: config.agent.description.clone(),
                version: config.agent.version.clone(),
                tool_count: config.tools.len(),
                tools: config.tools.iter().map(|t| t.name.clone()).collect(),
            }
        })
    }

    /// Reload all agents from disk.
    ///
    /// Useful for hot-reloading agent configurations without restarting the server.
    pub async fn reload_agents(&self) -> Result<usize> {
        info!("Reloading agents");

        let mut new_agents = HashMap::new();
        let mut loaded_count = 0;

        if self.agents_dir.exists() {
            // Scan for TOML files both at root level and in subdirectories
            for entry in std::fs::read_dir(&self.agents_dir)? {
                let entry = entry?;
                let path = entry.path();

                // Check if it's a TOML file at root level
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("toml") {
                    match TomlAgent::from_file(&path) {
                        Ok(agent) => {
                            let name = agent.name().to_string();
                            info!(agent = %name, "Reloaded agent");
                            new_agents.insert(name, Arc::new(agent));
                            loaded_count += 1;
                        }
                        Err(e) => {
                            warn!(path = %path.display(), error = %e, "Failed to reload agent");
                        }
                    }
                }
                // Check subdirectories for TOML files
                else if path.is_dir() {
                    if let Ok(subdir_entries) = std::fs::read_dir(&path) {
                        for subentry in subdir_entries.flatten() {
                            let subpath = subentry.path();
                            if subpath.is_file()
                                && subpath.extension().and_then(|s| s.to_str()) == Some("toml")
                            {
                                match TomlAgent::from_file(&subpath) {
                                    Ok(agent) => {
                                        let name = agent.name().to_string();
                                        info!(agent = %name, "Reloaded agent");
                                        new_agents.insert(name, Arc::new(agent));
                                        loaded_count += 1;
                                    }
                                    Err(e) => {
                                        warn!(path = %subpath.display(), error = %e, "Failed to reload agent");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut agents = self.agents.write().unwrap();
        *agents = new_agents;

        info!(count = loaded_count, "Agents reloaded");
        Ok(loaded_count)
    }

    /// Get statistics about the application state.
    pub async fn stats(&self) -> Result<StateStats> {
        Ok(StateStats {
            agent_count: self.list_agents().len(),
            session_count: self.sessions.session_count().await?,
        })
    }

    /// Get the agents directory path.
    pub fn agents_dir(&self) -> &std::path::Path {
        &self.agents_dir
    }

    /// Reload a single agent by name.
    ///
    /// Useful for hot-reloading a specific agent after configuration changes
    /// without reloading all agents.
    ///
    /// # Errors
    ///
    /// Returns an error if the agent file cannot be read or parsed.
    pub async fn reload_single_agent(&self, name: &str) -> Result<()> {
        info!(agent = %name, "Reloading single agent");

        // Search for the agent's TOML file by scanning the agents directory
        // It could be at agents/{name}.toml or agents/{subdir}/{name}.toml
        let mut found_path = None;

        // Check root level first
        let root_path = self.agents_dir.join(format!("{}.toml", name));
        if root_path.exists() {
            found_path = Some(root_path);
        } else {
            // Check all subdirectories
            if let Ok(entries) = std::fs::read_dir(&self.agents_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        // Check for TOML files in this subdirectory
                        if let Ok(subdir_entries) = std::fs::read_dir(&path) {
                            for subentry in subdir_entries.flatten() {
                                let subpath = subentry.path();
                                if subpath.is_file()
                                    && subpath.extension().and_then(|s| s.to_str()) == Some("toml")
                                {
                                    // Try to parse and check if it matches our agent name
                                    if let Ok(content) = std::fs::read_to_string(&subpath) {
                                        if let Ok(config) =
                                            toml::from_str::<crate::agent::config::AgentConfig>(
                                                &content,
                                            )
                                        {
                                            if config.agent.name == name {
                                                found_path = Some(subpath);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            if found_path.is_some() {
                                break;
                            }
                        }
                    }
                }
            }
        }

        let agent_path =
            found_path.ok_or_else(|| anyhow::anyhow!("Agent TOML file not found for: {}", name))?;

        match TomlAgent::from_file(&agent_path) {
            Ok(agent) => {
                let agent_name = agent.name().to_string();
                let mut agents = self.agents.write().unwrap();
                agents.insert(agent_name.clone(), Arc::new(agent));
                info!(agent = %agent_name, path = %agent_path.display(), "Reloaded agent");
                Ok(())
            }
            Err(e) => {
                warn!(path = %agent_path.display(), error = %e, "Failed to reload agent");
                Err(e)
            }
        }
    }
}

/// Agent information for API responses.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentInfo {
    /// The name of the agent.
    pub name: String,
    /// The model identifier used by the agent.
    pub model: String,
    /// Optional description of the agent.
    pub description: Option<String>,
    /// Optional version string.
    pub version: Option<String>,
    /// Number of tools configured for the agent.
    pub tool_count: usize,
    /// List of tool names.
    pub tools: Vec<String>,
}

/// Application statistics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateStats {
    /// Number of registered agents.
    pub agent_count: usize,
    /// Number of active sessions.
    pub session_count: usize,
}
