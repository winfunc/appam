//! API routes for agents, sessions, and chat.
//!
//! Provides RESTful endpoints for:
//! - Agent management and information
//! - Session creation and continuation
//! - Streaming chat with SSE

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{delete, get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

use crate::agent::config::AgentConfig;
use crate::llm::ChatMessage;
use crate::web::streaming::{EventSender, EventStream};

use super::fs_operations;
use super::session::{Session, SessionSummary};
use super::state::{AgentInfo, AppState, StateStats};

/// Agent-related routes.
pub fn agent_routes() -> Router<AppState> {
    Router::new()
        .route("/api/agents", get(list_agents))
        .route("/api/agents", post(create_agent))
        .route("/api/agents/:name", get(get_agent))
        .route("/api/agents/:name", put(update_agent))
        .route("/api/agents/:name", delete(delete_agent))
        .route("/api/agents/:name/config", get(get_agent_config))
        .route("/api/agents/:name/duplicate", post(duplicate_agent))
        .route("/api/agents/:name/reload", post(reload_agent))
        .route("/api/agents/:name/prompt", get(get_agent_prompt))
        .route("/api/agents/:name/prompt", put(update_agent_prompt))
        .route("/api/agents/:name/tools", post(create_tool))
        .route("/api/agents/:name/tools/:tool_name", get(get_tool))
        .route("/api/agents/:name/tools/:tool_name", put(update_tool))
        .route("/api/agents/:name/tools/:tool_name", delete(delete_tool))
        .route("/api/stats", get(get_stats))
}

/// Session-related routes.
pub fn session_routes() -> Router<AppState> {
    Router::new()
        .route("/api/sessions", get(list_sessions))
        .route("/api/sessions/:session_id", get(get_session))
        .route("/api/sessions/:session_id", delete(delete_session))
        .route("/api/agents/:name/sessions", get(list_agent_sessions))
}

/// Chat-related routes.
pub fn chat_routes() -> Router<AppState> {
    Router::new()
        .route("/api/agents/:name/chat", post(chat_stream))
        .route("/api/sessions/:session_id/chat", post(continue_chat_stream))
}

// ============================================================================
// Agent Endpoints
// ============================================================================

/// List all available agents.
///
/// GET /api/agents
async fn list_agents(State(state): State<AppState>) -> Json<Vec<AgentInfo>> {
    let agent_names = state.list_agents();
    let agents: Vec<AgentInfo> = agent_names
        .iter()
        .filter_map(|name| state.get_agent_info(name))
        .collect();

    Json(agents)
}

/// Get detailed information about a specific agent.
///
/// GET /api/agents/:name
async fn get_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<AgentInfo>, ApiError> {
    state
        .get_agent_info(&name)
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", name)))
}

/// Reload an agent from disk.
///
/// POST /api/agents/:name/reload
async fn reload_agent(
    State(state): State<AppState>,
    Path(_name): Path<String>,
) -> Result<Json<ReloadResponse>, ApiError> {
    let count = state
        .reload_agents()
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    Ok(Json(ReloadResponse {
        reloaded_count: count,
    }))
}

/// Get application statistics.
///
/// GET /api/stats
async fn get_stats(State(state): State<AppState>) -> Result<Json<StateStats>, ApiError> {
    state
        .stats()
        .await
        .map(Json)
        .map_err(|e| ApiError::Internal(e.to_string()))
}

// ============================================================================
// Agent Builder Endpoints
// ============================================================================

/// Get full agent configuration for builder UI.
///
/// GET /api/agents/:name/config
async fn get_agent_config(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<AgentConfigResponse>, ApiError> {
    info!(agent = %name, "Getting agent configuration");

    // Get agent to verify it exists
    let agent = state
        .get_agent(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", name)))?;

    // Read full configuration from disk
    let config_data = fs_operations::read_agent_config(agent.base_dir(), &name)
        .map_err(|e| ApiError::Internal(format!("Failed to read agent config: {}", e)))?;

    let response = AgentConfigResponse {
        name: config_data.name,
        config_toml: config_data.config_toml,
        config_parsed: config_data.config_parsed,
        prompt_content: config_data.prompt_content,
        tools_details: config_data
            .tools_details
            .into_iter()
            .map(|t| ToolDetailResponse {
                name: t.name,
                schema_json: t.schema_json,
                implementation_type: t.implementation_type,
                code: t.code,
            })
            .collect(),
        base_path: config_data.base_path,
    };

    Ok(Json(response))
}

/// Create a new agent.
///
/// POST /api/agents
async fn create_agent(
    State(state): State<AppState>,
    Json(request): Json<CreateAgentRequest>,
) -> Result<Json<AgentInfo>, ApiError> {
    info!(agent = %request.name, "Creating new agent");

    // Validate agent name
    fs_operations::validate_agent_name(&request.name)
        .map_err(|e| ApiError::BadRequest(format!("Invalid agent name: {}", e)))?;

    // Create agent directory
    let agent_dir = fs_operations::create_agent_directory(state.agents_dir(), &request.name)
        .map_err(|e| ApiError::Internal(format!("Failed to create agent directory: {}", e)))?;

    // Write system prompt
    fs_operations::write_system_prompt(&agent_dir, &request.system_prompt)
        .map_err(|e| ApiError::Internal(format!("Failed to write system prompt: {}", e)))?;

    // Create agent config
    let config = AgentConfig {
        agent: crate::agent::config::AgentMetadata {
            name: request.name.clone(),
            model: request.model,
            system_prompt: std::path::PathBuf::from("prompt.txt"),
            description: request.description,
            version: request.version,
        },
        tools: vec![], // Empty tools array is fine, serialization will handle it correctly
    };

    // Write TOML config
    fs_operations::write_agent_toml(&agent_dir, &config)
        .map_err(|e| ApiError::Internal(format!("Failed to write agent config: {}", e)))?;

    // Reload agents to include the new one
    state
        .reload_agents()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to reload agents: {}", e)))?;

    // Get agent info for response
    let agent_info = state
        .get_agent_info(&request.name)
        .ok_or_else(|| ApiError::Internal("Failed to load newly created agent".to_string()))?;

    info!(agent = %request.name, "Agent created successfully");
    Ok(Json(agent_info))
}

/// Update an existing agent's configuration.
///
/// PUT /api/agents/:name
async fn update_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<UpdateAgentRequest>,
) -> Result<Json<AgentInfo>, ApiError> {
    info!(agent = %name, "Updating agent");

    // Get agent to verify it exists
    let agent = state
        .get_agent(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", name)))?;

    let agent_dir = agent.base_dir();

    // Read current config
    let mut config_data = fs_operations::read_agent_config(agent_dir, &name)
        .map_err(|e| ApiError::Internal(format!("Failed to read agent config: {}", e)))?;

    // Update fields
    if let Some(model) = request.model {
        config_data.config_parsed.agent.model = Some(model);
    }
    if let Some(description) = request.description {
        config_data.config_parsed.agent.description = Some(description);
    }
    if let Some(version) = request.version {
        config_data.config_parsed.agent.version = Some(version);
    }

    // Write updated config to the original file path (preserve filename)
    let toml_path = std::path::PathBuf::from(&config_data.toml_file_path);
    fs_operations::write_agent_toml_to_path(&config_data.config_parsed, &toml_path)
        .map_err(|e| ApiError::Internal(format!("Failed to write agent config: {}", e)))?;

    // Reload this specific agent
    state
        .reload_single_agent(&name)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to reload agent: {}", e)))?;

    // Get updated agent info
    let agent_info = state
        .get_agent_info(&name)
        .ok_or_else(|| ApiError::Internal("Failed to load updated agent".to_string()))?;

    info!(agent = %name, "Agent updated successfully");
    Ok(Json(agent_info))
}

/// Delete an agent.
///
/// DELETE /api/agents/:name
async fn delete_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<DeleteResponse>, ApiError> {
    info!(agent = %name, "Deleting agent");

    // Get agent to verify it exists
    let agent = state
        .get_agent(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", name)))?;

    let agent_dir = agent.base_dir().to_path_buf();

    // Delete directory
    fs_operations::delete_agent_directory(&agent_dir)
        .map_err(|e| ApiError::Internal(format!("Failed to delete agent directory: {}", e)))?;

    // Reload agents
    state
        .reload_agents()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to reload agents: {}", e)))?;

    info!(agent = %name, "Agent deleted successfully");
    Ok(Json(DeleteResponse { success: true }))
}

/// Duplicate an agent with a new name.
///
/// POST /api/agents/:name/duplicate
async fn duplicate_agent(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<DuplicateAgentRequest>,
) -> Result<Json<AgentInfo>, ApiError> {
    info!(src_agent = %name, dst_agent = %request.new_name, "Duplicating agent");

    // Validate new agent name
    fs_operations::validate_agent_name(&request.new_name)
        .map_err(|e| ApiError::BadRequest(format!("Invalid agent name: {}", e)))?;

    // Verify source agent exists
    let _agent = state
        .get_agent(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", name)))?;

    // Duplicate directory
    fs_operations::duplicate_agent_directory(state.agents_dir(), &name, &request.new_name)
        .map_err(|e| ApiError::Internal(format!("Failed to duplicate agent: {}", e)))?;

    // Reload agents
    state
        .reload_agents()
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to reload agents: {}", e)))?;

    // Get new agent info
    let agent_info = state
        .get_agent_info(&request.new_name)
        .ok_or_else(|| ApiError::Internal("Failed to load duplicated agent".to_string()))?;

    info!(src_agent = %name, dst_agent = %request.new_name, "Agent duplicated successfully");
    Ok(Json(agent_info))
}

/// Get agent's system prompt.
///
/// GET /api/agents/:name/prompt
async fn get_agent_prompt(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<String, ApiError> {
    info!(agent = %name, "Getting agent prompt");

    let agent = state
        .get_agent(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", name)))?;

    let config_data = fs_operations::read_agent_config(agent.base_dir(), &name)
        .map_err(|e| ApiError::Internal(format!("Failed to read agent config: {}", e)))?;

    Ok(config_data.prompt_content)
}

/// Update agent's system prompt.
///
/// PUT /api/agents/:name/prompt
async fn update_agent_prompt(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<UpdatePromptRequest>,
) -> Result<Json<DeleteResponse>, ApiError> {
    info!(agent = %name, "Updating agent prompt");

    let agent = state
        .get_agent(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", name)))?;

    let agent_dir = agent.base_dir();

    // Write updated prompt
    fs_operations::write_system_prompt(agent_dir, &request.content)
        .map_err(|e| ApiError::Internal(format!("Failed to write prompt: {}", e)))?;

    // Reload agent
    state
        .reload_single_agent(&name)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to reload agent: {}", e)))?;

    info!(agent = %name, "Prompt updated successfully");
    Ok(Json(DeleteResponse { success: true }))
}

/// Get details of a specific tool.
///
/// GET /api/agents/:name/tools/:tool_name
async fn get_tool(
    State(state): State<AppState>,
    Path((agent_name, tool_name)): Path<(String, String)>,
) -> Result<Json<ToolDetailResponse>, ApiError> {
    info!(agent = %agent_name, tool = %tool_name, "Getting tool details");

    let agent = state
        .get_agent(&agent_name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", agent_name)))?;

    let config_data = fs_operations::read_agent_config(agent.base_dir(), &agent_name)
        .map_err(|e| ApiError::Internal(format!("Failed to read agent config: {}", e)))?;

    let tool_detail = config_data
        .tools_details
        .into_iter()
        .find(|t| t.name == tool_name)
        .ok_or_else(|| ApiError::NotFound(format!("Tool not found: {}", tool_name)))?;

    Ok(Json(ToolDetailResponse {
        name: tool_detail.name,
        schema_json: tool_detail.schema_json,
        implementation_type: tool_detail.implementation_type,
        code: tool_detail.code,
    }))
}

/// Create a new tool for an agent.
///
/// POST /api/agents/:name/tools
async fn create_tool(
    State(state): State<AppState>,
    Path(agent_name): Path<String>,
    Json(request): Json<CreateToolRequest>,
) -> Result<Json<DeleteResponse>, ApiError> {
    info!(agent = %agent_name, tool = %request.name, "Creating tool");

    // Validate tool name
    fs_operations::validate_tool_name(&request.name)
        .map_err(|e| ApiError::BadRequest(format!("Invalid tool name: {}", e)))?;

    let agent = state
        .get_agent(&agent_name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", agent_name)))?;

    let agent_dir = agent.base_dir();

    // Read current config
    let mut config_data = fs_operations::read_agent_config(agent_dir, &agent_name)
        .map_err(|e| ApiError::Internal(format!("Failed to read agent config: {}", e)))?;

    // Check if tool already exists
    if config_data
        .config_parsed
        .tools
        .iter()
        .any(|t| t.name == request.name)
    {
        return Err(ApiError::BadRequest(format!(
            "Tool '{}' already exists",
            request.name
        )));
    }

    // Write tool files
    let tool_info = fs_operations::ToolFileInfo {
        name: request.name.clone(),
        schema_json: request.schema_json,
        python_code: Some(request.python_code),
    };

    fs_operations::write_tool_files(agent_dir, &tool_info)
        .map_err(|e| ApiError::Internal(format!("Failed to write tool files: {}", e)))?;

    // Add tool to config
    config_data
        .config_parsed
        .tools
        .push(crate::tools::loader::ToolConfig {
            name: request.name.clone(),
            schema: std::path::PathBuf::from(format!("tools/{}.json", request.name)),
            implementation: crate::tools::loader::ToolImplementation::Python {
                script: std::path::PathBuf::from(format!("tools/{}.py", request.name)),
            },
        });

    // Write updated config to the original file path (preserve filename)
    let toml_path = std::path::PathBuf::from(&config_data.toml_file_path);
    fs_operations::write_agent_toml_to_path(&config_data.config_parsed, &toml_path)
        .map_err(|e| ApiError::Internal(format!("Failed to write agent config: {}", e)))?;

    // Reload agent
    state
        .reload_single_agent(&agent_name)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to reload agent: {}", e)))?;

    info!(agent = %agent_name, tool = %request.name, "Tool created successfully");
    Ok(Json(DeleteResponse { success: true }))
}

/// Update an existing tool.
///
/// PUT /api/agents/:name/tools/:tool_name
async fn update_tool(
    State(state): State<AppState>,
    Path((agent_name, tool_name)): Path<(String, String)>,
    Json(request): Json<UpdateToolRequest>,
) -> Result<Json<DeleteResponse>, ApiError> {
    info!(agent = %agent_name, tool = %tool_name, "Updating tool");

    let agent = state
        .get_agent(&agent_name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", agent_name)))?;

    let agent_dir = agent.base_dir();

    // Read current tool
    let config_data = fs_operations::read_agent_config(agent_dir, &agent_name)
        .map_err(|e| ApiError::Internal(format!("Failed to read agent config: {}", e)))?;

    let current_tool = config_data
        .tools_details
        .iter()
        .find(|t| t.name == tool_name)
        .ok_or_else(|| ApiError::NotFound(format!("Tool not found: {}", tool_name)))?;

    // Prepare updated tool info
    let schema_json = request
        .schema_json
        .unwrap_or(current_tool.schema_json.clone());
    let python_code = request.python_code.or_else(|| current_tool.code.clone());

    let tool_info = fs_operations::ToolFileInfo {
        name: tool_name.clone(),
        schema_json,
        python_code,
    };

    // Write updated tool files
    fs_operations::write_tool_files(agent_dir, &tool_info)
        .map_err(|e| ApiError::Internal(format!("Failed to write tool files: {}", e)))?;

    // Reload agent
    state
        .reload_single_agent(&agent_name)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to reload agent: {}", e)))?;

    info!(agent = %agent_name, tool = %tool_name, "Tool updated successfully");
    Ok(Json(DeleteResponse { success: true }))
}

/// Delete a tool from an agent.
///
/// DELETE /api/agents/:name/tools/:tool_name
async fn delete_tool(
    State(state): State<AppState>,
    Path((agent_name, tool_name)): Path<(String, String)>,
) -> Result<Json<DeleteResponse>, ApiError> {
    info!(agent = %agent_name, tool = %tool_name, "Deleting tool");

    let agent = state
        .get_agent(&agent_name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", agent_name)))?;

    let agent_dir = agent.base_dir();

    // Read current config
    let mut config_data = fs_operations::read_agent_config(agent_dir, &agent_name)
        .map_err(|e| ApiError::Internal(format!("Failed to read agent config: {}", e)))?;

    // Check if tool exists
    if !config_data
        .config_parsed
        .tools
        .iter()
        .any(|t| t.name == tool_name)
    {
        return Err(ApiError::NotFound(format!("Tool not found: {}", tool_name)));
    }

    // Delete tool files
    fs_operations::delete_tool_files(agent_dir, &tool_name)
        .map_err(|e| ApiError::Internal(format!("Failed to delete tool files: {}", e)))?;

    // Remove tool from config
    config_data
        .config_parsed
        .tools
        .retain(|t| t.name != tool_name);

    // Write updated config to the original file path (preserve filename)
    let toml_path = std::path::PathBuf::from(&config_data.toml_file_path);
    fs_operations::write_agent_toml_to_path(&config_data.config_parsed, &toml_path)
        .map_err(|e| ApiError::Internal(format!("Failed to write agent config: {}", e)))?;

    // Reload agent
    state
        .reload_single_agent(&agent_name)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to reload agent: {}", e)))?;

    info!(agent = %agent_name, tool = %tool_name, "Tool deleted successfully");
    Ok(Json(DeleteResponse { success: true }))
}

// ============================================================================
// Session Endpoints
// ============================================================================

/// List all active sessions.
///
/// GET /api/sessions
async fn list_sessions(
    State(state): State<AppState>,
) -> Result<Json<Vec<SessionSummary>>, ApiError> {
    state
        .sessions
        .list_sessions()
        .await
        .map(Json)
        .map_err(|e| ApiError::Internal(e.to_string()))
}

/// Get a specific session with full history.
///
/// GET /api/sessions/:session_id
async fn get_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Session>, ApiError> {
    state
        .sessions
        .get_session(&session_id)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("Session not found: {}", session_id)))
}

/// Delete a session.
///
/// DELETE /api/sessions/:session_id
async fn delete_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<DeleteResponse>, ApiError> {
    let deleted = state
        .sessions
        .delete_session(&session_id)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    if deleted {
        Ok(Json(DeleteResponse { success: true }))
    } else {
        Err(ApiError::NotFound(format!(
            "Session not found: {}",
            session_id
        )))
    }
}

/// List sessions for a specific agent.
///
/// GET /api/agents/:name/sessions
async fn list_agent_sessions(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Vec<SessionSummary>>, ApiError> {
    state
        .sessions
        .list_agent_sessions(&name)
        .await
        .map(Json)
        .map_err(|e| ApiError::Internal(e.to_string()))
}

// ============================================================================
// Chat Endpoints
// ============================================================================

/// Start a new chat session with streaming response.
///
/// POST /api/agents/:name/chat
async fn chat_stream(
    State(state): State<AppState>,
    Path(agent_name): Path<String>,
    Json(request): Json<ChatRequest>,
) -> Result<impl IntoResponse, ApiError> {
    info!(agent = %agent_name, "Starting new chat session");

    // Get agent
    let agent = state
        .get_agent(&agent_name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", agent_name)))?;

    // Create new session
    let session = state
        .sessions
        .create_session(agent_name.clone(), agent.model())
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    // Create streaming response
    let (tx, stream) = EventStream::new();
    let sender = EventSender::new(tx);

    // Spawn task to handle streaming
    tokio::spawn(async move {
        if let Err(e) = handle_chat_streaming(
            agent,
            session.id.clone(),
            request.message,
            None,
            &state,
            sender.clone(),
        )
        .await
        {
            error!(error = %e, "Chat streaming failed");
            let _ = sender.error(e.to_string());
        }
        let _ = sender.done();
    });

    Ok(stream.into_sse_response())
}

/// Continue an existing chat session with streaming response.
///
/// POST /api/sessions/:session_id/chat
async fn continue_chat_stream(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(request): Json<ChatRequest>,
) -> Result<impl IntoResponse, ApiError> {
    info!(session_id = %session_id, "Continuing chat session");

    // Get existing session
    let session = state
        .sessions
        .get_session(&session_id)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?
        .ok_or_else(|| ApiError::NotFound(format!("Session not found: {}", session_id)))?;

    // Get agent
    let agent = state
        .get_agent(&session.agent_name)
        .ok_or_else(|| ApiError::NotFound(format!("Agent not found: {}", session.agent_name)))?;

    // Create streaming response
    let (tx, stream) = EventStream::new();
    let sender = EventSender::new(tx);

    // Spawn task to handle streaming
    tokio::spawn(async move {
        if let Err(e) = handle_chat_streaming(
            agent,
            session.id.clone(),
            request.message,
            Some(session.messages),
            &state,
            sender.clone(),
        )
        .await
        {
            error!(error = %e, "Chat streaming failed");
            let _ = sender.error(e.to_string());
        }
        let _ = sender.done();
    });

    Ok(stream.into_sse_response())
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Chat request containing a user message.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// The user's message.
    pub message: String,
}

/// Response from reloading agents.
#[derive(Debug, Serialize)]
pub struct ReloadResponse {
    /// Number of agents reloaded.
    pub reloaded_count: usize,
}

/// Response from deleting an agent.
#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    /// Whether the deletion was successful.
    pub success: bool,
}

// ============================================================================
// Agent Builder Types
// ============================================================================

/// Full agent configuration response for builder UI.
#[derive(Debug, Serialize)]
pub struct AgentConfigResponse {
    /// The name of the agent.
    pub name: String,
    /// The raw TOML configuration content.
    pub config_toml: String,
    /// The parsed agent configuration.
    pub config_parsed: AgentConfig,
    /// The content of the system prompt file.
    pub prompt_content: String,
    /// Details of all tools configured for the agent.
    pub tools_details: Vec<ToolDetailResponse>,
    /// The base directory path of the agent.
    pub base_path: String,
}

/// Tool detail response.
#[derive(Debug, Serialize)]
pub struct ToolDetailResponse {
    /// The name of the tool.
    pub name: String,
    /// The JSON schema for the tool.
    pub schema_json: String,
    /// The implementation type (e.g., "python", "rust:module_name").
    pub implementation_type: String,
    /// Optional implementation code for Python tools.
    pub code: Option<String>,
}

/// Create agent request.
#[derive(Debug, Deserialize)]
pub struct CreateAgentRequest {
    /// The name of the new agent.
    pub name: String,
    /// Optional model identifier.
    pub model: Option<String>,
    /// Optional description of the agent.
    pub description: Option<String>,
    /// Optional version string.
    pub version: Option<String>,
    /// The system prompt for the agent.
    pub system_prompt: String,
}

/// Update agent request.
#[derive(Debug, Deserialize)]
pub struct UpdateAgentRequest {
    /// Optional new model identifier.
    pub model: Option<String>,
    /// Optional new description.
    pub description: Option<String>,
    /// Optional new version string.
    pub version: Option<String>,
}

/// Duplicate agent request.
#[derive(Debug, Deserialize)]
pub struct DuplicateAgentRequest {
    /// The name for the duplicated agent.
    pub new_name: String,
}

/// Create tool request.
#[derive(Debug, Deserialize)]
pub struct CreateToolRequest {
    /// The name of the new tool.
    pub name: String,
    /// The JSON schema for the tool.
    pub schema_json: String,
    /// The Python implementation code.
    pub python_code: String,
}

/// Update tool request.
#[derive(Debug, Deserialize)]
pub struct UpdateToolRequest {
    /// Optional new JSON schema.
    pub schema_json: Option<String>,
    /// Optional new Python implementation code.
    pub python_code: Option<String>,
}

/// Update prompt request.
#[derive(Debug, Deserialize)]
pub struct UpdatePromptRequest {
    /// The new prompt content.
    pub content: String,
}

// ============================================================================
// Error Handling
// ============================================================================

/// API error types.
#[derive(Debug)]
pub enum ApiError {
    /// Resource not found.
    NotFound(String),
    /// Bad request from client.
    BadRequest(String),
    /// Internal server error.
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

// ============================================================================
// Chat Streaming Handler
// ============================================================================

/// Handle the full chat streaming flow.
///
/// This orchestrates the agent execution with real-time streaming feedback
/// to the client via SSE events. Now uses the core streaming infrastructure.
async fn handle_chat_streaming(
    agent: Arc<crate::agent::TomlAgent>,
    session_id: String,
    user_message: String,
    previous_messages: Option<Vec<ChatMessage>>,
    state: &AppState,
    sender: EventSender,
) -> anyhow::Result<()> {
    use crate::agent::Agent;

    // If this is a new conversation, use run_streaming
    // If continuing, we need custom logic for message continuation
    let session = if let Some(mut previous_messages) = previous_messages {
        previous_messages.push(ChatMessage {
            role: crate::llm::Role::User,
            name: None,
            tool_call_id: None,
            content: Some(user_message),
            tool_calls: None,
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: None,
            timestamp: Some(chrono::Utc::now()),
            id: None,
            provider_response_id: None,
            status: None,
        });

        crate::agent::runtime::default_run_streaming_with_messages(
            agent.as_ref(),
            previous_messages,
            Box::new(sender.clone()),
        )
        .await?
    } else {
        // Simple case: new conversation - use the built-in streaming
        agent
            .run_streaming(&user_message, Box::new(sender.clone()))
            .await?
    };

    if let Some(mut stored_session) = state.sessions.get_session(&session_id).await? {
        stored_session.messages = session.messages;
        stored_session.updated_at = chrono::Utc::now();
        state.sessions.update_session(&stored_session).await?;
    }

    sender.turn_completed().map_err(|e| anyhow::anyhow!(e))?;
    Ok(())
}
