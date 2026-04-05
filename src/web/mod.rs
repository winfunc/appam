//! Experimental web API server with SSE streaming and session management.
//!
//! Provides a REST API for managing agents and chat sessions with:
//! - Server-Sent Events (SSE) for streaming responses
//! - Session persistence and continuation
//! - Multi-agent support
//! - Rate limiting and CORS
//! - Comprehensive error handling
//!
//! # Status
//!
//! This module is an old experimental surface and is currently disabled for
//! security reasons. The implementation is kept in-tree for future reference,
//! but [`serve`] will refuse to start the API until the surface is redesigned
//! with authentication and a tighter trust model.

pub mod fs_operations;
pub mod middleware;
pub mod routes;
pub mod session;
pub mod state;
pub mod streaming;
pub mod trace_models;
pub mod trace_parser;
pub mod trace_routes;

pub use state::AppState;

use axum::Router;
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;

/// Hard-disable the legacy web API while retaining the code for future work.
const EXPERIMENTAL_WEB_API_ENABLED: bool = false;

/// Start the web API server.
///
/// Creates an axum server with all routes, middleware, and state configured.
/// Binds to the specified host and port and runs until shutdown.
///
/// # Parameters
///
/// - `host`: Host address to bind to (e.g., "0.0.0.0")
/// - `port`: Port to listen on
/// - `agents_dir`: Directory containing agent TOML files
///
/// # Errors
///
/// Returns an error if the server fails to start or bind to the address.
pub async fn serve(host: String, port: u16, agents_dir: std::path::PathBuf) -> anyhow::Result<()> {
    if !EXPERIMENTAL_WEB_API_ENABLED {
        anyhow::bail!(
            "The Appam web API is experimental and currently disabled for security reasons."
        );
    }

    info!("Starting Appam Web API server");
    info!(host = %host, port = port, "Server configuration");

    // Create application state
    let state = AppState::new(agents_dir).await?;
    info!(agents = state.list_agents().len(), "Agents loaded");

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router with all routes
    let app = Router::new()
        .merge(routes::agent_routes())
        .merge(routes::session_routes())
        .merge(routes::chat_routes())
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .layer(middleware::rate_limit_layer())
        .with_state(state);

    // Bind and serve
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    info!("Server listening on http://{}", addr);
    info!("API Documentation: http://{}/api/docs", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}

/// Start the trace visualizer web server.
///
/// Creates a lightweight Axum server serving the trace visualizer UI and API
/// endpoints for listing and retrieving trace files.
///
/// # Parameters
///
/// - `host`: Host address to bind to (e.g., "0.0.0.0")
/// - `port`: Port to listen on
/// - `traces_dir`: Directory containing trace files
///
/// # Errors
///
/// Returns an error if the server fails to start or bind to the address.
///
/// # Examples
///
/// ```ignore
/// # use std::path::PathBuf;
/// appam::web::serve_tracing(
///     "127.0.0.1".to_string(),
///     8080,
///     PathBuf::from("logs")
/// ).await.unwrap();
/// ```
pub async fn serve_tracing(
    host: String,
    port: u16,
    traces_dir: std::path::PathBuf,
) -> anyhow::Result<()> {
    use std::sync::Arc;

    info!("Starting Appam Trace Visualizer");
    info!(host = %host, port = port, traces_dir = %traces_dir.display(), "Server configuration");

    // Ensure traces directory exists
    if !traces_dir.exists() {
        anyhow::bail!("Traces directory does not exist: {}", traces_dir.display());
    }

    // Create trace state
    let state = Arc::new(trace_routes::TraceState::new(traces_dir));

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router with trace routes
    let app = trace_routes::trace_routes()
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Bind and serve
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    info!("Trace Visualizer listening on http://{}", addr);
    info!("Open in browser: http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}
