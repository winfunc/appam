//! Appam CLI binary.
//!
//! Command-line interface for managing and running AI agents.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use appam::agent::{Agent, TomlAgent};

#[derive(Parser)]
#[command(name = "appam")]
#[command(version, about = "AI Agent Framework", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run an agent interactively
    Run {
        /// Path to agent TOML file
        #[arg(value_name = "AGENT")]
        agent: PathBuf,

        /// Optional user prompt (if not provided, agent runs in interactive mode)
        #[arg(value_name = "PROMPT")]
        prompt: Option<String>,

        /// Override the model specified in config
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Experimental web API server (currently disabled)
    Serve {
        /// Directory containing agent TOML files (default: ./agents)
        #[arg(short, long, default_value = "agents")]
        agents_dir: PathBuf,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value_t = 3000)]
        port: u16,
    },

    /// Validate an agent configuration
    Validate {
        /// Path to agent TOML file
        #[arg(value_name = "AGENT")]
        agent: PathBuf,
    },

    /// List all available agents
    List {
        /// Directory containing agent TOML files (default: ./agents)
        #[arg(short, long, default_value = "agents")]
        agents_dir: PathBuf,
    },

    /// Create a new agent from a template
    New {
        /// Agent name
        #[arg(value_name = "NAME")]
        name: String,

        /// Output directory (default: ./agents)
        #[arg(short, long, default_value = "agents")]
        output: PathBuf,
    },

    /// Start trace visualizer web server
    Tracing {
        /// Directory containing trace files (default: ./logs)
        #[arg(short, long, default_value = "logs")]
        traces_dir: PathBuf,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            agent,
            prompt,
            model,
        } => run_agent(agent, prompt, model).await?,

        Commands::Serve {
            agents_dir,
            host,
            port,
        } => serve_api(agents_dir, host, port).await?,

        Commands::Validate { agent } => validate_agent(agent)?,

        Commands::List { agents_dir } => list_agents(agents_dir)?,

        Commands::New { name, output } => create_agent(name, output)?,

        Commands::Tracing {
            traces_dir,
            host,
            port,
        } => serve_tracing(traces_dir, host, port).await?,
    }

    Ok(())
}

/// Run an agent with a prompt.
async fn run_agent(
    agent_path: PathBuf,
    prompt: Option<String>,
    model_override: Option<String>,
) -> Result<()> {
    println!("Loading agent from: {}", agent_path.display());

    let mut agent = TomlAgent::from_file(&agent_path)
        .with_context(|| format!("Failed to load agent: {}", agent_path.display()))?;

    if let Some(model) = model_override {
        agent = agent.with_model(model);
    }

    if let Some(user_prompt) = prompt {
        // Single-shot execution
        println!("Running agent with prompt: {}", user_prompt);
        agent.run(&user_prompt).await?;
    } else {
        // Interactive mode (to be implemented)
        println!("Interactive mode not yet implemented. Please provide a prompt.");
        println!(
            "Usage: appam run {} \"Your prompt here\"",
            agent_path.display()
        );
    }

    Ok(())
}

/// Start the web API server.
async fn serve_api(agents_dir: PathBuf, host: String, port: u16) -> Result<()> {
    use appam::web;

    web::serve(host, port, agents_dir).await?;

    Ok(())
}

/// Validate an agent configuration.
fn validate_agent(agent_path: PathBuf) -> Result<()> {
    println!("Validating agent: {}", agent_path.display());

    use appam::agent::config::AgentConfig;
    use appam::tools::loader::validate_tool_configs;

    let config = AgentConfig::from_file(&agent_path)
        .with_context(|| format!("Failed to load agent config: {}", agent_path.display()))?;

    let base_dir = agent_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid agent path"))?;

    // Validate agent config
    config
        .validate(base_dir)
        .context("Agent validation failed")?;

    // Validate tool configurations
    validate_tool_configs(&config.tools, base_dir).context("Tool validation failed")?;

    println!("✓ Agent configuration is valid");
    println!("  Name: {}", config.agent.name);
    println!(
        "  Model: {}",
        config.agent.model.as_deref().unwrap_or("default")
    );
    println!("  Tools: {}", config.tools.len());

    if let Some(desc) = &config.agent.description {
        println!("  Description: {}", desc);
    }

    Ok(())
}

/// List all available agents.
fn list_agents(agents_dir: PathBuf) -> Result<()> {
    if !agents_dir.exists() {
        println!("Agents directory not found: {}", agents_dir.display());
        println!("Create agents with: appam new <name>");
        return Ok(());
    }

    use appam::agent::config::AgentConfig;

    let mut agents = Vec::new();

    // Scan directory for TOML files
    for entry in std::fs::read_dir(&agents_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            if let Ok(config) = AgentConfig::from_file(&path) {
                agents.push((path.clone(), config));
            }
        }
    }

    if agents.is_empty() {
        println!("No agents found in: {}", agents_dir.display());
        println!("Create agents with: appam new <name>");
        return Ok(());
    }

    println!("Available agents in {}:", agents_dir.display());
    println!();

    for (path, config) in agents {
        println!("  • {} ({})", config.agent.name, path.display());
        if let Some(desc) = &config.agent.description {
            println!("    {}", desc);
        }
        println!(
            "    Model: {}",
            config.agent.model.as_deref().unwrap_or("default")
        );
        println!("    Tools: {}", config.tools.len());
        println!();
    }

    Ok(())
}

/// Create a new agent from a template.
fn create_agent(name: String, output_dir: PathBuf) -> Result<()> {
    use std::fs;

    println!("Creating new agent: {}", name);

    let agent_dir = output_dir.join(&name);
    if agent_dir.exists() {
        anyhow::bail!("Agent directory already exists: {}", agent_dir.display());
    }

    // Create directory structure
    fs::create_dir_all(&agent_dir)?;
    fs::create_dir_all(agent_dir.join("tools"))?;

    // Create agent config
    let config_path = agent_dir.join("agent.toml");
    fs::write(
        &config_path,
        format!(
            r#"[agent]
name = "{}"
model = "openai/gpt-5"
system_prompt = "prompt.txt"
description = "A new AI agent"

# Add tool configurations here
# [[tools]]
# name = "example"
# schema = "tools/example.json"
# implementation = {{ type = "python", script = "tools/example.py" }}
"#,
            name
        ),
    )?;

    // Create system prompt template
    let prompt_path = agent_dir.join("prompt.txt");
    fs::write(
        &prompt_path,
        format!(
            r#"You are {}, an AI assistant with specialized capabilities.

Your role is to help users by leveraging your available tools effectively.

Always:
- Be helpful, accurate, and concise
- Use tools when appropriate to provide factual information
- Explain your reasoning clearly
- Ask clarifying questions if needed

Available tools will be provided in your tool specifications.
"#,
            name
        ),
    )?;

    println!("✓ Agent created successfully!");
    println!();
    println!("Agent directory: {}", agent_dir.display());
    println!("Next steps:");
    println!(
        "  1. Edit {} to customize your agent",
        prompt_path.file_name().unwrap().to_string_lossy()
    );
    println!("  2. Add tools manually in the tools/ directory if needed");
    println!("  3. Update agent.toml with your tool configurations");
    println!("  4. Validate: appam validate {}", config_path.display());
    println!(
        "  5. Run: appam run {} \"Your prompt\"",
        config_path.display()
    );

    Ok(())
}

/// Start the trace visualizer web server.
async fn serve_tracing(traces_dir: PathBuf, host: String, port: u16) -> Result<()> {
    use appam::web;

    println!("🔍 Starting Appam Trace Visualizer");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Traces directory: {}", traces_dir.display());
    println!("Server address: http://{}:{}", host, port);
    println!();
    println!("Open in browser: http://{}:{}", host, port);
    println!();

    web::serve_tracing(host, port, traces_dir).await?;

    Ok(())
}
