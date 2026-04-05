//! TUI Coding Agent using Azure Anthropic with Claude Opus 4.6
//!
//! A minimal terminal interface coding assistant with:
//! - Multi-turn conversation loop
//! - File operations (read, write, list)
//! - Bash command execution
//! - Adaptive thinking with Anthropic tool calling
//! - Azure-hosted Anthropic Messages API streaming
//!
//! # Azure Anthropic Setup
//!
//! Azure Anthropic requires:
//! 1. A Claude deployment on Azure AI Foundry or Azure-hosted Anthropic endpoint
//! 2. A base endpoint such as `https://<resource>.services.ai.azure.com/anthropic`
//!    or `https://<resource>.openai.azure.com/anthropic`
//! 3. Either an Azure API key or a bearer token
//!
//! # Environment Variables
//!
//! **Required:**
//! - `AZURE_API_KEY` or `AZURE_ANTHROPIC_API_KEY` for `x-api-key` auth
//! - `AZURE_API_KEY` or `AZURE_ANTHROPIC_AUTH_TOKEN` for bearer auth
//! - `AZURE_ANTHROPIC_BASE_URL` or `AZURE_ANTHROPIC_RESOURCE`
//!
//! **Optional:**
//! - `AZURE_ANTHROPIC_AUTH_METHOD` - `x_api_key` (default) or `bearer`
//! - `AZURE_ANTHROPIC_MODEL` - Deployment/model name (default: `claude-opus-4-6`)
//!
//! # Usage
//!
//! ```bash
//! export AZURE_API_KEY="your-azure-key"
//! export AZURE_ANTHROPIC_BASE_URL="https://your-resource.services.ai.azure.com/anthropic"
//! export AZURE_ANTHROPIC_MODEL="claude-opus-4-6"  # optional
//! cargo run --example coding-agent-azure-anthropic
//! ```

use anyhow::{Context, Result};
use appam::prelude::*;
use rustyline::DefaultEditor;
use std::io::Write;
use std::process::Command;

// ============================================================================
// Tool Definitions
// ============================================================================

#[derive(Deserialize, Schema)]
struct ReadFileInput {
    #[description = "Path to the file to read"]
    file_path: String,
}

#[derive(Serialize)]
struct ReadFileOutput {
    success: bool,
    contents: Option<String>,
    file_path: String,
    size_bytes: Option<usize>,
    error: Option<String>,
}

/// Read the contents of a file from the filesystem
#[tool(description = "Read the contents of a file from the filesystem")]
fn read_file(input: ReadFileInput) -> Result<ReadFileOutput> {
    match std::fs::read_to_string(&input.file_path) {
        Ok(contents) => Ok(ReadFileOutput {
            success: true,
            contents: Some(contents.clone()),
            file_path: input.file_path,
            size_bytes: Some(contents.len()),
            error: None,
        }),
        Err(e) => Ok(ReadFileOutput {
            success: false,
            contents: None,
            file_path: input.file_path,
            size_bytes: None,
            error: Some(format!("Failed to read file: {}", e)),
        }),
    }
}

#[derive(Deserialize, Schema)]
struct WriteFileInput {
    #[description = "Path where the file should be written"]
    file_path: String,
    #[description = "Content to write to the file"]
    content: String,
}

#[derive(Serialize)]
struct WriteFileOutput {
    success: bool,
    message: Option<String>,
    file_path: String,
    bytes_written: Option<usize>,
    error: Option<String>,
}

/// Write content to a file, creating it if it doesn't exist
#[tool(description = "Write content to a file, creating it if it doesn't exist")]
fn write_file(input: WriteFileInput) -> Result<WriteFileOutput> {
    match std::fs::write(&input.file_path, &input.content) {
        Ok(_) => Ok(WriteFileOutput {
            success: true,
            message: Some(format!(
                "Successfully wrote {} bytes to {}",
                input.content.len(),
                input.file_path
            )),
            file_path: input.file_path,
            bytes_written: Some(input.content.len()),
            error: None,
        }),
        Err(e) => Ok(WriteFileOutput {
            success: false,
            message: None,
            file_path: input.file_path,
            bytes_written: None,
            error: Some(format!("Failed to write file: {}", e)),
        }),
    }
}

#[derive(Serialize)]
struct BashOutput {
    success: bool,
    exit_code: i32,
    stdout: String,
    stderr: String,
    command: String,
}

/// Execute a bash command and return its output
#[tool(description = "Execute a bash command and return its output")]
fn bash(#[arg(description = "The bash command to execute")] command: String) -> Result<BashOutput> {
    let output = Command::new("bash")
        .arg("-c")
        .arg(&command)
        .output()
        .context("Failed to execute bash command")?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok(BashOutput {
        success: output.status.success(),
        exit_code: output.status.code().unwrap_or(-1),
        stdout,
        stderr,
        command,
    })
}

#[derive(Deserialize, Schema)]
struct ListFilesInput {
    #[description = "Directory path to list"]
    directory: String,
    #[description = "Whether to list recursively"]
    #[serde(default)]
    recursive: bool,
}

#[derive(Serialize)]
struct ListFilesOutput {
    success: bool,
    directory: String,
    entries: Vec<serde_json::Value>,
    count: usize,
    error: Option<String>,
}

/// List files and subdirectories in a directory
#[tool(description = "List files and subdirectories in a directory")]
fn list_files(input: ListFilesInput) -> Result<ListFilesOutput> {
    let mut entries = Vec::new();

    if input.recursive {
        for entry in walkdir::WalkDir::new(&input.directory)
            .max_depth(5)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            entries.push(json!({
                "path": entry.path().display().to_string(),
                "is_dir": entry.file_type().is_dir(),
                "depth": entry.depth()
            }));
        }
        Ok(ListFilesOutput {
            success: true,
            directory: input.directory,
            count: entries.len(),
            entries,
            error: None,
        })
    } else {
        match std::fs::read_dir(&input.directory) {
            Ok(dir_entries) => {
                for entry in dir_entries.filter_map(|e| e.ok()) {
                    entries.push(json!({
                        "path": entry.path().display().to_string(),
                        "is_dir": entry.path().is_dir()
                    }));
                }
                Ok(ListFilesOutput {
                    success: true,
                    directory: input.directory,
                    count: entries.len(),
                    entries,
                    error: None,
                })
            }
            Err(e) => Ok(ListFilesOutput {
                success: false,
                directory: input.directory,
                entries: vec![],
                count: 0,
                error: Some(format!("Failed to read directory: {}", e)),
            }),
        }
    }
}

fn resolve_base_url() -> Result<String> {
    if let Ok(base_url) = std::env::var("AZURE_ANTHROPIC_BASE_URL") {
        if !base_url.trim().is_empty() {
            return Ok(base_url);
        }
    }

    let resource = std::env::var("AZURE_ANTHROPIC_RESOURCE").context(
        "AZURE_ANTHROPIC_BASE_URL or AZURE_ANTHROPIC_RESOURCE environment variable is required.",
    )?;
    appam::llm::anthropic::AzureAnthropicConfig::base_url_from_resource(&resource)
}

fn resolve_auth_method() -> Result<appam::llm::anthropic::AzureAnthropicAuthMethod> {
    Ok(std::env::var("AZURE_ANTHROPIC_AUTH_METHOD")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(appam::llm::anthropic::AzureAnthropicAuthMethod::XApiKey))
}

fn ensure_credentials_present(
    auth_method: &appam::llm::anthropic::AzureAnthropicAuthMethod,
) -> Result<()> {
    match auth_method {
        appam::llm::anthropic::AzureAnthropicAuthMethod::XApiKey => {
            if std::env::var("AZURE_ANTHROPIC_API_KEY").is_err()
                && std::env::var("AZURE_API_KEY").is_err()
            {
                anyhow::bail!(
                    "AZURE_ANTHROPIC_API_KEY or AZURE_API_KEY environment variable is required for x-api-key auth."
                );
            }
        }
        appam::llm::anthropic::AzureAnthropicAuthMethod::BearerToken => {
            if std::env::var("AZURE_ANTHROPIC_AUTH_TOKEN").is_err()
                && std::env::var("AZURE_API_KEY").is_err()
            {
                anyhow::bail!(
                    "AZURE_ANTHROPIC_AUTH_TOKEN or AZURE_API_KEY environment variable is required for bearer auth."
                );
            }
        }
    }

    Ok(())
}

// ============================================================================
// Main TUI Application
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let base_url = resolve_base_url()?;
    let auth_method = resolve_auth_method()?;
    ensure_credentials_present(&auth_method)?;
    let model =
        std::env::var("AZURE_ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-opus-4-6".to_string());

    println!("🚀 Coding Agent - Claude via Azure Anthropic\n");
    println!("   Base URL: {}", base_url);
    println!("   Model:    {}", model);
    println!("   Auth:     {}", auth_method.as_str());

    let agent = AgentBuilder::new("azure-anthropic-coding-assistant")
        .provider(LlmProvider::AzureAnthropic {
            base_url: base_url.clone(),
            auth_method: auth_method.clone(),
        })
        .model(&model)
        .system_prompt(
            "You are an advanced coding assistant powered by Claude Opus 4.6 via Azure Anthropic. \
             You have access to file operations, bash commands, and directory listing. \
             Use your adaptive thinking capabilities to reason through complex problems. \
             Always explain your reasoning process and provide detailed analysis.",
        )
        .thinking(appam::llm::anthropic::ThinkingConfig::adaptive())
        .effort(appam::llm::anthropic::EffortLevel::Max)
        .caching(appam::llm::anthropic::CachingConfig {
            enabled: true,
            ttl: appam::llm::anthropic::CacheTTL::OneHour,
        })
        .beta_features(appam::llm::anthropic::BetaFeatures {
            fine_grained_tool_streaming: true,
            context_management: true,
            interleaved_thinking: true,
            ..Default::default()
        })
        .tool_choice(appam::llm::anthropic::ToolChoiceConfig::Auto {
            disable_parallel_tool_use: false,
        })
        .rate_limiter(appam::llm::anthropic::RateLimiterConfig {
            enabled: true,
            tokens_per_minute: 1_800_000,
        })
        .retry(appam::llm::anthropic::RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 2000,
            max_backoff_ms: 60000,
            backoff_multiplier: 2.0,
            jitter: true,
        })
        .with_tool(Arc::new(read_file()))
        .with_tool(Arc::new(write_file()))
        .with_tool(Arc::new(bash()))
        .with_tool(Arc::new(list_files()))
        .max_tokens(20000)
        .build()?;

    println!("✓ Adaptive thinking enabled");
    println!("✓ Prompt caching active (1-hour TTL)");
    println!("✓ Tools: read_file, write_file, bash, list_files");
    println!("\nType 'exit' or 'quit' to end the session.\n");

    let mut rl = DefaultEditor::new()?;

    loop {
        let input = match rl.readline("You> ") {
            Ok(line) => line.trim().to_string(),
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("\nInterrupted. Type 'exit' to quit.");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\nGoodbye!");
                break;
            }
            Err(e) => return Err(e).context("Failed to read input"),
        };

        if input.is_empty() {
            continue;
        }

        if matches!(input.as_str(), "exit" | "quit" | "bye") {
            println!("Goodbye!");
            break;
        }

        let _ = rl.add_history_entry(&input);

        println!("\nAssistant> ");
        std::io::stdout().flush()?;

        let tool_calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let tool_calls_for_callback = Arc::clone(&tool_calls);

        let result = agent
            .stream(&input)
            .on_content(|content| {
                print!("{}", content);
                std::io::stdout().flush().ok();
            })
            .on_reasoning(|reasoning| {
                print!("\n[Thinking] {}\n", reasoning);
                std::io::stdout().flush().ok();
            })
            .on_tool_call(move |tool_name, args| {
                println!("\n🔧 Tool Call: {}", tool_name);
                println!(
                    "   Args: {}",
                    serde_json::to_string_pretty(&args).unwrap_or_default()
                );
                tool_calls_for_callback
                    .lock()
                    .unwrap()
                    .push(tool_name.to_string());
            })
            .on_tool_result(|tool_name, result| {
                println!("\n✓ {} completed", tool_name);
                if let Some(error) = result.get("error").and_then(|e| e.as_str()) {
                    println!("   Error: {}", error);
                }
            })
            .run()
            .await;

        match result {
            Ok(_) => {
                println!("\n");
                let executed_tools = tool_calls.lock().unwrap();
                if !executed_tools.is_empty() {
                    println!(
                        "Used tools: {}\n",
                        executed_tools
                            .iter()
                            .map(String::as_str)
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
            Err(e) => {
                eprintln!("\n❌ Error: {}\n", e);
            }
        }
    }

    Ok(())
}
