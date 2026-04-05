//! TUI Coding Agent using AWS Bedrock with Claude Opus 4.6
//!
//! A minimal terminal interface coding assistant with:
//! - Multi-turn conversation loop
//! - File operations (read, write, list)
//! - Bash command execution
//! - Adaptive thinking (Opus 4.6) with effort control
//! - Streaming responses via SigV4 authentication (default)
//! - EventStream binary frame decoding for Bedrock streaming
//!
//! # Authentication Methods
//!
//! ## SigV4 (Default, supports streaming)
//!
//! Uses standard AWS credentials:
//! ```bash
//! export AWS_ACCESS_KEY_ID="your-access-key"
//! export AWS_SECRET_ACCESS_KEY="your-secret-key"
//! export AWS_SESSION_TOKEN="your-session-token"  # optional, for temporary credentials
//! export AWS_REGION="us-east-1"  # optional, defaults to us-east-1
//! export AWS_BEDROCK_MODEL_ID="us.anthropic.claude-opus-4-6-v1"  # optional
//! cargo run --example coding-agent-bedrock
//! ```
//!
//! ## Bearer Token (Non-streaming only)
//!
//! Uses Bedrock API Keys:
//! ```bash
//! export AWS_BEARER_TOKEN_BEDROCK="your-token"
//! export AWS_BEDROCK_AUTH_METHOD="bearer_token"
//! export AWS_REGION="us-east-1"
//! cargo run --example coding-agent-bedrock
//! ```
//!
//! Usage:
//!   cargo run --example coding-agent-bedrock

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

// ============================================================================
// Main TUI Application
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Read configuration from environment
    let region = std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|_| "us-east-1".to_string());
    let model_id = std::env::var("AWS_BEDROCK_MODEL_ID")
        .unwrap_or_else(|_| "us.anthropic.claude-opus-4-6-v1".to_string());

    // Determine auth method (default to SigV4 for streaming support)
    let auth_method = std::env::var("AWS_BEDROCK_AUTH_METHOD")
        .map(|v| v.to_lowercase())
        .unwrap_or_else(|_| "sigv4".to_string());

    let use_bearer_token = auth_method == "bearer_token" || auth_method == "bearer";

    println!("🚀 Coding Agent - Claude via AWS Bedrock\n");
    println!("   Region: {}", region);
    println!("   Model:  {}", model_id);
    println!(
        "   Auth:   {} {}",
        if use_bearer_token {
            "Bearer Token"
        } else {
            "SigV4"
        },
        if use_bearer_token {
            "(non-streaming)"
        } else {
            "(streaming)"
        }
    );

    // Determine the auth method enum value
    let bedrock_auth = if use_bearer_token {
        appam::llm::anthropic::BedrockAuthMethod::BearerToken
    } else {
        appam::llm::anthropic::BedrockAuthMethod::SigV4
    };

    // Build agent with AWS Bedrock provider
    let agent = AgentBuilder::new("claude-bedrock")
        .provider(LlmProvider::Bedrock {
            region: region.clone(),
            model_id: model_id.clone(),
            auth_method: bedrock_auth,
        })
        // Model is specified in the provider, but we still need a placeholder
        // The actual model is determined by model_id in the Bedrock provider
        .model(&model_id)
        .system_prompt(
            "You are an advanced coding assistant powered by Claude Opus 4.6 via AWS Bedrock. \
             You have access to file operations, bash commands, and directory listing. \
             Use your adaptive thinking capabilities to reason through complex problems. \
             Always explain your reasoning process and provide detailed analysis.",
        )
        // Adaptive thinking for Opus 4.6 (replaces fixed budget_tokens)
        .thinking(appam::llm::anthropic::ThinkingConfig::adaptive())
        // Effort level: max for deepest reasoning
        .effort(appam::llm::anthropic::EffortLevel::Max)
        // Beta features for Bedrock
        .beta_features(appam::llm::anthropic::BetaFeatures {
            context_1m: true,
            effort: true,
            ..Default::default()
        })
        // Tool choice: let model decide
        .tool_choice(appam::llm::anthropic::ToolChoiceConfig::Auto {
            disable_parallel_tool_use: false,
        })
        // Retry configuration
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

    println!("✓ Adaptive thinking enabled (Opus 4.6)");
    println!("✓ Effort level: max");
    println!("✓ Tools: read_file, write_file, bash, list_files");
    println!("✓ Type 'exit', 'quit', or 'bye' to end conversation\n");

    // Initialize readline for multi-turn conversation
    let mut rl = DefaultEditor::new()?;

    loop {
        // Read user input
        let readline = rl.readline("You> ");
        match readline {
            Ok(line) => {
                let input = line.trim();

                // Check for exit commands
                if input.eq_ignore_ascii_case("exit")
                    || input.eq_ignore_ascii_case("quit")
                    || input.eq_ignore_ascii_case("bye")
                {
                    println!("\n👋 Goodbye!");
                    break;
                }

                if input.is_empty() {
                    continue;
                }

                // Add to history
                let _ = rl.add_history_entry(input);

                println!("\nAssistant:\n");

                // Track if we've shown thinking header
                let thinking_shown = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let thinking_shown_clone = Arc::clone(&thinking_shown);

                // Stream agent response
                match agent
                    .stream(input)
                    .on_content(|content| {
                        print!("{}", content);
                        std::io::stdout().flush().ok();
                    })
                    .on_reasoning(move |content| {
                        if !thinking_shown_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            println!("\n\n💭 Thinking:\n");
                            thinking_shown_clone.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        print!("{}", content);
                        std::io::stdout().flush().ok();
                    })
                    .on_tool_call(|tool_name, arguments| {
                        println!("\n\n🔧 {}", tool_name);
                        let args_str = arguments.to_string();
                        if args_str.len() > 200 {
                            println!("   Args: {}...", &args_str[..200]);
                        } else {
                            println!("   Args: {}", args_str);
                        }
                    })
                    .on_tool_result(|tool_name, result| {
                        println!("   ✓ {} completed", tool_name);
                        let result_str = serde_json::to_string_pretty(&result).unwrap_or_default();
                        if result_str.len() > 300 {
                            println!("   Result: {}...", &result_str[..300]);
                        } else {
                            println!("   Result: {}", result_str);
                        }
                    })
                    .run()
                    .await
                {
                    Ok(_) => {
                        println!("\n");
                    }
                    Err(e) => {
                        eprintln!("\n❌ Error: {}\n", e);
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("\n👋 Goodbye!");
                break;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\n👋 Goodbye!");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}
