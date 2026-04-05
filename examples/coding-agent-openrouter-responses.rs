//! TUI Coding Agent using OpenRouter Responses API with GPT-4o
//!
//! A minimal terminal interface coding assistant with:
//! - Multi-turn conversation loop
//! - File operations (read, write, list)
//! - Bash command execution
//! - High-effort reasoning
//!
//! Usage:
//!   export OPENROUTER_API_KEY="your-key"
//!   cargo run --example coding-agent-openrouter-responses

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
    println!("🚀 Coding Agent - GPT-4o via OpenRouter Responses API\n");

    // Build agent with OpenRouter Responses configuration
    let agent = AgentBuilder::new("coding-assistant")
        .provider(LlmProvider::OpenRouterResponses)
        .model("openai/gpt-4o")
        .system_prompt(
            "You are an expert coding assistant with access to file operations and bash commands. \
             Help users with code analysis, file management, and system tasks. \
             Always use the appropriate tools when working with files or executing commands.",
        )
        // High-effort reasoning (note: can only specify effort OR max_tokens, not both)
        .reasoning(appam::agent::ReasoningProvider::OpenRouter(
            appam::llm::openrouter::ReasoningConfig {
                enabled: Some(true),
                effort: Some(appam::llm::openrouter::ReasoningEffort::High),
                max_tokens: None, // Cannot combine with effort
                exclude: Some(false),
                summary: Some(appam::llm::openrouter::SummaryVerbosity::Detailed),
            },
        ))
        .with_tool(Arc::new(read_file()))
        .with_tool(Arc::new(write_file()))
        .with_tool(Arc::new(bash()))
        .with_tool(Arc::new(list_files()))
        .max_tokens(8192)
        .build()?;

    println!("✓ Reasoning: High effort");
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

                // Track if we've shown reasoning header
                let reasoning_shown = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let reasoning_shown_clone = Arc::clone(&reasoning_shown);

                // Stream agent response
                match agent
                    .stream(input)
                    .on_content(|content| {
                        print!("{}", content);
                        std::io::stdout().flush().ok();
                    })
                    .on_reasoning(move |content| {
                        if !reasoning_shown_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            println!("\n\n💭 Reasoning:\n");
                            reasoning_shown_clone.store(true, std::sync::atomic::Ordering::Relaxed);
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
