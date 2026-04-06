//! Default agent runtime and orchestration.
//!
//! Provides the standard conversation loop for agents: streaming LLM responses,
//! executing tool calls, and managing session state.

use anyhow::{Context, Result};
use futures::stream::{self, StreamExt};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use super::consumers::{ConsoleConsumer, TraceConsumer};
use super::errors::{extract_session_failure_kind, SessionFailureError, SessionFailureKind};
use super::history::SessionHistory;
use super::streaming::{MultiConsumer, StreamConsumer, StreamEvent};
use super::{Agent, Session};
use crate::llm::{
    ChatMessage, DynamicLlmClient, LlmClient, LlmProvider, Role, UnifiedMessage, UnifiedRole,
};
use crate::logging::write_session_log;
use crate::tools::{ToolConcurrency, ToolContext};

fn select_usage_model(cfg: &crate::config::AppConfig, provider: &LlmProvider) -> String {
    match provider {
        LlmProvider::Anthropic | LlmProvider::AzureAnthropic { .. } => cfg
            .anthropic
            .pricing_model
            .clone()
            .unwrap_or_else(|| cfg.anthropic.model.clone()),
        LlmProvider::OpenAI | LlmProvider::AzureOpenAI { .. } => cfg
            .openai
            .pricing_model
            .as_deref()
            .map(crate::llm::openai::normalize_openai_model)
            .unwrap_or_else(|| crate::llm::openai::normalize_openai_model(&cfg.openai.model)),
        LlmProvider::OpenAICodex => cfg
            .openai_codex
            .pricing_model
            .as_deref()
            .map(crate::llm::openai::normalize_openai_model)
            .unwrap_or_else(|| crate::llm::openai::normalize_openai_model(&cfg.openai_codex.model)),
        LlmProvider::Vertex => cfg.vertex.model.clone(),
        LlmProvider::OpenRouterCompletions | LlmProvider::OpenRouterResponses => {
            cfg.openrouter.model.clone()
        }
        LlmProvider::Bedrock { model_id, .. } => model_id.clone(),
    }
}

fn latest_provider_response_id(messages: &[ChatMessage]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        if message.role == Role::Assistant {
            message.provider_response_id.clone()
        } else {
            None
        }
    })
}

fn has_non_thinking_raw_content(blocks: &[crate::llm::UnifiedContentBlock]) -> bool {
    blocks
        .iter()
        .any(|block| !matches!(block, crate::llm::UnifiedContentBlock::Thinking { .. }))
}

fn blank_assistant_turn(
    streamed_text: &str,
    pending_tool_calls: &[crate::llm::ToolCall],
    raw_content_blocks: &[crate::llm::UnifiedContentBlock],
) -> bool {
    streamed_text.is_empty()
        && pending_tool_calls.is_empty()
        && !has_non_thinking_raw_content(raw_content_blocks)
}

/// Persist finalized tool calls for the current assistant turn.
///
/// Some streaming providers emit an additional empty finalized-tool-call batch
/// after the real tool calls have already been reported. Appam must ignore that
/// empty batch so it does not overwrite the assistant history entry with
/// `tool_calls: []`.
fn apply_finalized_tool_calls(
    pending_tool_calls: &mut Vec<crate::llm::ToolCall>,
    assistant_tool_msg: &mut Option<ChatMessage>,
    unified_calls: Vec<crate::llm::unified::UnifiedToolCall>,
) {
    if unified_calls.is_empty() {
        return;
    }

    let tool_calls: Vec<crate::llm::ToolCall> = unified_calls
        .iter()
        .map(|uc| crate::llm::ToolCall {
            id: uc.id.clone(),
            type_field: "function".to_string(),
            function: crate::llm::ToolCallFunction {
                name: uc.name.clone(),
                arguments: uc.raw_input_json.clone().unwrap_or_else(|| {
                    serde_json::to_string(&uc.input).unwrap_or_else(|_| uc.input.to_string())
                }),
            },
        })
        .collect();

    pending_tool_calls.extend(tool_calls.clone());
    *assistant_tool_msg = Some(ChatMessage {
        role: Role::Assistant,
        name: None,
        tool_call_id: None,
        content: None,
        tool_calls: Some(tool_calls),
        reasoning: None,
        raw_content_blocks: None,
        tool_metadata: None,
        timestamp: Some(chrono::Utc::now()),
        id: Some(ChatMessage::generate_id()),
        provider_response_id: None,
        status: Some(crate::llm::MessageStatus::Completed),
    });
}

fn called_required_completion_tool(messages: &[ChatMessage], required_tools: &[String]) -> bool {
    messages.iter().any(|msg| {
        msg.tool_calls.as_ref().is_some_and(|tool_calls| {
            tool_calls
                .iter()
                .any(|tool_call| required_tools.contains(&tool_call.function.name))
        })
    })
}

/// Runtime decision for agents that must call a completion tool before exiting.
///
/// Some agents, such as the init agent, are not allowed to finish with a plain
/// assistant response. They must eventually call a specific tool that persists
/// or finalizes their work. This enum captures the runtime action to take after
/// a turn ends without a successful completion-tool call.
///
/// # Design notes
///
/// The runtime keeps this decision explicit rather than encoding it in booleans
/// so call sites can distinguish between:
/// - a normal terminal state (`None`)
/// - a recoverable state that should inject another user continuation message
/// - an exhausted state that should raise a deterministic session failure
#[derive(Debug, Clone, PartialEq, Eq)]
enum RequiredCompletionContinuationDecision {
    /// Inject another continuation user message and let the session proceed.
    Inject {
        /// Exact continuation message that should be appended to the transcript.
        continuation_message: String,
        /// One-based attempt number for observability.
        continuation_attempt: usize,
        /// Required completion tools that remain outstanding.
        required_tools: Vec<String>,
    },
    /// Stop retrying because the continuation budget has been exhausted.
    Exhausted {
        /// Number of continuation messages already injected.
        continuation_count: usize,
        /// Required completion tools that were never called.
        required_tools: Vec<String>,
    },
    /// No continuation handling is needed.
    None,
}

/// Decide whether the runtime should auto-continue a session that requires a tool.
///
/// The runtime uses this helper after turns that end without tool calls and after
/// blank/reasoning-only turns. In both cases, the recovery strategy is identical:
/// if the agent still owes a required completion tool and continuation budget
/// remains, inject the configured continuation message instead of terminating the
/// session immediately.
///
/// # Parameters
///
/// - `agent`: Agent whose continuation policy should be enforced
/// - `messages`: Transcript accumulated so far, used to detect prior attempts
///
/// # Returns
///
/// A structured decision describing whether the runtime should inject another
/// continuation message, fail due to exhaustion, or do nothing.
///
/// # Edge cases
///
/// - Agents without required completion tools always return `None`
/// - If any required completion tool was already called, the helper returns `None`
/// - Exact-message matching is used intentionally so unrelated user messages do
///   not consume the continuation budget
fn required_completion_continuation_decision<A: Agent + ?Sized>(
    agent: &A,
    messages: &[ChatMessage],
) -> RequiredCompletionContinuationDecision {
    let Some(required_tools) = agent.required_completion_tools() else {
        return RequiredCompletionContinuationDecision::None;
    };

    if called_required_completion_tool(messages, required_tools) {
        return RequiredCompletionContinuationDecision::None;
    }

    let continuation_message = agent
        .continuation_message()
        .unwrap_or(
            "Continue your task. Please call one of the required completion tools to finish this session.",
        )
        .to_string();

    let continuation_count = messages
        .iter()
        .filter(|message| {
            message.role == Role::User
                && message
                    .content
                    .as_ref()
                    .map(|content| content == &continuation_message)
                    .unwrap_or(false)
        })
        .count();

    if continuation_count < agent.max_continuations() {
        RequiredCompletionContinuationDecision::Inject {
            continuation_message,
            continuation_attempt: continuation_count + 1,
            required_tools: required_tools.clone(),
        }
    } else {
        RequiredCompletionContinuationDecision::Exhausted {
            continuation_count,
            required_tools: required_tools.clone(),
        }
    }
}

/// Apply required-completion continuation handling for recoverable terminal turns.
///
/// This helper centralizes the runtime behavior for turns that ended before the
/// model called one of the required completion tools. That includes both:
/// - visible assistant turns that simply stopped without tool calls
/// - blank/reasoning-only turns where the provider produced no usable content
///
/// # Parameters
///
/// - `agent`: Agent whose completion policy is being enforced
/// - `messages`: Mutable transcript that may receive a continuation message
/// - `consumer`: Stream consumer used to emit structured error events on failure
/// - `client`: LLM client used to attach provider diagnostics to failures
/// - `completion_reason`: Short human-readable reason for logs
///
/// # Returns
///
/// - `Ok(true)` when a continuation message was appended and the caller should
///   immediately continue the main runtime loop
/// - `Ok(false)` when no continuation handling is required
/// - `Err(...)` when continuation attempts are exhausted and the session should fail
///
/// # Security considerations
///
/// The helper only injects the runtime-owned continuation message. It does not
/// reinterpret arbitrary model output as operator intent, which avoids accidental
/// privilege escalation through prompt-manipulated user messages.
fn handle_required_completion_gap<A: Agent + ?Sized>(
    agent: &A,
    messages: &mut Vec<ChatMessage>,
    consumer: &MultiConsumer,
    client: &DynamicLlmClient,
    completion_reason: &str,
) -> Result<bool> {
    match required_completion_continuation_decision(agent, messages) {
        RequiredCompletionContinuationDecision::Inject {
            continuation_message,
            continuation_attempt,
            required_tools,
        } => {
            info!(
                continuation_attempt = continuation_attempt,
                max_continuations = agent.max_continuations(),
                required_tools = ?required_tools,
                completion_reason = completion_reason,
                "Session continuation: injecting continuation message"
            );

            messages.push(ChatMessage {
                role: Role::User,
                name: None,
                tool_call_id: None,
                content: Some(continuation_message),
                tool_calls: None,
                reasoning: None,
                raw_content_blocks: None,
                tool_metadata: None,
                timestamp: Some(chrono::Utc::now()),
                id: None,
                provider_response_id: None,
                status: None,
            });
            Ok(true)
        }
        RequiredCompletionContinuationDecision::Exhausted {
            continuation_count,
            required_tools,
        } => {
            let error = anyhow::Error::new(SessionFailureError::new(
                SessionFailureKind::RequiredCompletionToolMissing,
                format!(
                    "Agent '{}' exhausted {} continuation attempts without calling required completion tools: {}",
                    agent.name(),
                    continuation_count,
                    required_tools.join(", ")
                ),
            ));
            emit_stream_error_event(consumer, client, &error);
            Err(error)
        }
        RequiredCompletionContinuationDecision::None => Ok(false),
    }
}

fn emit_stream_error_event(
    consumer: &MultiConsumer,
    client: &DynamicLlmClient,
    error: &anyhow::Error,
) {
    let provider_failure = client.take_last_failed_exchange();
    let event = StreamEvent::Error {
        message: error.to_string(),
        failure_kind: extract_session_failure_kind(error),
        provider: provider_failure
            .as_ref()
            .map(|capture| capture.provider.clone())
            .or_else(|| Some(client.provider_name().to_string())),
        model: provider_failure
            .as_ref()
            .map(|capture| capture.model.clone()),
        http_status: provider_failure
            .as_ref()
            .and_then(|capture| capture.http_status),
        request_payload: provider_failure
            .as_ref()
            .map(|capture| capture.request_payload.clone()),
        response_payload: provider_failure
            .as_ref()
            .map(|capture| capture.response_payload.clone()),
        provider_response_id: provider_failure
            .as_ref()
            .and_then(|capture| capture.provider_response_id.clone()),
    };

    if let Err(consumer_error) = consumer.on_event(&event) {
        warn!(
            error = %consumer_error,
            original_error = %error,
            "Failed to emit stream error event"
        );
    }
}

/// Apply per-agent parallel tool-call defaults to provider configuration.
///
/// Appam intentionally keeps provider-side tool batching disabled unless the
/// active agent explicitly enables it. Anthropic requires a slightly different
/// mapping because the parallel control lives inside `tool_choice`.
fn apply_parallel_tool_call_defaults<A: Agent + ?Sized>(
    agent: &A,
    cfg: &mut crate::config::AppConfig,
) {
    let parallel_tool_calls = agent.provider_parallel_tool_calls();
    cfg.openai.parallel_tool_calls = Some(parallel_tool_calls);
    cfg.openrouter.parallel_tool_calls = Some(parallel_tool_calls);
    cfg.openai_codex.parallel_tool_calls = Some(parallel_tool_calls);

    if cfg.anthropic.tool_choice.is_none() {
        cfg.anthropic.tool_choice = Some(crate::llm::anthropic::ToolChoiceConfig::Auto {
            disable_parallel_tool_use: !parallel_tool_calls,
        });
    }
}

#[derive(Debug)]
struct PreparedToolCall {
    call: crate::llm::ToolCall,
    args: serde_json::Value,
}

#[derive(Debug)]
struct ToolExecutionOutcome {
    call: crate::llm::ToolCall,
    result_json: serde_json::Value,
    success: bool,
    duration_ms: f64,
}

fn prepare_tool_calls(
    pending_tool_calls: &[crate::llm::ToolCall],
    multi_consumer: &MultiConsumer,
    emitted_tool_calls: &Arc<Mutex<HashSet<String>>>,
) -> Result<Vec<PreparedToolCall>> {
    let mut prepared = Vec::with_capacity(pending_tool_calls.len());

    for call in pending_tool_calls {
        info!(tool = %call.function.name, "Executing tool");

        let mut should_emit_start = false;
        let args_snapshot = call.function.arguments.clone();

        if !args_snapshot.trim().is_empty() {
            let mut seen = emitted_tool_calls
                .lock()
                .expect("tool call tracker poisoned");
            should_emit_start = seen.insert(call.id.clone());
        }

        if should_emit_start {
            multi_consumer.on_event(&StreamEvent::ToolCallStarted {
                tool_name: call.function.name.clone(),
                arguments: args_snapshot.clone(),
            })?;
        }

        debug!(
            tool = %call.function.name,
            args_len = call.function.arguments.len(),
            "Parsing tool arguments"
        );
        let args = serde_json::from_str(&call.function.arguments).with_context(|| {
            format!(
                "Failed to parse arguments for tool {} ({} bytes)",
                call.function.name,
                call.function.arguments.len()
            )
        })?;

        prepared.push(PreparedToolCall {
            call: call.clone(),
            args,
        });
    }

    Ok(prepared)
}

async fn run_tool_call<A: Agent + ?Sized>(
    agent: &A,
    session_id: &str,
    prepared: PreparedToolCall,
) -> ToolExecutionOutcome {
    let start_time = std::time::Instant::now();
    let ctx = ToolContext::new(
        session_id.to_string(),
        agent.name().to_string(),
        prepared.call.id.clone(),
    );
    let result = agent
        .execute_tool_with_context(&prepared.call.function.name, ctx, prepared.args)
        .await;
    let elapsed = start_time.elapsed();
    let duration_ms = elapsed.as_secs_f64() * 1000.0;

    match result {
        Ok(value) => ToolExecutionOutcome {
            call: prepared.call,
            result_json: value,
            success: true,
            duration_ms,
        },
        Err(error) => ToolExecutionOutcome {
            call: prepared.call,
            result_json: serde_json::json!({
                "success": false,
                "error": error.to_string()
            }),
            success: false,
            duration_ms,
        },
    }
}

async fn execute_pending_tool_calls<A: Agent + ?Sized>(
    agent: &A,
    session_id: &str,
    pending_tool_calls: &[crate::llm::ToolCall],
    messages: &mut Vec<ChatMessage>,
    multi_consumer: &MultiConsumer,
    emitted_tool_calls: &Arc<Mutex<HashSet<String>>>,
) -> Result<bool> {
    let required_completion_tools = agent.required_completion_tools().cloned();
    let prepared_calls =
        prepare_tool_calls(pending_tool_calls, multi_consumer, emitted_tool_calls)?;

    let should_run_parallel = agent.provider_parallel_tool_calls()
        && agent.max_concurrent_tool_executions() > 1
        && prepared_calls.len() > 1
        && prepared_calls.iter().all(|prepared| {
            agent.tool_concurrency(&prepared.call.function.name) == ToolConcurrency::ParallelSafe
        });

    let outcomes = if should_run_parallel {
        let max_concurrency = agent.max_concurrent_tool_executions();
        let mut indexed: Vec<(usize, ToolExecutionOutcome)> =
            stream::iter(prepared_calls.into_iter().enumerate().map(
                |(index, prepared)| async move {
                    (index, run_tool_call(agent, session_id, prepared).await)
                },
            ))
            .buffer_unordered(max_concurrency)
            .collect()
            .await;
        indexed.sort_by_key(|(index, _)| *index);
        indexed.into_iter().map(|(_, outcome)| outcome).collect()
    } else {
        let mut outcomes = Vec::with_capacity(prepared_calls.len());
        for prepared in prepared_calls {
            outcomes.push(run_tool_call(agent, session_id, prepared).await);
        }
        outcomes
    };

    let mut completed_via_required_tool = false;

    for outcome in outcomes {
        if outcome.success {
            info!(tool = %outcome.call.function.name, "Tool succeeded");
            multi_consumer.on_event(&StreamEvent::ToolCallCompleted {
                tool_name: outcome.call.function.name.clone(),
                result: outcome.result_json.clone(),
                success: true,
                duration_ms: outcome.duration_ms,
            })?;
        } else {
            let error_message = outcome.result_json["error"]
                .as_str()
                .unwrap_or("Tool execution failed")
                .to_string();
            info!(
                tool = %outcome.call.function.name,
                error = %error_message,
                "Tool failed"
            );
            multi_consumer.on_event(&StreamEvent::ToolCallFailed {
                tool_name: outcome.call.function.name.clone(),
                error: error_message,
            })?;
        }

        if outcome.success
            && required_completion_tools
                .as_ref()
                .is_some_and(|tools| tools.contains(&outcome.call.function.name))
        {
            completed_via_required_tool = true;
        }

        let content_str = match &outcome.result_json {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        messages.push(ChatMessage {
            role: Role::Tool,
            name: Some(outcome.call.function.name.clone()),
            tool_call_id: Some(outcome.call.id.clone()),
            content: Some(content_str),
            tool_calls: None,
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: Some(crate::llm::ToolExecutionMetadata {
                success: outcome.success,
                duration_ms: outcome.duration_ms,
                tool_name: outcome.call.function.name.clone(),
                arguments: outcome.call.function.arguments.clone(),
            }),
            timestamp: Some(chrono::Utc::now()),
            id: None,
            provider_response_id: None,
            status: None,
        });
    }

    Ok(completed_via_required_tool)
}

/// Default agent run implementation with console output.
///
/// Orchestrates a multi-turn conversation with tool calling and streams output
/// to the console with default formatting.
///
/// This is a convenience wrapper around `default_run_streaming()` that uses
/// `ConsoleConsumer` with default settings.
///
/// # Errors
///
/// Returns an error if:
/// - Configuration loading fails
/// - LLM request fails
/// - Tool execution fails
/// - Session logging fails
#[instrument(skip(agent), fields(agent = agent.name()))]
pub async fn default_run<A: Agent + ?Sized>(agent: &A, user_prompt: &str) -> Result<Session> {
    let consumer = ConsoleConsumer::default();
    default_run_streaming(agent, user_prompt, Box::new(consumer)).await
}

/// Default agent run implementation with custom streaming.
///
/// Orchestrates a multi-turn conversation with tool calling:
/// 1. Loads global configuration and initializes logging
/// 2. Creates a session and builds initial messages
/// 3. Streams LLM response with tool support via consumer
/// 4. Executes requested tools and appends results
/// 5. Repeats until LLM stops requesting tools
/// 6. Persists session transcript
///
/// # Tool Calling Loop
///
/// The agent continues streaming and executing tools until the LLM emits a
/// finish_reason other than "tool_calls". This enables multi-step reasoning
/// where the agent can call multiple tools sequentially.
///
/// # Parameters
///
/// - `agent`: The agent to run
/// - `user_prompt`: User's input message
/// - `consumer`: Stream consumer that receives events
///
/// # Errors
///
/// Returns an error if:
/// - Configuration loading fails
/// - LLM request fails
/// - Tool execution fails
/// - Consumer returns an error
/// - Session logging fails
#[instrument(skip(agent, consumer), fields(agent = agent.name()))]
pub async fn default_run_streaming<A: Agent + ?Sized>(
    agent: &A,
    user_prompt: &str,
    consumer: Box<dyn StreamConsumer>,
) -> Result<Session> {
    info!("Starting agent session");

    // Load configuration from defaults and environment variables only
    // (does NOT load appam.toml automatically - user must load explicitly if needed)
    let mut cfg = crate::config::load_config_from_env()?;

    // Override provider if agent specifies one
    if let Some(provider) = agent.provider() {
        info!(provider = %provider, "Using agent-specific provider override");
        cfg.provider = provider;
    }

    // Apply agent-specific configuration overrides (programmatic config has highest priority)
    agent.apply_config_overrides(&mut cfg);
    apply_parallel_tool_call_defaults(agent, &mut cfg);

    let logs_dir = crate::logging::init_logging(&cfg.logging)?;

    // Initialize session history
    let history = SessionHistory::new(cfg.history.clone()).await?;

    // Create session
    let session_id = Uuid::new_v4().to_string();
    let started_at = chrono::Utc::now();

    // Setup multi-consumer with trace consumer (if enabled)
    let multi_consumer = if cfg.logging.enable_traces {
        let trace_consumer = TraceConsumer::new(&logs_dir, &session_id, cfg.logging.trace_format)?;
        MultiConsumer::new()
            .add(Box::new(trace_consumer))
            .add(consumer)
    } else {
        MultiConsumer::new().add(consumer)
    };

    // Emit session started event
    multi_consumer.on_event(&StreamEvent::SessionStarted {
        session_id: session_id.clone(),
    })?;

    // Build initial messages
    let mut messages = agent.initial_messages(user_prompt)?;
    let tool_specs = agent.available_tools()?;

    debug!(
        session_id = %session_id,
        tools = tool_specs.len(),
        "Session initialized"
    );

    // Create LLM client based on configured provider
    let client = DynamicLlmClient::from_config(&cfg)?;
    client.set_previous_response_id(latest_provider_response_id(&messages));
    info!(provider = %client.provider(), "LLM client initialized");

    // Create usage tracker for this session
    let usage_tracker = crate::llm::usage::UsageTracker::new();
    let provider_variant = client.provider();
    let provider_usage_key = provider_variant.pricing_key().to_string();
    let model_name = select_usage_model(&cfg, &provider_variant);

    let emitted_tool_calls = Arc::new(Mutex::new(HashSet::<String>::new()));

    // Multi-round tool-calling loop
    loop {
        let mut streamed_text = String::new();
        let mut reasoning_text = String::new();
        let mut pending_tool_calls: Vec<crate::llm::ToolCall> = Vec::new();
        let mut assistant_tool_msg: Option<ChatMessage> = None;
        let mut raw_content_blocks: Vec<crate::llm::UnifiedContentBlock> = Vec::new();

        debug!("Starting LLM stream");

        let emitted_tool_calls_for_partial = Arc::clone(&emitted_tool_calls);

        // Convert to unified format for provider-agnostic client
        let unified_messages = chat_messages_to_unified(&messages);
        let unified_tools = tool_specs_to_unified(&tool_specs);

        // Stream response
        if let Err(error) = client
            .chat_with_tools_streaming(
                &unified_messages,
                &unified_tools,
                |chunk| {
                    // Content callback - stream via consumer
                    streamed_text.push_str(chunk);
                    multi_consumer.on_event(&StreamEvent::Content {
                        content: chunk.to_string(),
                    })?;
                    Ok(())
                },
                |unified_calls| {
                    apply_finalized_tool_calls(
                        &mut pending_tool_calls,
                        &mut assistant_tool_msg,
                        unified_calls,
                    );
                    Ok(())
                },
                |reason| {
                    // Reasoning callback - stream via consumer
                    reasoning_text.push_str(reason);
                    multi_consumer.on_event(&StreamEvent::Reasoning {
                        content: reason.to_string(),
                    })?;
                    Ok(())
                },
                |partial_calls| {
                    // Partial tool calls callback - emit tool call started events
                    for uc in partial_calls {
                        if !uc.name.is_empty() {
                            // Check if arguments are complete valid JSON
                            let args_str = uc.raw_input_json.clone().unwrap_or_else(|| {
                                serde_json::to_string(&uc.input)
                                    .unwrap_or_else(|_| uc.input.to_string())
                            });

                            if serde_json::from_str::<serde_json::Value>(&args_str).is_ok() {
                                let should_emit = {
                                    let mut seen = emitted_tool_calls_for_partial
                                        .lock()
                                        .expect("tool call tracker poisoned");
                                    seen.insert(uc.id.clone())
                                };

                                if should_emit {
                                    multi_consumer.on_event(&StreamEvent::ToolCallStarted {
                                        tool_name: uc.name.clone(),
                                        arguments: args_str,
                                    })?;
                                }
                            }
                        }
                    }
                    Ok(())
                },
                |content_block| {
                    // Complete content block callback - preserves thinking signatures
                    raw_content_blocks.push(content_block);
                    Ok(())
                },
                {
                    let tracker = usage_tracker.clone();
                    let provider = provider_usage_key.clone();
                    let model = model_name.clone();
                    let consumer = &multi_consumer;
                    move |usage| {
                        // Usage callback - track tokens and cost
                        tracker.add_usage(&usage, &provider, &model);
                        let snapshot = tracker.get_snapshot();
                        consumer.on_event(&StreamEvent::UsageUpdate { snapshot })?;
                        Ok(())
                    }
                },
            )
            .await
        {
            emit_stream_error_event(&multi_consumer, &client, &error);
            return Err(error);
        }

        let provider_response_id = client.latest_response_id();

        // Derive tool calls from raw content blocks when the streaming API
        // doesn't emit explicit tool call events (e.g., newer Responses formats).
        if pending_tool_calls.is_empty() {
            let derived_tool_calls: Vec<crate::llm::ToolCall> = raw_content_blocks
                .iter()
                .filter_map(|block| {
                    if let crate::llm::UnifiedContentBlock::ToolUse { id, name, input } = block {
                        match serde_json::to_string(input) {
                            Ok(arguments) => Some(crate::llm::ToolCall {
                                id: id.clone(),
                                type_field: "function".to_string(),
                                function: crate::llm::ToolCallFunction {
                                    name: name.clone(),
                                    arguments,
                                },
                            }),
                            Err(e) => {
                                debug!(tool = %name, error = %e, "Failed to serialize tool input");
                                None
                            }
                        }
                    } else {
                        None
                    }
                })
                .collect();

            if !derived_tool_calls.is_empty() {
                for call in &derived_tool_calls {
                    debug!(tool = %call.function.name, call_id = %call.id, "Derived tool call from raw content");
                }
                pending_tool_calls = derived_tool_calls.clone();

                if let Some(msg) = assistant_tool_msg.as_mut() {
                    msg.tool_calls = Some(derived_tool_calls.clone());
                } else {
                    assistant_tool_msg = Some(ChatMessage {
                        role: Role::Assistant,
                        name: None,
                        tool_call_id: None,
                        content: None,
                        tool_calls: Some(derived_tool_calls.clone()),
                        reasoning: None,
                        raw_content_blocks: None,
                        tool_metadata: None,
                        timestamp: Some(chrono::Utc::now()),
                        id: Some(ChatMessage::generate_id()),
                        provider_response_id: None,
                        status: Some(crate::llm::MessageStatus::Completed),
                    });
                }
            }
        }

        let has_tool_calls = !pending_tool_calls.is_empty();
        let has_streamed_text = !streamed_text.is_empty();
        let has_raw_blocks = !raw_content_blocks.is_empty();

        if has_tool_calls {
            // Preserve legacy behaviour: only add a separate assistant text message
            // when the model streamed visible content without raw blocks. When raw
            // blocks are present, they will be attached to the tool call message to
            // avoid duplicate messages.
            if has_streamed_text && !has_raw_blocks {
                messages.push(ChatMessage {
                    role: Role::Assistant,
                    name: None,
                    tool_call_id: None,
                    content: Some(streamed_text.clone()),
                    tool_calls: None,
                    reasoning: if reasoning_text.is_empty() {
                        None
                    } else {
                        Some(reasoning_text.clone())
                    },
                    raw_content_blocks: None,
                    tool_metadata: None,
                    timestamp: Some(chrono::Utc::now()),
                    id: Some(ChatMessage::generate_id()),
                    provider_response_id: provider_response_id.clone(),
                    status: Some(crate::llm::MessageStatus::Completed),
                });
            }
        } else if has_streamed_text || has_raw_blocks {
            // Final assistant turn (no tool calls). Capture whichever payload the
            // model produced: streamed text, raw content blocks, and reasoning.
            messages.push(ChatMessage {
                role: Role::Assistant,
                name: None,
                tool_call_id: None,
                content: if has_streamed_text {
                    Some(streamed_text.clone())
                } else {
                    None
                },
                tool_calls: None,
                reasoning: if reasoning_text.is_empty() {
                    None
                } else {
                    Some(reasoning_text.clone())
                },
                raw_content_blocks: if has_raw_blocks {
                    Some(raw_content_blocks.clone())
                } else {
                    None
                },
                tool_metadata: None,
                timestamp: Some(chrono::Utc::now()),
                id: Some(ChatMessage::generate_id()),
                provider_response_id: provider_response_id.clone(),
                status: Some(crate::llm::MessageStatus::Completed),
            });
        }

        if provider_response_id.is_some() {
            client.set_previous_response_id(provider_response_id.clone());
        }

        if blank_assistant_turn(&streamed_text, &pending_tool_calls, &raw_content_blocks) {
            if pending_tool_calls.is_empty()
                && handle_required_completion_gap(
                    agent,
                    &mut messages,
                    &multi_consumer,
                    &client,
                    "blank assistant turn",
                )?
            {
                continue;
            }

            let error = anyhow::Error::new(SessionFailureError::new(
                SessionFailureKind::BlankAssistantResponse,
                format!(
                    "Agent '{}' completed a turn without assistant text, tool calls, or non-thinking content",
                    agent.name()
                ),
            ));
            emit_stream_error_event(&multi_consumer, &client, &error);
            return Err(error);
        }

        // If no tool calls, check if continuation is needed
        if pending_tool_calls.is_empty() {
            debug!("No tool calls requested, checking completion status");

            if handle_required_completion_gap(
                agent,
                &mut messages,
                &multi_consumer,
                &client,
                "turn ended without tool calls",
            )? {
                continue;
            }

            debug!("Session complete");
            break;
        }

        debug!(
            tool_calls = pending_tool_calls.len(),
            "Executing tool calls"
        );

        // Add tool call message to history with reasoning and raw blocks attached
        if let Some(mut msg) = assistant_tool_msg.take() {
            // Attach any reasoning that was collected during streaming
            if !reasoning_text.is_empty() {
                msg.reasoning = Some(reasoning_text.clone());
            }
            // Attach raw content blocks to the tool call message (preserves thinking signatures)
            if !raw_content_blocks.is_empty() {
                msg.raw_content_blocks = Some(raw_content_blocks.clone());
            }
            msg.provider_response_id = provider_response_id.clone();
            messages.push(msg);
        }

        let completed_via_required_tool = execute_pending_tool_calls(
            agent,
            &session_id,
            &pending_tool_calls,
            &mut messages,
            &multi_consumer,
            &emitted_tool_calls,
        )
        .await?;

        if completed_via_required_tool {
            info!(
                agent = agent.name(),
                "Required completion tool executed successfully; ending session without additional continuation turn"
            );
            break;
        }

        // Loop continues: with tool results, prompt model again
    }

    let ended_at = chrono::Utc::now();

    // Create session object with usage tracking
    let session = Session {
        session_id: session_id.clone(),
        agent_name: agent.name().to_string(),
        model: model_name.clone(),
        messages: messages.clone(),
        started_at: Some(started_at),
        ended_at: Some(ended_at),
        usage: Some(usage_tracker.get_snapshot()),
    };

    // Save to history (if enabled and auto_save is true)
    if cfg.history.enabled && cfg.history.auto_save {
        history.save_session(&session).await?;
        info!(session_id = %session_id, "Session saved to history database");
    }

    // Write session log (legacy JSON format) - only if tracing enabled
    if cfg.logging.enable_traces {
        let log_path = write_session_log(&logs_dir, &session_id, &session)?;
        info!(path = %log_path.display(), "Session log written");
    }

    // Emit done event
    multi_consumer.on_event(&StreamEvent::Done)?;

    Ok(session)
}

/// Default agent run implementation with pre-built messages and custom streaming.
///
/// Like `default_run_streaming()`, but accepts a pre-built message list instead
/// of constructing messages from a user prompt. This is useful for multi-turn
/// conversations where the caller has already built the message history.
///
/// # Parameters
///
/// - `agent`: The agent to run
/// - `messages`: Pre-built message list (typically system + history + new user message)
/// - `consumer`: Stream consumer that receives events
///
/// # Errors
///
/// Returns an error if:
/// - Configuration loading fails
/// - LLM request fails
/// - Tool execution fails
/// - Consumer returns an error
/// - Session logging fails
#[instrument(skip(agent, messages, consumer), fields(agent = agent.name()))]
pub async fn default_run_streaming_with_messages<A: Agent + ?Sized>(
    agent: &A,
    messages: Vec<ChatMessage>,
    consumer: Box<dyn StreamConsumer>,
) -> Result<Session> {
    info!(
        message_count = messages.len(),
        "Starting agent session with pre-built messages"
    );

    // Load configuration from defaults and environment variables only
    let mut cfg = crate::config::load_config_from_env()?;

    // Override provider if agent specifies one
    if let Some(provider) = agent.provider() {
        info!(provider = %provider, "Using agent-specific provider override");
        cfg.provider = provider;
    }

    // Apply agent-specific configuration overrides
    agent.apply_config_overrides(&mut cfg);
    apply_parallel_tool_call_defaults(agent, &mut cfg);

    let logs_dir = crate::logging::init_logging(&cfg.logging)?;

    // Initialize session history
    let history = SessionHistory::new(cfg.history.clone()).await?;

    // Create session
    let session_id = Uuid::new_v4().to_string();
    let started_at = chrono::Utc::now();

    // Setup multi-consumer with trace consumer (if enabled)
    let multi_consumer = if cfg.logging.enable_traces {
        let trace_consumer = TraceConsumer::new(&logs_dir, &session_id, cfg.logging.trace_format)?;
        MultiConsumer::new()
            .add(Box::new(trace_consumer))
            .add(consumer)
    } else {
        MultiConsumer::new().add(consumer)
    };

    // Emit session started event
    multi_consumer.on_event(&StreamEvent::SessionStarted {
        session_id: session_id.clone(),
    })?;

    // Use pre-built messages instead of agent.initial_messages()
    let mut messages = messages;
    let tool_specs = agent.available_tools()?;

    debug!(
        session_id = %session_id,
        messages = messages.len(),
        tools = tool_specs.len(),
        "Session initialized with pre-built messages"
    );

    // Create LLM client based on configured provider
    let client = DynamicLlmClient::from_config(&cfg)?;
    client.set_previous_response_id(latest_provider_response_id(&messages));
    info!(provider = %client.provider(), "LLM client initialized");

    // Create usage tracker for this session
    let usage_tracker = crate::llm::usage::UsageTracker::new();
    let provider_variant = client.provider();
    let provider_usage_key = provider_variant.pricing_key().to_string();
    let model_name = select_usage_model(&cfg, &provider_variant);

    let emitted_tool_calls = Arc::new(Mutex::new(HashSet::<String>::new()));

    // Multi-round tool-calling loop
    loop {
        let mut streamed_text = String::new();
        let mut reasoning_text = String::new();
        let mut pending_tool_calls: Vec<crate::llm::ToolCall> = Vec::new();
        let mut assistant_tool_msg: Option<ChatMessage> = None;
        let mut raw_content_blocks: Vec<crate::llm::UnifiedContentBlock> = Vec::new();

        debug!("Starting LLM stream");

        let emitted_tool_calls_for_partial = Arc::clone(&emitted_tool_calls);

        // Convert to unified format for provider-agnostic client
        let unified_messages = chat_messages_to_unified(&messages);
        let unified_tools = tool_specs_to_unified(&tool_specs);

        // Stream response
        if let Err(error) = client
            .chat_with_tools_streaming(
                &unified_messages,
                &unified_tools,
                |chunk| {
                    streamed_text.push_str(chunk);
                    multi_consumer.on_event(&StreamEvent::Content {
                        content: chunk.to_string(),
                    })?;
                    Ok(())
                },
                |unified_calls| {
                    apply_finalized_tool_calls(
                        &mut pending_tool_calls,
                        &mut assistant_tool_msg,
                        unified_calls,
                    );
                    Ok(())
                },
                |reason| {
                    reasoning_text.push_str(reason);
                    multi_consumer.on_event(&StreamEvent::Reasoning {
                        content: reason.to_string(),
                    })?;
                    Ok(())
                },
                |partial_calls| {
                    for uc in partial_calls {
                        if !uc.name.is_empty() {
                            let args_str = uc.raw_input_json.clone().unwrap_or_else(|| {
                                serde_json::to_string(&uc.input)
                                    .unwrap_or_else(|_| uc.input.to_string())
                            });

                            if serde_json::from_str::<serde_json::Value>(&args_str).is_ok() {
                                let should_emit = {
                                    let mut seen = emitted_tool_calls_for_partial
                                        .lock()
                                        .expect("tool call tracker poisoned");
                                    seen.insert(uc.id.clone())
                                };

                                if should_emit {
                                    multi_consumer.on_event(&StreamEvent::ToolCallStarted {
                                        tool_name: uc.name.clone(),
                                        arguments: args_str,
                                    })?;
                                }
                            }
                        }
                    }
                    Ok(())
                },
                |content_block| {
                    raw_content_blocks.push(content_block);
                    Ok(())
                },
                {
                    let tracker = usage_tracker.clone();
                    let provider = provider_usage_key.clone();
                    let model = model_name.clone();
                    let consumer = &multi_consumer;
                    move |usage| {
                        tracker.add_usage(&usage, &provider, &model);
                        let snapshot = tracker.get_snapshot();
                        consumer.on_event(&StreamEvent::UsageUpdate { snapshot })?;
                        Ok(())
                    }
                },
            )
            .await
        {
            emit_stream_error_event(&multi_consumer, &client, &error);
            return Err(error);
        }

        let provider_response_id = client.latest_response_id();

        // Derive tool calls from raw content blocks when needed
        if pending_tool_calls.is_empty() {
            let derived_tool_calls: Vec<crate::llm::ToolCall> = raw_content_blocks
                .iter()
                .filter_map(|block| {
                    if let crate::llm::UnifiedContentBlock::ToolUse { id, name, input } = block {
                        match serde_json::to_string(input) {
                            Ok(arguments) => Some(crate::llm::ToolCall {
                                id: id.clone(),
                                type_field: "function".to_string(),
                                function: crate::llm::ToolCallFunction {
                                    name: name.clone(),
                                    arguments,
                                },
                            }),
                            Err(e) => {
                                debug!(tool = %name, error = %e, "Failed to serialize tool input");
                                None
                            }
                        }
                    } else {
                        None
                    }
                })
                .collect();

            if !derived_tool_calls.is_empty() {
                for call in &derived_tool_calls {
                    debug!(tool = %call.function.name, call_id = %call.id, "Derived tool call from raw content");
                }
                pending_tool_calls = derived_tool_calls.clone();

                if let Some(msg) = assistant_tool_msg.as_mut() {
                    msg.tool_calls = Some(derived_tool_calls.clone());
                } else {
                    assistant_tool_msg = Some(ChatMessage {
                        role: Role::Assistant,
                        name: None,
                        tool_call_id: None,
                        content: None,
                        tool_calls: Some(derived_tool_calls.clone()),
                        reasoning: None,
                        raw_content_blocks: None,
                        tool_metadata: None,
                        timestamp: Some(chrono::Utc::now()),
                        id: Some(ChatMessage::generate_id()),
                        provider_response_id: None,
                        status: Some(crate::llm::MessageStatus::Completed),
                    });
                }
            }
        }

        let has_tool_calls = !pending_tool_calls.is_empty();
        let has_streamed_text = !streamed_text.is_empty();
        let has_raw_blocks = !raw_content_blocks.is_empty();

        if has_tool_calls {
            if has_streamed_text && !has_raw_blocks {
                messages.push(ChatMessage {
                    role: Role::Assistant,
                    name: None,
                    tool_call_id: None,
                    content: Some(streamed_text.clone()),
                    tool_calls: None,
                    reasoning: if reasoning_text.is_empty() {
                        None
                    } else {
                        Some(reasoning_text.clone())
                    },
                    raw_content_blocks: None,
                    tool_metadata: None,
                    timestamp: Some(chrono::Utc::now()),
                    id: Some(ChatMessage::generate_id()),
                    provider_response_id: provider_response_id.clone(),
                    status: Some(crate::llm::MessageStatus::Completed),
                });
            }
        } else if has_streamed_text || has_raw_blocks {
            messages.push(ChatMessage {
                role: Role::Assistant,
                name: None,
                tool_call_id: None,
                content: if has_streamed_text {
                    Some(streamed_text.clone())
                } else {
                    None
                },
                tool_calls: None,
                reasoning: if reasoning_text.is_empty() {
                    None
                } else {
                    Some(reasoning_text.clone())
                },
                raw_content_blocks: if has_raw_blocks {
                    Some(raw_content_blocks.clone())
                } else {
                    None
                },
                tool_metadata: None,
                timestamp: Some(chrono::Utc::now()),
                id: Some(ChatMessage::generate_id()),
                provider_response_id: provider_response_id.clone(),
                status: Some(crate::llm::MessageStatus::Completed),
            });
        }

        if provider_response_id.is_some() {
            client.set_previous_response_id(provider_response_id.clone());
        }

        if blank_assistant_turn(&streamed_text, &pending_tool_calls, &raw_content_blocks) {
            if pending_tool_calls.is_empty()
                && handle_required_completion_gap(
                    agent,
                    &mut messages,
                    &multi_consumer,
                    &client,
                    "blank assistant turn",
                )?
            {
                continue;
            }

            let error = anyhow::Error::new(SessionFailureError::new(
                SessionFailureKind::BlankAssistantResponse,
                format!(
                    "Agent '{}' completed a turn without assistant text, tool calls, or non-thinking content",
                    agent.name()
                ),
            ));
            emit_stream_error_event(&multi_consumer, &client, &error);
            return Err(error);
        }

        // If no tool calls, check if continuation is needed
        if pending_tool_calls.is_empty() {
            debug!("No tool calls requested, checking completion status");

            if handle_required_completion_gap(
                agent,
                &mut messages,
                &multi_consumer,
                &client,
                "turn ended without tool calls",
            )? {
                continue;
            }

            debug!("Session complete");
            break;
        }

        debug!(
            tool_calls = pending_tool_calls.len(),
            "Executing tool calls"
        );

        // Add tool call message to history with reasoning and raw blocks attached
        if let Some(mut msg) = assistant_tool_msg.take() {
            if !reasoning_text.is_empty() {
                msg.reasoning = Some(reasoning_text.clone());
            }
            if !raw_content_blocks.is_empty() {
                msg.raw_content_blocks = Some(raw_content_blocks.clone());
            }
            msg.provider_response_id = provider_response_id.clone();
            messages.push(msg);
        }

        let completed_via_required_tool = execute_pending_tool_calls(
            agent,
            &session_id,
            &pending_tool_calls,
            &mut messages,
            &multi_consumer,
            &emitted_tool_calls,
        )
        .await?;

        if completed_via_required_tool {
            info!(
                agent = agent.name(),
                "Required completion tool executed successfully; ending session without additional continuation turn"
            );
            break;
        }
    }

    let ended_at = chrono::Utc::now();

    // Create session object with usage tracking
    let session = Session {
        session_id: session_id.clone(),
        agent_name: agent.name().to_string(),
        model: model_name.clone(),
        messages: messages.clone(),
        started_at: Some(started_at),
        ended_at: Some(ended_at),
        usage: Some(usage_tracker.get_snapshot()),
    };

    // Save to history (if enabled and auto_save is true)
    if cfg.history.enabled && cfg.history.auto_save {
        history.save_session(&session).await?;
        info!(session_id = %session_id, "Session saved to history database");
    }

    // Write session log (legacy JSON format) - only if tracing enabled
    if cfg.logging.enable_traces {
        let log_path = write_session_log(&logs_dir, &session_id, &session)?;
        info!(path = %log_path.display(), "Session log written");
    }

    // Emit done event
    multi_consumer.on_event(&StreamEvent::Done)?;

    Ok(session)
}

/// Continue an existing session with a new user prompt.
///
/// Loads the session from history and continues the conversation with
/// the new prompt, preserving all previous messages and context.
///
/// # Parameters
///
/// - `agent`: The agent to run
/// - `session_id`: ID of the session to continue
/// - `user_prompt`: New user input
///
/// # Errors
///
/// Returns an error if:
/// - Session history is not enabled
/// - Session ID does not exist
/// - LLM request fails
/// - Tool execution fails
#[instrument(skip(agent), fields(agent = agent.name(), session_id = %session_id))]
pub async fn continue_session_run<A: Agent + ?Sized>(
    agent: &A,
    session_id: &str,
    user_prompt: &str,
) -> Result<Session> {
    let consumer = ConsoleConsumer::default();
    continue_session_streaming(agent, session_id, user_prompt, Box::new(consumer)).await
}

/// Continue an existing session with custom streaming.
///
/// Like `continue_session_run()`, but with a custom stream consumer.
///
/// # Parameters
///
/// - `agent`: The agent to run
/// - `session_id`: ID of the session to continue
/// - `user_prompt`: New user input
/// - `consumer`: Stream consumer for events
///
/// # Errors
///
/// Returns an error if:
/// - Session history is not enabled
/// - Session ID does not exist
/// - LLM request fails
/// - Tool execution fails
/// - Consumer returns an error
#[instrument(skip(agent, consumer), fields(agent = agent.name(), session_id = %session_id))]
pub async fn continue_session_streaming<A: Agent + ?Sized>(
    agent: &A,
    session_id: &str,
    user_prompt: &str,
    consumer: Box<dyn StreamConsumer>,
) -> Result<Session> {
    info!("Continuing existing session");

    // Load configuration from defaults and environment variables only
    // (does NOT load appam.toml automatically - user must load explicitly if needed)
    let mut cfg = crate::config::load_config_from_env()?;

    // Override provider if agent specifies one
    if let Some(provider) = agent.provider() {
        info!(provider = %provider, "Using agent-specific provider override for continuation");
        cfg.provider = provider;
    }

    // Apply agent-specific configuration overrides (programmatic config has highest priority)
    agent.apply_config_overrides(&mut cfg);
    apply_parallel_tool_call_defaults(agent, &mut cfg);

    let logs_dir = crate::logging::init_logging(&cfg.logging)?;

    // Initialize session history
    let history = SessionHistory::new(cfg.history.clone()).await?;

    // Load existing session
    let mut session = history
        .load_session(session_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

    info!(
        session_id = %session_id,
        existing_messages = session.messages.len(),
        "Continuing session with existing messages"
    );

    // Setup multi-consumer with trace consumer (if enabled)
    let multi_consumer = if cfg.logging.enable_traces {
        let trace_consumer = TraceConsumer::new(&logs_dir, session_id, cfg.logging.trace_format)?;
        MultiConsumer::new()
            .add(Box::new(trace_consumer))
            .add(consumer)
    } else {
        MultiConsumer::new().add(consumer)
    };

    // Emit session started event (continuation)
    multi_consumer.on_event(&StreamEvent::SessionStarted {
        session_id: session_id.to_string(),
    })?;

    // Add new user message to existing messages
    session.messages.push(ChatMessage {
        role: Role::User,
        name: None,
        tool_call_id: None,
        content: Some(user_prompt.to_string()),
        tool_calls: None,
        reasoning: None,
        raw_content_blocks: None,
        tool_metadata: None,
        timestamp: Some(chrono::Utc::now()),
        id: None,
        provider_response_id: None,
        status: None,
    });

    let mut messages = session.messages.clone();
    let tool_specs = agent.available_tools()?;

    debug!(
        session_id = %session_id,
        total_messages = messages.len(),
        tools = tool_specs.len(),
        "Session continuation initialized"
    );

    // Create LLM client based on configured provider
    let client = DynamicLlmClient::from_config(&cfg)?;
    client.set_previous_response_id(latest_provider_response_id(&messages));
    info!(provider = %client.provider(), "LLM client initialized for session continuation");

    // Create or restore usage tracker for this session
    let usage_tracker = session
        .usage
        .clone()
        .map(|u| {
            let tracker = crate::llm::usage::UsageTracker::new();
            tracker.inner.lock().unwrap().clone_from(&u);
            tracker
        })
        .unwrap_or_else(crate::llm::usage::UsageTracker::new);
    let provider_variant = client.provider();
    let provider_usage_key = provider_variant.pricing_key().to_string();
    let model_name = select_usage_model(&cfg, &provider_variant);

    // Ensure the session metadata reflects the active provider/model pair.
    session.model = model_name.clone();

    let emitted_tool_calls = Arc::new(Mutex::new(HashSet::<String>::new()));

    // Multi-round tool-calling loop (same as default_run_streaming)
    loop {
        let mut streamed_text = String::new();
        let mut reasoning_text = String::new();
        let mut pending_tool_calls: Vec<crate::llm::ToolCall> = Vec::new();
        let mut assistant_tool_msg: Option<ChatMessage> = None;
        let mut raw_content_blocks: Vec<crate::llm::UnifiedContentBlock> = Vec::new();

        debug!("Starting LLM stream");

        let emitted_tool_calls_for_partial = Arc::clone(&emitted_tool_calls);

        // Convert to unified format for provider-agnostic client
        let unified_messages = chat_messages_to_unified(&messages);
        let unified_tools = tool_specs_to_unified(&tool_specs);

        // Stream response
        if let Err(error) = client
            .chat_with_tools_streaming(
                &unified_messages,
                &unified_tools,
                |chunk| {
                    streamed_text.push_str(chunk);
                    multi_consumer.on_event(&StreamEvent::Content {
                        content: chunk.to_string(),
                    })?;
                    Ok(())
                },
                |unified_calls| {
                    apply_finalized_tool_calls(
                        &mut pending_tool_calls,
                        &mut assistant_tool_msg,
                        unified_calls,
                    );
                    Ok(())
                },
                |reason| {
                    reasoning_text.push_str(reason);
                    multi_consumer.on_event(&StreamEvent::Reasoning {
                        content: reason.to_string(),
                    })?;
                    Ok(())
                },
                |partial_calls| {
                    // Partial tool calls callback - emit tool call started events
                    for uc in partial_calls {
                        if !uc.name.is_empty() {
                            let args_str = uc.raw_input_json.clone().unwrap_or_else(|| {
                                serde_json::to_string(&uc.input)
                                    .unwrap_or_else(|_| uc.input.to_string())
                            });
                            if serde_json::from_str::<serde_json::Value>(&args_str).is_ok() {
                                let should_emit = {
                                    let mut seen = emitted_tool_calls_for_partial
                                        .lock()
                                        .expect("tool call tracker poisoned");
                                    seen.insert(uc.id.clone())
                                };

                                if should_emit {
                                    multi_consumer.on_event(&StreamEvent::ToolCallStarted {
                                        tool_name: uc.name.clone(),
                                        arguments: args_str,
                                    })?;
                                }
                            }
                        }
                    }
                    Ok(())
                },
                |content_block| {
                    // Complete content block callback - preserves thinking signatures
                    raw_content_blocks.push(content_block);
                    Ok(())
                },
                {
                    let tracker = usage_tracker.clone();
                    let provider = provider_usage_key.clone();
                    let model = model_name.clone();
                    let consumer = &multi_consumer;
                    move |usage| {
                        // Usage callback - track tokens and cost
                        tracker.add_usage(&usage, &provider, &model);
                        let snapshot = tracker.get_snapshot();
                        consumer.on_event(&StreamEvent::UsageUpdate { snapshot })?;
                        Ok(())
                    }
                },
            )
            .await
        {
            emit_stream_error_event(&multi_consumer, &client, &error);
            return Err(error);
        }

        let provider_response_id = client.latest_response_id();

        if pending_tool_calls.is_empty() {
            let derived_tool_calls: Vec<crate::llm::ToolCall> = raw_content_blocks
                .iter()
                .filter_map(|block| {
                    if let crate::llm::UnifiedContentBlock::ToolUse { id, name, input } = block {
                        match serde_json::to_string(input) {
                            Ok(arguments) => Some(crate::llm::ToolCall {
                                id: id.clone(),
                                type_field: "function".to_string(),
                                function: crate::llm::ToolCallFunction {
                                    name: name.clone(),
                                    arguments,
                                },
                            }),
                            Err(e) => {
                                debug!(tool = %name, error = %e, "Failed to serialize tool input");
                                None
                            }
                        }
                    } else {
                        None
                    }
                })
                .collect();

            if !derived_tool_calls.is_empty() {
                for call in &derived_tool_calls {
                    debug!(tool = %call.function.name, call_id = %call.id, "Derived tool call from raw content");
                }
                pending_tool_calls = derived_tool_calls.clone();

                if let Some(msg) = assistant_tool_msg.as_mut() {
                    msg.tool_calls = Some(derived_tool_calls.clone());
                } else {
                    assistant_tool_msg = Some(ChatMessage {
                        role: Role::Assistant,
                        name: None,
                        tool_call_id: None,
                        content: None,
                        tool_calls: Some(derived_tool_calls.clone()),
                        reasoning: None,
                        raw_content_blocks: None,
                        tool_metadata: None,
                        timestamp: Some(chrono::Utc::now()),
                        id: Some(ChatMessage::generate_id()),
                        provider_response_id: None,
                        status: Some(crate::llm::MessageStatus::Completed),
                    });
                }
            }
        }

        let has_tool_calls = !pending_tool_calls.is_empty();
        let has_streamed_text = !streamed_text.is_empty();
        let has_raw_blocks = !raw_content_blocks.is_empty();

        if has_tool_calls {
            if has_streamed_text && !has_raw_blocks {
                messages.push(ChatMessage {
                    role: Role::Assistant,
                    name: None,
                    tool_call_id: None,
                    content: Some(streamed_text.clone()),
                    tool_calls: None,
                    reasoning: if reasoning_text.is_empty() {
                        None
                    } else {
                        Some(reasoning_text.clone())
                    },
                    raw_content_blocks: None,
                    tool_metadata: None,
                    timestamp: Some(chrono::Utc::now()),
                    id: Some(ChatMessage::generate_id()),
                    provider_response_id: provider_response_id.clone(),
                    status: Some(crate::llm::MessageStatus::Completed),
                });
            }
        } else if has_streamed_text || has_raw_blocks {
            messages.push(ChatMessage {
                role: Role::Assistant,
                name: None,
                tool_call_id: None,
                content: if has_streamed_text {
                    Some(streamed_text.clone())
                } else {
                    None
                },
                tool_calls: None,
                reasoning: if reasoning_text.is_empty() {
                    None
                } else {
                    Some(reasoning_text.clone())
                },
                raw_content_blocks: if has_raw_blocks {
                    Some(raw_content_blocks.clone())
                } else {
                    None
                },
                tool_metadata: None,
                timestamp: Some(chrono::Utc::now()),
                id: Some(ChatMessage::generate_id()),
                provider_response_id: provider_response_id.clone(),
                status: Some(crate::llm::MessageStatus::Completed),
            });
        }

        if provider_response_id.is_some() {
            client.set_previous_response_id(provider_response_id.clone());
        }

        if blank_assistant_turn(&streamed_text, &pending_tool_calls, &raw_content_blocks) {
            if pending_tool_calls.is_empty()
                && handle_required_completion_gap(
                    agent,
                    &mut messages,
                    &multi_consumer,
                    &client,
                    "blank assistant turn",
                )?
            {
                continue;
            }

            let error = anyhow::Error::new(SessionFailureError::new(
                SessionFailureKind::BlankAssistantResponse,
                format!(
                    "Agent '{}' completed a turn without assistant text, tool calls, or non-thinking content",
                    agent.name()
                ),
            ));
            emit_stream_error_event(&multi_consumer, &client, &error);
            return Err(error);
        }

        // If no tool calls, check if continuation is needed
        if pending_tool_calls.is_empty() {
            debug!("No tool calls requested, checking completion status");

            if handle_required_completion_gap(
                agent,
                &mut messages,
                &multi_consumer,
                &client,
                "turn ended without tool calls",
            )? {
                continue;
            }

            debug!("Session complete");
            break;
        }

        debug!(
            tool_calls = pending_tool_calls.len(),
            "Executing tool calls"
        );

        // Add tool call message to history with reasoning attached
        if let Some(mut msg) = assistant_tool_msg.take() {
            if !reasoning_text.is_empty() {
                msg.reasoning = Some(reasoning_text.clone());
            }
            if !raw_content_blocks.is_empty() {
                msg.raw_content_blocks = Some(raw_content_blocks.clone());
            }
            msg.provider_response_id = provider_response_id.clone();
            messages.push(msg);
        }

        let completed_via_required_tool = execute_pending_tool_calls(
            agent,
            session_id,
            &pending_tool_calls,
            &mut messages,
            &multi_consumer,
            &emitted_tool_calls,
        )
        .await?;

        if completed_via_required_tool {
            info!(
                agent = agent.name(),
                "Required completion tool executed successfully; ending session without additional continuation turn"
            );
            break;
        }

        // Loop continues: with tool results, prompt model again
    }

    let ended_at = chrono::Utc::now();

    // Update session object with usage tracking
    session.messages = messages;
    session.ended_at = Some(ended_at);
    session.usage = Some(usage_tracker.get_snapshot());

    // Save to history (if enabled and auto_save is true)
    if cfg.history.enabled && cfg.history.auto_save {
        history.save_session(&session).await?;
        info!(session_id = %session_id, "Session updated in history database");
    }

    // Write session log (legacy JSON format) - only if tracing enabled
    if cfg.logging.enable_traces {
        let log_path = write_session_log(&logs_dir, session_id, &session)?;
        info!(path = %log_path.display(), "Session log written");
    }

    // Emit done event
    multi_consumer.on_event(&StreamEvent::Done)?;

    Ok(session)
}

/// Convert ChatMessage to UnifiedMessage for provider-agnostic client calls.
///
/// Maps the legacy ChatMessage format to the unified format that works
/// with both OpenRouter and Anthropic providers.
///
/// # Signature Preservation
///
/// When `raw_content_blocks` is present, it's used directly to preserve:
/// - Thinking block signatures (required for Anthropic tool use)
/// - Redacted thinking blocks
/// - Multimodal content (images, documents)
/// - Exact block ordering from API responses
///
/// Otherwise, content is reconstructed from legacy fields (backward compatibility).
fn chat_messages_to_unified(messages: &[ChatMessage]) -> Vec<UnifiedMessage> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::System => UnifiedRole::System,
                Role::User => UnifiedRole::User,
                Role::Assistant => UnifiedRole::Assistant,
                Role::Developer => UnifiedRole::System, // Map developer to system
                Role::Tool => UnifiedRole::User,        // Tool results as user messages
            };

            // FAST PATH: Use raw blocks if available (preserves thinking signatures!)
            // This takes precedence over all legacy fields to avoid duplication
            if let Some(ref raw_blocks) = msg.raw_content_blocks {
                // When raw blocks are present, use them exclusively
                // This preserves exact API response structure with signatures
                return UnifiedMessage {
                    role,
                    content: raw_blocks.clone(),
                    id: msg.id.clone(),
                    timestamp: msg.timestamp,
                    reasoning: msg.reasoning.clone(),
                    reasoning_details: None, // TODO: Add if needed
                };
            }

            // LEGACY PATH: Reconstruct from individual fields (backward compatibility)
            let mut content_blocks = Vec::new();

            // Add text content if present
            if let Some(ref text) = msg.content {
                if !text.trim().is_empty() {
                    content_blocks
                        .push(crate::llm::UnifiedContentBlock::Text { text: text.clone() });
                }
            }

            // Add tool calls if present
            if let Some(ref tool_calls) = msg.tool_calls {
                for tc in tool_calls {
                    let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::json!({}));

                    content_blocks.push(crate::llm::UnifiedContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        input,
                    });
                }
            }

            // Add reasoning if present (legacy - no signature)
            if let Some(ref reasoning) = msg.reasoning {
                content_blocks.push(crate::llm::UnifiedContentBlock::Thinking {
                    thinking: reasoning.clone(),
                    signature: None,
                    encrypted_content: None,
                    redacted: false,
                });
            }

            // Handle tool result messages
            if msg.role == Role::Tool {
                if let (Some(ref content), Some(ref tool_call_id)) =
                    (&msg.content, &msg.tool_call_id)
                {
                    content_blocks = vec![crate::llm::UnifiedContentBlock::ToolResult {
                        tool_use_id: tool_call_id.clone(),
                        content: serde_json::json!(content),
                        is_error: Some(false),
                    }];
                }
            }

            UnifiedMessage {
                role,
                content: content_blocks,
                id: msg.id.clone(),
                timestamp: msg.timestamp,
                reasoning: msg.reasoning.clone(),
                reasoning_details: None, // TODO: Add if needed
            }
        })
        .collect()
}

/// Convert ToolSpec to UnifiedTool.
fn tool_specs_to_unified(specs: &[crate::llm::ToolSpec]) -> Vec<crate::llm::UnifiedTool> {
    specs
        .iter()
        .map(|spec| crate::llm::UnifiedTool {
            name: spec.name.clone(),
            description: spec.description.clone(),
            parameters: spec.parameters.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        apply_finalized_tool_calls, execute_pending_tool_calls,
        required_completion_continuation_decision, select_usage_model,
        RequiredCompletionContinuationDecision,
    };
    use crate::agent::AgentBuilder;
    use crate::config::AppConfig;
    use crate::llm::{ChatMessage, Role, ToolCall, ToolCallFunction, ToolSpec};
    use crate::tools::{AsyncTool, Tool, ToolConcurrency, ToolContext};
    use anyhow::Result;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// Minimal tool used to exercise completion-tool continuation logic.
    ///
    /// The tests only need the tool's schema and name because continuation
    /// decisions are derived from transcript state, not tool execution.
    struct TestCompletionTool {
        name: String,
    }

    impl TestCompletionTool {
        fn new(name: impl Into<String>) -> Self {
            Self { name: name.into() }
        }
    }

    impl Tool for TestCompletionTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn spec(&self) -> Result<ToolSpec> {
            Ok(serde_json::from_value(json!({
                "type": "function",
                "name": self.name,
                "description": "Test completion tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                }
            }))?)
        }

        fn execute(&self, _args: serde_json::Value) -> Result<serde_json::Value> {
            Ok(json!({"success": true}))
        }
    }

    fn user_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            name: None,
            tool_call_id: None,
            content: Some(content.to_string()),
            tool_calls: None,
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: None,
            timestamp: None,
            id: None,
            provider_response_id: None,
            status: None,
        }
    }

    #[test]
    fn apply_finalized_tool_calls_ignores_empty_batches() {
        let mut pending = Vec::new();
        let mut assistant_tool_msg = None;

        apply_finalized_tool_calls(&mut pending, &mut assistant_tool_msg, Vec::new());

        assert!(pending.is_empty());
        assert!(assistant_tool_msg.is_none());
    }

    #[test]
    fn apply_finalized_tool_calls_preserves_non_empty_batches() {
        let mut pending = Vec::new();
        let mut assistant_tool_msg = None;

        apply_finalized_tool_calls(
            &mut pending,
            &mut assistant_tool_msg,
            vec![crate::llm::unified::UnifiedToolCall {
                id: "call_1".to_string(),
                name: "bash".to_string(),
                input: json!({"command": "mkdir poem_generator"}),
                raw_input_json: Some("{\"command\":\"mkdir poem_generator\"}".to_string()),
            }],
        );

        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].function.name, "bash");
        let stored = assistant_tool_msg.expect("assistant tool message should be stored");
        assert_eq!(stored.tool_calls.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn select_usage_model_prefers_openai_pricing_model_override() {
        let mut cfg = AppConfig::default();
        cfg.openai.model = "gpt-5.4-fast".to_string();
        cfg.openai.pricing_model = Some("gpt-5.4".to_string());

        let selected = select_usage_model(&cfg, &crate::llm::LlmProvider::OpenAI);
        assert_eq!(selected, "gpt-5.4");
    }

    #[test]
    fn select_usage_model_prefers_anthropic_pricing_model_override() {
        let mut cfg = AppConfig::default();
        cfg.anthropic.model = "claude-4-6-opus".to_string();
        cfg.anthropic.pricing_model = Some("claude-opus-4-6".to_string());

        let selected = select_usage_model(
            &cfg,
            &crate::llm::LlmProvider::AzureAnthropic {
                base_url: "https://example.services.ai.azure.com/anthropic".to_string(),
                auth_method: crate::llm::anthropic::AzureAnthropicAuthMethod::BearerToken,
            },
        );
        assert_eq!(selected, "claude-opus-4-6");
    }

    #[test]
    fn required_completion_continuation_injects_before_budget_is_exhausted() {
        let completion_tool = Arc::new(TestCompletionTool::new("store_custom_prompt"));
        let agent = AgentBuilder::new("init-agent-test")
            .system_prompt("Test")
            .require_completion_tools(vec![completion_tool as Arc<dyn Tool>])
            .continuation_message("Call the completion tool now.")
            .max_continuations(2)
            .build()
            .expect("agent should build");

        let messages = vec![user_message("Analyze the codebase.")];

        let decision = required_completion_continuation_decision(&agent, &messages);
        assert_eq!(
            decision,
            RequiredCompletionContinuationDecision::Inject {
                continuation_message: "Call the completion tool now.".to_string(),
                continuation_attempt: 1,
                required_tools: vec!["store_custom_prompt".to_string()],
            }
        );
    }

    #[test]
    fn required_completion_continuation_exhausts_at_configured_limit() {
        let completion_tool = Arc::new(TestCompletionTool::new("store_custom_prompt"));
        let agent = AgentBuilder::new("init-agent-test")
            .system_prompt("Test")
            .require_completion_tools(vec![completion_tool as Arc<dyn Tool>])
            .continuation_message("Call the completion tool now.")
            .max_continuations(2)
            .build()
            .expect("agent should build");

        let messages = vec![
            user_message("Analyze the codebase."),
            user_message("Call the completion tool now."),
            user_message("Call the completion tool now."),
        ];

        let decision = required_completion_continuation_decision(&agent, &messages);
        assert_eq!(
            decision,
            RequiredCompletionContinuationDecision::Exhausted {
                continuation_count: 2,
                required_tools: vec!["store_custom_prompt".to_string()],
            }
        );
    }

    #[test]
    fn required_completion_continuation_stops_after_required_tool_call() {
        let completion_tool = Arc::new(TestCompletionTool::new("store_custom_prompt"));
        let agent = AgentBuilder::new("init-agent-test")
            .system_prompt("Test")
            .require_completion_tools(vec![completion_tool as Arc<dyn Tool>])
            .continuation_message("Call the completion tool now.")
            .max_continuations(2)
            .build()
            .expect("agent should build");

        let messages = vec![ChatMessage {
            role: Role::Assistant,
            name: None,
            tool_call_id: None,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                type_field: "function".to_string(),
                function: ToolCallFunction {
                    name: "store_custom_prompt".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            reasoning: None,
            raw_content_blocks: None,
            tool_metadata: None,
            timestamp: None,
            id: None,
            provider_response_id: None,
            status: None,
        }];

        let decision = required_completion_continuation_decision(&agent, &messages);
        assert_eq!(decision, RequiredCompletionContinuationDecision::None);
    }

    struct SleepTool {
        name: &'static str,
        concurrency: ToolConcurrency,
    }

    #[async_trait]
    impl AsyncTool for SleepTool {
        fn name(&self) -> &str {
            self.name
        }

        fn spec(&self) -> Result<ToolSpec> {
            Ok(serde_json::from_value(json!({
                "type": "function",
                "name": self.name,
                "description": "Sleep tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "delay_ms": {
                            "type": "number",
                            "description": "Delay in milliseconds"
                        }
                    },
                    "required": ["delay_ms"]
                }
            }))?)
        }

        fn concurrency(&self) -> ToolConcurrency {
            self.concurrency
        }

        async fn execute(
            &self,
            _ctx: ToolContext,
            args: serde_json::Value,
        ) -> Result<serde_json::Value> {
            let delay_ms = args["delay_ms"].as_u64().unwrap_or(0);
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            Ok(json!({
                "tool": self.name,
                "delay_ms": delay_ms
            }))
        }
    }

    fn tool_call(name: &str, id: &str, delay_ms: u64) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            type_field: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: json!({ "delay_ms": delay_ms }).to_string(),
            },
        }
    }

    #[tokio::test]
    async fn execute_pending_tool_calls_runs_parallel_safe_batches_concurrently() {
        let agent = AgentBuilder::new("parallel-agent")
            .system_prompt("test")
            .with_async_tools(vec![
                Arc::new(SleepTool {
                    name: "sleep_a",
                    concurrency: ToolConcurrency::ParallelSafe,
                }) as Arc<dyn AsyncTool>,
                Arc::new(SleepTool {
                    name: "sleep_b",
                    concurrency: ToolConcurrency::ParallelSafe,
                }) as Arc<dyn AsyncTool>,
            ])
            .enable_parallel_tool_calls(4)
            .build()
            .unwrap();

        let mut messages = Vec::new();
        let pending = vec![
            tool_call("sleep_a", "call-1", 80),
            tool_call("sleep_b", "call-2", 80),
        ];
        let consumer = crate::agent::streaming::MultiConsumer::new();
        let emitted = Arc::new(Mutex::new(std::collections::HashSet::new()));

        let started = Instant::now();
        let completed = execute_pending_tool_calls(
            &agent,
            "session-parallel",
            &pending,
            &mut messages,
            &consumer,
            &emitted,
        )
        .await
        .unwrap();
        let elapsed = started.elapsed();

        assert!(!completed);
        assert!(elapsed < Duration::from_millis(140));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call-2"));
    }

    #[tokio::test]
    async fn execute_pending_tool_calls_serializes_mixed_batches() {
        let agent = AgentBuilder::new("serial-agent")
            .system_prompt("test")
            .with_async_tools(vec![
                Arc::new(SleepTool {
                    name: "sleep_parallel",
                    concurrency: ToolConcurrency::ParallelSafe,
                }) as Arc<dyn AsyncTool>,
                Arc::new(SleepTool {
                    name: "sleep_serial",
                    concurrency: ToolConcurrency::SerialOnly,
                }) as Arc<dyn AsyncTool>,
            ])
            .enable_parallel_tool_calls(4)
            .build()
            .unwrap();

        let mut messages = Vec::new();
        let pending = vec![
            tool_call("sleep_parallel", "call-1", 70),
            tool_call("sleep_serial", "call-2", 70),
        ];
        let consumer = crate::agent::streaming::MultiConsumer::new();
        let emitted = Arc::new(Mutex::new(std::collections::HashSet::new()));

        let started = Instant::now();
        execute_pending_tool_calls(
            &agent,
            "session-serial",
            &pending,
            &mut messages,
            &consumer,
            &emitted,
        )
        .await
        .unwrap();
        let elapsed = started.elapsed();

        assert!(elapsed >= Duration::from_millis(130));
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call-2"));
    }
}
