//! OpenRouter Responses API client with streaming support.
//!
//! Provides a client for the OpenRouter Responses API (Beta) that supports
//! streaming Server-Sent Events (SSE) responses with tool calling, enhanced
//! reasoning, and structured output items.
//!
//! Key differences from Chat Completions API:
//! - Endpoint: `/api/v1/responses` instead of `/api/v1/chat/completions`
//! - Input: Structured message arrays with typed content
//! - Output: Array of typed items (messages, reasoning, function_calls)
//! - Streaming: Event-based with granular delta types
//! - Reasoning: Separate output items with configurable effort levels
//! - Full usage tracking with detailed token counts and costs

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use tracing::{debug, error, info, warn};

// Use shared config and types
use super::config::OpenRouterConfig;
use super::types::ToolSpec;
use crate::llm::{ChatMessage, InputItem, Role, ToolCall, ToolCallFunction};

// All configuration types now in super::config module

/// OpenRouter Responses API streaming client.
///
/// Handles authentication, request construction, and SSE parsing for
/// responses with tool calling and reasoning support.
#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    http: reqwest::Client,
    cfg: OpenRouterConfig,
}

impl OpenRouterClient {
    /// Create a new OpenRouter Responses API client.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be constructed.
    pub fn new(cfg: OpenRouterConfig) -> Result<Self> {
        let http = crate::http::client_pool::get_or_init_client(&cfg.base_url, |ctx| {
            // Configure HTTP client with optimizations:
            // - http2_prior_knowledge: Use HTTP/2 for multiplexing and better performance
            // - connect_timeout: 30s to prevent hanging on connection establishment
            // - pool_idle_timeout: 120s to clean up stale connections
            // - pool_max_idle_per_host: Limit to 10 idle connections per host
            // - tcp_keepalive: 60s to detect dead connections
            // - tcp_nodelay: Disable Nagle's algorithm for lower latency
            // - gzip: Enable compression to reduce bandwidth
            // - resolve_to_addrs: Cache DNS lookups to prevent flaky DNS errors
            // - NO read timeout: Allow infinite active connection time for streaming responses
            let mut builder = reqwest::Client::builder()
                .http2_prior_knowledge()
                .connect_timeout(std::time::Duration::from_secs(30))
                .pool_idle_timeout(std::time::Duration::from_secs(120))
                .pool_max_idle_per_host(10)
                .tcp_keepalive(std::time::Duration::from_secs(60))
                .tcp_nodelay(true)
                .gzip(true)
                .user_agent("appam/0.1.0");

            if let Some(addrs) = ctx.resolved_addrs() {
                builder = builder.resolve_to_addrs(ctx.host(), addrs);
            }

            builder
                .build()
                .context("Failed to create OpenRouter HTTP client")
        })?;

        Ok(Self { http, cfg })
    }

    /// Build HTTP headers for OpenRouter requests.
    ///
    /// Includes authentication, content-type, and optional attribution headers.
    /// Never logs the authorization header.
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is missing or cannot be formatted.
    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));

        let key = self
            .cfg
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow!("Missing OpenRouter API key; set via env or config"))?;

        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", key)).context("Invalid API key header")?,
        );

        if let Some(ref referer) = self.cfg.http_referer {
            headers.insert("HTTP-Referer", HeaderValue::from_str(referer)?);
        }

        if let Some(ref title) = self.cfg.x_title {
            headers.insert("X-Title", HeaderValue::from_str(title)?);
        }

        Ok(headers)
    }

    /// Stream a response with tool calling support using the Responses API (internal).
    ///
    /// Internal implementation using legacy ChatMessage format.
    /// External callers should use the `LlmClient` trait implementation.
    ///
    /// Makes a streaming request to OpenRouter Responses API and invokes callbacks for:
    /// - Content chunks (streamed assistant response)
    /// - Tool call requests (when model decides to use tools)
    /// - Reasoning tokens (extended thinking, if enabled)
    /// - Partial tool calls (incremental argument accumulation)
    ///
    /// # Parameters
    ///
    /// - `messages`: Conversation history (converted to API format internally)
    /// - `tools`: Available tool specifications
    /// - `on_content`: Callback for each content chunk
    /// - `on_tool_calls`: Callback when tool calls are finalized
    /// - `on_reasoning`: Callback for reasoning tokens
    /// - `on_tool_calls_partial`: Callback for incremental tool call updates
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails, authentication is invalid, or
    /// the response cannot be parsed.
    #[allow(clippy::too_many_arguments)]
    async fn chat_with_tools_streaming_internal<FContent, FTool, FReason, FToolPartial, FUsage>(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
        on_content: FContent,
        on_tool_calls: FTool,
        on_reasoning: FReason,
        on_tool_calls_partial: FToolPartial,
        on_usage: FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<ToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[ToolCall]) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        // Convert internal messages to API input format
        let input_items = self.messages_to_input(messages);

        // Build request body with usage tracking enabled
        let mut body = serde_json::json!({
            "model": self.cfg.model,
            "input": input_items,
            "stream": true,
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "usage": {"include": true},
        });

        // Add tools if provided
        if !tools.is_empty() {
            body["tools"] = serde_json::json!(tools);
        }

        // Add reasoning config if enabled
        if let Some(ref reasoning) = self.cfg.reasoning {
            body["reasoning"] = serde_json::to_value(reasoning)?;
        }

        // Add optional parameters
        if let Some(max_tokens) = self.cfg.max_output_tokens {
            body["max_output_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(temp) = self.cfg.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = self.cfg.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }

        info!(target: "openrouter", "Sending Responses API request");
        debug!(target: "openrouter", "Request body: {}", serde_json::to_string_pretty(&body)?);

        let res = self
            .http
            .post(format!("{}/responses", self.cfg.base_url))
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await
            .context("OpenRouter Responses API request failed")?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
            error!(status=?status, body=%text, "OpenRouter error response");
            return Err(anyhow!("OpenRouter error ({}): {}", status, text));
        }

        // Debug instrumentation
        let debug_enabled = std::env::var("OPENROUTER_DEBUG")
            .ok()
            .map(|v| {
                let l = v.to_lowercase();
                !(l == "0" || l == "false" || l.is_empty())
            })
            .unwrap_or(false);

        if debug_enabled {
            eprintln!("[openrouter] status: {}", res.status());
            eprintln!("[openrouter] headers:");
            for (k, v) in res.headers().iter() {
                if k.as_str().eq_ignore_ascii_case("authorization") {
                    continue;
                }
                eprintln!("  {}: {}", k, v.to_str().unwrap_or("<bin>"));
            }
        }

        // Parse SSE stream
        self.parse_stream(
            res,
            debug_enabled,
            on_content,
            on_tool_calls,
            on_reasoning,
            on_tool_calls_partial,
            on_usage,
        )
        .await
    }

    /// Parse SSE event stream from Responses API.
    ///
    /// The Responses API uses a different event structure than Chat Completions:
    /// - Events are typed with a `type` field
    /// - Content comes in deltas per output item
    /// - Reasoning is a separate output item type
    /// - Function calls are separate items (not embedded in messages)
    /// - Usage data comes in `response.done` or `response.completed` events
    ///
    /// # Error Recovery
    ///
    /// If the stream is interrupted by a recoverable error (EOF, connection reset, etc.),
    /// this method logs a warning and returns successfully with partial content rather than
    /// failing completely. This allows the caller to process whatever content was received.
    #[allow(clippy::too_many_arguments)]
    async fn parse_stream<FContent, FTool, FReason, FToolPartial, FUsage>(
        &self,
        res: reqwest::Response,
        debug_enabled: bool,
        mut on_content: FContent,
        mut on_tool_calls: FTool,
        mut on_reasoning: FReason,
        mut on_tool_calls_partial: FToolPartial,
        mut on_usage: FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<ToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[ToolCall]) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        let mut stream = res.bytes_stream();
        let mut text_buf = String::new();
        let mut event_data_lines: Vec<String> = Vec::new();

        // Track output items by index
        let mut output_items: Vec<StreamOutputItem> = Vec::new();
        let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
        let mut done = false;

        // Track bytes received for error reporting
        let mut total_bytes_received: usize = 0;
        let mut events_processed: usize = 0;

        while let Some(chunk) = stream.next().await {
            // Handle chunk reading with graceful error recovery
            let bytes = match chunk.context("Failed to read stream chunk") {
                Ok(b) => b,
                Err(e) => {
                    // Check if this is a recoverable network error during chunk reading
                    if Self::is_chunk_error_recoverable(&e) {
                        warn!(
                            target: "openrouter",
                            bytes_received = total_bytes_received,
                            events_processed = events_processed,
                            error = %e,
                            "Stream interrupted by recoverable error, returning partial response"
                        );
                        // Return success with partial content - the caller already received
                        // content via callbacks before the interruption
                        return Ok(());
                    }
                    // Non-recoverable error - propagate immediately
                    return Err(e);
                }
            };
            total_bytes_received += bytes.len();
            let part = String::from_utf8_lossy(&bytes);
            text_buf.push_str(&part);

            while let Some(pos) = text_buf.find('\n') {
                let mut line = text_buf[..pos].to_string();
                text_buf = text_buf[pos + 1..].to_string();

                // Trim trailing CR
                if line.ends_with('\r') {
                    line.pop();
                }

                if debug_enabled {
                    eprintln!("[openrouter::sse] {}", line);
                }

                if line.is_empty() {
                    // End of event; process accumulated data
                    if !event_data_lines.is_empty() {
                        let data_payload = event_data_lines.join("\n");
                        event_data_lines.clear();

                        if data_payload == "[DONE]" {
                            done = true;
                            break;
                        }

                        // Parse event JSON
                        match self.handle_event(
                            &data_payload,
                            &mut output_items,
                            &mut pending_tool_calls,
                            &mut on_content,
                            &mut on_tool_calls,
                            &mut on_reasoning,
                            &mut on_tool_calls_partial,
                            &mut on_usage,
                            debug_enabled,
                        ) {
                            Ok(()) => {
                                events_processed += 1;
                            }
                            Err(e) => {
                                if debug_enabled {
                                    eprintln!("[openrouter::error] Event parse error: {}", e);
                                    eprintln!("[openrouter::error] Event data: {}", data_payload);
                                }
                                error!(target: "openrouter", "Event parse error: {}", e);
                            }
                        }
                    }
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    event_data_lines.push(rest.to_string());
                } else if line.starts_with(':') {
                    // Comment/keepalive; ignore
                }
            }

            if done {
                break;
            }
        }

        Ok(())
    }

    /// Handle a single streaming event from the Responses API.
    ///
    /// Event types:
    /// - `response.created`: Response started
    /// - `response.output_item.added`: New output item (message, reasoning, function_call)
    /// - `response.content_part.added`: New content part in current item
    /// - `response.content_part.delta`: Incremental content update
    /// - `response.reasoning.delta`: Reasoning token
    /// - `response.function_call_arguments.delta`: Function arguments chunk
    /// - `response.function_call_arguments.done`: Function call complete
    /// - `response.output_item.done`: Output item complete
    /// - `response.done`: Response complete
    #[allow(clippy::too_many_arguments)]
    fn handle_event<FContent, FTool, FReason, FToolPartial, FUsage>(
        &self,
        data: &str,
        output_items: &mut Vec<StreamOutputItem>,
        pending_tool_calls: &mut Vec<ToolCall>,
        on_content: &mut FContent,
        on_tool_calls: &mut FTool,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
        on_usage: &mut FUsage,
        debug_enabled: bool,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()>,
        FTool: FnMut(Vec<ToolCall>) -> Result<()>,
        FReason: FnMut(&str) -> Result<()>,
        FToolPartial: FnMut(&[ToolCall]) -> Result<()>,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()>,
    {
        let event: StreamEvent =
            serde_json::from_str(data).context("Failed to parse stream event")?;

        if debug_enabled {
            eprintln!("[openrouter::event] type={}", event.event_type);
        }

        match event.event_type.as_str() {
            "response.created" => {
                debug!(target: "openrouter", "Response created");
            }
            "response.output_item.added" => {
                // New output item started
                if let Some(item) = event.item {
                    let index = event.output_index.unwrap_or(output_items.len());
                    while output_items.len() <= index {
                        output_items.push(StreamOutputItem::default());
                    }
                    output_items[index].item_type = item.item_type.clone();
                    output_items[index].id = item.id.clone();

                    if let Some(name) = item.name.clone() {
                        output_items[index].function_name = Some(name.clone());

                        // If this is a function call, emit ToolCallStarted event
                        if item.item_type.as_deref() == Some("function_call") {
                            if let Some(call_id) = item.call_id.clone() {
                                output_items[index].call_id = Some(call_id.clone());

                                // Emit partial tool call started (arguments will come later)
                                let partial = ToolCall {
                                    id: call_id,
                                    type_field: "function".to_string(),
                                    function: ToolCallFunction {
                                        name,
                                        arguments: String::new(),
                                    },
                                };
                                on_tool_calls_partial(&[partial])?;
                            }
                        }
                    } else if let Some(call_id) = item.call_id {
                        output_items[index].call_id = Some(call_id);
                    }
                }
            }
            "response.content_part.added" => {
                // New content part in current output item
                debug!(target: "openrouter", "Content part added");
            }
            "response.content_part.delta" | "response.output_text.delta" => {
                // Content delta (API sends both formats)
                if let Some(delta) = event.delta {
                    on_content(&delta)?;
                }
            }
            "response.reasoning.delta"
            | "response.reasoning_text.delta"
            | "response.reasoning_summary_text.delta" => {
                // Reasoning token (multiple formats: Claude, GPT-5, etc.)
                if let Some(delta) = event.delta {
                    on_reasoning(&delta)?;

                    // Mark that we've emitted reasoning deltas for this item
                    let index = event.output_index.unwrap_or(0);
                    while output_items.len() <= index {
                        output_items.push(StreamOutputItem::default());
                    }
                    output_items[index].reasoning_emitted = true;
                }
            }
            "response.function_call_arguments.delta" => {
                // Function arguments chunk
                if let Some(delta) = event.delta {
                    let index = event.output_index.unwrap_or(0);
                    if index < output_items.len() {
                        output_items[index].arguments.push_str(&delta);

                        // Build partial tool call for callback
                        if let (Some(call_id), Some(name)) = (
                            &output_items[index].call_id,
                            &output_items[index].function_name,
                        ) {
                            let partial = ToolCall {
                                id: call_id.clone(),
                                type_field: "function".to_string(),
                                function: ToolCallFunction {
                                    name: name.clone(),
                                    arguments: output_items[index].arguments.clone(),
                                },
                            };
                            on_tool_calls_partial(&[partial])?;
                        }
                    }
                }
            }
            "response.function_call_arguments.done" => {
                // Function call finalized with complete arguments
                if let Some(arguments) = event.arguments.clone() {
                    let index = event.output_index.unwrap_or(0);
                    if index < output_items.len() {
                        output_items[index].arguments = arguments.clone();

                        // Emit updated partial with complete arguments
                        if let (Some(call_id), Some(name)) = (
                            &output_items[index].call_id,
                            &output_items[index].function_name,
                        ) {
                            let partial = ToolCall {
                                id: call_id.clone(),
                                type_field: "function".to_string(),
                                function: ToolCallFunction {
                                    name: name.clone(),
                                    arguments,
                                },
                            };
                            on_tool_calls_partial(&[partial])?;
                        }
                    }
                }
            }
            "response.output_item.done" => {
                // Output item complete - but don't emit tool calls yet
                // Arguments may still be empty and will be filled in response.completed
                debug!(target: "openrouter", "Output item done");
            }
            "response.done" | "response.completed" => {
                // Response complete - extract function calls and usage from full response
                if let Some(response_data) = &event.response {
                    // Extract function calls from output
                    if let Some(output) = response_data.get("output").and_then(|v| v.as_array()) {
                        for item in output {
                            if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                                if let (Some(call_id), Some(name), Some(arguments)) = (
                                    item.get("call_id").and_then(|v| v.as_str()),
                                    item.get("name").and_then(|v| v.as_str()),
                                    item.get("arguments").and_then(|v| v.as_str()),
                                ) {
                                    let call_id_str = call_id.to_string();
                                    // Check if already added
                                    if !pending_tool_calls.iter().any(|tc| tc.id == call_id_str) {
                                        pending_tool_calls.push(ToolCall {
                                            id: call_id_str,
                                            type_field: "function".to_string(),
                                            function: ToolCallFunction {
                                                name: name.to_string(),
                                                arguments: arguments.to_string(),
                                            },
                                        });
                                    }
                                }
                            }
                        }
                    }

                    // Extract usage data
                    if let Some(usage_value) = response_data.get("usage") {
                        match serde_json::from_value::<super::types::Usage>(usage_value.clone()) {
                            Ok(usage) => {
                                debug!(
                                    target: "openrouter",
                                    prompt_tokens = usage.prompt_tokens,
                                    completion_tokens = usage.completion_tokens,
                                    total_tokens = usage.total_tokens,
                                    "Received usage data"
                                );

                                // Convert to unified format and emit
                                let unified_usage = usage.to_unified();
                                on_usage(unified_usage)?;
                            }
                            Err(e) => {
                                debug!(target: "openrouter", "Failed to parse usage data: {}", e);
                            }
                        }
                    }
                }

                // Emit any pending tool calls
                if !pending_tool_calls.is_empty() {
                    let calls = std::mem::take(pending_tool_calls);
                    on_tool_calls(calls)?;
                }
                debug!(target: "openrouter", "Response done");
            }
            "response.output_text.done" | "response.content_part.done" => {
                // Text content completed - already handled in output_item.done
                debug!(target: "openrouter", "Text content done");
            }
            "response.reasoning_text.done"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_summary_part.done" => {
                // Reasoning completed
                // Skip emitting full text if we've already streamed deltas
                let index = event.output_index.unwrap_or(0);
                let already_emitted =
                    index < output_items.len() && output_items[index].reasoning_emitted;

                if !already_emitted {
                    // Only emit if we haven't streamed deltas (fallback for non-streaming case)
                    if let Some(text) = event.text.as_ref() {
                        on_reasoning(text)?;
                    } else if let Some(part) = event.part.as_ref() {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            on_reasoning(text)?;
                        }
                    }
                }
                debug!(target: "openrouter", "Reasoning done");
            }
            _ => {
                if debug_enabled {
                    eprintln!("[openrouter::unknown] event type: {}", event.event_type);
                }
                debug!(target: "openrouter", "Unknown event type: {}", event.event_type);
            }
        }

        Ok(())
    }

    /// Convert internal ChatMessage history to Responses API input format.
    ///
    /// Handles conversion of the unified internal format to the structured
    /// input required by the Responses API.
    fn messages_to_input(&self, messages: &[ChatMessage]) -> Vec<InputItem> {
        let mut input_items = Vec::new();

        for msg in messages {
            input_items.extend(msg.to_input_items());
        }

        input_items
    }

    /// Check if a stream chunk reading error is recoverable.
    ///
    /// Returns true for transient network errors during stream reading:
    /// - EOF errors (connection closed unexpectedly)
    /// - Connection resets
    /// - Incomplete chunk reads
    /// - DNS resolution failures
    ///
    /// These errors indicate the connection was interrupted during streaming.
    /// When recoverable, we return partial content rather than failing completely.
    ///
    /// # Arguments
    ///
    /// * `error` - The error that occurred while reading a stream chunk
    ///
    /// # Returns
    ///
    /// True if the error is recoverable and partial content should be returned.
    fn is_chunk_error_recoverable(error: &anyhow::Error) -> bool {
        let error_str = format!("{:#}", error);
        let error_str_lower = error_str.to_lowercase();

        // Check for known recoverable patterns in reqwest/hyper errors
        error_str_lower.contains("unexpected eof")
            || error_str_lower.contains("connection reset")
            || error_str_lower.contains("broken pipe")
            || error_str_lower.contains("connection closed")
            || error_str_lower.contains("incomplete")
            || error_str_lower.contains("chunk size")
            || error_str_lower.contains("dns error")
            || error_str_lower.contains("failed to lookup address")
            || error_str_lower.contains("nodename nor servname provided")
            || error_str_lower.contains("decoding response body")
            || error_str_lower.contains("reading a body from connection")
    }
}

/// Streaming event from Responses API.
///
/// Events are JSON objects with a `type` field that determines the structure.
#[derive(Debug, Clone, Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    #[allow(dead_code)]
    response_id: Option<String>,
    #[serde(default)]
    output_index: Option<usize>,
    #[serde(default)]
    #[allow(dead_code)]
    content_index: Option<usize>,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    name: Option<String>,
    #[serde(default)]
    item: Option<StreamItem>,
    #[serde(default)]
    arguments: Option<String>,
    #[serde(default)]
    response: Option<serde_json::Value>,
    #[serde(default)]
    part: Option<serde_json::Value>,
}

/// Streaming item in event.
#[derive(Debug, Clone, Deserialize)]
struct StreamItem {
    #[serde(rename = "type")]
    item_type: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    role: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    status: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    content: Option<serde_json::Value>,
}

/// Accumulated output item during streaming.
#[derive(Debug, Clone, Default)]
struct StreamOutputItem {
    item_type: Option<String>,
    id: Option<String>,
    #[allow(dead_code)]
    content: String,
    function_name: Option<String>,
    call_id: Option<String>,
    arguments: String,
    reasoning_emitted: bool,
}

// Implement LlmClient trait for unified interface
#[async_trait]
impl crate::llm::provider::LlmClient for OpenRouterClient {
    async fn chat_with_tools_streaming<
        FContent,
        FTool,
        FReason,
        FToolPartial,
        FContentBlock,
        FUsage,
    >(
        &self,
        messages: &[crate::llm::unified::UnifiedMessage],
        tools: &[crate::llm::unified::UnifiedTool],
        on_content: FContent,
        on_tool_calls: FTool,
        on_reasoning: FReason,
        on_tool_calls_partial: FToolPartial,
        _on_content_block_complete: FContentBlock,
        on_usage: FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<crate::llm::unified::UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[crate::llm::unified::UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(crate::llm::unified::UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        // Convert unified messages to ChatMessage format
        let chat_messages = unified_to_chat_messages(messages);
        let tool_specs = unified_tools_to_specs(tools);

        // Wrap callbacks to convert from ToolCall to UnifiedToolCall
        let on_tool_calls_adapted = {
            let mut on_tool_calls = on_tool_calls;
            move |calls: Vec<ToolCall>| {
                let unified_calls = calls
                    .into_iter()
                    .map(|tc| crate::llm::unified::UnifiedToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        input: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({})),
                        raw_input_json: Some(tc.function.arguments),
                    })
                    .collect();
                on_tool_calls(unified_calls)
            }
        };

        let on_tool_calls_partial_adapted = {
            let mut on_tool_calls_partial = on_tool_calls_partial;
            move |calls: &[ToolCall]| {
                let unified_calls: Vec<_> = calls
                    .iter()
                    .map(|tc| crate::llm::unified::UnifiedToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        input: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({})),
                        raw_input_json: Some(tc.function.arguments.clone()),
                    })
                    .collect();
                on_tool_calls_partial(&unified_calls)
            }
        };

        // Call internal implementation with usage tracking
        self.chat_with_tools_streaming_internal(
            &chat_messages,
            &tool_specs,
            on_content,
            on_tool_calls_adapted,
            on_reasoning,
            on_tool_calls_partial_adapted,
            on_usage,
        )
        .await
    }

    fn provider_name(&self) -> &str {
        "openrouter"
    }
}

/// Convert unified messages to ChatMessage format (OpenRouter legacy).
///
/// This function handles the conversion of unified messages to the ChatMessage format
/// required by the OpenRouter Responses API. It properly handles:
/// - Text content
/// - Tool calls (function calls)
/// - Tool results (function call outputs)
/// - Reasoning content
fn unified_to_chat_messages(messages: &[crate::llm::unified::UnifiedMessage]) -> Vec<ChatMessage> {
    let mut result = Vec::new();

    for msg in messages {
        let role = match msg.role {
            crate::llm::unified::UnifiedRole::System => Role::System,
            crate::llm::unified::UnifiedRole::User => Role::User,
            crate::llm::unified::UnifiedRole::Assistant => Role::Assistant,
        };

        // Extract content components
        let content = msg.extract_text();
        let tool_calls = msg.extract_tool_calls();
        let reasoning = msg.extract_reasoning();

        // Extract tool results from the message
        let tool_results: Vec<_> = msg
            .content
            .iter()
            .filter_map(|block| {
                if let crate::llm::unified::UnifiedContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } = block
                {
                    Some((tool_use_id.clone(), content.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Convert tool calls
        let converted_tool_calls = if !tool_calls.is_empty() {
            Some(
                tool_calls
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        type_field: "function".to_string(),
                        function: ToolCallFunction {
                            name: tc.name,
                            arguments: tc.input.to_string(),
                        },
                    })
                    .collect(),
            )
        } else {
            None
        };

        // If this message has tool results, create separate Tool messages for each
        if !tool_results.is_empty() {
            for (tool_use_id, tool_content) in tool_results {
                // Convert content to string
                let content_str = match &tool_content {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };

                result.push(ChatMessage {
                    role: Role::Tool,
                    name: None,
                    tool_call_id: Some(tool_use_id),
                    content: Some(content_str),
                    tool_calls: None,
                    reasoning: None,
                    raw_content_blocks: None,
                    tool_metadata: None,
                    timestamp: msg.timestamp,
                    id: None,
                    provider_response_id: None,
                    status: None,
                });
            }
        } else {
            // Regular message (no tool results)
            result.push(ChatMessage {
                role,
                name: None,
                tool_call_id: None,
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                tool_calls: converted_tool_calls,
                reasoning,
                raw_content_blocks: None,
                tool_metadata: None,
                timestamp: msg.timestamp,
                id: msg.id.clone(),
                provider_response_id: None,
                status: None,
            });
        }
    }

    result
}

/// Convert unified tools to ToolSpec format (OpenRouter).
fn unified_tools_to_specs(tools: &[crate::llm::unified::UnifiedTool]) -> Vec<ToolSpec> {
    tools
        .iter()
        .map(|tool| ToolSpec {
            type_field: "function".to_string(),
            name: tool.name.clone(),
            description: Some(tool.description.clone()),
            parameters: tool.parameters.clone(),
            strict: None,
        })
        .collect()
}
