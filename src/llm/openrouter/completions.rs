//! OpenRouter Chat Completions API client with streaming support.
//!
//! Provides a client for the OpenRouter Chat Completions API (standard, stable)
//! that supports streaming Server-Sent Events (SSE) responses with tool calling,
//! reasoning tokens, and provider routing.
//!
//! Key features:
//! - Endpoint: `/api/v1/chat/completions` (OpenAI-compatible)
//! - Automatic reasoning preservation across multi-turn conversations
//! - Provider routing with fallbacks and cost optimization
//! - Automatic prompt caching (provider-dependent)
//! - Internal conversion of structured reasoning to simple strings
//! - Full usage tracking with detailed token counts and costs

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json;
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

use super::config::{OpenRouterConfig, ProviderPreferences, ReasoningConfig};
use super::types::{
    ChatCompletionsTool, ChatCompletionsToolFunction, CompletionChunk, CompletionMessage,
    CompletionRequest, ReasoningDetail, ToolCall, ToolCallBuilder, ToolCallDelta, ToolCallFunction,
    ToolChoice,
};

/// OpenRouter Chat Completions API streaming client.
///
/// Handles authentication, request construction, SSE parsing, and automatic
/// reasoning preservation for multi-turn conversations.
#[derive(Debug, Clone)]
pub struct OpenRouterCompletionsClient {
    http: reqwest::Client,
    cfg: OpenRouterConfig,
    reasoning_cfg: Option<ReasoningConfig>,
    provider_preferences: Option<ProviderPreferences>,
}

impl OpenRouterCompletionsClient {
    /// Create a new OpenRouter Completions API client.
    ///
    /// # Arguments
    ///
    /// * `cfg` - Base configuration (API key, model, etc.)
    /// * `reasoning_cfg` - Optional reasoning configuration (defaults to enabled)
    /// * `provider_preferences` - Optional provider routing preferences
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be constructed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use appam::llm::openrouter::{OpenRouterConfig, ReasoningConfig};
    /// use appam::llm::openrouter::completions::OpenRouterCompletionsClient;
    ///
    /// let cfg = OpenRouterConfig::default();
    /// let reasoning = Some(ReasoningConfig::default());
    /// let client = OpenRouterCompletionsClient::new(cfg, reasoning, None)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new(
        cfg: OpenRouterConfig,
        reasoning_cfg: Option<ReasoningConfig>,
        provider_preferences: Option<ProviderPreferences>,
    ) -> Result<Self> {
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

        // Reasoning is optional for Completions API - only use if explicitly provided
        // Not all models support reasoning, and sending it can cause 500 errors
        Ok(Self {
            http,
            cfg,
            reasoning_cfg,
            provider_preferences,
        })
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

        let key = self.cfg.api_key.as_ref().ok_or_else(|| {
            anyhow!("Missing OpenRouter API key; set via env OPENROUTER_API_KEY or config")
        })?;

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

    /// Convert UnifiedMessage to CompletionMessage(s) with automatic reasoning preservation.
    ///
    /// This method automatically preserves `reasoning` and `reasoning_details` from
    /// the unified message, enabling seamless multi-turn conversations with reasoning.
    ///
    /// Returns multiple messages when tool results are present, as the Chat Completions
    /// API requires separate "tool" role messages for each tool result.
    fn unified_to_completion_messages(
        &self,
        msg: &crate::llm::unified::UnifiedMessage,
    ) -> Vec<CompletionMessage> {
        let mut messages = Vec::new();

        let role = match msg.role {
            crate::llm::unified::UnifiedRole::System => "system",
            crate::llm::unified::UnifiedRole::User => "user",
            crate::llm::unified::UnifiedRole::Assistant => "assistant",
        }
        .to_string();

        // Extract content, tool calls, and tool results
        let content = msg.extract_text();
        let tool_calls = msg.extract_tool_calls();

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

        // If this message has tool results, create separate "tool" messages for each
        if !tool_results.is_empty() {
            for (tool_use_id, tool_content) in tool_results {
                // Convert content to string
                let content_str = match &tool_content {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };

                messages.push(CompletionMessage {
                    role: "tool".to_string(),
                    content: Some(content_str),
                    tool_calls: None,
                    reasoning: None,
                    reasoning_details: None,
                    name: None,
                    tool_call_id: Some(tool_use_id),
                });
            }
        } else {
            // Regular message (no tool results)
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

            messages.push(CompletionMessage {
                role,
                // Some OpenAI-compatible providers behind OpenRouter reject
                // `assistant` tool-call messages when `content` is serialized
                // as JSON null. Preserve the tool call and send an empty string
                // instead of omitting the field for this specific shape.
                content: if content.is_empty() {
                    if converted_tool_calls.is_some() {
                        Some(String::new())
                    } else {
                        None
                    }
                } else {
                    Some(content)
                },
                tool_calls: converted_tool_calls,
                // AUTOMATIC PRESERVATION: Pass through reasoning from unified message
                reasoning: msg.reasoning.clone(),
                reasoning_details: msg.reasoning_details.clone(),
                name: None,
                tool_call_id: None,
            });
        }

        messages
    }

    /// Stream a response with tool calling support using the Completions API.
    ///
    /// Makes a streaming request to `/api/v1/chat/completions` and invokes callbacks for:
    /// - Content chunks (streamed assistant response)
    /// - Tool call requests (when model decides to use tools)
    /// - Reasoning tokens (automatically converted from structured to simple strings)
    /// - Partial tool calls (incremental argument accumulation)
    ///
    /// # Parameters
    ///
    /// - `messages`: Conversation history (converted to API format internally)
    /// - `tools`: Available tool specifications
    /// - `on_content`: Callback for each content chunk
    /// - `on_tool_calls`: Callback when tool calls are finalized
    /// - `on_reasoning`: Callback for reasoning tokens (simple strings)
    /// - `on_tool_calls_partial`: Callback for incremental tool call updates
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails, authentication is invalid, or
    /// the response cannot be parsed.
    #[allow(clippy::too_many_arguments)]
    async fn chat_with_tools_streaming_internal<FContent, FTool, FReason, FToolPartial, FUsage>(
        &self,
        messages: &[crate::llm::unified::UnifiedMessage],
        tools: &[crate::llm::unified::UnifiedTool],
        mut on_content: FContent,
        mut on_tool_calls: FTool,
        mut on_reasoning: FReason,
        mut on_tool_calls_partial: FToolPartial,
        mut on_usage: FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<crate::llm::unified::UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[crate::llm::unified::UnifiedToolCall]) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        // Convert unified messages to Completions API format (with automatic preservation)
        // Uses flat_map because tool result messages expand into multiple "tool" role messages
        let completion_messages: Vec<CompletionMessage> = messages
            .iter()
            .flat_map(|msg| self.unified_to_completion_messages(msg))
            .collect();

        // Convert tools to Chat Completions format (nested function object)
        let tool_specs = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|tool| ChatCompletionsTool {
                        type_field: "function".to_string(),
                        function: ChatCompletionsToolFunction {
                            name: tool.name.clone(),
                            description: Some(tool.description.clone()),
                            parameters: tool.parameters.clone(),
                            strict: None,
                        },
                    })
                    .collect(),
            )
        };

        // Build request with usage tracking enabled
        let request = CompletionRequest {
            model: self.cfg.model.clone(),
            messages: completion_messages,
            tools: tool_specs,
            tool_choice: Some(ToolChoice::String("auto".to_string())),
            parallel_tool_calls: Some(true),
            temperature: self.cfg.temperature,
            max_tokens: self.cfg.max_output_tokens,
            top_p: self.cfg.top_p,
            reasoning: self
                .reasoning_cfg
                .as_ref()
                .map(|rc| serde_json::to_value(rc).unwrap()),
            provider: self.provider_preferences.clone(),
            stream: true,
            usage: Some(serde_json::json!({"include": true})),
            ..Default::default()
        };

        info!(target: "openrouter::completions", "Sending Chat Completions API request");
        debug!(
            target: "openrouter::completions",
            "Request body: {}",
            serde_json::to_string_pretty(&request)?
        );

        // Send request
        let res = self
            .http
            .post(format!("{}/chat/completions", self.cfg.base_url))
            .headers(self.headers()?)
            .json(&request)
            .send()
            .await
            .context("OpenRouter Completions API request failed")?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
            error!(status=?status, body=%text, "OpenRouter Completions error response");
            return Err(anyhow!(
                "OpenRouter Completions error ({}): {}",
                status,
                text
            ));
        }

        // Parse SSE stream
        self.parse_stream(
            res,
            &mut on_content,
            &mut on_tool_calls,
            &mut on_reasoning,
            &mut on_tool_calls_partial,
            &mut on_usage,
        )
        .await
    }

    /// Parse SSE event stream from Completions API.
    ///
    /// The Completions API uses standard OpenAI-compatible SSE format:
    /// - `data: {json}` lines containing CompletionChunk
    /// - `data: [DONE]` to signal completion
    /// - Reasoning comes in `delta.reasoning_details` array (converted to strings)
    /// - Usage data comes in the final chunk when `usage: {include: true}` is set
    ///
    /// # Error Recovery
    ///
    /// If the stream is interrupted by a recoverable error (EOF, connection reset, etc.),
    /// this method logs a warning and returns successfully with partial content rather than
    /// failing completely. This allows the caller to process whatever content was received.
    async fn parse_stream<FContent, FTool, FReason, FToolPartial, FUsage>(
        &self,
        res: reqwest::Response,
        on_content: &mut FContent,
        on_tool_calls: &mut FTool,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
        on_usage: &mut FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<crate::llm::unified::UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[crate::llm::unified::UnifiedToolCall]) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        let mut stream = res.bytes_stream();
        let mut buffer = String::new();

        // Tool call accumulation
        let mut accumulated_tool_calls: HashMap<usize, ToolCallBuilder> = HashMap::new();

        // Reasoning accumulation (for preservation)
        let mut accumulated_reasoning_details: Vec<ReasoningDetail> = Vec::new();

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
                            target: "openrouter::completions",
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
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // Process complete lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                // Skip empty lines and comments
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                // Parse SSE data
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        debug!(target: "openrouter::completions", "Stream completed");
                        break;
                    }

                    // Parse completion chunk
                    let chunk: CompletionChunk = match serde_json::from_str(data) {
                        Ok(chunk) => chunk,
                        Err(e) => {
                            error!(target: "openrouter::completions", "Failed to parse chunk: {}", e);
                            continue;
                        }
                    };
                    events_processed += 1;

                    for choice in chunk.choices {
                        // Check for errors
                        if let Some(error) = choice.error {
                            error!(target: "openrouter::completions", "Stream error: {}", error.message);
                            return Err(anyhow!("Stream error: {}", error.message));
                        }

                        // Handle content streaming
                        if let Some(content) = choice.delta.content {
                            on_content(&content)?;
                        }

                        // Handle reasoning streaming (AUTOMATIC CONVERSION)
                        if let Some(reasoning_details) = choice.delta.reasoning_details {
                            for detail in reasoning_details {
                                // Convert structured reasoning to simple string for callback
                                on_reasoning(detail.extract_text())?;

                                // Accumulate for preservation
                                accumulated_reasoning_details.push(detail);
                            }
                        } else if let Some(reasoning) = choice.delta.reasoning {
                            // Legacy simple reasoning string
                            on_reasoning(&reasoning)?;
                        }

                        // Handle tool call streaming (partial)
                        if let Some(tool_call_deltas) = choice.delta.tool_calls {
                            for delta in tool_call_deltas {
                                self.accumulate_tool_call(&mut accumulated_tool_calls, delta);
                            }

                            // Emit partial tool calls
                            let partial_calls =
                                self.get_partial_tool_calls(&accumulated_tool_calls);
                            on_tool_calls_partial(&partial_calls)?;
                        }

                        // Handle completion
                        if let Some(finish_reason) = choice.finish_reason {
                            if finish_reason == "tool_calls" {
                                // Finalize and emit tool calls
                                let final_calls =
                                    self.finalize_tool_calls(&mut accumulated_tool_calls);
                                if !final_calls.is_empty() {
                                    on_tool_calls(final_calls)?;
                                }
                            }
                        }
                    }

                    // Handle usage data (comes in final chunk when usage tracking enabled)
                    if let Some(usage) = chunk.usage {
                        debug!(
                            target: "openrouter::completions",
                            prompt_tokens = usage.prompt_tokens,
                            completion_tokens = usage.completion_tokens,
                            total_tokens = usage.total_tokens,
                            "Received usage data"
                        );

                        // Convert to unified format and emit
                        let unified_usage = usage.to_unified();
                        on_usage(unified_usage)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Accumulate a tool call delta.
    fn accumulate_tool_call(
        &self,
        accumulated: &mut HashMap<usize, ToolCallBuilder>,
        delta: ToolCallDelta,
    ) {
        let builder = accumulated
            .entry(delta.index)
            .or_insert_with(|| ToolCallBuilder::new(delta.id.clone().unwrap_or_default()));

        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                builder.name = name;
            }
            if let Some(arguments) = function.arguments {
                builder.arguments.push_str(&arguments);
            }
        }
    }

    /// Get partial tool calls (for incremental updates).
    fn get_partial_tool_calls(
        &self,
        accumulated: &HashMap<usize, ToolCallBuilder>,
    ) -> Vec<crate::llm::unified::UnifiedToolCall> {
        accumulated
            .values()
            .map(|builder| crate::llm::unified::UnifiedToolCall {
                id: builder.id.clone(),
                name: builder.name.clone(),
                input: serde_json::from_str(&builder.arguments).unwrap_or(serde_json::json!({})),
                raw_input_json: Some(builder.arguments.clone()),
            })
            .collect()
    }

    /// Finalize accumulated tool calls.
    fn finalize_tool_calls(
        &self,
        accumulated: &mut HashMap<usize, ToolCallBuilder>,
    ) -> Vec<crate::llm::unified::UnifiedToolCall> {
        accumulated
            .drain()
            .map(|(_, builder)| crate::llm::unified::UnifiedToolCall {
                id: builder.id,
                name: builder.name,
                input: serde_json::from_str(&builder.arguments).unwrap_or(serde_json::json!({})),
                raw_input_json: Some(builder.arguments),
            })
            .collect()
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

// Implement LlmClient trait for unified interface
#[async_trait]
impl crate::llm::provider::LlmClient for OpenRouterCompletionsClient {
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
        // Call internal implementation with usage tracking enabled
        self.chat_with_tools_streaming_internal(
            messages,
            tools,
            on_content,
            on_tool_calls,
            on_reasoning,
            on_tool_calls_partial,
            on_usage,
        )
        .await
    }

    fn provider_name(&self) -> &str {
        "openrouter-completions"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::unified::{UnifiedContentBlock, UnifiedMessage, UnifiedRole};
    use serde_json::json;

    #[test]
    fn test_assistant_tool_call_message_uses_empty_string_content() {
        let client = OpenRouterCompletionsClient::new(OpenRouterConfig::default(), None, None)
            .expect("client should initialize");

        let messages = client.unified_to_completion_messages(&UnifiedMessage {
            role: UnifiedRole::Assistant,
            content: vec![UnifiedContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                input: json!({"file_path": "README.md"}),
            }],
            id: None,
            timestamp: None,
            reasoning: None,
            reasoning_details: None,
        });

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "assistant");
        assert_eq!(messages[0].content.as_deref(), Some(""));
        assert!(messages[0].tool_calls.is_some());
    }

    #[test]
    fn test_plain_assistant_message_without_tool_calls_keeps_empty_content_omitted() {
        let client = OpenRouterCompletionsClient::new(OpenRouterConfig::default(), None, None)
            .expect("client should initialize");

        let messages = client.unified_to_completion_messages(&UnifiedMessage {
            role: UnifiedRole::Assistant,
            content: vec![],
            id: None,
            timestamp: None,
            reasoning: None,
            reasoning_details: None,
        });

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, None);
        assert!(messages[0].tool_calls.is_none());
    }
}
