//! Vertex AI client implementation.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER};
use reqwest::{StatusCode, Url};
use tokio::time::sleep;
use tracing::{debug, error, warn};

use super::config::VertexConfig;
use super::convert::{from_unified_messages, from_unified_tools};
use super::types::{
    VertexErrorResponse, VertexFunctionCall, VertexGenerateContentRequest,
    VertexGenerateContentResponse, VertexPart, VertexPartialArg, VertexUsageMetadata,
};
use crate::llm::provider::LlmClient;
use crate::llm::unified::{UnifiedContentBlock, UnifiedMessage, UnifiedTool, UnifiedToolCall};

/// HTTP client wrapper for Google Vertex Gemini inference.
///
/// This client translates appam's provider-agnostic `UnifiedMessage` /
/// `UnifiedTool` structures into Vertex `generateContent` payloads and converts
/// streamed responses back into unified callbacks used by the runtime.
///
/// # Design Notes
///
/// - Supports both SSE streaming (`alt=sse`) and non-streaming JSON responses.
/// - Handles full function-call arguments and incremental `partialArgs`
///   reconstruction when `streamFunctionCallArguments` is enabled.
/// - Preserves thought signatures by emitting signature metadata blocks so that
///   follow-up turns can replay model/tool history without continuity errors.
/// - Applies bounded retries for transient HTTP failures.
#[derive(Debug, Clone)]
pub struct VertexClient {
    http: reqwest::Client,
    config: VertexConfig,
}

#[derive(Debug, Clone)]
struct ResolvedAuth {
    access_token: Option<String>,
    api_key: Option<String>,
}

#[derive(Debug, Clone)]
struct PendingToolCall {
    id: String,
    name: String,
    args: serde_json::Value,
    signature: Option<String>,
    string_fragments: HashMap<String, String>,
    finalized: bool,
}

#[derive(Debug, Default)]
struct StreamState {
    text_buffer: String,
    text_signature: Option<String>,
    pending_tool_calls: Vec<PendingToolCall>,
    finalized_tool_calls: Vec<UnifiedToolCall>,
    finalized_fingerprints: HashSet<String>,
    content_blocks: Vec<UnifiedContentBlock>,
    latest_usage: Option<VertexUsageMetadata>,
    next_tool_call_index: usize,
}

#[derive(Debug, Clone)]
enum PathSegment {
    Key(String),
    Index(usize),
}

impl StreamState {
    fn flush_text_block(&mut self) {
        if self.text_buffer.is_empty() {
            return;
        }

        let text = std::mem::take(&mut self.text_buffer);
        self.content_blocks.push(UnifiedContentBlock::Text { text });

        if let Some(signature) = self.text_signature.take() {
            self.content_blocks.push(UnifiedContentBlock::Thinking {
                thinking: String::new(),
                signature: Some(signature),
                encrypted_content: None,
                redacted: false,
            });
        }
    }

    fn ensure_pending_call(&mut self, name: Option<&str>, signature: Option<String>) -> usize {
        let target = if let Some(name) = name {
            self.pending_tool_calls
                .iter()
                .enumerate()
                .rev()
                .find(|(_, call)| !call.finalized && call.name == name)
                .map(|(idx, _)| idx)
        } else {
            self.pending_tool_calls
                .iter()
                .position(|call| !call.finalized)
        };

        if let Some(idx) = target {
            if let Some(name) = name {
                if !name.trim().is_empty() {
                    self.pending_tool_calls[idx].name = name.to_string();
                }
            }
            if self.pending_tool_calls[idx].signature.is_none() {
                self.pending_tool_calls[idx].signature = signature;
            }
            return idx;
        }

        let id = format!("vertex_call_{}", self.next_tool_call_index);
        self.next_tool_call_index += 1;

        let call = PendingToolCall {
            id,
            name: name
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .unwrap_or("unnamed_tool")
                .to_string(),
            args: serde_json::json!({}),
            signature,
            string_fragments: HashMap::new(),
            finalized: false,
        };

        self.pending_tool_calls.push(call);
        self.pending_tool_calls.len() - 1
    }

    fn apply_partial_args(&mut self, idx: usize, partial_args: &[VertexPartialArg]) {
        let call = &mut self.pending_tool_calls[idx];
        for partial in partial_args {
            let Some(path) = partial.json_path.as_deref() else {
                continue;
            };

            if let Some(fragment) = partial.string_value.as_deref() {
                if fragment.is_empty() {
                    continue;
                }

                let entry = call.string_fragments.entry(path.to_string()).or_default();
                entry.push_str(fragment);
                set_json_path(
                    &mut call.args,
                    path,
                    serde_json::Value::String(entry.clone()),
                );
                continue;
            }

            if let Some(number) = partial.number_value {
                if let Some(number_value) = serde_json::Number::from_f64(number) {
                    set_json_path(
                        &mut call.args,
                        path,
                        serde_json::Value::Number(number_value),
                    );
                }
                continue;
            }

            if let Some(value) = partial.bool_value {
                set_json_path(&mut call.args, path, serde_json::Value::Bool(value));
                continue;
            }

            if partial.null_value.is_some() {
                set_json_path(&mut call.args, path, serde_json::Value::Null);
                continue;
            }

            if let Some(value) = partial.struct_value.clone() {
                set_json_path(&mut call.args, path, value);
                continue;
            }

            if let Some(value) = partial.list_value.clone() {
                set_json_path(&mut call.args, path, value);
            }
        }
    }

    fn pending_calls_snapshot(&self) -> Vec<UnifiedToolCall> {
        self.pending_tool_calls
            .iter()
            .filter(|call| !call.finalized)
            .map(|call| UnifiedToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                input: call.args.clone(),
                raw_input_json: Some(call.args.to_string()),
            })
            .collect()
    }

    fn finalize_pending_call(&mut self, idx: usize) {
        if self.pending_tool_calls[idx].finalized {
            return;
        }

        let call = &mut self.pending_tool_calls[idx];
        call.finalized = true;

        let fingerprint = format!("{}:{}", call.name, call.args);
        if !self.finalized_fingerprints.insert(fingerprint) {
            return;
        }

        let unified = UnifiedToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.args.clone(),
            raw_input_json: Some(call.args.to_string()),
        };

        self.finalized_tool_calls.push(unified.clone());
        self.content_blocks.push(UnifiedContentBlock::ToolUse {
            id: unified.id,
            name: unified.name,
            input: unified.input,
        });

        if let Some(signature) = call.signature.clone() {
            self.content_blocks.push(UnifiedContentBlock::Thinking {
                thinking: String::new(),
                signature: Some(signature),
                encrypted_content: None,
                redacted: false,
            });
        }
    }

    fn finalize_all_pending_calls(&mut self) {
        let len = self.pending_tool_calls.len();
        for idx in 0..len {
            if !self.pending_tool_calls[idx].finalized {
                self.finalize_pending_call(idx);
            }
        }
    }
}

impl VertexClient {
    /// Create a new Vertex client with connection pooling and DNS-aware host
    /// resolution from the shared HTTP client pool.
    ///
    /// # Parameters
    ///
    /// - `config`: Provider configuration (model, endpoint, auth hints, retry)
    ///
    /// # Returns
    ///
    /// Returns an initialized client ready to execute chat/tool requests.
    ///
    /// # Errors
    ///
    /// Returns an error when configuration validation fails or the underlying
    /// HTTP client cannot be constructed.
    pub fn new(config: VertexConfig) -> Result<Self> {
        config.validate()?;

        let base_url = if config.project_id.is_some()
            && config.base_url == "https://aiplatform.googleapis.com"
        {
            format!("https://{}-aiplatform.googleapis.com", config.location)
        } else {
            config.base_url.clone()
        };

        let http = crate::http::client_pool::get_or_init_client(&base_url, |ctx| {
            let mut builder = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .pool_idle_timeout(Duration::from_secs(120))
                .pool_max_idle_per_host(10)
                .tcp_keepalive(Duration::from_secs(60))
                .tcp_nodelay(true)
                .gzip(true)
                .user_agent("appam/0.1.1");

            if let Some(addrs) = ctx.resolved_addrs() {
                builder = builder.resolve_to_addrs(ctx.host(), addrs);
            }

            builder
                .build()
                .context("Failed to create Vertex HTTP client")
        })?;

        Ok(Self { http, config })
    }

    fn effective_base_url(&self) -> String {
        if self.config.project_id.is_some()
            && self.config.base_url == "https://aiplatform.googleapis.com"
        {
            format!("https://{}-aiplatform.googleapis.com", self.config.location)
        } else {
            self.config.base_url.clone()
        }
    }

    fn resolve_auth(&self) -> Result<ResolvedAuth> {
        let access_token = self
            .config
            .access_token
            .clone()
            .or_else(|| std::env::var("GOOGLE_VERTEX_ACCESS_TOKEN").ok())
            .filter(|value| !value.trim().is_empty());

        let api_key = self
            .config
            .api_key
            .clone()
            .or_else(|| std::env::var("GOOGLE_VERTEX_API_KEY").ok())
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .filter(|value| !value.trim().is_empty());

        if access_token.is_none() && api_key.is_none() {
            return Err(anyhow!(
                "Missing Vertex credentials. Set GOOGLE_VERTEX_ACCESS_TOKEN or GOOGLE_VERTEX_API_KEY/GOOGLE_API_KEY/GEMINI_API_KEY"
            ));
        }

        Ok(ResolvedAuth {
            access_token,
            api_key,
        })
    }

    fn build_headers(&self, streaming: bool, access_token: Option<&str>) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        headers.insert(
            ACCEPT,
            HeaderValue::from_static(if streaming {
                "text/event-stream"
            } else {
                "application/json"
            }),
        );

        if let Some(token) = access_token {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", token))
                    .context("Invalid Vertex authorization header")?,
            );
        }

        Ok(headers)
    }

    fn build_endpoint_url(&self, streaming: bool, api_key: Option<&str>) -> Result<Url> {
        let base = self.effective_base_url();
        let method = if streaming {
            "streamGenerateContent"
        } else {
            "generateContent"
        };

        let endpoint = if let Some(project_id) = self.config.project_id.as_ref() {
            format!(
                "{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
                base.trim_end_matches('/'),
                project_id,
                self.config.location,
                self.config.model,
                method
            )
        } else {
            format!(
                "{}/v1/publishers/google/models/{}:{}",
                base.trim_end_matches('/'),
                self.config.model,
                method
            )
        };

        let mut url = Url::parse(&endpoint)
            .with_context(|| format!("Failed to build Vertex endpoint URL from '{}'", endpoint))?;

        {
            let mut query = url.query_pairs_mut();
            if streaming {
                query.append_pair("alt", "sse");
            }

            if let Some(key) = api_key {
                query.append_pair("key", key);
            }
        }

        Ok(url)
    }

    fn retry_config(&self) -> crate::llm::openai::RetryConfig {
        self.config.retry.clone().unwrap_or_default()
    }

    fn should_retry_status(status: StatusCode) -> bool {
        matches!(
            status,
            StatusCode::TOO_MANY_REQUESTS
                | StatusCode::INTERNAL_SERVER_ERROR
                | StatusCode::BAD_GATEWAY
                | StatusCode::SERVICE_UNAVAILABLE
                | StatusCode::GATEWAY_TIMEOUT
                | StatusCode::REQUEST_TIMEOUT
        )
    }

    fn should_retry_reqwest_error(error: &reqwest::Error) -> bool {
        error.is_timeout() || error.is_connect() || error.is_request() || error.is_body()
    }

    fn retry_after_from_headers(headers: &HeaderMap) -> Option<Duration> {
        headers
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|raw| raw.parse::<u64>().ok())
            .map(Duration::from_secs)
    }

    fn compute_retry_delay(
        retry_config: &crate::llm::openai::RetryConfig,
        attempt: u32,
        retry_after: Option<Duration>,
    ) -> Duration {
        if let Some(delay) = retry_after {
            return std::cmp::min(delay, Duration::from_millis(retry_config.max_backoff_ms));
        }

        Duration::from_millis(retry_config.calculate_backoff(attempt))
    }

    fn build_request_body(
        &self,
        messages: &[UnifiedMessage],
        tools: &[UnifiedTool],
    ) -> VertexGenerateContentRequest {
        let conversation = from_unified_messages(messages);
        let vertex_tools = from_unified_tools(tools);

        let generation_config = Some(super::types::VertexGenerationConfig {
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            max_output_tokens: self.config.max_output_tokens,
            thinking_config: self.config.thinking.clone(),
        });

        let tool_config = if tools.is_empty() {
            None
        } else {
            Some(super::types::VertexToolConfig {
                function_calling_config: super::types::VertexFunctionCallingConfig {
                    mode: Some(
                        match self.config.function_calling_mode {
                            super::config::VertexFunctionCallingMode::Auto => "AUTO",
                            super::config::VertexFunctionCallingMode::Any => "ANY",
                            super::config::VertexFunctionCallingMode::None => "NONE",
                        }
                        .to_string(),
                    ),
                    allowed_function_names: self.config.allowed_function_names.clone(),
                    stream_function_call_arguments: Some(
                        self.config.stream_function_call_arguments,
                    ),
                },
            })
        };

        VertexGenerateContentRequest {
            contents: conversation.contents,
            system_instruction: conversation.system_instruction,
            tools: if vertex_tools.is_empty() {
                None
            } else {
                Some(vertex_tools)
            },
            tool_config,
            generation_config,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_response_chunk<FContent, FReason, FToolPartial>(
        &self,
        chunk: &VertexGenerateContentResponse,
        state: &mut StreamState,
        on_content: &mut FContent,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
    {
        if let Some(usage) = chunk.usage_metadata.clone() {
            state.latest_usage = Some(usage);
        }

        for candidate in &chunk.candidates {
            if let Some(content) = &candidate.content {
                for part in &content.parts {
                    self.process_part(
                        part,
                        state,
                        on_content,
                        on_reasoning,
                        on_tool_calls_partial,
                    )?;
                }
            }

            if candidate.finish_reason.is_some() {
                state.flush_text_block();
                state.finalize_all_pending_calls();
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn process_part<FContent, FReason, FToolPartial>(
        &self,
        part: &VertexPart,
        state: &mut StreamState,
        on_content: &mut FContent,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
    {
        if let Some(text) = part.text.as_deref() {
            if part.thought.unwrap_or(false) {
                if !text.is_empty() {
                    on_reasoning(text)?;
                }

                state.content_blocks.push(UnifiedContentBlock::Thinking {
                    thinking: text.to_string(),
                    signature: part.thought_signature.clone(),
                    encrypted_content: None,
                    redacted: false,
                });
            } else if !text.is_empty() {
                on_content(text)?;
                state.text_buffer.push_str(text);
                if let Some(signature) = part.thought_signature.clone() {
                    state.text_signature = Some(signature);
                }
            }
        }

        if let Some(function_call) = part.function_call.as_ref() {
            state.flush_text_block();
            self.handle_function_call(
                function_call,
                part.thought_signature.clone(),
                state,
                on_tool_calls_partial,
            )?;
        }

        Ok(())
    }

    fn handle_function_call<FToolPartial>(
        &self,
        function_call: &VertexFunctionCall,
        signature: Option<String>,
        state: &mut StreamState,
        on_tool_calls_partial: &mut FToolPartial,
    ) -> Result<()>
    where
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
    {
        let is_empty_call = function_call.name.is_none()
            && function_call.args.is_none()
            && function_call.partial_args.is_none()
            && function_call.will_continue.is_none();

        if is_empty_call {
            return Ok(());
        }

        let idx = state.ensure_pending_call(function_call.name.as_deref(), signature);

        if let Some(args) = function_call.args.clone() {
            state.pending_tool_calls[idx].args = args;
            state.pending_tool_calls[idx].string_fragments.clear();
        }

        let mut should_emit_partial = false;
        if let Some(partial_args) = function_call.partial_args.as_ref() {
            state.apply_partial_args(idx, partial_args);
            should_emit_partial = true;
        }

        if function_call.name.is_some()
            && function_call.args.is_none()
            && function_call.partial_args.is_none()
            && function_call.will_continue.unwrap_or(true)
        {
            should_emit_partial = true;
        }

        if should_emit_partial {
            let snapshot = state.pending_calls_snapshot();
            if !snapshot.is_empty() {
                on_tool_calls_partial(&snapshot)?;
            }
        }

        let should_finalize = function_call.will_continue == Some(false)
            || (function_call.will_continue.is_none()
                && function_call.args.is_some()
                && function_call.partial_args.is_none());

        if should_finalize {
            state.finalize_pending_call(idx);
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn parse_sse_stream<FContent, FTool, FReason, FToolPartial, FContentBlock, FUsage>(
        &self,
        response: reqwest::Response,
        on_content: &mut FContent,
        on_tool_calls: &mut FTool,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
        on_content_block_complete: &mut FContentBlock,
        on_usage: &mut FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut state = StreamState::default();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.context("Failed to read Vertex stream chunk")?;
            let chunk_str =
                std::str::from_utf8(&chunk).context("Vertex stream payload was not valid UTF-8")?;
            buffer.push_str(chunk_str);

            while let Some((event_end, delimiter_len)) = find_sse_event_boundary(&buffer) {
                let event = buffer[..event_end].to_string();
                buffer = buffer[event_end + delimiter_len..].to_string();

                if let Some(payload) = extract_sse_payload(&event) {
                    self.process_payload(
                        &payload,
                        &mut state,
                        on_content,
                        on_reasoning,
                        on_tool_calls_partial,
                    )?;
                }
            }
        }

        if !buffer.trim().is_empty() {
            if let Some(payload) = extract_sse_payload(&buffer).or_else(|| {
                if buffer.trim_start().starts_with('{') {
                    Some(buffer.trim().to_string())
                } else {
                    None
                }
            }) {
                self.process_payload(
                    &payload,
                    &mut state,
                    on_content,
                    on_reasoning,
                    on_tool_calls_partial,
                )?;
            }
        }

        self.finish_stream_state(state, on_tool_calls, on_content_block_complete, on_usage)
    }

    #[allow(clippy::too_many_arguments)]
    async fn parse_non_streaming_response<
        FContent,
        FTool,
        FReason,
        FToolPartial,
        FContentBlock,
        FUsage,
    >(
        &self,
        response: reqwest::Response,
        on_content: &mut FContent,
        on_tool_calls: &mut FTool,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
        on_content_block_complete: &mut FContentBlock,
        on_usage: &mut FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        let body = response
            .text()
            .await
            .context("Failed to read Vertex non-streaming body")?;

        let mut state = StreamState::default();
        self.process_payload(
            body.trim(),
            &mut state,
            on_content,
            on_reasoning,
            on_tool_calls_partial,
        )?;

        self.finish_stream_state(state, on_tool_calls, on_content_block_complete, on_usage)
    }

    fn process_payload<FContent, FReason, FToolPartial>(
        &self,
        payload: &str,
        state: &mut StreamState,
        on_content: &mut FContent,
        on_reasoning: &mut FReason,
        on_tool_calls_partial: &mut FToolPartial,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
    {
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            return Ok(());
        }

        if payload.starts_with('{') {
            if let Ok(error_response) = serde_json::from_str::<VertexErrorResponse>(payload) {
                return Err(anyhow!(
                    "Vertex API error{}: {}",
                    error_response
                        .error
                        .status
                        .as_deref()
                        .map(|status| format!(" ({})", status))
                        .unwrap_or_default(),
                    error_response.error.message
                ));
            }

            let split_chunks: Vec<&str> = payload
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .collect();

            // Some Vertex transports send multiple JSON documents in one SSE data
            // payload line group. Process each document independently.
            if split_chunks.len() > 1
                && split_chunks
                    .iter()
                    .all(|line| line.starts_with('{') || line.starts_with('['))
            {
                for line in split_chunks {
                    self.process_payload(
                        line,
                        state,
                        on_content,
                        on_reasoning,
                        on_tool_calls_partial,
                    )?;
                }
                return Ok(());
            }

            let chunk = serde_json::from_str::<VertexGenerateContentResponse>(payload)
                .with_context(|| format!("Failed to parse Vertex stream payload: {}", payload))?;

            return self.process_response_chunk(
                &chunk,
                state,
                on_content,
                on_reasoning,
                on_tool_calls_partial,
            );
        }

        if payload.starts_with('[') {
            let chunks = serde_json::from_str::<Vec<VertexGenerateContentResponse>>(payload)
                .with_context(|| {
                    format!("Failed to parse Vertex payload array response: {}", payload)
                })?;

            for chunk in chunks {
                self.process_response_chunk(
                    &chunk,
                    state,
                    on_content,
                    on_reasoning,
                    on_tool_calls_partial,
                )?;
            }
            return Ok(());
        }

        debug!(payload = %payload, "Ignoring non-JSON Vertex stream payload");
        Ok(())
    }

    fn finish_stream_state<FTool, FContentBlock, FUsage>(
        &self,
        mut state: StreamState,
        on_tool_calls: &mut FTool,
        on_content_block_complete: &mut FContentBlock,
        on_usage: &mut FUsage,
    ) -> Result<()>
    where
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FContentBlock: FnMut(UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        state.flush_text_block();
        state.finalize_all_pending_calls();

        if !state.finalized_tool_calls.is_empty() {
            on_tool_calls(state.finalized_tool_calls.clone())?;
        }

        for block in state.content_blocks {
            on_content_block_complete(block)?;
        }

        if let Some(usage) = state.latest_usage {
            let unified_usage = crate::llm::unified::UnifiedUsage {
                input_tokens: usage.prompt_token_count.unwrap_or_default(),
                output_tokens: usage.candidates_token_count.unwrap_or_default(),
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                reasoning_tokens: usage.thoughts_token_count.filter(|value| *value > 0),
            };
            on_usage(unified_usage)?;
        }

        Ok(())
    }
}

fn extract_sse_payload(event: &str) -> Option<String> {
    let mut payload = String::new();

    for line in event.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !payload.is_empty() {
                payload.push('\n');
            }
            payload.push_str(rest.trim_start());
        }
    }

    if payload.trim().is_empty() {
        None
    } else {
        Some(payload)
    }
}

fn find_sse_event_boundary(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|idx| (idx, 2));
    let crlf = buffer.find("\r\n\r\n").map(|idx| (idx, 4));

    match (lf, crlf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn parse_json_path(path: &str) -> Vec<PathSegment> {
    let mut segments = Vec::new();
    let mut chars = path.trim().chars().peekable();

    if chars.next() != Some('$') {
        return segments;
    }

    while let Some(ch) = chars.next() {
        match ch {
            '.' => {
                let mut key = String::new();
                while let Some(next) = chars.peek() {
                    if *next == '.' || *next == '[' {
                        break;
                    }
                    key.push(*next);
                    chars.next();
                }
                if !key.is_empty() {
                    segments.push(PathSegment::Key(key));
                }
            }
            '[' => {
                let mut raw_index = String::new();
                for next in chars.by_ref() {
                    if next == ']' {
                        break;
                    }
                    raw_index.push(next);
                }

                if let Ok(index) = raw_index.parse::<usize>() {
                    segments.push(PathSegment::Index(index));
                }
            }
            _ => {}
        }
    }

    segments
}

fn set_json_path(root: &mut serde_json::Value, path: &str, value: serde_json::Value) {
    let segments = parse_json_path(path);
    if segments.is_empty() {
        *root = value;
        return;
    }

    set_json_segments(root, &segments, value);
}

fn set_json_segments(
    current: &mut serde_json::Value,
    segments: &[PathSegment],
    value: serde_json::Value,
) {
    if segments.is_empty() {
        *current = value;
        return;
    }

    match &segments[0] {
        PathSegment::Key(key) => {
            if !current.is_object() {
                *current = serde_json::json!({});
            }

            let object = current
                .as_object_mut()
                .expect("object conversion must succeed");

            if segments.len() == 1 {
                object.insert(key.clone(), value);
            } else {
                let next = object
                    .entry(key.clone())
                    .or_insert_with(|| serde_json::Value::Null);
                set_json_segments(next, &segments[1..], value);
            }
        }
        PathSegment::Index(index) => {
            if !current.is_array() {
                *current = serde_json::json!([]);
            }

            let array = current
                .as_array_mut()
                .expect("array conversion must succeed");

            if array.len() <= *index {
                array.resize(*index + 1, serde_json::Value::Null);
            }

            if segments.len() == 1 {
                array[*index] = value;
            } else {
                set_json_segments(&mut array[*index], &segments[1..], value);
            }
        }
    }
}

#[async_trait]
impl LlmClient for VertexClient {
    async fn chat_with_tools_streaming<
        FContent,
        FTool,
        FReason,
        FToolPartial,
        FContentBlock,
        FUsage,
    >(
        &self,
        messages: &[UnifiedMessage],
        tools: &[UnifiedTool],
        on_content: FContent,
        on_tool_calls: FTool,
        on_reasoning: FReason,
        on_tool_calls_partial: FToolPartial,
        on_content_block_complete: FContentBlock,
        on_usage: FUsage,
    ) -> Result<()>
    where
        FContent: FnMut(&str) -> Result<()> + Send,
        FTool: FnMut(Vec<UnifiedToolCall>) -> Result<()> + Send,
        FReason: FnMut(&str) -> Result<()> + Send,
        FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
        FContentBlock: FnMut(UnifiedContentBlock) -> Result<()> + Send,
        FUsage: FnMut(crate::llm::unified::UnifiedUsage) -> Result<()> + Send,
    {
        let request_body = self.build_request_body(messages, tools);
        let auth = self.resolve_auth()?;

        debug!(
            target: "vertex",
            model = %self.config.model,
            stream = self.config.stream,
            "Sending Vertex request"
        );

        let retry_config = self.retry_config();
        let max_attempts = retry_config.max_retries.saturating_add(1).max(1);
        let mut attempt: u32 = 0;

        let mut on_content = on_content;
        let mut on_tool_calls = on_tool_calls;
        let mut on_reasoning = on_reasoning;
        let mut on_tool_calls_partial = on_tool_calls_partial;
        let mut on_content_block_complete = on_content_block_complete;
        let mut on_usage = on_usage;

        loop {
            attempt += 1;

            let endpoint = self.build_endpoint_url(
                self.config.stream,
                if auth.access_token.is_some() {
                    None
                } else {
                    auth.api_key.as_deref()
                },
            )?;

            let headers = self.build_headers(self.config.stream, auth.access_token.as_deref())?;

            let response = match self
                .http
                .post(endpoint)
                .headers(headers)
                .json(&request_body)
                .send()
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    if attempt < max_attempts && Self::should_retry_reqwest_error(&error) {
                        let wait = Self::compute_retry_delay(&retry_config, attempt, None);
                        warn!(
                            target: "vertex",
                            attempt,
                            max_attempts,
                            wait_secs = wait.as_secs_f64(),
                            error = %error,
                            "Vertex request failed, retrying"
                        );
                        sleep(wait).await;
                        continue;
                    }

                    return Err(error).context("Vertex API request failed");
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let headers = response.headers().clone();
                let body = response.text().await.unwrap_or_default();

                error!(target: "vertex", status = %status, body = %body, "Vertex API error response");

                if attempt < max_attempts && Self::should_retry_status(status) {
                    let retry_after = Self::retry_after_from_headers(&headers);
                    let wait = Self::compute_retry_delay(&retry_config, attempt, retry_after);
                    sleep(wait).await;
                    continue;
                }

                if let Ok(error_response) = serde_json::from_str::<VertexErrorResponse>(&body) {
                    return Err(anyhow!(
                        "Vertex API error{}: {}",
                        error_response
                            .error
                            .status
                            .as_deref()
                            .map(|status| format!(" ({})", status))
                            .unwrap_or_default(),
                        error_response.error.message
                    ));
                }

                return Err(anyhow!("Vertex API error ({}): {}", status, body));
            }

            let result = if self.config.stream {
                self.parse_sse_stream(
                    response,
                    &mut on_content,
                    &mut on_tool_calls,
                    &mut on_reasoning,
                    &mut on_tool_calls_partial,
                    &mut on_content_block_complete,
                    &mut on_usage,
                )
                .await
            } else {
                self.parse_non_streaming_response(
                    response,
                    &mut on_content,
                    &mut on_tool_calls,
                    &mut on_reasoning,
                    &mut on_tool_calls_partial,
                    &mut on_content_block_complete,
                    &mut on_usage,
                )
                .await
            };

            match result {
                Ok(()) => return Ok(()),
                Err(error) => {
                    let reqwest_retryable = error
                        .chain()
                        .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
                        .map(Self::should_retry_reqwest_error)
                        .unwrap_or(false);

                    if attempt < max_attempts && reqwest_retryable {
                        let wait = Self::compute_retry_delay(&retry_config, attempt, None);
                        warn!(
                            target: "vertex",
                            attempt,
                            max_attempts,
                            wait_secs = wait.as_secs_f64(),
                            error = %error,
                            "Vertex streaming parse failed, retrying"
                        );
                        sleep(wait).await;
                        continue;
                    }

                    return Err(error);
                }
            }
        }
    }

    fn provider_name(&self) -> &str {
        "vertex"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_partial_args_reconstructs_nested_json() {
        let mut state = StreamState::default();
        let idx = state.ensure_pending_call(Some("search_docs"), Some("sig-partial".to_string()));

        state.apply_partial_args(
            idx,
            &[
                VertexPartialArg {
                    json_path: Some("$.query".to_string()),
                    string_value: Some("rust ".to_string()),
                    ..Default::default()
                },
                VertexPartialArg {
                    json_path: Some("$.query".to_string()),
                    string_value: Some("streaming".to_string()),
                    ..Default::default()
                },
                VertexPartialArg {
                    json_path: Some("$.filters[0].name".to_string()),
                    string_value: Some("provider".to_string()),
                    ..Default::default()
                },
                VertexPartialArg {
                    json_path: Some("$.filters[0].enabled".to_string()),
                    bool_value: Some(true),
                    ..Default::default()
                },
            ],
        );

        assert_eq!(
            state.pending_tool_calls[idx].args,
            serde_json::json!({
                "query": "rust streaming",
                "filters": [
                    {
                        "name": "provider",
                        "enabled": true
                    }
                ]
            })
        );
    }

    #[test]
    fn test_handle_function_call_partial_streaming_lifecycle() -> Result<()> {
        let client = VertexClient::new(VertexConfig::default())?;
        let mut state = StreamState::default();
        let mut partial_events: Vec<Vec<UnifiedToolCall>> = Vec::new();

        client.handle_function_call(
            &VertexFunctionCall {
                name: Some("run_query".to_string()),
                partial_args: Some(vec![VertexPartialArg {
                    json_path: Some("$.sql".to_string()),
                    string_value: Some("SELECT ".to_string()),
                    ..Default::default()
                }]),
                will_continue: Some(true),
                ..Default::default()
            },
            Some("sig-vertex-call".to_string()),
            &mut state,
            &mut |calls| {
                partial_events.push(calls.to_vec());
                Ok(())
            },
        )?;

        client.handle_function_call(
            &VertexFunctionCall {
                name: Some("run_query".to_string()),
                partial_args: Some(vec![VertexPartialArg {
                    json_path: Some("$.sql".to_string()),
                    string_value: Some("1".to_string()),
                    ..Default::default()
                }]),
                will_continue: Some(false),
                ..Default::default()
            },
            None,
            &mut state,
            &mut |calls| {
                partial_events.push(calls.to_vec());
                Ok(())
            },
        )?;

        assert_eq!(partial_events.len(), 2);
        assert_eq!(state.finalized_tool_calls.len(), 1);
        assert_eq!(state.finalized_tool_calls[0].name, "run_query");
        assert_eq!(
            state.finalized_tool_calls[0].input,
            serde_json::json!({"sql": "SELECT 1"})
        );

        assert!(matches!(
            state.content_blocks.first(),
            Some(UnifiedContentBlock::ToolUse { name, .. }) if name == "run_query"
        ));
        assert!(matches!(
            state.content_blocks.get(1),
            Some(UnifiedContentBlock::Thinking { signature, .. })
                if signature.as_deref() == Some("sig-vertex-call")
        ));

        Ok(())
    }

    #[test]
    fn test_process_payload_handles_multiple_json_documents() -> Result<()> {
        let client = VertexClient::new(VertexConfig::default())?;
        let mut state = StreamState::default();
        let mut content = String::new();

        let first = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "hello "}]
                }
            }]
        });
        let second = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "vertex"}]
                },
                "finishReason": "STOP"
            }]
        });
        let payload = format!("{}\n{}", first, second);

        client.process_payload(
            &payload,
            &mut state,
            &mut |chunk| {
                content.push_str(chunk);
                Ok(())
            },
            &mut |_| Ok(()),
            &mut |_| Ok(()),
        )?;

        assert_eq!(content, "hello vertex");
        assert_eq!(state.content_blocks.len(), 1);
        assert!(matches!(
            state.content_blocks.first(),
            Some(UnifiedContentBlock::Text { text }) if text == "hello vertex"
        ));

        Ok(())
    }

    #[test]
    fn test_find_sse_event_boundary_supports_lf_and_crlf() {
        assert_eq!(find_sse_event_boundary("data:a\n\ndata:b"), Some((6, 2)));
        assert_eq!(
            find_sse_event_boundary("data:a\r\n\r\ndata:b"),
            Some((6, 4))
        );
    }
}
