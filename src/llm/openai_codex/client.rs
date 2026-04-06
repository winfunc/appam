//! OpenAI Codex subscription client with SSE streaming support.
//!
//! This client targets the ChatGPT Codex backend rather than the public
//! `api.openai.com` Responses API. The high-level callback contract matches the
//! rest of Appam's providers, but the transport uses Codex-specific headers and
//! slightly different streaming event semantics.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER};
use reqwest::StatusCode;
use serde_json::Value;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

use super::auth::resolve_openai_codex_auth;
use super::config::{resolve_reasoning_effort_for_codex_model, OpenAICodexConfig};
use crate::llm::openai::convert::{
    extract_instructions, from_unified_messages, from_unified_tools,
};
use crate::llm::openai::streaming::is_chunk_error_recoverable;
use crate::llm::openai::types::{Reasoning, ResponseCreateParams, ResponseTextConfig, ToolChoice};
use crate::llm::openai::{
    model_supports_sampling_parameters, normalize_openai_model, ReasoningEffort, RetryConfig,
    TextVerbosity,
};
use crate::llm::provider::{LlmClient, ProviderFailureCapture};
use crate::llm::unified::{UnifiedContentBlock, UnifiedMessage, UnifiedTool, UnifiedToolCall};

/// OpenAI Codex client implementation.
///
/// The client is intentionally separate from `OpenAIClient` because the Codex
/// backend uses ChatGPT OAuth authentication and a different request envelope.
#[derive(Debug, Clone)]
pub struct OpenAICodexClient {
    http_client: reqwest::Client,
    config: OpenAICodexConfig,
    session_id: String,
    latest_response_id: Arc<Mutex<Option<String>>>,
    last_failed_exchange: Arc<Mutex<Option<ProviderFailureCapture>>>,
}

impl OpenAICodexClient {
    /// Create a new OpenAI Codex client.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration is invalid or the HTTP client
    /// cannot be created.
    pub fn new(config: OpenAICodexConfig) -> Result<Self> {
        config.validate()?;

        let http_client = crate::http::client_pool::get_or_init_client(&config.base_url, |ctx| {
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
                .context("Failed to create OpenAI Codex HTTP client")
        })?;

        Ok(Self {
            http_client,
            config,
            session_id: uuid::Uuid::new_v4().to_string(),
            latest_response_id: Arc::new(Mutex::new(None)),
            last_failed_exchange: Arc::new(Mutex::new(None)),
        })
    }

    /// Return the latest response identifier observed from Codex.
    pub fn latest_response_id(&self) -> Option<String> {
        self.latest_response_id
            .lock()
            .expect("openai codex latest_response_id mutex poisoned")
            .clone()
    }

    /// Retrieve and clear the most recent failed provider exchange.
    pub fn take_last_failed_exchange(&self) -> Option<ProviderFailureCapture> {
        self.last_failed_exchange
            .lock()
            .expect("openai codex last_failed_exchange mutex poisoned")
            .take()
    }

    fn clear_last_failed_exchange(&self) {
        *self
            .last_failed_exchange
            .lock()
            .expect("openai codex last_failed_exchange mutex poisoned") = None;
    }

    fn record_failed_exchange(
        &self,
        http_status: Option<StatusCode>,
        request_payload: &str,
        response_payload: impl Into<String>,
    ) {
        let capture = ProviderFailureCapture {
            provider: "openai-codex".to_string(),
            model: normalize_openai_model(&self.config.model),
            http_status: http_status.map(|status| status.as_u16()),
            request_payload: request_payload.to_string(),
            response_payload: response_payload.into(),
            provider_response_id: self.latest_response_id(),
        };

        *self
            .last_failed_exchange
            .lock()
            .expect("openai codex last_failed_exchange mutex poisoned") = Some(capture);
    }

    fn build_endpoint_url(&self) -> String {
        let trimmed = self.config.base_url.trim_end_matches('/');
        format!("{trimmed}/codex/responses")
    }

    async fn build_headers(&self) -> Result<HeaderMap> {
        let resolved_auth =
            resolve_openai_codex_auth(self.config.access_token.as_deref(), &self.config.auth_file)
                .await?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", resolved_auth.access_token))
                .context("Invalid OpenAI Codex authorization header")?,
        );
        headers.insert(
            "chatgpt-account-id",
            HeaderValue::from_str(&resolved_auth.account_id)
                .context("Invalid OpenAI Codex chatgpt-account-id header")?,
        );
        headers.insert(
            "originator",
            HeaderValue::from_str(&self.config.originator)
                .context("Invalid OpenAI Codex originator header")?,
        );
        headers.insert(
            "OpenAI-Beta",
            HeaderValue::from_static("responses=experimental"),
        );
        headers.insert(
            "session_id",
            HeaderValue::from_str(&self.session_id).context("Invalid OpenAI Codex session_id")?,
        );

        Ok(headers)
    }

    fn build_request_body(
        &self,
        messages: &[UnifiedMessage],
        tools: &[UnifiedTool],
    ) -> Result<ResponseCreateParams> {
        let instructions = extract_instructions(messages);
        let non_system_messages = prepare_codex_messages(messages);
        let input = from_unified_messages(&non_system_messages, None);

        let normalized_model = normalize_openai_model(&self.config.model);
        let requested_effort = self.reasoning_effort();
        let resolved_effort =
            self.config.reasoning.as_ref().map(|_| {
                resolve_reasoning_effort_for_codex_model(&normalized_model, requested_effort)
            });
        let sampling_supported =
            model_supports_sampling_parameters(&normalized_model, resolved_effort);

        Ok(ResponseCreateParams {
            model: normalized_model,
            input: Some(input),
            instructions,
            // The ChatGPT Codex backend currently rejects `max_output_tokens`.
            // Keep the config field for API parity, but omit it from the wire
            // format until Codex supports it.
            max_output_tokens: None,
            tools: if tools.is_empty() {
                None
            } else {
                Some(from_unified_tools(tools))
            },
            tool_choice: if tools.is_empty() {
                None
            } else {
                Some(ToolChoice::String("auto".to_string()))
            },
            parallel_tool_calls: Some(self.config.parallel_tool_calls.unwrap_or(false)),
            max_tool_calls: None,
            temperature: sampling_supported
                .then_some(self.config.temperature)
                .flatten(),
            top_p: sampling_supported.then_some(self.config.top_p).flatten(),
            stream: Some(self.config.stream),
            stream_options: None,
            text: self
                .config
                .text_verbosity
                .map(|verbosity| ResponseTextConfig {
                    format: None,
                    verbosity: Some(map_text_verbosity(verbosity)),
                }),
            reasoning: self.config.reasoning.as_ref().map(|reasoning| Reasoning {
                effort: resolved_effort.map(reasoning_effort_to_string),
                summary: reasoning.summary.map(reasoning_summary_to_string),
            }),
            service_tier: None,
            conversation: None,
            previous_response_id: None,
            background: None,
            store: Some(false),
            include: Some(vec!["reasoning.encrypted_content".to_string()]),
            truncation: None,
            top_logprobs: None,
            metadata: None,
            prompt_cache_key: Some(self.session_id.clone()),
            safety_identifier: None,
        })
    }

    fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.config
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort)
    }

    fn retry_config(&self) -> RetryConfig {
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
        if error.is_timeout() || error.is_connect() || error.is_request() || error.is_body() {
            return true;
        }

        let error_msg = error.to_string().to_ascii_lowercase();
        error_msg.contains("dns error")
            || error_msg.contains("failed to lookup address")
            || error_msg.contains("nodename nor servname provided")
    }

    fn retry_after_from_headers(headers: &HeaderMap) -> Option<Duration> {
        headers
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|raw| raw.parse::<u64>().ok())
            .map(Duration::from_secs)
    }

    fn compute_retry_delay(
        retry_config: &RetryConfig,
        attempt: u32,
        retry_after: Option<Duration>,
    ) -> Duration {
        if let Some(delay) = retry_after {
            let max_backoff = Duration::from_millis(retry_config.max_backoff_ms);
            return std::cmp::min(delay, max_backoff);
        }

        Duration::from_millis(retry_config.calculate_backoff(attempt))
    }

    #[allow(clippy::too_many_arguments)]
    async fn process_stream<FContent, FTool, FReason, FToolPartial, FContentBlock, FUsage>(
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
        let mut pending_bytes = Vec::new();
        let mut buffer = String::new();
        let mut state = CodexStreamState::default();

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    let anyhow_error = anyhow!(error);
                    if is_chunk_error_recoverable(&anyhow_error) {
                        warn!("OpenAI Codex stream ended with a recoverable chunk error");
                        break;
                    }
                    return Err(anyhow_error).context("OpenAI Codex stream read failed");
                }
            };

            pending_bytes.extend_from_slice(&chunk);

            match std::str::from_utf8(&pending_bytes) {
                Ok(valid_str) => {
                    buffer.push_str(valid_str);
                    pending_bytes.clear();
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0 {
                        let valid_str = std::str::from_utf8(&pending_bytes[..valid_up_to])
                            .expect("valid UTF-8 prefix");
                        buffer.push_str(valid_str);
                        pending_bytes.drain(..valid_up_to);
                    }

                    if error.error_len().is_some() {
                        bail!(
                            "Invalid UTF-8 in OpenAI Codex stream: encountered invalid byte sequence"
                        );
                    }
                }
            }

            while let Some(event_end) = buffer.find("\n\n") {
                let event_data = buffer[..event_end].to_string();
                buffer = buffer[event_end + 2..].to_string();

                let mut data_payload = String::new();
                for line in event_data.lines() {
                    if let Some(rest) = line.strip_prefix("data: ") {
                        if !data_payload.is_empty() {
                            data_payload.push('\n');
                        }
                        data_payload.push_str(rest);
                    }
                }

                if data_payload.is_empty() || data_payload == "[DONE]" {
                    continue;
                }

                let payload: Value = match serde_json::from_str(&data_payload) {
                    Ok(payload) => payload,
                    Err(error) => {
                        debug!(
                            error = %error,
                            payload = %data_payload,
                            "Skipping unparsable OpenAI Codex SSE payload"
                        );
                        continue;
                    }
                };

                let action = handle_codex_payload(
                    &payload,
                    &mut state,
                    on_content,
                    on_reasoning,
                    on_tool_calls_partial,
                )?;

                match action {
                    CodexPayloadAction::Continue => {}
                    CodexPayloadAction::Terminal => {
                        state.saw_terminal = true;
                        break;
                    }
                }
            }

            if state.saw_terminal {
                break;
            }
        }

        if !state.saw_terminal {
            bail!("OpenAI Codex stream ended before a terminal response event");
        }

        *self
            .latest_response_id
            .lock()
            .expect("openai codex latest_response_id mutex poisoned") =
            state.latest_response_id.clone();

        if !state.completed_tool_calls.is_empty() {
            on_tool_calls(state.completed_tool_calls.clone())?;
        }

        if let Some(usage) = state.usage.clone() {
            on_usage(usage)?;
        }

        for block in state.into_content_blocks() {
            on_content_block_complete(block)?;
        }

        Ok(())
    }
}

#[async_trait]
impl LlmClient for OpenAICodexClient {
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
        *self
            .latest_response_id
            .lock()
            .expect("openai codex latest_response_id mutex poisoned") = None;
        self.clear_last_failed_exchange();

        let request_body = self.build_request_body(messages, tools)?;
        let request_payload = serde_json::to_string_pretty(&request_body)?;
        let retry_config = self.retry_config();
        let max_attempts = retry_config.max_retries.saturating_add(1).max(1);

        let mut on_content = on_content;
        let mut on_tool_calls = on_tool_calls;
        let mut on_reasoning = on_reasoning;
        let mut on_tool_calls_partial = on_tool_calls_partial;
        let mut on_content_block_complete = on_content_block_complete;
        let mut on_usage = on_usage;

        let headers = self.build_headers().await?;

        for attempt in 1..=max_attempts {
            debug!(
                attempt = attempt,
                max_attempts = max_attempts,
                "Sending OpenAI Codex request"
            );

            let response = match self
                .http_client
                .post(self.build_endpoint_url())
                .headers(headers.clone())
                .json(&request_body)
                .send()
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    if attempt < max_attempts && Self::should_retry_reqwest_error(&error) {
                        let wait = Self::compute_retry_delay(&retry_config, attempt, None);
                        warn!(
                            attempt = attempt,
                            max_attempts = max_attempts,
                            wait_secs = wait.as_secs_f64(),
                            error = %error,
                            "OpenAI Codex request failed, retrying after backoff"
                        );
                        sleep(wait).await;
                        continue;
                    }

                    self.record_failed_exchange(None, &request_payload, error.to_string());
                    return Err(error).context("OpenAI Codex request failed");
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let response_headers = response.headers().clone();
                let body = response.text().await.unwrap_or_default();
                error!(
                    status = %status,
                    attempt = attempt,
                    body = %body,
                    "OpenAI Codex error response"
                );

                if attempt < max_attempts && Self::should_retry_status(status) {
                    let retry_after = Self::retry_after_from_headers(&response_headers);
                    let wait = Self::compute_retry_delay(&retry_config, attempt, retry_after);
                    info!(
                        attempt = attempt,
                        max_attempts = max_attempts,
                        wait_secs = wait.as_secs_f64(),
                        status = %status,
                        "Retrying OpenAI Codex request after API error"
                    );
                    sleep(wait).await;
                    continue;
                }

                let friendly = parse_error_response(status, &body);
                self.record_failed_exchange(Some(status), &request_payload, body);
                return Err(anyhow!(friendly));
            }

            let processing_result = self
                .process_stream(
                    response,
                    &mut on_content,
                    &mut on_tool_calls,
                    &mut on_reasoning,
                    &mut on_tool_calls_partial,
                    &mut on_content_block_complete,
                    &mut on_usage,
                )
                .await;

            match processing_result {
                Ok(()) => return Ok(()),
                Err(error) => {
                    if attempt < max_attempts {
                        let wait = Self::compute_retry_delay(&retry_config, attempt, None);
                        warn!(
                            attempt = attempt,
                            max_attempts = max_attempts,
                            wait_secs = wait.as_secs_f64(),
                            error = %error,
                            "OpenAI Codex streaming failed, retrying"
                        );
                        sleep(wait).await;
                        continue;
                    }

                    self.record_failed_exchange(None, &request_payload, format!("{error:#}"));
                    return Err(error).context("OpenAI Codex stream processing failed");
                }
            }
        }

        Err(anyhow!("OpenAI Codex request exceeded retry budget"))
    }

    fn provider_name(&self) -> &str {
        "openai-codex"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodexPayloadAction {
    Continue,
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
enum BlockKey {
    Text(i32, i32),
    Thinking(i32, i32),
    ToolUse(i32, usize),
}

#[derive(Debug, Default)]
struct CodexStreamState {
    saw_terminal: bool,
    latest_response_id: Option<String>,
    text_buffers: BTreeMap<(i32, i32), String>,
    text_streamed_keys: HashSet<(i32, i32)>,
    reasoning_buffers: BTreeMap<(i32, i32), String>,
    reasoning_streamed_keys: HashSet<(i32, i32)>,
    function_arg_buffers: HashMap<String, String>,
    function_call_meta: HashMap<String, (String, String, i32)>,
    completed_tool_calls: Vec<UnifiedToolCall>,
    block_order: Vec<BlockKey>,
    block_set: HashSet<BlockKey>,
    usage: Option<crate::llm::unified::UnifiedUsage>,
}

impl CodexStreamState {
    fn register_block(&mut self, block: BlockKey) {
        if self.block_set.insert(block) {
            self.block_order.push(block);
        }
    }

    fn into_content_blocks(self) -> Vec<UnifiedContentBlock> {
        let mut blocks = Vec::new();

        for block in self.block_order {
            match block {
                BlockKey::Text(output_index, content_index) => {
                    if let Some(text) = self.text_buffers.get(&(output_index, content_index)) {
                        if !text.is_empty() {
                            blocks.push(UnifiedContentBlock::Text { text: text.clone() });
                        }
                    }
                }
                BlockKey::Thinking(output_index, content_index) => {
                    if let Some(thinking) =
                        self.reasoning_buffers.get(&(output_index, content_index))
                    {
                        if !thinking.is_empty() {
                            blocks.push(UnifiedContentBlock::Thinking {
                                thinking: thinking.clone(),
                                signature: None,
                                encrypted_content: None,
                                redacted: false,
                            });
                        }
                    }
                }
                BlockKey::ToolUse(_, index) => {
                    if let Some(tool_call) = self.completed_tool_calls.get(index) {
                        blocks.push(UnifiedContentBlock::ToolUse {
                            id: tool_call.id.clone(),
                            name: tool_call.name.clone(),
                            input: tool_call.input.clone(),
                        });
                    }
                }
            }
        }

        blocks
    }
}

fn handle_codex_payload<FContent, FReason, FToolPartial>(
    payload: &Value,
    state: &mut CodexStreamState,
    on_content: &mut FContent,
    on_reasoning: &mut FReason,
    on_tool_calls_partial: &mut FToolPartial,
) -> Result<CodexPayloadAction>
where
    FContent: FnMut(&str) -> Result<()> + Send,
    FReason: FnMut(&str) -> Result<()> + Send,
    FToolPartial: FnMut(&[UnifiedToolCall]) -> Result<()> + Send,
{
    let Some(event_type) = payload.get("type").and_then(Value::as_str) else {
        return Ok(CodexPayloadAction::Continue);
    };

    match event_type {
        "error" | "response.error" => {
            let message = payload
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .or_else(|| payload.get("message").and_then(Value::as_str))
                .unwrap_or("OpenAI Codex stream returned an error event");
            bail!("{message}");
        }
        "response.failed" => {
            let message = payload
                .get("response")
                .and_then(|response| response.get("error"))
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("OpenAI Codex response failed");
            bail!("{message}");
        }
        "response.output_item.added" => {
            if let Some(item) = payload.get("item") {
                let output_index = payload
                    .get("output_index")
                    .and_then(Value::as_i64)
                    .unwrap_or(0) as i32;

                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    let item_id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    if !item_id.is_empty() {
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        state
                            .function_call_meta
                            .insert(item_id, (call_id, name, output_index));
                    }
                }
            }
        }
        "response.output_text.delta" => {
            let delta = payload
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let output_index = payload
                .get("output_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let content_index = payload
                .get("content_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let key = (output_index, content_index);
            state.register_block(BlockKey::Text(output_index, content_index));
            state.text_buffers.entry(key).or_default().push_str(delta);
            state.text_streamed_keys.insert(key);
            on_content(delta)?;
        }
        "response.output_text.done" => {
            let text = payload
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let output_index = payload
                .get("output_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let content_index = payload
                .get("content_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let key = (output_index, content_index);
            state.register_block(BlockKey::Text(output_index, content_index));
            let buffer = state.text_buffers.entry(key).or_default();
            if buffer.is_empty() {
                buffer.push_str(text);
            }
            if !text.is_empty() && !state.text_streamed_keys.contains(&key) {
                on_content(text)?;
                state.text_streamed_keys.insert(key);
            }
        }
        "response.reasoning_text.delta" => {
            let delta = payload
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let output_index = payload
                .get("output_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let content_index = payload
                .get("content_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let key = (output_index, content_index);
            state.register_block(BlockKey::Thinking(output_index, content_index));
            state
                .reasoning_buffers
                .entry(key)
                .or_default()
                .push_str(delta);
            state.reasoning_streamed_keys.insert(key);
            on_reasoning(delta)?;
        }
        "response.reasoning_text.done" => {
            let text = payload
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let output_index = payload
                .get("output_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let content_index = payload
                .get("content_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let key = (output_index, content_index);
            state.register_block(BlockKey::Thinking(output_index, content_index));
            let buffer = state.reasoning_buffers.entry(key).or_default();
            if buffer.is_empty() {
                buffer.push_str(text);
            }
            if !text.is_empty() && !state.reasoning_streamed_keys.contains(&key) {
                on_reasoning(text)?;
                state.reasoning_streamed_keys.insert(key);
            }
        }
        "response.reasoning_summary_text.delta" => {
            let delta = payload
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let output_index = payload
                .get("output_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let summary_index = payload
                .get("summary_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let key = (output_index, 10_000 + summary_index);
            state.register_block(BlockKey::Thinking(output_index, 10_000 + summary_index));
            state
                .reasoning_buffers
                .entry(key)
                .or_default()
                .push_str(delta);
            state.reasoning_streamed_keys.insert(key);
            on_reasoning(delta)?;
        }
        "response.reasoning_summary_text.done" => {
            let text = payload
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let output_index = payload
                .get("output_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let summary_index = payload
                .get("summary_index")
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32;
            let key = (output_index, 10_000 + summary_index);
            state.register_block(BlockKey::Thinking(output_index, 10_000 + summary_index));
            let buffer = state.reasoning_buffers.entry(key).or_default();
            if buffer.is_empty() {
                buffer.push_str(text);
            }
            if !text.is_empty() && !state.reasoning_streamed_keys.contains(&key) {
                on_reasoning(text)?;
                state.reasoning_streamed_keys.insert(key);
            }
        }
        "response.function_call_arguments.delta" => {
            let item_id = payload
                .get("item_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let delta = payload
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !item_id.is_empty() {
                let current = state
                    .function_arg_buffers
                    .entry(item_id.clone())
                    .or_default();
                current.push_str(delta);
                if let Some((call_id, name, _)) = state.function_call_meta.get(&item_id) {
                    let partial = UnifiedToolCall {
                        id: call_id.clone(),
                        name: name.clone(),
                        input: serde_json::from_str(current)
                            .unwrap_or_else(|_| serde_json::json!({})),
                        raw_input_json: Some(current.clone()),
                    };
                    on_tool_calls_partial(&[partial])?;
                }
            }
        }
        "response.function_call_arguments.done" => {
            let item_id = payload
                .get("item_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let arguments = payload
                .get("arguments")
                .and_then(Value::as_str)
                .or_else(|| {
                    state
                        .function_arg_buffers
                        .get(&item_id)
                        .map(std::string::String::as_str)
                })
                .unwrap_or("{}");
            let (call_id, name, output_index) = payload
                .get("call_id")
                .and_then(Value::as_str)
                .map(|call_id| {
                    (
                        call_id.to_string(),
                        payload
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        payload
                            .get("output_index")
                            .and_then(Value::as_i64)
                            .unwrap_or(0) as i32,
                    )
                })
                .or_else(|| state.function_call_meta.get(&item_id).cloned())
                .unwrap_or_else(|| {
                    (
                        item_id.clone(),
                        payload
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        payload
                            .get("output_index")
                            .and_then(Value::as_i64)
                            .unwrap_or(0) as i32,
                    )
                });

            let tool_call = UnifiedToolCall {
                id: call_id,
                name,
                input: serde_json::from_str(arguments).unwrap_or_else(|_| serde_json::json!({})),
                raw_input_json: Some(arguments.to_string()),
            };

            let index = state.completed_tool_calls.len();
            state.completed_tool_calls.push(tool_call);
            state.register_block(BlockKey::ToolUse(output_index, index));
        }
        "response.completed" | "response.done" | "response.incomplete" => {
            if let Some(response) = payload.get("response") {
                state.latest_response_id = response
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                state.usage = parse_usage(response);
            }
            return Ok(CodexPayloadAction::Terminal);
        }
        _ => {}
    }

    Ok(CodexPayloadAction::Continue)
}

fn parse_usage(response: &Value) -> Option<crate::llm::unified::UnifiedUsage> {
    let usage = response.get("usage")?;
    let input_tokens = usage.get("input_tokens")?.as_u64()? as u32;
    let output_tokens = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let cache_read_tokens = usage
        .get("input_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let reasoning_tokens = usage
        .get("output_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;

    Some(crate::llm::unified::UnifiedUsage {
        input_tokens,
        output_tokens,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: (cache_read_tokens > 0).then_some(cache_read_tokens),
        reasoning_tokens: (reasoning_tokens > 0).then_some(reasoning_tokens),
    })
}

fn parse_error_response(status: StatusCode, body: &str) -> String {
    let mut message = if body.trim().is_empty() {
        status.to_string()
    } else {
        body.to_string()
    };

    if let Ok(parsed) = serde_json::from_str::<Value>(body) {
        if let Some(error) = parsed.get("error") {
            let code = error
                .get("code")
                .and_then(Value::as_str)
                .or_else(|| error.get("type").and_then(Value::as_str))
                .unwrap_or_default();
            let response_message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or(message.as_str());

            if status == StatusCode::TOO_MANY_REQUESTS
                || matches!(
                    code,
                    "usage_limit_reached" | "usage_not_included" | "rate_limit_exceeded"
                )
            {
                let plan = error
                    .get("plan_type")
                    .and_then(Value::as_str)
                    .map(|plan| format!(" ({})", plan.to_lowercase()))
                    .unwrap_or_default();
                return format!("You have hit your ChatGPT usage limit{plan}. {response_message}")
                    .trim()
                    .to_string();
            }

            message = response_message.to_string();
        }
    }

    message
}

fn prepare_codex_messages(messages: &[UnifiedMessage]) -> Vec<UnifiedMessage> {
    messages
        .iter()
        .filter(|message| !matches!(message.role, crate::llm::UnifiedRole::System))
        .filter_map(|message| {
            if !matches!(message.role, crate::llm::UnifiedRole::Assistant) {
                return Some(message.clone());
            }

            let has_tool_use = message
                .content
                .iter()
                .any(|block| matches!(block, UnifiedContentBlock::ToolUse { .. }));

            let filtered_content: Vec<UnifiedContentBlock> = message
                .content
                .iter()
                .filter_map(|block| match block {
                    UnifiedContentBlock::Thinking {
                        encrypted_content: Some(encrypted_content),
                        ..
                    } => Some(UnifiedContentBlock::Thinking {
                        thinking: String::new(),
                        signature: None,
                        encrypted_content: Some(encrypted_content.clone()),
                        redacted: false,
                    }),
                    UnifiedContentBlock::Thinking { .. } => None,
                    UnifiedContentBlock::Text { .. } if has_tool_use => None,
                    _ => Some(block.clone()),
                })
                .collect();

            (!filtered_content.is_empty()).then(|| UnifiedMessage {
                role: message.role,
                content: filtered_content,
                id: message.id.clone(),
                timestamp: message.timestamp,
                reasoning: message.reasoning.clone(),
                reasoning_details: message.reasoning_details.clone(),
            })
        })
        .collect()
}

fn map_text_verbosity(verbosity: TextVerbosity) -> crate::llm::openai::types::TextVerbosity {
    match verbosity {
        TextVerbosity::Low => crate::llm::openai::types::TextVerbosity::Low,
        TextVerbosity::Medium => crate::llm::openai::types::TextVerbosity::Medium,
        TextVerbosity::High => crate::llm::openai::types::TextVerbosity::High,
    }
}

fn reasoning_effort_to_string(effort: ReasoningEffort) -> String {
    match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
    }
    .to_string()
}

fn reasoning_summary_to_string(summary: crate::llm::openai::ReasoningSummary) -> String {
    match summary {
        crate::llm::openai::ReasoningSummary::Auto => "auto",
        crate::llm::openai::ReasoningSummary::Concise => "concise",
        crate::llm::openai::ReasoningSummary::Detailed => "detailed",
    }
    .to_string()
}

use futures::StreamExt;

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;

    #[test]
    fn test_build_endpoint_url() {
        let client = OpenAICodexClient::new(OpenAICodexConfig::default()).unwrap();
        assert_eq!(
            client.build_endpoint_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    fn mock_token(account_id: &str) -> String {
        let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"none"}"#);
        let payload = URL_SAFE_NO_PAD.encode(
            format!(r#"{{"https://api.openai.com/auth":{{"chatgpt_account_id":"{account_id}"}}}}"#)
                .as_bytes(),
        );
        format!("{header}.{payload}.sig")
    }

    #[tokio::test]
    async fn test_build_headers_uses_codex_specific_fields() {
        let client = OpenAICodexClient::new(OpenAICodexConfig {
            access_token: Some(mock_token("acc_headers")),
            ..Default::default()
        })
        .unwrap();

        let headers = client.build_headers().await.unwrap();
        assert_eq!(headers.get("chatgpt-account-id").unwrap(), "acc_headers");
        assert_eq!(
            headers.get("OpenAI-Beta").unwrap(),
            "responses=experimental"
        );
        assert_eq!(headers.get("originator").unwrap(), "pi");
        assert!(headers.contains_key("session_id"));
    }

    #[test]
    fn test_build_request_body_uses_instructions_and_omits_system_messages() {
        let client = OpenAICodexClient::new(OpenAICodexConfig {
            access_token: Some(mock_token("acc_body")),
            ..Default::default()
        })
        .unwrap();
        let messages = vec![
            UnifiedMessage::system("Follow the system prompt"),
            UnifiedMessage::user("Hello"),
        ];
        let request = client.build_request_body(&messages, &[]).unwrap();
        let serialized = serde_json::to_value(&request).unwrap();

        assert_eq!(
            serialized
                .get("instructions")
                .and_then(Value::as_str)
                .unwrap(),
            "Follow the system prompt"
        );
        let input = serialized
            .get("input")
            .and_then(Value::as_array)
            .expect("structured input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0].get("role").and_then(Value::as_str), Some("user"));
        assert_eq!(
            serialized
                .get("parallel_tool_calls")
                .and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            serialized.get("store").and_then(Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn test_extract_instructions_omits_non_system_messages() {
        let messages = vec![
            UnifiedMessage::system("System A"),
            UnifiedMessage::user("Hello"),
            UnifiedMessage::system("System B"),
        ];

        assert_eq!(
            extract_instructions(&messages).as_deref(),
            Some("System A\n\nSystem B")
        );
    }

    #[test]
    fn test_prepare_codex_messages_drops_text_on_tool_turns() {
        let messages = vec![
            UnifiedMessage::assistant("Thinking aloud"),
            UnifiedMessage {
                role: crate::llm::UnifiedRole::Assistant,
                content: vec![
                    UnifiedContentBlock::Text {
                        text: "I will call a tool".to_string(),
                    },
                    UnifiedContentBlock::ToolUse {
                        id: "call_1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "mkdir -p poem_generator"}),
                    },
                ],
                id: None,
                timestamp: None,
                reasoning: None,
                reasoning_details: None,
            },
        ];

        let prepared = prepare_codex_messages(&messages);
        assert_eq!(prepared.len(), 2);
        assert!(matches!(
            prepared[0].content.as_slice(),
            [UnifiedContentBlock::Text { .. }]
        ));
        assert!(matches!(
            prepared[1].content.as_slice(),
            [UnifiedContentBlock::ToolUse { .. }]
        ));
    }

    #[test]
    fn test_parse_usage_extracts_cached_and_reasoning_tokens() {
        let usage = parse_usage(&serde_json::json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "input_tokens_details": { "cached_tokens": 25 },
                "output_tokens_details": { "reasoning_tokens": 10 }
            }
        }))
        .unwrap();

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_read_input_tokens, Some(25));
        assert_eq!(usage.reasoning_tokens, Some(10));
    }

    #[test]
    fn test_handle_codex_payload_streams_text_and_terminal_events() {
        let mut state = CodexStreamState::default();
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut partials = Vec::<Vec<UnifiedToolCall>>::new();

        let action = handle_codex_payload(
            &serde_json::json!({
                "type": "response.output_text.delta",
                "delta": "Hello"
            }),
            &mut state,
            &mut |chunk| {
                content.push_str(chunk);
                Ok(())
            },
            &mut |chunk| {
                reasoning.push_str(chunk);
                Ok(())
            },
            &mut |calls| {
                partials.push(calls.to_vec());
                Ok(())
            },
        )
        .unwrap();

        assert!(matches!(action, CodexPayloadAction::Continue));
        assert_eq!(content, "Hello");
        assert!(reasoning.is_empty());
        assert!(partials.is_empty());

        let action = handle_codex_payload(
            &serde_json::json!({
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "input_tokens_details": { "cached_tokens": 0 },
                        "output_tokens_details": { "reasoning_tokens": 0 }
                    }
                }
            }),
            &mut state,
            &mut |_| Ok(()),
            &mut |_| Ok(()),
            &mut |_| Ok(()),
        )
        .unwrap();

        assert!(matches!(action, CodexPayloadAction::Terminal));
        assert!(state.usage.is_some());
    }

    #[test]
    fn test_handle_codex_payload_builds_tool_call_completion() {
        let mut state = CodexStreamState::default();

        handle_codex_payload(
            &serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "read_file"
                }
            }),
            &mut state,
            &mut |_| Ok(()),
            &mut |_| Ok(()),
            &mut |_| Ok(()),
        )
        .unwrap();

        handle_codex_payload(
            &serde_json::json!({
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "arguments": "{\"path\":\"src/lib.rs\"}"
            }),
            &mut state,
            &mut |_| Ok(()),
            &mut |_| Ok(()),
            &mut |_| Ok(()),
        )
        .unwrap();

        assert_eq!(state.completed_tool_calls.len(), 1);
        assert_eq!(state.completed_tool_calls[0].id, "call_1");
        assert_eq!(state.completed_tool_calls[0].name, "read_file");
    }
}
