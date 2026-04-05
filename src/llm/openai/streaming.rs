//! Streaming support for OpenAI Responses API.
//!
//! Handles Server-Sent Events (SSE) parsing and stream accumulation for
//! incremental response processing.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::types::*;

/// SSE stream event types from OpenAI Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    // Response lifecycle events
    /// Response created
    #[serde(rename = "response.created")]
    ResponseCreated {
        /// Response snapshot
        response: Response,
        /// Sequence number
        sequence_number: i32,
    },

    /// Response in progress
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        /// Sequence number
        sequence_number: i32,
    },

    /// Response completed
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        /// Final response
        response: Response,
        /// Sequence number
        sequence_number: i32,
    },

    /// Response failed
    #[serde(rename = "response.failed")]
    ResponseFailed {
        /// Error details
        error: ResponseError,
        /// Sequence number
        sequence_number: i32,
    },

    /// Response incomplete
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        /// Sequence number
        sequence_number: i32,
    },

    // Output item events
    /// Output item added
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        /// Output item
        item: OutputItem,
        /// Output index
        output_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    /// Output item done
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        /// Output index
        output_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    // Content part events
    /// Content part added
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        /// Content part
        part: ContentPart,
        /// Output index
        output_index: i32,
        /// Content index
        content_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    /// Content part done
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        /// Output index
        output_index: i32,
        /// Content index
        content_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    // Text delta events
    /// Text delta
    #[serde(rename = "response.output_text.delta")]
    ResponseTextDelta {
        /// Text delta
        delta: String,
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Content index
        content_index: i32,
        /// Sequence number
        sequence_number: i32,
        /// Logprobs
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<Logprob>>,
    },

    /// Text done
    #[serde(rename = "response.output_text.done")]
    ResponseTextDone {
        /// Complete text
        text: String,
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Content index
        content_index: i32,
        /// Sequence number
        sequence_number: i32,
        /// Annotations
        #[serde(default)]
        annotations: Vec<Annotation>,
        /// Logprobs
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<Logprob>>,
    },

    // Function call events
    /// Function call arguments delta
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        /// Arguments delta
        delta: String,
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    /// Function call arguments done
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        /// Complete arguments
        arguments: String,
        /// Item ID
        item_id: String,
        /// Call ID
        #[serde(default)]
        call_id: Option<String>,
        /// Function name
        #[serde(default)]
        name: Option<String>,
        /// Output index
        output_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    // Reasoning events
    /// Reasoning summary part added
    #[serde(rename = "response.reasoning_summary_part.added")]
    ResponseReasoningSummaryPartAdded {
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Summary index
        summary_index: i32,
        /// Summary part payload
        part: Value,
        /// Sequence number
        sequence_number: i32,
    },

    /// Reasoning summary part done
    #[serde(rename = "response.reasoning_summary_part.done")]
    ResponseReasoningSummaryPartDone {
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Summary index
        summary_index: i32,
        /// Summary part payload
        part: Value,
        /// Sequence number
        sequence_number: i32,
    },

    /// Reasoning summary text delta
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ResponseReasoningSummaryTextDelta {
        /// Text delta
        delta: String,
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Summary index
        summary_index: i32,
        /// Sequence number
        sequence_number: i32,
        /// Optional obfuscation token
        #[serde(skip_serializing_if = "Option::is_none")]
        obfuscation: Option<String>,
    },

    /// Reasoning summary text done
    #[serde(rename = "response.reasoning_summary_text.done")]
    ResponseReasoningSummaryTextDone {
        /// Full summary text
        text: String,
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Summary index
        summary_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    /// Reasoning text delta
    #[serde(rename = "response.reasoning_text.delta")]
    ResponseReasoningTextDelta {
        /// Reasoning delta
        delta: String,
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Content index
        content_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    /// Reasoning text done
    #[serde(rename = "response.reasoning_text.done")]
    ResponseReasoningTextDone {
        /// Complete reasoning text
        text: String,
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Content index
        content_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    // Refusal events
    /// Refusal delta
    #[serde(rename = "response.refusal.delta")]
    ResponseRefusalDelta {
        /// Refusal delta
        delta: String,
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Content index
        content_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    /// Refusal done
    #[serde(rename = "response.refusal.done")]
    ResponseRefusalDone {
        /// Complete refusal
        refusal: String,
        /// Item ID
        item_id: String,
        /// Output index
        output_index: i32,
        /// Content index
        content_index: i32,
        /// Sequence number
        sequence_number: i32,
    },

    // Error event
    /// Response error
    #[serde(rename = "response.error")]
    ResponseError {
        /// Error details
        error: ResponseError,
        /// Sequence number
        sequence_number: i32,
    },
}

/// Stream accumulator for managing partial streaming state.
///
/// Accumulates deltas and maintains snapshots of the response as it streams.
pub struct StreamAccumulator {
    /// Current response snapshot
    pub response_snapshot: Option<Response>,

    /// Text buffers by (output_index, content_index)
    text_buffers: HashMap<(usize, usize), String>,

    /// Function argument buffers by output_index
    function_arg_buffers: HashMap<usize, String>,

    /// Reasoning buffers by (output_index, content_index)
    reasoning_buffers: HashMap<(usize, usize), String>,
}

impl StreamAccumulator {
    /// Create a new stream accumulator.
    pub fn new() -> Self {
        Self {
            response_snapshot: None,
            text_buffers: HashMap::new(),
            function_arg_buffers: HashMap::new(),
            reasoning_buffers: HashMap::new(),
        }
    }

    /// Handle a stream event and update internal state.
    ///
    /// Returns the accumulated text for the event, if applicable.
    pub fn handle_event(&mut self, event: &StreamEvent) -> Option<String> {
        match event {
            StreamEvent::ResponseCreated { response, .. } => {
                self.response_snapshot = Some(response.clone());
                None
            }
            StreamEvent::ResponseTextDelta {
                delta,
                output_index,
                content_index,
                ..
            } => {
                let key = (*output_index as usize, *content_index as usize);
                let buffer = self.text_buffers.entry(key).or_default();
                buffer.push_str(delta);
                Some(delta.clone())
            }
            StreamEvent::ResponseFunctionCallArgumentsDelta {
                delta,
                output_index,
                ..
            } => {
                let buffer = self
                    .function_arg_buffers
                    .entry(*output_index as usize)
                    .or_default();
                buffer.push_str(delta);
                None
            }
            StreamEvent::ResponseReasoningTextDelta {
                delta,
                output_index,
                content_index,
                ..
            } => {
                let key = (*output_index as usize, *content_index as usize);
                let buffer = self.reasoning_buffers.entry(key).or_default();
                buffer.push_str(delta);
                Some(delta.clone())
            }
            StreamEvent::ResponseCompleted { response, .. } => {
                self.response_snapshot = Some(response.clone());
                None
            }
            _ => None,
        }
    }

    /// Get the final response snapshot.
    pub fn get_final_response(&self) -> Option<&Response> {
        self.response_snapshot.as_ref()
    }

    /// Get accumulated text for a specific output/content index.
    pub fn get_text(&self, output_index: usize, content_index: usize) -> Option<&String> {
        self.text_buffers.get(&(output_index, content_index))
    }

    /// Get accumulated function arguments for a specific output index.
    pub fn get_function_args(&self, output_index: usize) -> Option<&String> {
        self.function_arg_buffers.get(&output_index)
    }

    /// Get accumulated reasoning for a specific output/content index.
    pub fn get_reasoning(&self, output_index: usize, content_index: usize) -> Option<&String> {
        self.reasoning_buffers.get(&(output_index, content_index))
    }
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self::new()
    }
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
pub fn is_chunk_error_recoverable(error: &anyhow::Error) -> bool {
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
