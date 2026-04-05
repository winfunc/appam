//! Rich error types with enhanced context
//!
//! This module provides detailed error types that help developers quickly identify
//! and fix issues with tool execution, parameter mismatches, and other failures.
//!
//! # Examples
//!
//! ```
//! use appam::agent::errors::{ToolExecutionError, analyze_tool_error};
//! use anyhow::anyhow;
//! use serde_json::json;
//!
//! let args = json!({"path": "file.txt"});  // Wrong field name!
//! let error = anyhow!("Missing field `file_path`");
//!
//! let suggestion = analyze_tool_error("read_file", &args, &error);
//! assert!(suggestion.is_some());
//! ```

use anyhow::Error;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

/// Canonical failure kinds for agent sessions that should not be treated as silent success.
///
/// These variants are intentionally narrow and stable so downstream orchestration
/// layers can make retry and DLQ decisions without string-matching provider errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionFailureKind {
    /// The model completed a turn without emitting visible assistant text,
    /// tool calls, or non-thinking content blocks.
    BlankAssistantResponse,
    /// The model exhausted continuation attempts without calling a required
    /// completion tool.
    RequiredCompletionToolMissing,
}

impl fmt::Display for SessionFailureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlankAssistantResponse => write!(f, "blank_assistant_response"),
            Self::RequiredCompletionToolMissing => {
                write!(f, "required_completion_tool_missing")
            }
        }
    }
}

/// Structured runtime failure used for deterministic retry and diagnostics handling.
///
/// Wrapping these failures in a concrete error type avoids string matching in the
/// runtime, orchestrators, and DLQ persistence layers.
#[derive(Debug)]
pub struct SessionFailureError {
    /// Machine-readable failure classification.
    pub kind: SessionFailureKind,
    /// Human-readable explanation for logs and operators.
    pub message: String,
}

impl SessionFailureError {
    /// Create a new structured session failure.
    pub fn new(kind: SessionFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for SessionFailureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SessionFailureError {}

/// Extract a structured session failure kind from an `anyhow::Error` chain.
///
/// Returns `None` when the error does not contain a wrapped `SessionFailureError`.
pub fn extract_session_failure_kind(error: &Error) -> Option<SessionFailureKind> {
    error.chain().find_map(|cause| {
        cause
            .downcast_ref::<SessionFailureError>()
            .map(|failure| failure.kind)
    })
}

/// Rich error type for tool execution failures
///
/// Provides detailed context about what went wrong, including:
/// - Tool name and call ID
/// - Arguments that were provided
/// - The underlying error
/// - Helpful suggestions for fixing the issue
#[derive(Debug)]
pub struct ToolExecutionError {
    /// Name of the tool that failed
    pub tool_name: String,
    /// Unique call ID (if available)
    pub call_id: Option<String>,
    /// Arguments provided to the tool
    pub arguments: Value,
    /// The underlying error
    pub error: Error,
    /// Helpful suggestion for fixing the error
    pub suggestion: Option<String>,
}

impl ToolExecutionError {
    /// Create a new tool execution error with automatic suggestion generation
    pub fn new(tool_name: impl Into<String>, args: Value, error: Error) -> Self {
        let tool_name = tool_name.into();
        let suggestion = analyze_tool_error(&tool_name, &args, &error);

        Self {
            tool_name,
            call_id: None,
            arguments: args,
            error,
            suggestion,
        }
    }

    /// Set the call ID
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        self.call_id = Some(call_id.into());
        self
    }

    /// Set a custom suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

impl fmt::Display for ToolExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tool execution failed")?;
        writeln!(f, "  ├─ Tool: {}", self.tool_name)?;

        if let Some(call_id) = &self.call_id {
            writeln!(f, "  ├─ Call ID: {}", call_id)?;
        }

        // Pretty print arguments (truncate if too long)
        let args_str = serde_json::to_string_pretty(&self.arguments)
            .unwrap_or_else(|_| format!("{:?}", self.arguments));

        if args_str.len() > 200 {
            writeln!(f, "  ├─ Arguments: {}...", &args_str[..200])?;
            writeln!(f, "  │  (truncated, {} bytes total)", args_str.len())?;
        } else {
            writeln!(f, "  ├─ Arguments: {}", args_str)?;
        }

        writeln!(f, "  └─ Reason: {}", self.error)?;

        if let Some(suggestion) = &self.suggestion {
            writeln!(f)?;
            writeln!(f, "  Help: {}", suggestion)?;
        }

        Ok(())
    }
}

impl std::error::Error for ToolExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

/// Analyze tool execution error and provide helpful suggestions
///
/// This function examines the error message and arguments to provide
/// context-aware suggestions for fixing common issues.
///
/// # Examples
///
/// ```
/// use appam::agent::errors::analyze_tool_error;
/// use anyhow::anyhow;
/// use serde_json::json;
///
/// // Missing field
/// let args = json!({"path": "file.txt"});
/// let error = anyhow!("Missing field `file_path`");
/// let suggestion = analyze_tool_error("read_file", &args, &error);
/// assert!(suggestion.unwrap().contains("file_path"));
///
/// // Type mismatch
/// let args2 = json!({"count": "not a number"});
/// let error2 = anyhow!("invalid type: string, expected number");
/// let suggestion2 = analyze_tool_error("counter", &args2, &error2);
/// assert!(suggestion2.unwrap().contains("type mismatch"));
/// ```
pub fn analyze_tool_error(tool_name: &str, args: &Value, error: &Error) -> Option<String> {
    let error_str = error.to_string().to_lowercase();

    // Missing required field
    if error_str.contains("missing field") {
        if let Some(field) = extract_field_name(&error_str) {
            // Check if similar field exists in args
            if let Some(similar) = find_similar_field(args, &field) {
                return Some(format!(
                    "The tool '{}' expects field '{}' but received '{}'. \n        \
                     Check the tool schema or ensure the LLM provides the correct field name.",
                    tool_name, field, similar
                ));
            }
            return Some(format!(
                "The tool '{}' requires field '{}' which was not provided. \n        \
                 Ensure the LLM includes all required parameters.",
                tool_name, field
            ));
        }
    }

    // Type mismatch
    if error_str.contains("invalid type") || error_str.contains("type mismatch") {
        return Some(format!(
            "Parameter type mismatch for tool '{}'. \n        \
             Ensure the LLM provides correct types (string, number, boolean, object, array). \n        \
             Check the tool's parameter schema and LLM prompt.",
            tool_name
        ));
    }

    // Deserialization error
    if error_str.contains("failed to parse") || error_str.contains("deserialization") {
        return Some(format!(
            "Failed to parse parameters for tool '{}'. \n        \
             The LLM may have provided malformed JSON or incorrect parameter structure. \n        \
             Review the tool schema and improve the system prompt.",
            tool_name
        ));
    }

    // File not found
    if error_str.contains("no such file") || error_str.contains("not found") {
        return Some(format!(
            "Tool '{}' could not find the specified file or resource. \n        \
             Verify the path is correct and the file exists. \n        \
             Consider adding file existence checks in your tool implementation.",
            tool_name
        ));
    }

    // Permission denied
    if error_str.contains("permission denied") || error_str.contains("access denied") {
        return Some(format!(
            "Tool '{}' encountered a permission error. \n        \
             Check file permissions or consider running with appropriate privileges. \n        \
             You may need to add permission checks or user confirmation.",
            tool_name
        ));
    }

    None
}

/// Extract field name from error message like "missing field `field_name`"
fn extract_field_name(error: &str) -> Option<String> {
    // Common patterns:
    // - "missing field `field_name`"
    // - "missing field 'field_name'"
    // - "missing field \"field_name\""

    if let Some(start) = error.find('`') {
        if let Some(end) = error[start + 1..].find('`') {
            return Some(error[start + 1..start + 1 + end].to_string());
        }
    }

    if let Some(start) = error.find('\'') {
        if let Some(end) = error[start + 1..].find('\'') {
            return Some(error[start + 1..start + 1 + end].to_string());
        }
    }

    if let Some(start) = error.find('"') {
        if let Some(end) = error[start + 1..].find('"') {
            return Some(error[start + 1..start + 1 + end].to_string());
        }
    }

    None
}

/// Find a similar field name in the arguments (for typo detection)
fn find_similar_field(args: &Value, target: &str) -> Option<String> {
    if let Some(obj) = args.as_object() {
        for key in obj.keys() {
            // Check if one name contains the other (common pattern: file_path vs path)
            if target.contains(key) || key.contains(target) {
                return Some(key.clone());
            }

            let distance = levenshtein_distance(key, target);
            // If distance is small (1-2 edits), it's likely a typo
            if distance > 0 && distance <= 2 {
                return Some(key.clone());
            }
        }
    }
    None
}

/// Calculate Levenshtein distance between two strings
///
/// Used for detecting typos in field names.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let len_a = a_chars.len();
    let len_b = b_chars.len();

    if len_a == 0 {
        return len_b;
    }
    if len_b == 0 {
        return len_a;
    }

    let mut matrix = vec![vec![0; len_b + 1]; len_a + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(len_a + 1) {
        row[0] = i;
    }
    for (j, value) in matrix[0].iter_mut().enumerate().take(len_b + 1) {
        *value = j;
    }

    for i in 1..=len_a {
        for j in 1..=len_b {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = std::cmp::min(
                std::cmp::min(
                    matrix[i - 1][j] + 1, // deletion
                    matrix[i][j - 1] + 1, // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );
        }
    }

    matrix[len_a][len_b]
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use serde_json::json;

    #[test]
    fn test_extract_field_name() {
        assert_eq!(
            extract_field_name("missing field `file_path`"),
            Some("file_path".to_string())
        );
        assert_eq!(
            extract_field_name("missing field 'username'"),
            Some("username".to_string())
        );
        assert_eq!(
            extract_field_name("missing field \"count\""),
            Some("count".to_string())
        );
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("file_path", "filepath"), 1);
        assert_eq!(levenshtein_distance("file_path", "file_pat"), 1);
        assert_eq!(levenshtein_distance("count", "cont"), 1);
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("hello", "world"), 4);
    }

    #[test]
    fn test_find_similar_field() {
        let args = json!({
            "path": "file.txt",
            "count": 10,
        });

        assert_eq!(
            find_similar_field(&args, "file_path"),
            Some("path".to_string())
        );
        assert_eq!(find_similar_field(&args, "cont"), Some("count".to_string()));
        assert_eq!(find_similar_field(&args, "totally_different"), None);
    }

    #[test]
    fn test_analyze_missing_field() {
        let args = json!({"path": "file.txt"});
        let error = anyhow!("Missing field `file_path`");
        let suggestion = analyze_tool_error("read_file", &args, &error);

        assert!(suggestion.is_some());
        let msg = suggestion.unwrap();
        assert!(msg.contains("file_path"));
        assert!(msg.contains("path"));
    }

    #[test]
    fn test_analyze_type_mismatch() {
        let args = json!({"count": "not a number"});
        let error = anyhow!("invalid type: string \"not a number\", expected number");
        let suggestion = analyze_tool_error("counter", &args, &error);

        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("type mismatch"));
    }

    #[test]
    fn test_tool_execution_error_display() {
        let args = json!({"path": "test.txt"});
        let error = anyhow!("Missing field `file_path`");
        let err = ToolExecutionError::new("read_file", args, error).with_call_id("abc123");

        let display = format!("{}", err);
        assert!(display.contains("Tool: read_file"));
        assert!(display.contains("Call ID: abc123"));
        assert!(display.contains("Arguments"));
        assert!(display.contains("Help:"));
    }

    #[test]
    fn test_extract_session_failure_kind_from_error_chain() {
        let error = anyhow::Error::new(SessionFailureError::new(
            SessionFailureKind::RequiredCompletionToolMissing,
            "required completion tool missing",
        ));

        assert_eq!(
            extract_session_failure_kind(&error),
            Some(SessionFailureKind::RequiredCompletionToolMissing)
        );
    }
}
