//! Tool utilities for Anthropic API.
//!
//! Helpers for creating and managing tools, including server tools.
//!
//! # Client Tools
//!
//! User-defined tools that execute on the client side.
//!
//! # Server Tools
//!
//! Anthropic-hosted tools that execute on their servers:
//! - Web Search: Search the web for information
//! - Web Fetch: Fetch and parse web page content
//! - Bash: Execute bash commands
//! - Code Execution: Run Python code
//! - Text Editor: Edit files with str_replace

use super::types::Tool;

/// Create a web search server tool.
///
/// # Parameters
///
/// - `max_uses`: Maximum number of searches in one request (optional)
///
/// # Examples
///
/// ```ignore
/// let tool = create_web_search_tool(Some(5));
/// ```
pub fn create_web_search_tool(max_uses: Option<u32>) -> Tool {
    Tool::web_search(max_uses)
}

/// Create a web fetch server tool.
///
/// Fetches and parses web page content.
pub fn create_web_fetch_tool() -> Tool {
    Tool::web_fetch()
}

/// Create a bash server tool.
///
/// Executes bash commands in a containerized environment.
pub fn create_bash_tool() -> Tool {
    Tool {
        tool_type: Some("bash_20250124".to_string()),
        name: "bash".to_string(),
        description: None,
        input_schema: None,
        cache_control: None,
        max_uses: None,
        allowed_domains: None,
        blocked_domains: None,
    }
}

/// Create a code execution server tool.
///
/// Executes Python code in a containerized environment.
pub fn create_code_execution_tool() -> Tool {
    Tool {
        tool_type: Some("code_execution_20250825".to_string()),
        name: "code_execution".to_string(),
        description: None,
        input_schema: None,
        cache_control: None,
        max_uses: None,
        allowed_domains: None,
        blocked_domains: None,
    }
}

/// Create a text editor server tool.
///
/// String-replace based file editing tool.
pub fn create_text_editor_tool() -> Tool {
    Tool {
        tool_type: Some("text_editor_20250728".to_string()),
        name: "str_replace_based_edit_tool".to_string(),
        description: None,
        input_schema: None,
        cache_control: None,
        max_uses: None,
        allowed_domains: None,
        blocked_domains: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_web_search_tool() {
        let tool = create_web_search_tool(Some(5));
        assert_eq!(tool.name, "web_search");
        assert_eq!(tool.tool_type.as_ref().unwrap(), "web_search_20250305");
        assert_eq!(tool.max_uses, Some(5));
    }

    #[test]
    fn test_create_bash_tool() {
        let tool = create_bash_tool();
        assert_eq!(tool.name, "bash");
        assert_eq!(tool.tool_type.as_ref().unwrap(), "bash_20250124");
    }
}
