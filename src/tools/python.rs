//! Python tool bridge via PyO3.
//!
//! Enables tools to be implemented in Python scripts and invoked from the Rust
//! runtime. Handles argument marshaling, result unmarshaling, and error conversion.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde_json::Value;
use tracing::{debug, error, warn};

use super::Tool;
use crate::llm::ToolSpec;

/// Initialize Python interpreter (once per process).
static PYTHON_INIT: Once = Once::new();

fn ensure_python_initialized() {
    PYTHON_INIT.call_once(|| {
        Python::initialize();
        debug!("Python interpreter initialized");
    });
}

/// Python-based tool implementation.
///
/// Wraps a Python script that implements the tool execution logic. The script
/// must define an `execute(args: dict) -> dict` function that accepts JSON-like
/// arguments and returns JSON-like results.
///
/// # Python Script Interface
///
/// ```python
/// def execute(args):
///     """
///     Execute the tool with the given arguments.
///
///     Args:
///         args (dict): Arguments matching the JSON schema
///
///     Returns:
///         dict: Tool results (must be JSON-serializable)
///
///     Raises:
///         Exception: On execution failure
///     """
///     # Implementation here
///     return {"success": True, "result": ...}
/// ```
///
/// # Security
///
/// Python tools execute in the same process with full access to the filesystem
/// and network. Ensure Python scripts are trusted and validated before loading.
/// Consider sandboxing techniques for untrusted code.
pub struct PythonTool {
    /// Tool name (must match spec)
    name: String,
    /// Path to Python script
    script_path: PathBuf,
    /// Tool specification (loaded separately)
    spec: ToolSpec,
    /// Optional execution timeout
    timeout: Option<Duration>,
}

impl PythonTool {
    /// Create a new Python tool.
    ///
    /// # Parameters
    ///
    /// - `name`: Tool name (must match the name in the spec)
    /// - `script_path`: Path to the Python script
    /// - `spec`: Tool specification (JSON schema)
    ///
    /// # Errors
    ///
    /// Returns an error if the script path is invalid or the spec is malformed.
    pub fn new(name: String, script_path: impl AsRef<Path>, spec: ToolSpec) -> Result<Self> {
        ensure_python_initialized();

        let script_path = script_path.as_ref().to_path_buf();
        if !script_path.exists() {
            return Err(anyhow!(
                "Python script not found: {}",
                script_path.display()
            ));
        }

        Ok(Self {
            name,
            script_path,
            spec,
            timeout: Some(Duration::from_secs(60)),
        })
    }

    /// Set execution timeout for this tool.
    ///
    /// If set to None, the tool can run indefinitely (not recommended).
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Load and execute the Python script.
    ///
    /// Loads the script module, calls the `execute` function with marshaled
    /// arguments, and unmarshals the result back to JSON.
    fn execute_python(&self, args: Value) -> Result<Value> {
        Python::attach(|py| {
            // Read script source
            let script_source = std::fs::read_to_string(&self.script_path).with_context(|| {
                format!(
                    "Failed to read Python script: {}",
                    self.script_path.display()
                )
            })?;

            let code = CString::new(script_source)
                .context("Python script source contains an embedded NUL byte")?;
            let file_name = CString::new(self.script_path.to_string_lossy().as_ref())
                .context("Python script path contains an embedded NUL byte")?;
            let module_name = CString::new(self.name.as_str())
                .context("Python module name contains an embedded NUL byte")?;

            // Create a module from the script
            let module = PyModule::from_code(
                py,
                code.as_c_str(),
                file_name.as_c_str(),
                module_name.as_c_str(),
            )
            .with_context(|| "Failed to load Python module")?;

            // Get the execute function
            let execute_fn = module
                .getattr("execute")
                .with_context(|| "Python script must define an 'execute' function")?;

            // Marshal JSON args to Python dict
            let py_args = Self::json_to_python(py, &args)?;

            // Call execute function
            let py_result = execute_fn
                .call1((py_args,))
                .with_context(|| "Python execute function failed")?;

            // Unmarshal Python result to JSON
            let result = Self::python_to_json(&py_result)?;

            Ok(result)
        })
    }

    /// Convert JSON value to Python object.
    fn json_to_python<'py>(py: Python<'py>, value: &Value) -> Result<Bound<'py, PyAny>> {
        use pyo3::IntoPyObjectExt;

        match value {
            Value::Null => Ok(py.None().into_bound(py)),
            Value::Bool(b) => Ok((*b).into_bound_py_any(py)?),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(i.into_bound_py_any(py)?)
                } else if let Some(u) = n.as_u64() {
                    Ok(u.into_bound_py_any(py)?)
                } else if let Some(f) = n.as_f64() {
                    Ok(f.into_bound_py_any(py)?)
                } else {
                    Err(anyhow!("Unsupported number type"))
                }
            }
            Value::String(s) => Ok(s.as_str().into_bound_py_any(py)?),
            Value::Array(arr) => {
                let py_list = pyo3::types::PyList::empty(py);
                for item in arr {
                    let py_item = Self::json_to_python(py, item)?;
                    py_list.append(py_item)?;
                }
                Ok(py_list.into_any())
            }
            Value::Object(obj) => {
                let py_dict = PyDict::new(py);
                for (key, value) in obj {
                    let py_value = Self::json_to_python(py, value)?;
                    py_dict.set_item(key, py_value)?;
                }
                Ok(py_dict.into_any())
            }
        }
    }

    /// Convert Python object to JSON value.
    fn python_to_json(obj: &Bound<'_, PyAny>) -> Result<Value> {
        if obj.is_none() {
            Ok(Value::Null)
        } else if let Ok(b) = obj.extract::<bool>() {
            Ok(Value::Bool(b))
        } else if let Ok(i) = obj.extract::<i64>() {
            Ok(serde_json::to_value(i)?)
        } else if let Ok(f) = obj.extract::<f64>() {
            Ok(serde_json::to_value(f)?)
        } else if let Ok(s) = obj.extract::<String>() {
            Ok(Value::String(s))
        } else if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
            let mut arr = Vec::new();
            for item in list.iter() {
                arr.push(Self::python_to_json(&item)?);
            }
            Ok(Value::Array(arr))
        } else if let Ok(dict) = obj.cast::<pyo3::types::PyDict>() {
            let mut map = serde_json::Map::new();
            for (key, value) in dict.iter() {
                let key_str = key.extract::<String>()?;
                let value_json = Self::python_to_json(&value)?;
                map.insert(key_str, value_json);
            }
            Ok(Value::Object(map))
        } else {
            // Fallback: try to convert to string
            warn!("Unsupported Python type, converting to string");
            let s = obj.str()?.extract::<String>()?;
            Ok(Value::String(s))
        }
    }
}

impl Tool for PythonTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec(&self) -> Result<ToolSpec> {
        Ok(self.spec.clone())
    }

    fn execute(&self, args: Value) -> Result<Value> {
        debug!(
            tool = %self.name,
            script = %self.script_path.display(),
            "Executing Python tool"
        );

        let result = self.execute_python(args);

        match result {
            Ok(value) => {
                debug!(tool = %self.name, "Python tool succeeded");
                Ok(value)
            }
            Err(e) => {
                error!(
                    tool = %self.name,
                    error = %e,
                    "Python tool execution failed"
                );
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_script(code: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(code.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_python_tool_basic() {
        let script = create_test_script(
            r#"
def execute(args):
    message = args.get("message", "")
    return {"output": f"Echo: {message}"}
"#,
        );

        let spec = serde_json::from_value(json!({
            "type": "function",
            "name": "echo",
            "description": "Echo tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                }
            }
        }))
        .unwrap();

        let tool = PythonTool::new("echo".to_string(), script.path(), spec).unwrap();

        let result = tool.execute(json!({"message": "hello"})).unwrap();
        assert_eq!(result["output"], "Echo: hello");
    }

    #[test]
    fn test_python_tool_error_handling() {
        let script = create_test_script(
            r#"
def execute(args):
    raise ValueError("Test error")
"#,
        );

        let spec = serde_json::from_value(json!({
            "type": "function",
            "name": "error",
            "description": "Error tool",
            "parameters": {"type": "object", "properties": {}}
        }))
        .unwrap();

        let tool = PythonTool::new("error".to_string(), script.path(), spec).unwrap();

        let result = tool.execute(json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_json_python_conversion() {
        let script = create_test_script(
            r#"
def execute(args):
    return args
"#,
        );

        let spec = serde_json::from_value(json!({
            "type": "function",
            "name": "passthrough",
            "description": "Passthrough tool",
            "parameters": {"type": "object"}
        }))
        .unwrap();

        let tool = PythonTool::new("passthrough".to_string(), script.path(), spec).unwrap();

        let input = json!({
            "string": "hello",
            "number": 42,
            "bool": true,
            "null": null,
            "array": [1, 2, 3],
            "object": {"nested": "value"}
        });

        let result = tool.execute(input.clone()).unwrap();
        assert_eq!(result, input);
    }
}
