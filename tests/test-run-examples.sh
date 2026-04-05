#!/usr/bin/env bash
# =============================================================================
# test-run-examples.sh
# =============================================================================
# Tests all coding agent examples to verify streaming and tool execution work
# correctly. Each example is run with a prompt that exercises all tools:
#   - bash: Create a directory
#   - write_file: Write a Python file
#   - read_file: Read the file back
#   - list_files: List the directory contents
#
# Usage:
#   ./test-run-examples.sh              # Run all examples
#   ./test-run-examples.sh --help       # Show help
#   ./test-run-examples.sh --example coding-agent-anthropic  # Run specific example
#   ./test-run-examples.sh --verbose    # Show full output (not truncated)
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="poem_generator"
TIMEOUT_SECONDS=180  # 3 minutes per example
VERBOSE=false
SPECIFIC_EXAMPLE=""

# Examples to test (in order)
ALL_EXAMPLES=(
    "coding-agent-anthropic"
    "coding-agent-azure-anthropic"
    "coding-agent-vertex"
    "coding-agent-openai-codex"
    "coding-agent-openai-responses"
    "coding-agent-openrouter-completions"
    "coding-agent-openrouter-responses"
)

# Required environment variables per example
declare -A EXAMPLE_ENV_VARS
EXAMPLE_ENV_VARS["coding-agent-anthropic"]="ANTHROPIC_API_KEY"
EXAMPLE_ENV_VARS["coding-agent-azure-anthropic"]="AZURE_API_KEY"
EXAMPLE_ENV_VARS["coding-agent-vertex"]="GOOGLE_VERTEX_API_KEY"
EXAMPLE_ENV_VARS["coding-agent-openai-codex"]="OPENAI_CODEX_ACCESS_TOKEN"
EXAMPLE_ENV_VARS["coding-agent-openai-responses"]="OPENAI_API_KEY"
EXAMPLE_ENV_VARS["coding-agent-openrouter-completions"]="OPENROUTER_API_KEY"
EXAMPLE_ENV_VARS["coding-agent-openrouter-responses"]="OPENROUTER_API_KEY"

# Provider names for display
declare -A EXAMPLE_PROVIDERS
EXAMPLE_PROVIDERS["coding-agent-anthropic"]="Anthropic (Claude)"
EXAMPLE_PROVIDERS["coding-agent-azure-anthropic"]="Azure Anthropic (Claude)"
EXAMPLE_PROVIDERS["coding-agent-vertex"]="Google Vertex (Gemini)"
EXAMPLE_PROVIDERS["coding-agent-openai-codex"]="OpenAI Codex Subscription"
EXAMPLE_PROVIDERS["coding-agent-openai-responses"]="OpenAI (GPT)"
EXAMPLE_PROVIDERS["coding-agent-openrouter-completions"]="OpenRouter Completions"
EXAMPLE_PROVIDERS["coding-agent-openrouter-responses"]="OpenRouter Responses"

# Test prompt that exercises all tools:
# - bash: mkdir for creating directory
# - write_file: write the Python poem generator
# - read_file: verify the file content
# - list_files: list the directory to confirm structure
TEST_PROMPT="Create a directory called '$ARTIFACT_DIR' using bash mkdir, then write a simple Python poem generator script to '$ARTIFACT_DIR/generator.py'. After writing, read the file to verify it was created correctly, and list the directory contents to confirm. The Python script should have a generate_poem() function that returns a random haiku. Keep it simple - about 20-30 lines."

# Track results
declare -A RESULTS
PASSED=0
FAILED=0
SKIPPED=0

# =============================================================================
# Utility Functions
# =============================================================================

show_help() {
    cat << EOF
${CYAN}Appam Examples Test Suite${NC}

Tests all coding agent examples to verify streaming and tool execution work correctly.

${YELLOW}USAGE:${NC}
    $0 [OPTIONS]

${YELLOW}OPTIONS:${NC}
    -h, --help              Show this help message
    -e, --example NAME      Run only the specified example
    -v, --verbose           Show full output (not truncated to last 100 lines)
    -t, --timeout SECONDS   Set timeout per example (default: 180)
    -l, --list              List available examples

${YELLOW}AVAILABLE EXAMPLES:${NC}
    coding-agent-anthropic           - Uses Anthropic API (Claude)
    coding-agent-azure-anthropic     - Uses Azure Anthropic Messages API (Claude)
    coding-agent-vertex              - Uses Google Vertex API (Gemini)
    coding-agent-openai-codex        - Uses OpenAI Codex subscription auth (ChatGPT)
    coding-agent-openai-responses    - Uses OpenAI Responses API (GPT)
    coding-agent-openrouter-completions - Uses OpenRouter Completions API
    coding-agent-openrouter-responses   - Uses OpenRouter Responses API

${YELLOW}REQUIRED ENVIRONMENT VARIABLES:${NC}
    ANTHROPIC_API_KEY      For Anthropic examples
    AZURE_API_KEY          For Azure Anthropic examples
    GOOGLE_VERTEX_API_KEY  For Vertex examples
    OPENAI_CODEX_ACCESS_TOKEN For OpenAI Codex example (or cached auth file)
    OPENAI_API_KEY         For OpenAI examples
    OPENROUTER_API_KEY     For OpenRouter examples

${YELLOW}EXAMPLES:${NC}
    $0                                  # Test all examples
    $0 --example coding-agent-anthropic # Test only Anthropic example
    $0 --verbose                        # Show full output
    $0 --timeout 300                    # Set 5 minute timeout

EOF
    exit 0
}

list_examples() {
    echo -e "${CYAN}Available examples:${NC}"
    echo ""
    for example in "${ALL_EXAMPLES[@]}"; do
        local provider="${EXAMPLE_PROVIDERS[$example]}"
        local env_var="${EXAMPLE_ENV_VARS[$example]}"
        local status="${RED}not set${NC}"
        if [ -n "${!env_var:-}" ]; then
            status="${GREEN}set${NC}"
        fi
        echo -e "  ${BLUE}$example${NC}"
        echo -e "    Provider: $provider"
        echo -e "    Requires: $env_var ($status)"
        echo ""
    done
    exit 0
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
}

# Clean up artifacts from previous runs
cleanup_artifacts() {
    if [ -d "$PROJECT_DIR/$ARTIFACT_DIR" ]; then
        log_info "Cleaning up artifact directory: $ARTIFACT_DIR"
        rm -rf "$PROJECT_DIR/$ARTIFACT_DIR"
    fi
}

# Check if required environment variable is set
check_env_var() {
    local var_name="$1"
    if [ -z "${!var_name:-}" ]; then
        return 1
    fi
    return 0
}

check_azure_anthropic_env() {
    local auth_method="${AZURE_ANTHROPIC_AUTH_METHOD:-x_api_key}"
    local has_base=false
    local has_auth=false

    if [ -n "${AZURE_ANTHROPIC_BASE_URL:-}" ] || [ -n "${AZURE_ANTHROPIC_RESOURCE:-}" ]; then
        has_base=true
    fi

    case "${auth_method,,}" in
        bearer|bearer_token|bearer-token)
            if [ -n "${AZURE_ANTHROPIC_AUTH_TOKEN:-}" ] || [ -n "${AZURE_API_KEY:-}" ]; then
                has_auth=true
            fi
            ;;
        *)
            if [ -n "${AZURE_ANTHROPIC_API_KEY:-}" ] || [ -n "${AZURE_API_KEY:-}" ]; then
                has_auth=true
            fi
            ;;
    esac

    if [ "$has_base" = true ] && [ "$has_auth" = true ]; then
        return 0
    fi

    return 1
}

check_openai_codex_env() {
    if [ -n "${OPENAI_CODEX_ACCESS_TOKEN:-}" ]; then
        return 0
    fi

    local auth_file="${OPENAI_CODEX_AUTH_FILE:-$HOME/.appam/auth.json}"
    if [ -f "$auth_file" ] && grep -q '"openai-codex"' "$auth_file"; then
        return 0
    fi

    return 1
}

# Run a single example with timeout
run_example() {
    local example="$1"
    local env_var="${EXAMPLE_ENV_VARS[$example]}"
    local provider="${EXAMPLE_PROVIDERS[$example]}"
    local timeout_seconds="$TIMEOUT_SECONDS"

    if [ "$example" = "coding-agent-openai-codex" ]; then
        timeout_seconds=300
    fi

    log_header "Testing: $example"
    log_info "Provider: $provider"

    # Check for required environment variable
    if [ "$example" = "coding-agent-azure-anthropic" ]; then
        if ! check_azure_anthropic_env; then
            log_warning "Skipping $example - requires Azure Anthropic credentials plus AZURE_ANTHROPIC_BASE_URL or AZURE_ANTHROPIC_RESOURCE"
            RESULTS["$example"]="SKIPPED"
            ((SKIPPED++))
            return 0
        fi
        log_info "Environment: Azure Anthropic requirements are set ✓"
    elif [ "$example" = "coding-agent-openai-codex" ]; then
        if ! check_openai_codex_env; then
            log_warning "Skipping $example - requires OPENAI_CODEX_ACCESS_TOKEN or an auth cache entry in \${OPENAI_CODEX_AUTH_FILE:-$HOME/.appam/auth.json}"
            RESULTS["$example"]="SKIPPED"
            ((SKIPPED++))
            return 0
        fi
        log_info "Environment: OpenAI Codex auth is available ✓"
    else
        if ! check_env_var "$env_var"; then
            log_warning "Skipping $example - $env_var not set"
            RESULTS["$example"]="SKIPPED"
            ((SKIPPED++))
            return 0
        fi

        log_info "Environment: $env_var is set ✓"
    fi
    log_info "Building and running example..."

    # Clean up before test
    cleanup_artifacts

    # Create a temporary file for output
    local output_file
    output_file=$(mktemp)
    local exit_code=0
    local -a example_env_prefix=()

    if [ "$example" = "coding-agent-azure-anthropic" ] \
        && [ -z "${AZURE_ANTHROPIC_AUTH_METHOD:-}" ] \
        && [ -z "${AZURE_ANTHROPIC_API_KEY:-}" ] \
        && [ -n "${AZURE_API_KEY:-}" ]; then
        example_env_prefix=(env AZURE_ANTHROPIC_AUTH_METHOD=bearer)
        log_info "Harness override: using bearer auth for Azure Anthropic because AZURE_API_KEY is set"
    fi

    # Run the example with input piped via stdin
    # We use printf to send the prompt followed by "exit" to cleanly terminate
    # The timeout command ensures we don't hang forever
    (
        cd "$PROJECT_DIR"
        # Give the prompt, wait, then exit
        printf '%s\nexit\n' "$TEST_PROMPT" | timeout "$timeout_seconds" "${example_env_prefix[@]}" cargo run --example "$example" 2>&1
    ) > "$output_file" 2>&1 || exit_code=$?

    # Check results
    local success=false

    # Display output
    echo ""
    if [ "$VERBOSE" = true ]; then
        log_info "Full output:"
    else
        log_info "Output (showing last 100 lines, use --verbose for full):"
    fi
    echo "─────────────────────────────────────────────────────────────────────"
    if [ "$VERBOSE" = true ]; then
        cat "$output_file"
    else
        tail -n 100 "$output_file"
    fi
    echo "─────────────────────────────────────────────────────────────────────"
    echo ""

    # Determine success based on multiple factors:
    # 1. Exit code (0 or 124 for timeout with partial success)
    # 2. Evidence of SUCCESSFUL tool usage (not just tool mentions in logs)
    # 3. Artifact creation
    # 4. No critical errors in output

    local tool_calls_found=false
    local artifact_created=false
    local has_errors=false

    # Check for critical errors first
    if grep -qE "❌ Error:|API error|authentication_error|error_type=|Failed to" "$output_file"; then
        has_errors=true
        log_error "Errors detected in output!"
    fi

    # Check for SUCCESSFUL tool execution (look for "✓ X completed" pattern)
    if grep -q "✓.*completed" "$output_file"; then
        tool_calls_found=true
        log_info "Successful tool calls detected in output ✓"
    fi

    # Check if artifact was created
    if [ -d "$PROJECT_DIR/$ARTIFACT_DIR" ] && [ -f "$PROJECT_DIR/$ARTIFACT_DIR/generator.py" ]; then
        artifact_created=true
        log_info "Artifact created successfully ✓"
        log_info "Contents of $ARTIFACT_DIR:"
        ls -la "$PROJECT_DIR/$ARTIFACT_DIR" 2>/dev/null || true
    fi

    # Determine final result
    # Success requires: no critical errors AND (successful tool calls OR artifact creation)
    if [ "$has_errors" = false ]; then
        if [ "$exit_code" -eq 0 ] || [ "$exit_code" -eq 124 ]; then
            # Exit 0 = clean exit, 124 = timeout (might still be success if tools ran)
            if [ "$tool_calls_found" = true ] || [ "$artifact_created" = true ]; then
                success=true
            fi
        fi
    fi

    # Clean up temp file
    rm -f "$output_file"

    # Clean up artifacts for next test
    cleanup_artifacts

    # Record result
    if [ "$success" = true ]; then
        log_success "$example passed!"
        RESULTS["$example"]="PASSED"
        ((PASSED++))
        return 0
    else
        log_error "$example failed! (exit code: $exit_code)"
        RESULTS["$example"]="FAILED"
        ((FAILED++))
        return 1
    fi
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                ;;
            -l|--list)
                list_examples
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -e|--example)
                if [[ -n "${2:-}" ]]; then
                    SPECIFIC_EXAMPLE="$2"
                    shift 2
                else
                    echo -e "${RED}Error: --example requires an argument${NC}" >&2
                    exit 1
                fi
                ;;
            -t|--timeout)
                if [[ -n "${2:-}" ]]; then
                    TIMEOUT_SECONDS="$2"
                    shift 2
                else
                    echo -e "${RED}Error: --timeout requires an argument${NC}" >&2
                    exit 1
                fi
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}" >&2
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    parse_args "$@"

    log_header "Appam Examples Test Suite"

    log_info "Project directory: $PROJECT_DIR"
    log_info "Artifact directory: $ARTIFACT_DIR"
    log_info "Timeout per example: ${TIMEOUT_SECONDS}s"
    log_info "Verbose mode: $VERBOSE"
    log_info "Test prompt: ${TEST_PROMPT:0:80}..."
    echo ""

    # Determine which examples to run
    local examples_to_run=("${ALL_EXAMPLES[@]}")
    if [ -n "$SPECIFIC_EXAMPLE" ]; then
        # Validate the specific example exists
        local found=false
        for ex in "${ALL_EXAMPLES[@]}"; do
            if [ "$ex" = "$SPECIFIC_EXAMPLE" ]; then
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            log_error "Unknown example: $SPECIFIC_EXAMPLE"
            echo "Available examples:"
            for ex in "${ALL_EXAMPLES[@]}"; do
                echo "  - $ex"
            done
            exit 1
        fi
        examples_to_run=("$SPECIFIC_EXAMPLE")
        log_info "Running specific example: $SPECIFIC_EXAMPLE"
    fi

    # Initial cleanup
    cleanup_artifacts

    # Build examples first (to catch compile errors early)
    log_info "Pre-building examples..."
    (
        cd "$PROJECT_DIR"
        if [ -n "$SPECIFIC_EXAMPLE" ]; then
            cargo build --example "$SPECIFIC_EXAMPLE" 2>&1
        else
            cargo build --examples 2>&1
        fi
    ) || {
        log_error "Failed to build examples!"
        exit 1
    }
    log_success "Examples built successfully"

    # Run each example
    local overall_exit=0
    for example in "${examples_to_run[@]}"; do
        if ! run_example "$example"; then
            overall_exit=1
        fi
    done

    # Final cleanup
    cleanup_artifacts

    # Summary
    log_header "Test Summary"
    echo ""
    for example in "${examples_to_run[@]}"; do
        local result="${RESULTS[$example]:-UNKNOWN}"
        local provider="${EXAMPLE_PROVIDERS[$example]}"
        case "$result" in
            PASSED)
                echo -e "  ${GREEN}✓${NC} $example ${CYAN}($provider)${NC}"
                ;;
            FAILED)
                echo -e "  ${RED}✗${NC} $example ${CYAN}($provider)${NC}"
                ;;
            SKIPPED)
                echo -e "  ${YELLOW}○${NC} $example ${CYAN}($provider)${NC} - skipped (missing required configuration)"
                ;;
            *)
                echo -e "  ${RED}?${NC} $example ${CYAN}($provider)${NC} - unknown"
                ;;
        esac
    done

    echo ""
    echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}, ${YELLOW}$SKIPPED skipped${NC}"
    echo ""

    if [ "$FAILED" -gt 0 ]; then
        log_error "Some examples failed!"
        exit 1
    elif [ "$PASSED" -eq 0 ]; then
        log_warning "No examples were tested (missing API keys?)"
        echo ""
        echo "Required environment variables:"
        echo "  - ANTHROPIC_API_KEY for coding-agent-anthropic"
        echo "  - AZURE_API_KEY plus AZURE_ANTHROPIC_BASE_URL or AZURE_ANTHROPIC_RESOURCE for coding-agent-azure-anthropic"
        echo "  - OPENAI_CODEX_ACCESS_TOKEN or an auth cache entry for coding-agent-openai-codex"
        echo "  - OPENAI_API_KEY for coding-agent-openai-responses"
        echo "  - OPENROUTER_API_KEY for coding-agent-openrouter-completions"
        echo "  - OPENROUTER_API_KEY for coding-agent-openrouter-responses"
        exit 0
    else
        log_success "All tested examples passed!"
        exit 0
    fi
}

# Run main
main "$@"
