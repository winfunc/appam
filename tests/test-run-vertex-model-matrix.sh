#!/usr/bin/env bash
# =============================================================================
# test-run-vertex-model-matrix.sh
# =============================================================================
# Runs the Vertex coding-agent example across a matrix of Gemini models.
# Reuses test-run-examples.sh for each model to ensure consistent assertions.
#
# Usage:
#   bash tests/test-run-vertex-model-matrix.sh
#   bash tests/test-run-vertex-model-matrix.sh --models "gemini-2.5-flash,gemini-2.5-pro"
#   bash tests/test-run-vertex-model-matrix.sh --verbose --include-thoughts
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_MODELS=(
    "gemini-2.5-flash"
    "gemini-2.5-pro"
    "gemini-3.1-pro-preview"
)

MODELS=("${DEFAULT_MODELS[@]}")
TIMEOUT_SECONDS=180
VERBOSE=false
INCLUDE_THOUGHTS=false
THINKING_LEVEL=""

declare -A RESULTS
PASSED=0
FAILED=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
}

show_help() {
    cat << EOF
${CYAN}Vertex Model Matrix Smoke Test${NC}

Runs coding-agent-vertex against multiple Gemini models.

${YELLOW}USAGE:${NC}
    $0 [OPTIONS]

${YELLOW}OPTIONS:${NC}
    -h, --help                Show this help
    -m, --models LIST         Comma-separated model list
    -t, --timeout SECONDS     Timeout per model run (default: 180)
    -v, --verbose             Show full example output
    --include-thoughts        Set GOOGLE_VERTEX_INCLUDE_THOUGHTS=true
    --thinking-level LEVEL    Optional GOOGLE_VERTEX_THINKING_LEVEL value

${YELLOW}EXAMPLES:${NC}
    $0
    $0 --models "gemini-2.5-flash,gemini-3.1-pro-preview"
    $0 --verbose --include-thoughts

EOF
    exit 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                ;;
            -m|--models)
                if [[ -z "${2:-}" ]]; then
                    log_error "--models requires a value"
                    exit 1
                fi
                IFS=',' read -r -a MODELS <<< "$2"
                shift 2
                ;;
            -t|--timeout)
                if [[ -z "${2:-}" ]]; then
                    log_error "--timeout requires a value"
                    exit 1
                fi
                TIMEOUT_SECONDS="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --include-thoughts)
                INCLUDE_THOUGHTS=true
                shift
                ;;
            --thinking-level)
                if [[ -z "${2:-}" ]]; then
                    log_error "--thinking-level requires a value"
                    exit 1
                fi
                THINKING_LEVEL="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

run_for_model() {
    local model="$1"
    local status=0

    log_header "Vertex Smoke: $model"

    (
        cd "$PROJECT_DIR"
        export GOOGLE_VERTEX_MODEL="$model"
        if [[ "$INCLUDE_THOUGHTS" == true ]]; then
            export GOOGLE_VERTEX_INCLUDE_THOUGHTS=true
            if [[ -n "$THINKING_LEVEL" ]]; then
                export GOOGLE_VERTEX_THINKING_LEVEL="$THINKING_LEVEL"
            else
                unset GOOGLE_VERTEX_THINKING_LEVEL
            fi
        fi

        args=(--example coding-agent-vertex --timeout "$TIMEOUT_SECONDS")
        if [[ "$VERBOSE" == true ]]; then
            args+=(--verbose)
        fi

        bash tests/test-run-examples.sh "${args[@]}"
    ) || status=$?

    if [[ $status -eq 0 ]]; then
        RESULTS["$model"]="PASSED"
        ((PASSED++))
        log_success "$model passed"
    else
        RESULTS["$model"]="FAILED"
        ((FAILED++))
        log_error "$model failed"
    fi
}

main() {
    parse_args "$@"

    if [[ -z "${GOOGLE_VERTEX_API_KEY:-}" ]]; then
        log_error "GOOGLE_VERTEX_API_KEY is required"
        exit 1
    fi

    log_header "Vertex Model Matrix"
    log_info "Project directory: $PROJECT_DIR"
    log_info "Models: ${MODELS[*]}"
    log_info "Timeout per run: ${TIMEOUT_SECONDS}s"
    log_info "Include thoughts: $INCLUDE_THOUGHTS"
    if [[ -n "$THINKING_LEVEL" ]]; then
        log_info "Thinking level: $THINKING_LEVEL"
    fi

    local overall_status=0
    for model in "${MODELS[@]}"; do
        model="$(echo "$model" | xargs)"
        if [[ -z "$model" ]]; then
            continue
        fi
        if ! run_for_model "$model"; then
            overall_status=1
        fi
    done

    log_header "Vertex Matrix Summary"
    for model in "${MODELS[@]}"; do
        model="$(echo "$model" | xargs)"
        if [[ -z "$model" ]]; then
            continue
        fi

        if [[ "${RESULTS[$model]:-UNKNOWN}" == "PASSED" ]]; then
            echo -e "  ${GREEN}✓${NC} $model"
        elif [[ "${RESULTS[$model]:-UNKNOWN}" == "FAILED" ]]; then
            echo -e "  ${RED}✗${NC} $model"
        fi
    done
    echo ""
    echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"

    if [[ $FAILED -gt 0 ]]; then
        exit 1
    fi

    exit $overall_status
}

main "$@"
