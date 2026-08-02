#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

readonly BASE_COMPOSE_FILE="docker-compose.yml"
readonly GPU_NVIDIA_COMPOSE_FILE="docker-compose.gpu-nvidia.yml"
readonly GPU_METAL_COMPOSE_FILE="docker-compose.gpu-metal.yml"
readonly PUBLIC_KEY_FILE=".keys/jwt_ed25519_public.pem"
readonly API_HOST_PORT="27800"
readonly DEFAULT_NGINX_HOST_PORT="${XIMA_AGENT_NGINX_PORT:-27801}"
readonly DEFAULT_WORKSPACES_HOST_PATH="${XIMA_WORKSPACES_HOST_PATH:-./workspaces}"
readonly DEFAULT_UVICORN_WORKERS="${UVICORN_WORKERS:-2}"
readonly DEFAULT_COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-}"
readonly DEFAULT_HEALTH_RETRIES="${XIMA_LAUNCHER_HEALTH_RETRIES:-20}"
readonly DEFAULT_HEALTH_INTERVAL_SEC="${XIMA_LAUNCHER_HEALTH_INTERVAL_SEC:-2}"
readonly LAUNCHER_CONFIG_FILE=".launcher-config.env"
readonly NATIVE_VENV_DIR=".venv"
readonly NATIVE_PYTHON="${AGENT_DIR}/${NATIVE_VENV_DIR}/bin/python"
readonly NATIVE_CELERY="${AGENT_DIR}/${NATIVE_VENV_DIR}/bin/celery"
readonly NATIVE_RUNTIME_DIR="${AGENT_DIR}/api/state/native-metal"
readonly NATIVE_API_PID_FILE="${NATIVE_RUNTIME_DIR}/api.pid"
readonly NATIVE_WORKER_PID_FILE="${NATIVE_RUNTIME_DIR}/worker.pid"
readonly NATIVE_API_LOG_FILE="${NATIVE_RUNTIME_DIR}/api.log"
readonly NATIVE_WORKER_LOG_FILE="${NATIVE_RUNTIME_DIR}/worker.log"

COMPOSE_CMD=()
API_HEALTH_METHOD="none"
API_HEALTH_STATUS="not-checked"
NGINX_HEALTH_STATUS="not-checked"
LAST_COMMAND="-"
SUMMARY_COMMAND="-"
API_HEALTH_URL=""
NGINX_HEALTH_URL=""
NGINX_HOST_PORT="${DEFAULT_NGINX_HOST_PORT}"
WORKSPACES_HOST_PATH="${DEFAULT_WORKSPACES_HOST_PATH}"
UVICORN_WORKERS_VALUE="${DEFAULT_UVICORN_WORKERS}"
COMPOSE_PROJECT_NAME_VALUE="${DEFAULT_COMPOSE_PROJECT_NAME}"
HEALTH_RETRIES_VALUE="${DEFAULT_HEALTH_RETRIES}"
HEALTH_INTERVAL_SEC_VALUE="${DEFAULT_HEALTH_INTERVAL_SEC}"
ADVANCED_MODE="no"
RUNTIME_ENV_VARS=()

log_info() {
  printf '[INFO] %s\n' "$*"
}

log_ok() {
  printf '[ OK ] %s\n' "$*"
}

log_warn() {
  printf '[WARN] %s\n' "$*" >&2
}

log_error() {
  printf '[ERR ] %s\n' "$*" >&2
}

cmd_to_string() {
  local out
  printf -v out '%q ' "$@"
  printf '%s' "${out% }"
}

print_section() {
  echo "============================================================"
}

compose_project_label() {
  if [[ -n "${COMPOSE_PROJECT_NAME_VALUE}" ]]; then
    printf '%s' "${COMPOSE_PROJECT_NAME_VALUE}"
    return
  fi
  printf 'auto'
}

print_result_summary() {
  local result="$1"
  echo
  print_section
  echo "xima-core launcher result"
  print_section
  printf "Result        : %s\n" "${result}"
  printf "Operation     : %s\n" "${OPERATION:-unknown}"

  case "${OPERATION:-}" in
    start|restart|rebuild)
    printf "Auth mode     : %s\n" "${AUTH_MODE:-n/a}"
    printf "Runtime       : %s\n" "${GPU_PROFILE:-n/a}"
    printf "Build         : %s\n" "${BUILD_FLAG:-n/a}"
    printf "Advanced mode : %s\n" "${ADVANCED_MODE}"
    printf "nginx port    : %s\n" "${NGINX_HOST_PORT}"
    printf "workspaces    : %s\n" "${WORKSPACES_HOST_PATH}"
    printf "compose proj  : %s\n" "$(compose_project_label)"
    printf "uvicorn worker: %s\n" "${UVICORN_WORKERS_VALUE}"
    printf "health wait   : retries=%s interval=%ss\n" "${HEALTH_RETRIES_VALUE}" "${HEALTH_INTERVAL_SEC_VALUE}"
    printf "API health    : %s\n" "${API_HEALTH_STATUS}"
    printf "nginx health  : %s\n" "${NGINX_HEALTH_STATUS}"
    printf "API endpoint  : %s\n" "${API_HEALTH_URL%/health}"
    printf "nginx endpoint: %s\n" "${NGINX_HEALTH_URL%/health}"
      ;;
  esac

  if [[ "${OPERATION:-}" == "health" ]]; then
    printf "API health    : %s\n" "${API_HEALTH_STATUS}"
    printf "nginx health  : %s\n" "${NGINX_HEALTH_STATUS}"
  fi

  printf "Command       : %s\n" "${SUMMARY_COMMAND}"
  print_section
}

print_cmd() {
  LAST_COMMAND="$(cmd_to_string "$@")"
  printf '[CMD ] %s\n' "${LAST_COMMAND}"
}

refresh_health_urls() {
  API_HEALTH_URL="http://127.0.0.1:${API_HOST_PORT}/health"
  NGINX_HEALTH_URL="http://127.0.0.1:${NGINX_HOST_PORT}/health"
}

build_runtime_env_vars() {
  RUNTIME_ENV_VARS=(
    "XIMA_AGENT_AUTH_MODE=${AUTH_MODE}"
    "XIMA_AGENT_NGINX_PORT=${NGINX_HOST_PORT}"
    "XIMA_WORKSPACES_HOST_PATH=${WORKSPACES_HOST_PATH}"
    "UVICORN_WORKERS=${UVICORN_WORKERS_VALUE}"
  )
  if [[ -n "${COMPOSE_PROJECT_NAME_VALUE}" ]]; then
    RUNTIME_ENV_VARS+=("COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME_VALUE}")
  fi
}

apply_default_launcher_config() {
  AUTH_MODE="external"
  GPU_PROFILE="cpu"
  NGINX_HOST_PORT="${DEFAULT_NGINX_HOST_PORT}"
  WORKSPACES_HOST_PATH="${DEFAULT_WORKSPACES_HOST_PATH}"
  UVICORN_WORKERS_VALUE="${DEFAULT_UVICORN_WORKERS}"
  COMPOSE_PROJECT_NAME_VALUE="${DEFAULT_COMPOSE_PROJECT_NAME}"
  HEALTH_RETRIES_VALUE="${DEFAULT_HEALTH_RETRIES}"
  HEALTH_INTERVAL_SEC_VALUE="${DEFAULT_HEALTH_INTERVAL_SEC}"
}

normalize_launcher_config_values() {
  AUTH_MODE="${AUTH_MODE:-external}"
  GPU_PROFILE="${GPU_PROFILE:-cpu}"
  NGINX_HOST_PORT="${NGINX_HOST_PORT:-${DEFAULT_NGINX_HOST_PORT}}"
  WORKSPACES_HOST_PATH="${WORKSPACES_HOST_PATH:-${DEFAULT_WORKSPACES_HOST_PATH}}"
  UVICORN_WORKERS_VALUE="${UVICORN_WORKERS_VALUE:-${DEFAULT_UVICORN_WORKERS}}"
  COMPOSE_PROJECT_NAME_VALUE="${COMPOSE_PROJECT_NAME_VALUE:-${DEFAULT_COMPOSE_PROJECT_NAME}}"
  HEALTH_RETRIES_VALUE="${HEALTH_RETRIES_VALUE:-${DEFAULT_HEALTH_RETRIES}}"
  HEALTH_INTERVAL_SEC_VALUE="${HEALTH_INTERVAL_SEC_VALUE:-${DEFAULT_HEALTH_INTERVAL_SEC}}"
}

save_launcher_config() {
  {
    echo "# xima-core launcher config (auto-generated)"
    echo "# Edit values manually if you want to change defaults for start/restart/rebuild."
    printf 'AUTH_MODE=%q\n' "${AUTH_MODE}"
    printf 'GPU_PROFILE=%q\n' "${GPU_PROFILE}"
    printf 'NGINX_HOST_PORT=%q\n' "${NGINX_HOST_PORT}"
    printf 'WORKSPACES_HOST_PATH=%q\n' "${WORKSPACES_HOST_PATH}"
    printf 'COMPOSE_PROJECT_NAME_VALUE=%q\n' "${COMPOSE_PROJECT_NAME_VALUE}"
    printf 'UVICORN_WORKERS_VALUE=%q\n' "${UVICORN_WORKERS_VALUE}"
    printf 'HEALTH_RETRIES_VALUE=%q\n' "${HEALTH_RETRIES_VALUE}"
    printf 'HEALTH_INTERVAL_SEC_VALUE=%q\n' "${HEALTH_INTERVAL_SEC_VALUE}"
  } > "${LAUNCHER_CONFIG_FILE}"
}

load_or_init_launcher_config() {
  apply_default_launcher_config
  if [[ -f "${LAUNCHER_CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${LAUNCHER_CONFIG_FILE}"
    normalize_launcher_config_values
    return
  fi

  save_launcher_config
  log_info "Initialized launcher config: ${LAUNCHER_CONFIG_FILE}"
}

print_runtime_plan() {
  log_info "Runtime config:"
  echo "  - auth mode : ${AUTH_MODE}"
  echo "  - profile   : ${GPU_PROFILE}"
  echo "  - nginx port: ${NGINX_HOST_PORT}"
  echo "  - workspaces: ${WORKSPACES_HOST_PATH}"
  echo "  - compose   : $(compose_project_label)"
  echo "  - uvicorn   : ${UVICORN_WORKERS_VALUE}"
  echo "  - health    : retries=${HEALTH_RETRIES_VALUE}, interval=${HEALTH_INTERVAL_SEC_VALUE}s"
  echo "  - config    : ${LAUNCHER_CONFIG_FILE}"
}

to_lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    log_error "Missing command: ${cmd}"
    exit 1
  fi
}

ensure_compose_available() {
  require_command docker
  if ! docker compose version >/dev/null 2>&1; then
    log_error "docker compose is not available. Please install Docker Compose v2."
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon is not reachable. Please start Docker Desktop / dockerd."
    exit 1
  fi
}

build_compose_cmd() {
  local gpu_profile="$1"
  COMPOSE_CMD=(docker compose -f "${BASE_COMPOSE_FILE}")
  case "${gpu_profile}" in
    gpu-nvidia)
      COMPOSE_CMD+=(-f "${GPU_NVIDIA_COMPOSE_FILE}")
      ;;
    gpu-metal)
      COMPOSE_CMD+=(-f "${GPU_METAL_COMPOSE_FILE}")
      ;;
  esac
}

prompt_operation() {
  local answer
  print_section
  echo "xima-core interactive launcher"
  print_section
  while true; do
    cat <<'EOF'
Select operation:
  1) start
  2) stop
  3) restart
  4) rebuild
  5) log
  6) health
  7) demo   (create a demo workspace with sample images)
EOF
    read -r -p "Operation [1]: " answer
    answer="$(to_lower "${answer:-1}")"
    case "${answer}" in
      1|start)
        OPERATION="start"
        return
        ;;
      2|stop)
        OPERATION="stop"
        return
        ;;
      3|restart)
        OPERATION="restart"
        return
        ;;
      4|rebuild)
        OPERATION="rebuild"
        return
        ;;
      5|log|logs)
        OPERATION="log"
        return
        ;;
      6|health)
        OPERATION="health"
        return
        ;;
      7|demo)
        OPERATION="demo"
        return
        ;;
      *)
        log_warn "Invalid operation: ${answer}"
        ;;
    esac
  done
}

prompt_auth_mode() {
  local answer
  while true; do
    cat <<'EOF'
Select auth mode:
  1) external (recommended)
  2) open
EOF
    read -r -p "Auth mode [1]: " answer
    answer="$(to_lower "${answer:-1}")"
    case "${answer}" in
      1|external)
        AUTH_MODE="external"
        return
        ;;
      2|open)
        AUTH_MODE="open"
        return
        ;;
      *)
        log_warn "Invalid auth mode: ${answer}"
        ;;
    esac
  done
}

prompt_gpu_profile() {
  local answer
  while true; do
    cat <<'EOF'
Select runtime profile:
  1) cpu
  2) gpu-nvidia
  3) gpu-metal
EOF
    read -r -p "Profile [1]: " answer
    answer="$(to_lower "${answer:-1}")"
    case "${answer}" in
      1|cpu)
        GPU_PROFILE="cpu"
        return
        ;;
      2|gpu|gpu-nvidia|nvidia)
        GPU_PROFILE="gpu-nvidia"
        return
        ;;
      3|gpu-metal|metal|mps)
        GPU_PROFILE="gpu-metal"
        return
        ;;
      *)
        log_warn "Invalid profile: ${answer}"
        ;;
    esac
  done
}

prompt_build_flag() {
  local answer
  while true; do
    read -r -p "Build images before start? [Y/n]: " answer
    answer="$(to_lower "${answer:-y}")"
    case "${answer}" in
      y|yes)
        BUILD_FLAG="yes"
        return
        ;;
      n|no)
        BUILD_FLAG="no"
        return
        ;;
      *)
        log_warn "Please answer yes or no."
        ;;
    esac
  done
}

prompt_advanced_mode() {
  local answer
  while true; do
    read -r -p "Use advanced options? [y/N]: " answer
    answer="$(to_lower "${answer:-n}")"
    case "${answer}" in
      y|yes)
        ADVANCED_MODE="yes"
        return
        ;;
      n|no)
        ADVANCED_MODE="no"
        return
        ;;
      *)
        log_warn "Please answer yes or no."
        ;;
    esac
  done
}

prompt_nginx_port() {
  local answer
  while true; do
    read -r -p "nginx host port [${DEFAULT_NGINX_HOST_PORT}]: " answer
    answer="${answer:-${DEFAULT_NGINX_HOST_PORT}}"
    if [[ "${answer}" =~ ^[0-9]+$ ]] && ((answer >= 1 && answer <= 65535)); then
      NGINX_HOST_PORT="${answer}"
      return
    fi
    log_warn "Please enter a valid port number (1-65535)."
  done
}

prompt_workspaces_host_path() {
  local answer
  while true; do
    read -r -p "workspaces host path [${DEFAULT_WORKSPACES_HOST_PATH}]: " answer
    answer="${answer:-${DEFAULT_WORKSPACES_HOST_PATH}}"
    if [[ "${answer}" == *:* ]]; then
      log_warn "Path must not contain ':' for Docker volume syntax."
      continue
    fi
    WORKSPACES_HOST_PATH="${answer}"
    return
  done
}

prompt_compose_project_name() {
  local answer
  local default_label
  default_label="$(printf '%s' "${DEFAULT_COMPOSE_PROJECT_NAME}")"
  if [[ -z "${default_label}" ]]; then
    default_label="auto"
  fi
  while true; do
    read -r -p "compose project name [${default_label}]: " answer
    answer="${answer:-${DEFAULT_COMPOSE_PROJECT_NAME}}"
    if [[ -z "${answer}" ]]; then
      COMPOSE_PROJECT_NAME_VALUE=""
      return
    fi
    if [[ "${answer}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
      COMPOSE_PROJECT_NAME_VALUE="${answer}"
      return
    fi
    log_warn "Use [a-zA-Z0-9_.-], starting with alnum."
  done
}

prompt_uvicorn_workers() {
  local answer
  while true; do
    read -r -p "uvicorn workers [${DEFAULT_UVICORN_WORKERS}]: " answer
    answer="${answer:-${DEFAULT_UVICORN_WORKERS}}"
    if [[ "${answer}" =~ ^[0-9]+$ ]] && ((answer >= 1 && answer <= 128)); then
      UVICORN_WORKERS_VALUE="${answer}"
      return
    fi
    log_warn "Please enter an integer between 1 and 128."
  done
}

prompt_health_wait_settings() {
  local answer
  while true; do
    read -r -p "health retries [${DEFAULT_HEALTH_RETRIES}]: " answer
    answer="${answer:-${DEFAULT_HEALTH_RETRIES}}"
    if [[ "${answer}" =~ ^[0-9]+$ ]] && ((answer >= 1 && answer <= 300)); then
      HEALTH_RETRIES_VALUE="${answer}"
      break
    fi
    log_warn "Please enter an integer between 1 and 300."
  done
  while true; do
    read -r -p "health interval sec [${DEFAULT_HEALTH_INTERVAL_SEC}]: " answer
    answer="${answer:-${DEFAULT_HEALTH_INTERVAL_SEC}}"
    if [[ "${answer}" =~ ^[0-9]+$ ]] && ((answer >= 1 && answer <= 30)); then
      HEALTH_INTERVAL_SEC_VALUE="${answer}"
      return
    fi
    log_warn "Please enter an integer between 1 and 30."
  done
}

prompt_logs_service() {
  local answer
  while true; do
    cat <<'EOF'
Select log target:
  1) all
  2) api
  3) worker
  4) nginx
  5) redis
EOF
    read -r -p "Logs [1]: " answer
    answer="$(to_lower "${answer:-1}")"
    case "${answer}" in
      1|all)
        LOG_SERVICE="all"
        return
        ;;
      2|api)
        LOG_SERVICE="api"
        return
        ;;
      3|worker)
        LOG_SERVICE="worker"
        return
        ;;
      4|nginx)
        LOG_SERVICE="nginx"
        return
        ;;
      5|redis)
        LOG_SERVICE="redis"
        return
        ;;
      *)
        log_warn "Invalid log target: ${answer}"
        ;;
    esac
  done
}

confirm_start_plan() {
  local answer
  echo "Start plan:"
  echo "  auth mode : ${AUTH_MODE}"
  echo "  profile   : ${GPU_PROFILE}"
  echo "  build     : ${BUILD_FLAG}"
  echo "  advanced  : ${ADVANCED_MODE}"
  if [[ "${GPU_PROFILE}" == "gpu-metal" ]]; then
    echo "  native    : api/worker from ${NATIVE_VENV_DIR}"
  fi
  if [[ "${ADVANCED_MODE}" == "yes" ]]; then
    echo "  nginx port: ${NGINX_HOST_PORT}"
    echo "  workspaces: ${WORKSPACES_HOST_PATH}"
    echo "  compose   : $(compose_project_label)"
    echo "  uvicorn   : ${UVICORN_WORKERS_VALUE}"
    echo "  health    : retries=${HEALTH_RETRIES_VALUE}, interval=${HEALTH_INTERVAL_SEC_VALUE}s"
  fi
  while true; do
    read -r -p "Proceed? [Y/n]: " answer
    answer="$(to_lower "${answer:-y}")"
    case "${answer}" in
      y|yes)
        return
        ;;
      n|no)
        log_info "Canceled."
        exit 0
        ;;
      *)
        log_warn "Please answer yes or no."
        ;;
    esac
  done
}

validate_local_files() {
  local gpu_profile="$1"
  local auth_mode="$2"

  if [[ ! -f "${BASE_COMPOSE_FILE}" ]]; then
    log_error "Missing file: ${BASE_COMPOSE_FILE}"
    exit 1
  fi
  if [[ "${gpu_profile}" == "gpu-nvidia" && ! -f "${GPU_NVIDIA_COMPOSE_FILE}" ]]; then
    log_error "Missing file: ${GPU_NVIDIA_COMPOSE_FILE}"
    exit 1
  fi
  if [[ "${gpu_profile}" == "gpu-metal" && ! -f "${GPU_METAL_COMPOSE_FILE}" ]]; then
    log_error "Missing file: ${GPU_METAL_COMPOSE_FILE}"
    exit 1
  fi
  if [[ "${auth_mode}" == "external" && ! -f "${PUBLIC_KEY_FILE}" ]]; then
    log_error "external mode requires ${PUBLIC_KEY_FILE}"
    exit 1
  fi
}

ensure_workspaces_host_path() {
  if [[ -d "${WORKSPACES_HOST_PATH}" ]]; then
    return
  fi
  log_info "Creating workspaces directory: ${WORKSPACES_HOST_PATH}"
  mkdir -p "${WORKSPACES_HOST_PATH}"
}

validate_runtime_settings() {
  if ! [[ "${NGINX_HOST_PORT}" =~ ^[0-9]+$ ]] || ((NGINX_HOST_PORT < 1 || NGINX_HOST_PORT > 65535)); then
    log_error "Invalid nginx host port: ${NGINX_HOST_PORT}"
    exit 1
  fi
  if [[ "${WORKSPACES_HOST_PATH}" == *:* ]]; then
    log_error "Invalid workspaces host path (contains ':'): ${WORKSPACES_HOST_PATH}"
    exit 1
  fi
  if ! [[ "${UVICORN_WORKERS_VALUE}" =~ ^[0-9]+$ ]] || ((UVICORN_WORKERS_VALUE < 1 || UVICORN_WORKERS_VALUE > 128)); then
    log_error "Invalid uvicorn workers: ${UVICORN_WORKERS_VALUE}"
    exit 1
  fi
  if ! [[ "${HEALTH_RETRIES_VALUE}" =~ ^[0-9]+$ ]] || ((HEALTH_RETRIES_VALUE < 1 || HEALTH_RETRIES_VALUE > 300)); then
    log_error "Invalid health retries: ${HEALTH_RETRIES_VALUE}"
    exit 1
  fi
  if ! [[ "${HEALTH_INTERVAL_SEC_VALUE}" =~ ^[0-9]+$ ]] || ((HEALTH_INTERVAL_SEC_VALUE < 1 || HEALTH_INTERVAL_SEC_VALUE > 30)); then
    log_error "Invalid health interval sec: ${HEALTH_INTERVAL_SEC_VALUE}"
    exit 1
  fi
  if [[ -n "${COMPOSE_PROJECT_NAME_VALUE}" ]] && ! [[ "${COMPOSE_PROJECT_NAME_VALUE}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    log_error "Invalid compose project name: ${COMPOSE_PROJECT_NAME_VALUE}"
    exit 1
  fi
}

warn_gpu_runtime_if_needed() {
  if [[ "${GPU_PROFILE}" != "gpu-nvidia" ]]; then
    return
  fi
  if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    log_warn "NVIDIA runtime was not detected in docker info. GPU startup may fail."
  fi
}

ensure_native_metal_env() {
  local needs_install="no"

  require_command python3

  if [[ ! -x "${NATIVE_PYTHON}" ]]; then
    log_info "Creating native venv for gpu-metal: ${NATIVE_VENV_DIR}"
    print_cmd python3 -m venv "${NATIVE_VENV_DIR}"
    python3 -m venv "${NATIVE_VENV_DIR}"
    needs_install="yes"
  fi

  if [[ ! -f "${AGENT_DIR}/api/requirements.base.txt" || ! -f "${AGENT_DIR}/api/requirements.jobs.txt" ]]; then
    log_error "Missing requirements files under api/."
    exit 1
  fi

  if [[ "${needs_install}" != "yes" ]]; then
    if [[ ! -x "${NATIVE_CELERY}" ]]; then
      needs_install="yes"
    elif ! "${NATIVE_PYTHON}" -c "import fastapi, uvicorn, celery, torch, clip" >/dev/null 2>&1; then
      needs_install="yes"
    fi
  fi

  if [[ "${needs_install}" == "yes" ]]; then
    log_info "Installing native dependencies into ${NATIVE_VENV_DIR}..."
    print_cmd "${NATIVE_PYTHON}" -m pip install --upgrade pip
    "${NATIVE_PYTHON}" -m pip install --upgrade pip
    print_cmd "${NATIVE_PYTHON}" -m pip install -r "${AGENT_DIR}/api/requirements.base.txt" -r "${AGENT_DIR}/api/requirements.jobs.txt"
    "${NATIVE_PYTHON}" -m pip install -r "${AGENT_DIR}/api/requirements.base.txt" -r "${AGENT_DIR}/api/requirements.jobs.txt"
  fi

  if [[ ! -x "${NATIVE_CELERY}" ]]; then
    log_error "gpu-metal setup failed: missing celery entrypoint at ${NATIVE_CELERY}"
    exit 1
  fi
  if ! "${NATIVE_PYTHON}" -c "import uvicorn, celery" >/dev/null 2>&1; then
    log_error "gpu-metal setup failed: missing uvicorn/celery in ${NATIVE_VENV_DIR}"
    exit 1
  fi
}

abs_path_under_agent() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s' "${path}"
    return
  fi
  printf '%s/%s' "${AGENT_DIR}" "${path#./}"
}

ensure_native_runtime_dir() {
  mkdir -p "${NATIVE_RUNTIME_DIR}"
}

pid_from_file() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi
  tr -d '[:space:]' < "${pid_file}"
}

is_pid_running() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" >/dev/null 2>&1
}

stop_native_process() {
  local pid_file="$1"
  local name="$2"
  local pid=""
  pid="$(pid_from_file "${pid_file}" || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    return
  fi
  if ! is_pid_running "${pid}"; then
    rm -f "${pid_file}"
    return
  fi
  log_info "Stopping native ${name} (pid=${pid})..."
  kill "${pid}" >/dev/null 2>&1 || true
  local i
  for ((i = 1; i <= 20; i++)); do
    if ! is_pid_running "${pid}"; then
      rm -f "${pid_file}"
      log_ok "Stopped native ${name}."
      return
    fi
    sleep 0.2
  done
  log_warn "Force killing native ${name} (pid=${pid})..."
  kill -9 "${pid}" >/dev/null 2>&1 || true
  rm -f "${pid_file}"
}

stop_native_metal_services() {
  stop_native_process "${NATIVE_WORKER_PID_FILE}" "worker"
  stop_native_process "${NATIVE_API_PID_FILE}" "api"
}

start_native_metal_services() {
  local workspaces_root_abs
  local public_key_abs
  local broker_url
  local queue_name
  local worker_concurrency

  ensure_native_runtime_dir
  stop_native_metal_services

  workspaces_root_abs="$(abs_path_under_agent "${WORKSPACES_HOST_PATH}")"
  public_key_abs="$(abs_path_under_agent "${PUBLIC_KEY_FILE}")"
  broker_url="${XIMA_CELERY_BROKER_URL:-redis://127.0.0.1:6379/0}"
  queue_name="${XIMA_CELERY_QUEUE:-xima_jobs}"
  worker_concurrency="${XIMA_CELERY_WORKER_CONCURRENCY:-1}"

  log_info "Starting native api/worker for gpu-metal profile..."
  : > "${NATIVE_API_LOG_FILE}"
  : > "${NATIVE_WORKER_LOG_FILE}"

  (
    cd "${AGENT_DIR}/api"
    env \
      XIMA_WORKSPACES_ROOT="${workspaces_root_abs}" \
      XIMA_AGENT_PORT="${API_HOST_PORT}" \
      XIMA_AGENT_AUTH_MODE="${AUTH_MODE}" \
      XIMA_AGENT_JWT_AUDIENCE="${XIMA_AGENT_JWT_AUDIENCE:-xima-agent}" \
      XIMA_AGENT_JWT_ISSUER="${XIMA_AGENT_JWT_ISSUER:-xima-saas}" \
      XIMA_JWT_PUBLIC_KEY_PATH="${public_key_abs}" \
      XIMA_JWT_ALG="${XIMA_JWT_ALG:-EdDSA}" \
      XIMA_CELERY_BROKER_URL="${broker_url}" \
      XIMA_CELERY_QUEUE="${queue_name}" \
      "${NATIVE_PYTHON}" -m uvicorn app.main:app --host 127.0.0.1 --port "${API_HOST_PORT}" --workers "${UVICORN_WORKERS_VALUE}" \
      >> "${NATIVE_API_LOG_FILE}" 2>&1 &
    echo "$!" > "${NATIVE_API_PID_FILE}"
  )

  (
    cd "${AGENT_DIR}/api"
    env \
      XIMA_WORKSPACES_ROOT="${workspaces_root_abs}" \
      XIMA_AGENT_PORT="${API_HOST_PORT}" \
      XIMA_CELERY_BROKER_URL="${broker_url}" \
      XIMA_CELERY_QUEUE="${queue_name}" \
      "${NATIVE_CELERY}" -A app.jobs.celery_app.celery_app worker --loglevel=INFO --queues "${queue_name}" --concurrency "${worker_concurrency}" \
      >> "${NATIVE_WORKER_LOG_FILE}" 2>&1 &
    echo "$!" > "${NATIVE_WORKER_PID_FILE}"
  )

  local api_pid
  local worker_pid
  api_pid="$(pid_from_file "${NATIVE_API_PID_FILE}" || true)"
  worker_pid="$(pid_from_file "${NATIVE_WORKER_PID_FILE}" || true)"

  sleep 1
  if ! is_pid_running "${api_pid}"; then
    log_error "Failed to start native api. See ${NATIVE_API_LOG_FILE}"
    return 1
  fi
  if ! is_pid_running "${worker_pid}"; then
    log_error "Failed to start native worker. See ${NATIVE_WORKER_LOG_FILE}"
    return 1
  fi
  log_ok "Native api/worker started (api pid=${api_pid}, worker pid=${worker_pid})."
}

native_metal_active() {
  local api_pid
  local worker_pid
  api_pid="$(pid_from_file "${NATIVE_API_PID_FILE}" || true)"
  worker_pid="$(pid_from_file "${NATIVE_WORKER_PID_FILE}" || true)"
  is_pid_running "${api_pid}" || is_pid_running "${worker_pid}"
}

check_http_once() {
  local url="$1"
  curl -fsS --max-time 3 "${url}" >/dev/null 2>&1
}

check_api_once() {
  if check_http_once "${API_HEALTH_URL}"; then
    API_HEALTH_METHOD="host:27800"
    return 0
  fi
  if "${COMPOSE_CMD[@]}" exec -T api curl -fsS --max-time 3 http://127.0.0.1:27800/health >/dev/null 2>&1; then
    API_HEALTH_METHOD="container:api"
    return 0
  fi
  API_HEALTH_METHOD="none"
  return 1
}

wait_for_api_health() {
  local i
  for ((i = 1; i <= HEALTH_RETRIES_VALUE; i++)); do
    if check_api_once; then
      return 0
    fi
    sleep "${HEALTH_INTERVAL_SEC_VALUE}"
  done
  return 1
}

wait_for_nginx_health() {
  local i
  for ((i = 1; i <= HEALTH_RETRIES_VALUE; i++)); do
    if check_http_once "${NGINX_HEALTH_URL}"; then
      return 0
    fi
    sleep "${HEALTH_INTERVAL_SEC_VALUE}"
  done
  return 1
}

show_health_snapshot() {
  API_HEALTH_STATUS="NG"
  NGINX_HEALTH_STATUS="NG"

  if check_api_once; then
    API_HEALTH_STATUS="OK (${API_HEALTH_METHOD})"
  fi
  if check_http_once "${NGINX_HEALTH_URL}"; then
    NGINX_HEALTH_STATUS="OK (host:${NGINX_HOST_PORT})"
  fi

  log_info "Health snapshot:"
  echo "  - API   : ${API_HEALTH_STATUS}"
  echo "  - nginx : ${NGINX_HEALTH_STATUS}"
}

stop_all_services() {
  log_info "Stopping native services..."
  stop_native_metal_services
  log_info "Stopping containers..."
  ensure_compose_available
  build_runtime_env_vars
  build_compose_cmd "gpu-nvidia"
  print_cmd env "${RUNTIME_ENV_VARS[@]}" "${COMPOSE_CMD[@]}" stop
  env "${RUNTIME_ENV_VARS[@]}" "${COMPOSE_CMD[@]}" stop || true
  build_compose_cmd "gpu-metal"
  print_cmd env "${RUNTIME_ENV_VARS[@]}" "${COMPOSE_CMD[@]}" stop
  env "${RUNTIME_ENV_VARS[@]}" "${COMPOSE_CMD[@]}" stop || true
  build_compose_cmd "cpu"
  print_cmd env "${RUNTIME_ENV_VARS[@]}" "${COMPOSE_CMD[@]}" stop
  env "${RUNTIME_ENV_VARS[@]}" "${COMPOSE_CMD[@]}" stop || true
}

run_up() {
  local build_mode="$1" # yes|no
  local config_mode="${2:-load}" # load|current

  refresh_health_urls
  API_HEALTH_STATUS="waiting"
  NGINX_HEALTH_STATUS="waiting"

  log_info "Checking Docker / Compose availability..."
  ensure_compose_available
  require_command curl

  if [[ "${config_mode}" == "load" ]]; then
    load_or_init_launcher_config
  fi
  validate_runtime_settings
  refresh_health_urls
  validate_local_files "${GPU_PROFILE}" "${AUTH_MODE}"
  ensure_workspaces_host_path
  build_compose_cmd "${GPU_PROFILE}"
  build_runtime_env_vars
  warn_gpu_runtime_if_needed
  print_runtime_plan

  if [[ "${GPU_PROFILE}" == "gpu-metal" ]]; then
    ensure_native_metal_env
  fi

  local up_args=(up -d)
  BUILD_FLAG="no"
  if [[ "${build_mode}" == "yes" ]]; then
    up_args+=(--build)
    BUILD_FLAG="yes"
  else
    # Keep existing containers and only start/create missing ones.
    up_args+=(--no-recreate)
  fi

  if [[ "${GPU_PROFILE}" != "gpu-metal" ]]; then
    stop_native_metal_services
  fi

  log_info "Starting containers..."
  print_cmd env "${RUNTIME_ENV_VARS[@]}" "${COMPOSE_CMD[@]}" "${up_args[@]}"
  SUMMARY_COMMAND="${LAST_COMMAND}"
  env "${RUNTIME_ENV_VARS[@]}" "${COMPOSE_CMD[@]}" "${up_args[@]}"

  if [[ "${GPU_PROFILE}" == "gpu-metal" ]]; then
    if ! start_native_metal_services; then
      print_result_summary "FAILED"
      exit 1
    fi
  fi

  log_info "Waiting for health endpoints..."
  local failed=0
  if wait_for_api_health; then
    API_HEALTH_STATUS="OK (${API_HEALTH_METHOD})"
    log_ok "API health OK (${API_HEALTH_METHOD})"
  else
    API_HEALTH_STATUS="NG (${API_HEALTH_URL} + container exec)"
    log_error "API health NG (${API_HEALTH_URL} and container exec)"
    failed=1
  fi
  if wait_for_nginx_health; then
    NGINX_HEALTH_STATUS="OK (host:${NGINX_HOST_PORT})"
    log_ok "nginx health OK (host:${NGINX_HOST_PORT})"
  else
    NGINX_HEALTH_STATUS="NG (${NGINX_HEALTH_URL})"
    log_error "nginx health NG (${NGINX_HEALTH_URL})"
    failed=1
  fi

  if [[ "${failed}" -ne 0 ]]; then
    log_error "Startup completed but health checks failed. Inspect status/logs below."
    print_cmd "${COMPOSE_CMD[@]}" ps
    "${COMPOSE_CMD[@]}" ps || true
    if [[ "${GPU_PROFILE}" == "gpu-metal" ]]; then
      print_cmd "${COMPOSE_CMD[@]}" logs --tail=120 nginx redis
      "${COMPOSE_CMD[@]}" logs --tail=120 nginx redis || true
      echo "--- native api log (tail) ---"
      tail -n 120 "${NATIVE_API_LOG_FILE}" || true
      echo "--- native worker log (tail) ---"
      tail -n 120 "${NATIVE_WORKER_LOG_FILE}" || true
    else
      print_cmd "${COMPOSE_CMD[@]}" logs --tail=120 api nginx
      "${COMPOSE_CMD[@]}" logs --tail=120 api nginx || true
    fi
    print_result_summary "FAILED"
    exit 1
  fi

  print_result_summary "SUCCESS"
}

prompt_runtime_config_for_rebuild() {
  apply_default_launcher_config
  BUILD_FLAG="yes"
  ADVANCED_MODE="no"

  prompt_auth_mode
  prompt_gpu_profile
  prompt_advanced_mode

  if [[ "${ADVANCED_MODE}" == "yes" ]]; then
    prompt_nginx_port
    prompt_workspaces_host_path
    prompt_compose_project_name
    prompt_uvicorn_workers
    prompt_health_wait_settings
  fi

  validate_runtime_settings
  refresh_health_urls
  confirm_start_plan
  save_launcher_config
  log_info "Updated launcher config: ${LAUNCHER_CONFIG_FILE}"
}

run_start() {
  OPERATION="start"
  run_up "no" "load"
}

run_stop() {
  OPERATION="stop"
  load_or_init_launcher_config
  validate_runtime_settings
  refresh_health_urls
  stop_all_services
  SUMMARY_COMMAND="${LAST_COMMAND}"
  log_ok "Containers stopped (not removed)."
  print_result_summary "SUCCESS"
}

run_restart() {
  OPERATION="restart"
  load_or_init_launcher_config
  validate_runtime_settings
  refresh_health_urls
  log_info "Restarting all services..."
  stop_all_services
  log_ok "All services stopped. Proceeding to start flow."
  run_up "no" "load"
}

run_rebuild() {
  OPERATION="rebuild"
  prompt_runtime_config_for_rebuild
  log_info "Rebuilding selected runtime services..."
  stop_all_services
  log_ok "All services stopped. Proceeding to rebuild/start flow."
  run_up "yes" "current"
}

run_logs() {
  OPERATION="log"
  ensure_compose_available
  load_or_init_launcher_config
  validate_runtime_settings
  build_compose_cmd "${GPU_PROFILE}"
  prompt_logs_service
  if [[ "${GPU_PROFILE}" == "gpu-metal" && "${LOG_SERVICE}" != "all" && "${LOG_SERVICE}" != "nginx" && "${LOG_SERVICE}" != "redis" ]] && ! native_metal_active; then
    log_error "gpu-metal profile requires native api/worker running for '${LOG_SERVICE}' logs."
    log_error "Use 'start' first, or select nginx/redis logs."
    exit 1
  fi
  if native_metal_active; then
    case "${LOG_SERVICE}" in
      api)
        log_info "Streaming native api logs. Press Ctrl+C to stop."
        print_cmd tail -n 200 -f "${NATIVE_API_LOG_FILE}"
        tail -n 200 -f "${NATIVE_API_LOG_FILE}"
        return
        ;;
      worker)
        log_info "Streaming native worker logs. Press Ctrl+C to stop."
        print_cmd tail -n 200 -f "${NATIVE_WORKER_LOG_FILE}"
        tail -n 200 -f "${NATIVE_WORKER_LOG_FILE}"
        return
        ;;
      all)
        log_info "Streaming native api/worker + container nginx/redis logs. Press Ctrl+C to stop."
        print_cmd "${COMPOSE_CMD[@]}" logs -f --tail=200 nginx redis
        "${COMPOSE_CMD[@]}" logs -f --tail=200 nginx redis &
        local compose_logs_pid="$!"
        tail -n 200 -f "${NATIVE_API_LOG_FILE}" "${NATIVE_WORKER_LOG_FILE}"
        kill "${compose_logs_pid}" >/dev/null 2>&1 || true
        return
        ;;
    esac
  fi
  log_info "Streaming logs. Press Ctrl+C to stop."
  if [[ "${LOG_SERVICE}" == "all" ]]; then
    print_cmd "${COMPOSE_CMD[@]}" logs -f --tail=200
    "${COMPOSE_CMD[@]}" logs -f --tail=200
  else
    print_cmd "${COMPOSE_CMD[@]}" logs -f --tail=200 "${LOG_SERVICE}"
    "${COMPOSE_CMD[@]}" logs -f --tail=200 "${LOG_SERVICE}"
  fi
}

run_health() {
  OPERATION="health"
  load_or_init_launcher_config
  validate_runtime_settings
  refresh_health_urls
  log_info "Checking health endpoints..."
  ensure_compose_available
  require_command curl
  build_compose_cmd "${GPU_PROFILE}"
  show_health_snapshot
  SUMMARY_COMMAND="curl ${API_HEALTH_URL} && curl ${NGINX_HEALTH_URL}"
  if [[ "${API_HEALTH_STATUS}" == OK* && "${NGINX_HEALTH_STATUS}" == OK* ]]; then
    print_result_summary "SUCCESS"
  else
    print_result_summary "FAILED"
    exit 1
  fi
}

# Create a demo workspace/experiment with generated sample images so a new user
# can try labeling -> training without preparing their own dataset.
run_demo() {
  OPERATION="demo"
  load_or_init_launcher_config
  validate_runtime_settings
  refresh_health_urls
  ensure_compose_available
  require_command curl

  log_info "Creating demo workspace via http://127.0.0.1:${API_HOST_PORT}/demo ..."
  local response
  if ! response="$(curl -fsS -X POST "http://127.0.0.1:${API_HOST_PORT}/demo" 2>&1)"; then
    log_error "Failed to create demo. Is the agent running? (try: start)"
    log_error "${response}"
    SUMMARY_COMMAND="curl -X POST http://127.0.0.1:${API_HOST_PORT}/demo"
    print_result_summary "FAILED"
    exit 1
  fi

  echo "${response}"
  log_info "Demo created. Open the UI and select the demo workspace."
  SUMMARY_COMMAND="curl -X POST http://127.0.0.1:${API_HOST_PORT}/demo"
  print_result_summary "SUCCESS"
}

main() {
  cd "${AGENT_DIR}"
  prompt_operation
  case "${OPERATION}" in
    start)
      run_start
      ;;
    stop)
      run_stop
      ;;
    log)
      run_logs
      ;;
    restart)
      run_restart
      ;;
    rebuild)
      run_rebuild
      ;;
    health)
      run_health
      ;;
    demo)
      run_demo
      ;;
    *)
      log_error "Unsupported operation: ${OPERATION}"
      exit 1
      ;;
  esac
}

main "$@"
