#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.prod.yml"
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE=(docker compose --project-directory "${SCRIPT_DIR}" --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}")

info() {
  printf '\033[1;34m[deploy]\033[0m %s\n' "$*"
}

success() {
  printf '\033[1;32m[ok]\033[0m %s\n' "$*"
}

fail() {
  printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
EdgeMind production deployment

Usage:
  ./deploy.sh deploy   Build, start, and verify all services (default)
  ./deploy.sh smoke    Verify the backend and frontend proxy
  ./deploy.sh status   Show container status
  ./deploy.sh logs     Follow service logs
  ./deploy.sh restart  Restart application containers
  ./deploy.sh stop     Stop containers without deleting database data
  ./deploy.sh help     Show this help

Optional .env values:
  APP_PORT=5173       Public Web UI port
  BACKEND_PORT=8000   Backend port bound to 127.0.0.1
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "找不到必要指令：$1"
}

env_value() {
  local key="$1"
  local fallback="${2:-}"
  local value
  value="$(sed -n -E "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*(.*)$/\\1/p" "${ENV_FILE}" | tail -n 1)"
  value="${value%$'\r'}"
  value="${value#\"}"
  value="${value%\"}"
  value="${value#\'}"
  value="${value%\'}"
  printf '%s' "${value:-${fallback}}"
}

validate_environment() {
  [[ -f "${ENV_FILE}" ]] || fail ".env 不存在；請先執行 cp .env.example .env 並填入設定"

  local required key value
  required=(GEMINI_API_KEY POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB DATABASE_URL)
  for key in "${required[@]}"; do
    value="$(env_value "${key}")"
    [[ -n "${value}" ]] || fail ".env 缺少 ${key}"
    case "${value}" in
      *CHANGE_ME*|*replace_with_your_key*|*replace_with_a_strong_password*)
        fail ".env 的 ${key} 仍是範例值"
        ;;
    esac
  done

  if [[ "$(env_value DATABASE_URL)" != *"@db:5432/"* ]]; then
    fail "Docker 部署的 DATABASE_URL 必須使用 db:5432"
  fi

  "${COMPOSE[@]}" config --quiet
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local attempts="${3:-60}"
  local index
  for ((index = 1; index <= attempts; index += 1)); do
    if curl --fail --silent --show-error --max-time 4 "${url}" >/dev/null 2>&1; then
      success "${name} 已就緒：${url}"
      return 0
    fi
    sleep 2
  done
  return 1
}

show_failure_logs() {
  local exit_code=$?
  if ((exit_code != 0)); then
    local fail_message
    printf '\n' >&2
    fail_message="部署失敗，以下是最近的容器狀態與日誌"
    printf '\033[1;31m[error]\033[0m %s\n' "${fail_message}" >&2
    "${COMPOSE[@]}" ps >&2 || true
    "${COMPOSE[@]}" logs --tail=120 db backend frontend >&2 || true
  fi
  exit "${exit_code}"
}

smoke_test() {
  require_command curl
  local app_port backend_port
  app_port="$(env_value APP_PORT 5173)"
  backend_port="$(env_value BACKEND_PORT 8000)"

  if ! wait_for_url "後端 API" "http://127.0.0.1:${backend_port}/api/health" 30; then
    printf '\033[1;31m[error]\033[0m 後端 API 在等待時間內未就緒\n' >&2
    return 1
  fi
  if ! wait_for_url "前端與 API 代理" "http://127.0.0.1:${app_port}/api/health" 30; then
    printf '\033[1;31m[error]\033[0m 前端或 /api 代理在等待時間內未就緒\n' >&2
    return 1
  fi
}

deploy() {
  require_command docker
  require_command curl
  docker compose version >/dev/null 2>&1 || fail "Docker Compose plugin 無法使用"
  validate_environment

  mkdir -p "${SCRIPT_DIR}/backend/outputs"
  info "建置 production images"
  "${COMPOSE[@]}" build --pull
  info "啟動 PostgreSQL、FastAPI 與 Nginx"
  "${COMPOSE[@]}" up -d --remove-orphans
  smoke_test
  "${COMPOSE[@]}" ps

  local app_port
  app_port="$(env_value APP_PORT 5173)"
  success "EdgeMind 部署完成：http://localhost:${app_port}"
  info "推論報表位置：${SCRIPT_DIR}/backend/outputs/inference_runs"
}

main() {
  local command="${1:-deploy}"
  cd "${SCRIPT_DIR}"
  case "${command}" in
    deploy)
      trap show_failure_logs ERR
      deploy
      trap - ERR
      ;;
    smoke)
      validate_environment
      smoke_test
      ;;
    status)
      validate_environment
      "${COMPOSE[@]}" ps
      ;;
    logs)
      validate_environment
      "${COMPOSE[@]}" logs --tail=200 --follow db backend frontend
      ;;
    restart)
      validate_environment
      "${COMPOSE[@]}" restart backend frontend
      smoke_test
      ;;
    stop)
      validate_environment
      "${COMPOSE[@]}" down
      success "服務已停止；PostgreSQL volume 與 backend/outputs 均保留"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      usage >&2
      fail "未知指令：${command}"
      ;;
  esac
}

main "$@"
