#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required." >&2
  exit 1
}
docker compose version >/dev/null
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is unavailable or this account cannot access its socket." >&2
  echo "Start Docker and grant this account Docker access, then rerun ./deploy.sh." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example to .env and replace CHANGE_ME values." >&2
  exit 1
fi
if grep -Eq '(^|=)CHANGE_ME' .env; then
  echo ".env still contains CHANGE_ME placeholders." >&2
  exit 1
fi
for required_name in \
  GEMINI_API_KEY POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB DATABASE_URL; do
  if ! grep -Eq "^${required_name}=.+" .env; then
    echo ".env must define a non-empty ${required_name}." >&2
    exit 1
  fi
done
if ! grep -Eq '^DATABASE_URL=.+@db:5432/' .env; then
  echo "DATABASE_URL must use the Compose database host db:5432." >&2
  exit 1
fi

docker compose config --quiet
docker compose build --pull
docker compose up -d --remove-orphans

for attempt in $(seq 1 60); do
  unhealthy=""
  for service in db backend frontend; do
    container_id="$(docker compose ps -q "$service")"
    if [[ -z "$container_id" ]]; then
      unhealthy="${unhealthy}${service} "
      continue
    fi
    state="$(docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' "$container_id")"
    if [[ "$state" != "running healthy" ]]; then
      unhealthy="${unhealthy}${service} "
    fi
  done
  if [[ -z "$unhealthy" ]]; then
    docker compose ps
    echo "Deployment is healthy."
    exit 0
  fi
  sleep 2
done

docker compose ps
docker compose logs --tail=100 db backend frontend
echo "Deployment did not become healthy within 120 seconds." >&2
exit 1
