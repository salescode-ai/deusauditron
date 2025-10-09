# Configuration & Environment

Deusauditron reads configuration from environment variables.

## Core

- `HOST` (default: 0.0.0.0)
- `PORT` (default: 8081)
- `CORS_ORIGINS` (comma-separated)

## Backends

- `STATE_BACKEND`: `local` or `redis`
- `QUEUE_BACKEND`: `local` or `redis`
- `LOCK_BACKEND`: `local` or `redis`
- `REDIS_URL`: e.g., `redis://localhost:6379/0`

## LLM

- Provider-specific API keys can be supplied in the request payload (`api_keys`) or via env vars used by your adapter.

## Tracing

- `TRACING_ENABLED` (true/false)
- `PHOENIX_COLLECTOR_ENDPOINT` (http://host:6006)

## Persistence

- `persist_path` (request payload): `bucket/key.json` for S3 persistence (if implemented)
