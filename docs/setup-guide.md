# Setup Guide for Co-developers

This guide helps you set up, run, and test Deusauditron locally.

## Prerequisites

- Python 3.11+
- make, gcc/clang toolchain
- Optional: Redis (if using Redis-backed state/queue/locks)

## Quick Start

```bash
cd /path/to/deusauditron
make venv
source venv/bin/activate
make install
make run
```

Service runs at http://localhost:8080/api/v1 (Swagger: http://localhost:8080/docs)

## Running Tests

```bash
source venv/bin/activate
make test
```

## Optional Environment Variables

- `HOST` (default: 0.0.0.0)
- `PORT` (default: 8080)
- `REDIS_URL` (e.g., redis://localhost:6379/0)
- `STATE_BACKEND` (local|redis; default based on config)
- `QUEUE_BACKEND` (local|redis; default based on config)
- `LOCK_BACKEND` (local|redis; default based on config)
- `TRACING_ENABLED` (true|false; default: false)
- `PHOENIX_COLLECTOR_ENDPOINT` (if tracing enabled)

Export these in your shell or a `.env` file loaded by your IDE.

## Helpful Commands

```bash
make run          # start uvicorn with the service
make test         # run all tests
make lint         # linter
make format       # format code
```
