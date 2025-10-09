# Deployment Guide

## Local

```bash
make venv && source venv/bin/activate
make install
make run
```

## Docker (example)

```bash
docker build -t deusauditron:latest .
docker run -p 8081:8081 --env-file .env deusauditron:latest
```

## Configuration

- Provide env vars for backends (STATE/QUEUE/LOCK/REDIS_URL)
- Disable tracing unless collector is configured
