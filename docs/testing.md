# Testing Guide

Run tests via Make or pytest.

## Quick start

```bash
source venv/bin/activate
make test
# or
pytest -q
```

## Structure

- `deusauditron/tests/` contains API and unit tests
- Uses httpx AsyncClient with ASGITransport

## Writing tests

- Prefer async tests with pytest-asyncio
- Seed state via `StateManager().set_eval_state(...)` and any other setup
- Use local backends for hermetic tests unless you specifically test Redis
