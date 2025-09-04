# Performance & Scaling

## Async-first

- All I/O is async; avoid blocking calls on the event loop
- Parallelize eval strategies via `asyncio.gather`

## Backends

- Use Redis backends for queue, lock, and state under load
- Size Redis connection pool appropriately

## Caching

- Consider caching static prompt content if needed

## Metrics

- Use tracing spans to measure model latencies and queue times

## Concurrency

- Background consumer runs continuously; tune concurrency by running multiple workers or pods
