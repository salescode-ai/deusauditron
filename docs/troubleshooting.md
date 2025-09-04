# Troubleshooting

## Evaluation stuck In_Progress

- Check provider API keys
- Verify queue backend (local/redis) is running
- Inspect logs; enable debug if needed

## 404 on GET

- Evaluation may not have been requested
- Ensure correct tenant/agent/run IDs

## 409 on DELETE

- Evaluation still In_Progress; wait for completion

## S3 persistence errors

- `write_to_s3` may not be implemented in local builds (by design). Disable persist or implement real S3.
