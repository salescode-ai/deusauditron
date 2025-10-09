# Make Commands

Common developer workflow:

```bash
make venv      # create virtualenv
source venv/bin/activate
make install   # install deps
make run       # run service on 0.0.0.0:8081
make test      # run pytest
make lint      # run linter
make format    # format code
```

If you use Redis-backed backends, ensure `REDIS_URL` is set.
