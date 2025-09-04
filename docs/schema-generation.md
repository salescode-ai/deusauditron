# Schema Generation (Pydantic Autogen)

Deusauditron ships with auto-generated Pydantic models under `schemas/autogen/`. These are generated from source JSON schemas in `schemas/json/`.

## When to Regenerate

- After any change to `schemas/json/*.json` or `schemas/json/references/*.json`

## Requirements

```bash
pip install -r requirements-dev.txt
```

## Command

```bash
bash scripts/generate_models.sh
```

This will:

- Copy JSON schema files to temp dirs
- Run `datamodel-codegen` for references then top-level schemas
- Regenerate `schemas/autogen/` files
- Create `__init__.py` shims and run Black formatting

## Notes

- Do not manually edit files in `schemas/autogen/` (they are overwritten)
- Commit both the JSON source changes and regenerated Python models
