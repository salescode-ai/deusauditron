#!/bin/bash
set -e
set -x

# Pre-reqs:
#   pip install datamodel-code-generator black

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"

SCHEMA_DIR="$PROJECT_ROOT/schemas/json"
REFERENCES_DIR="$SCHEMA_DIR/references"
OUTPUT_DIR="$PROJECT_ROOT/schemas/autogen"
REF_OUTPUT_DIR="$OUTPUT_DIR/references"

mkdir -p "$REF_OUTPUT_DIR"
rm -f "$REF_OUTPUT_DIR"/*.py || true
rm -f "$OUTPUT_DIR"/*.py || true

TEMP_SCHEMA_DIR=$(mktemp -d)
TEMP_REF_DIR=$(mktemp -d)

rsync -a --include='*/' --include='*.json' --exclude='*' "$SCHEMA_DIR/" "$TEMP_SCHEMA_DIR/"
rsync -a --include='*/' --include='*.json' --exclude='*' "$REFERENCES_DIR/" "$TEMP_REF_DIR/"

echo "Generating reference models → $REF_OUTPUT_DIR"
datamodel-codegen \
  --input "$TEMP_REF_DIR" \
  --input-file-type jsonschema \
  --output "$REF_OUTPUT_DIR" \
  --output-model-type pydantic_v2.BaseModel \
  --use-schema-description \
  --disable-timestamp

echo "Generating top-level models → $OUTPUT_DIR"
datamodel-codegen \
  --input "$TEMP_SCHEMA_DIR" \
  --input-file-type jsonschema \
  --output "$OUTPUT_DIR" \
  --output-model-type pydantic_v2.BaseModel \
  --use-schema-description \
  --disable-timestamp

rm -rf "$TEMP_SCHEMA_DIR" "$TEMP_REF_DIR"

echo "Auto-generating __init__.py files"
echo "# Auto-generated init" > "$OUTPUT_DIR/__init__.py"
for file in "$OUTPUT_DIR"/*.py; do
  modname=$(basename "$file" .py)
  if [[ "$modname" != "__init__" ]]; then
    echo "from .${modname} import *" >> "$OUTPUT_DIR/__init__.py"
  fi
done

echo "# Auto-generated init" > "$REF_OUTPUT_DIR/__init__.py"
for file in "$REF_OUTPUT_DIR"/*.py; do
  modname=$(basename "$file" .py)
  if [[ "$modname" != "__init__" ]]; then
    echo "from .${modname} import *" >> "$REF_OUTPUT_DIR/__init__.py"
  fi
done

echo "Formatting"
black "$OUTPUT_DIR"

echo "✅ Model generation complete."