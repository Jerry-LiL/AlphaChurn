#!/bin/sh
set -e

if [ "$STARTUP_INGEST" = "1" ]; then
  python -m documind.ingest --paths "${DOCS_PATH:-/app/documind/sample_docs}"
fi

exec uvicorn documind.api:app --host 0.0.0.0 --port 8000
