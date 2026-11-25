FROM python:3.12-slim

WORKDIR /app

COPY documind/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY documind /app/documind
RUN chmod +x /app/documind/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    LLM_PROVIDER=local \
    EMBEDDING_PROVIDER=local \
    CHROMA_PATH=/app/data/chroma \
    CATALOG_PATH=/app/data/catalog.sqlite \
    CHUNK_STORE=/app/data/chunks.json

EXPOSE 8000

ENTRYPOINT ["/app/documind/entrypoint.sh"]
