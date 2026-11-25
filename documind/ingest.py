"""Load PDF and Markdown files, chunk them, and index the chunks."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documind.config import Settings
from documind.embeddings import build_embedding
from documind.pdftext import read_pdf_pages
from documind.sqltool import init_catalog, upsert_document

COLLECTION = "documents"
_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def load_pages(path: Path, root: Path | None = None, category: str | None = None) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _markdown_pages(path, root, category)
    if suffix == ".pdf":
        return _pdf_pages(path, root, category)
    raise ValueError(f"Unsupported file type: {path.name}")


def chunk_pages(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for page in pages:
        parts = splitter.split_text(page["text"])
        for index, part in enumerate(parts):
            text = part.strip()
            if not text:
                continue
            chunks.append(
                {
                    "id": f"{page['doc']}:{page['page']}:{index}",
                    "doc": page["doc"],
                    "page": int(page["page"]),
                    "text": text,
                    "source": page["source"],
                    "file_type": page["file_type"],
                    "category": page["category"],
                    "chunk_index": index,
                }
            )
    return chunks


def ingest_paths(
    paths: list[Path],
    settings: Settings,
    category: str | None = None,
) -> dict:
    files = _expand(paths)
    stems = [path.stem for path in files]
    if len(stems) != len(set(stems)):
        raise ValueError("Document file names must be unique so citations stay unambiguous.")

    root = _ingest_root(paths, files)
    stored = _read_store(settings.chunk_store)
    total_chunks = 0
    for path in files:
        pages = load_pages(path, root=root, category=category)
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            continue
        doc = chunks[0]["doc"]
        stored = [row for row in stored if row["doc"] != doc]
        stored.extend(chunks)
        _replace_in_chroma(settings, doc, chunks)
        upsert_document(
            settings.catalog_path,
            {
                "doc": doc,
                "title": doc,
                "category": chunks[0]["category"],
                "file_type": chunks[0]["file_type"],
                "pages": max(int(page["page"]) for page in pages),
                "chunks": len(chunks),
                "source": chunks[0]["source"],
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        total_chunks += len(chunks)
    _write_store(settings.chunk_store, stored)
    init_catalog(settings.catalog_path)
    return {"documents": len(files), "chunks": total_chunks}


def load_chunks(settings: Settings) -> list[dict]:
    return _read_store(settings.chunk_store)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index PDF and Markdown documents.")
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--category", default=None)
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    stats = ingest_paths([Path(path) for path in args.paths], settings, category=args.category)
    print(json.dumps(stats))


def _markdown_pages(path: Path, root: Path | None, category: str | None) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    meta, body = _frontmatter(raw)
    chosen = category or meta.get("category") or _folder_category(path, root)
    pages = body.split("\f")
    return [
        _page(path, index, text, "markdown", chosen)
        for index, text in enumerate(pages, start=1)
        if text.strip()
    ]


def _pdf_pages(path: Path, root: Path | None, category: str | None) -> list[dict]:
    chosen = category or _folder_category(path, root)
    return [
        _page(path, index, text, "pdf", chosen)
        for index, text in enumerate(read_pdf_pages(path), start=1)
        if text.strip()
    ]


def _page(path: Path, page: int, text: str, file_type: str, category: str) -> dict:
    return {
        "doc": path.stem,
        "page": page,
        "text": text.strip(),
        "source": str(path),
        "file_type": file_type,
        "category": category,
    }


def _frontmatter(text: str) -> tuple[dict, str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, text[match.end() :]


def _folder_category(path: Path, root: Path | None) -> str:
    if root is not None and path.parent.resolve() != root.resolve():
        return path.parent.name
    if root is None and path.parent.name:
        return path.parent.name
    return "general"


def _expand(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(
                    item
                    for item in path.rglob("*")
                    if item.suffix.lower() in {".md", ".markdown", ".pdf"} and item.is_file()
                )
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError("No PDF or Markdown files found.")
    return files


def _ingest_root(paths: list[Path], files: list[Path]) -> Path:
    directories = [path.resolve() for path in paths if path.is_dir()]
    if len(directories) == 1:
        return directories[0]
    return _common_root(files)


def _common_root(files: list[Path]) -> Path:
    parents = [path.parent.resolve() for path in files]
    shared = Path(parents[0])
    def outside(path: Path) -> bool:
        return shared not in path.parents and path != shared

    while shared != shared.parent and any(outside(path) for path in parents):
        shared = shared.parent
    return shared


def _read_store(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("chunks", []))


def _write_store(path: Path, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"chunks": chunks}, indent=2), encoding="utf-8")


def _replace_in_chroma(settings: Settings, doc: str, chunks: list[dict]) -> None:
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=build_embedding(settings),
    )
    existing = collection.get(where={"doc": doc})
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "doc": chunk["doc"],
                "page": int(chunk["page"]),
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "category": chunk["category"],
                "chunk_index": int(chunk["chunk_index"]),
            }
            for chunk in chunks
        ],
    )


if __name__ == "__main__":
    main()
