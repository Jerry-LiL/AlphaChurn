"""Small text-only PDF writer and reader.

The writer exists so the sample billing document and the tests can build a
multi-page PDF without another dependency. Ingestion reads PDFs with pypdf.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def write_text_pdf(path: Path, pages: list[str]) -> None:
    if not pages:
        raise ValueError("A PDF needs at least one page.")
    objects: dict[int, bytes] = {}
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    page_ids: list[int] = []
    next_id = 4
    for page_text in pages:
        content_id = next_id
        page_id = next_id + 1
        next_id += 2
        lines = page_text.splitlines() or [""]
        commands = ["BT", "/F1 12 Tf", "72 740 Td", "16 TL"]
        for index, line in enumerate(lines):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if index == 0:
                commands.append(f"({escaped}) Tj")
            else:
                commands.append(f"T* ({escaped}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", errors="replace")
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode()
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[2] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode()
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    blob = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    max_id = max(objects)
    for obj_id in range(1, max_id + 1):
        offsets.append(len(blob))
        blob.extend(f"{obj_id} 0 obj\n".encode())
        blob.extend(objects[obj_id])
        blob.extend(b"\nendobj\n")
    xref = len(blob)
    blob.extend(f"xref\n0 {max_id + 1}\n".encode())
    blob.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, max_id + 1):
        blob.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode())
    blob.extend(
        f"trailer << /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(blob))


def read_pdf_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    return [(page.extract_text() or "").strip() for page in reader.pages]
