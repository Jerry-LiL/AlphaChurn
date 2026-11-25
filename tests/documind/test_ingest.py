
from documind.ingest import chunk_pages, ingest_paths, load_pages
from documind.pdftext import read_pdf_pages, write_text_pdf


def test_pdf_round_trip_keeps_page_text(tmp_path):
    path = tmp_path / "billing.pdf"
    write_text_pdf(
        path,
        [
            "Invoices are issued on the first business day of each month.",
            "Late fees of 1.5 percent apply after 15 days.",
        ],
    )
    pages = read_pdf_pages(path)
    assert len(pages) == 2
    assert "first business day" in pages[0]
    assert "Late fees" in pages[1]


def test_markdown_form_feed_sets_page_numbers(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("Alpha is on the first page.\fBeta is on the second page.\n", encoding="utf-8")
    pages = load_pages(path)
    assert [page["page"] for page in pages] == [1, 2]
    assert pages[1]["doc"] == "notes"
    assert pages[0]["file_type"] == "markdown"


def test_recursive_chunks_keep_page_metadata():
    page = {
        "doc": "handbook",
        "page": 3,
        "text": " ".join(["policy"] * 500),
        "source": "handbook.md",
        "file_type": "markdown",
        "category": "handbook",
    }
    chunks = chunk_pages([page], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert {chunk["page"] for chunk in chunks} == {3}
    assert {chunk["category"] for chunk in chunks} == {"handbook"}
    assert all(chunk["text"] for chunk in chunks)
    assert all(chunk["chunk_index"] == index for index, chunk in enumerate(chunks))


def test_ingest_tags_category_from_the_folder(settings, tmp_path):
    root = tmp_path / "docs"
    policy = root / "policies"
    policy.mkdir(parents=True)
    (policy / "refunds.md").write_text(
        "Plus plan customers have a refund window of 14 days from the charge date.\n",
        encoding="utf-8",
    )
    stats = ingest_paths([root], settings)
    assert stats["documents"] == 1
    assert stats["chunks"] >= 1
    stored = settings.chunk_store.read_text(encoding="utf-8")
    assert '"category": "policies"' in stored
    assert '"file_type": "markdown"' in stored
