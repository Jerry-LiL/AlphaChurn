from documind.ingest import ingest_paths
from documind.retrieve import hybrid_search, reciprocal_rank_fusion


def test_rrf_promotes_a_chunk_both_rankers_like():
    fused = reciprocal_rank_fusion(
        [
            ["a", "b", "c"],
            ["b", "c", "a"],
        ]
    )
    assert fused[0][0] == "b"
    assert fused[0][1] > fused[1][1]


def test_hybrid_search_finds_the_lexical_page(settings, tmp_path):
    billing = tmp_path / "billing.md"
    billing.write_text(
        "Invoices are issued on the first business day of each month.\f"
        "Late fees of 1.5 percent apply after 15 days.\n",
        encoding="utf-8",
    )
    other = tmp_path / "onboarding.md"
    other.write_text("Laptops are issued on day one by workplace IT.\n", encoding="utf-8")
    ingest_paths([billing, other], settings)
    hits = hybrid_search(settings, "late fees after 15 days", k=2)
    assert hits
    assert hits[0]["doc"] == "billing"
    assert hits[0]["page"] == 2
