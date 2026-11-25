import pytest
from pydantic import ValidationError

from documind.schemas import AnswerBody, Citation


def test_inline_citation_is_required():
    AnswerBody(
        route="retrieve",
        answer="Plus plan customers have a refund window of 14 days [refunds:1].",
        citations=[Citation(doc="refunds", page=1, quote="14 days")],
        grounded=True,
    )
    with pytest.raises(ValidationError):
        AnswerBody(
            route="retrieve",
            answer="Plus plan customers have a refund window of 14 days.",
            citations=[Citation(doc="refunds", page=1, quote="14 days")],
            grounded=True,
        )


def test_clarification_cannot_cite():
    AnswerBody(route="clarify", answer="Which topic should I look up?", citations=[], grounded=True)
    with pytest.raises(ValidationError):
        AnswerBody(
            route="clarify",
            answer="See [refunds:1].",
            citations=[],
            grounded=True,
        )
