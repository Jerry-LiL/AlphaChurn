"""Answer contract. Citations have to show up in the text as [Doc:Page]."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+):(\d+)\]")


class Citation(BaseModel):
    doc: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)


class AnswerBody(BaseModel):
    route: str = Field(pattern=r"^(retrieve|sql|clarify)$")
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool

    @model_validator(mode="after")
    def citations_match_the_text(self) -> AnswerBody:
        found = {(doc, int(page)) for doc, page in CITATION_RE.findall(self.answer)}
        if self.route == "clarify":
            if found or self.citations:
                raise ValueError("A clarification does not cite a document page.")
            return self
        if not self.citations:
            raise ValueError("An answer needs at least one [Doc:Page] citation.")
        declared = {(item.doc, item.page) for item in self.citations}
        for item in self.citations:
            token = f"[{item.doc}:{item.page}]"
            if token not in self.answer:
                raise ValueError(f"Missing inline citation {token}.")
        if found != declared:
            raise ValueError("Inline citations and the citation list do not match.")
        return self


class QuestionIn(BaseModel):
    question: str = Field(min_length=1)
