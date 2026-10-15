"""Pydantic data models for datasets, search results and answers."""

from pydantic import BaseModel, model_validator, Field
from typing_extensions import Self
import uuid


class MinimalSource(BaseModel):
    """A span of characters inside a file of the corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def check_indexes(self) -> Self:
        """Check that the span is not empty or reversed."""
        if self.last_character_index <= self.first_character_index:
            raise ValueError("last_character_index can't be lower"
                             " or equal to first_character_index")
        return self


class UnansweredQuestion(BaseModel):
    """A question received as input, without its answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference answer and reference sources."""

    answer: str
    sources: list[MinimalSource]
    difficulty: str | None = None
    is_valid: bool = False


class RagDataset(BaseModel):
    """A dataset of questions, with or without their answers."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and answers for a whole dataset."""

    search_results: list[MinimalAnswer]
    k: int
