"""Compute recall@k of search results against a ground-truth dataset."""

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

IOU_THRESHOLD: float = 0.05


def iou(first: MinimalSource, second: MinimalSource) -> float:
    """Return the intersection over union of two spans (0 if other file)."""
    if first.file_path != second.file_path:
        return 0.0
    overlap = (min(first.last_character_index, second.last_character_index)
               - max(first.first_character_index,
                     second.first_character_index))
    if overlap <= 0:
        return 0.0
    union = ((first.last_character_index - first.first_character_index)
             + (second.last_character_index - second.first_character_index)
             - overlap)
    return overlap / union


def question_recall(gold: list[MinimalSource],
                    retrieved: list[MinimalSource], k: int) -> float:
    """Return the share of gold sources found in the first k results."""
    if not gold:
        return 0.0
    found = 0
    for gold_source in gold:
        for result in retrieved[:k]:
            if iou(gold_source, result) >= IOU_THRESHOLD:
                found += 1
                break
    return found / len(gold)


def dataset_recall(results: StudentSearchResults, dataset: RagDataset,
                   k: int) -> float:
    """Return the mean recall@k over all answered questions."""
    retrieved_by_id: dict[str, list[MinimalSource]] = {}
    for search_result in results.search_results:
        retrieved_by_id[search_result.question_id] = (
            search_result.retrieved_sources)
    total = 0.0
    count = 0
    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            continue
        retrieved = retrieved_by_id.get(question.question_id, [])
        total += question_recall(question.sources, retrieved, k)
        count += 1
    if count == 0:
        return 0.0
    return total / count
