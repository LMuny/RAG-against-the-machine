"""Load the BM25 index and answer top-k queries."""

import os
import bm25s
from src.models import MinimalSource


class Retriever:
    """Load the BM25 index once and answer top-k queries."""

    def __init__(self, processed_directory: str) -> None:
        """Load the index and the corpus saved by the indexer."""
        if not os.path.isfile(os.path.join(processed_directory,
                                           "params.index.json")):
            raise OSError(
                f"Index not found in {processed_directory}, "
                "run 'index' first")
        self.retriever = bm25s.BM25.load(processed_directory,
                                         load_corpus=True)
        self.chunk_count = len(self.retriever.corpus)

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the k most relevant sources for the query."""
        if k <= 0 or not query.strip():
            return []
        k = min(k, self.chunk_count)
        query_tokens = bm25s.tokenize(query, stopwords="en",
                                      return_ids=False,
                                      show_progress=False)
        if len(query_tokens[0]) == 0:
            return []
        documents, scores = self.retriever.retrieve(query_tokens, k=k,
                                                    show_progress=False)
        sources: list[MinimalSource] = []
        for rank in range(documents.shape[1]):
            if scores[0][rank] <= 0:
                continue
            sources.append(MinimalSource(**documents[0][rank]))
        return sources
