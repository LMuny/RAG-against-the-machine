"""Command-line interface of the RAG system, built with Python Fire."""

import fire
from uuid import uuid4
from src.indexer import build_index
from src.retriever import Retriever
from src.models import MinimalSearchResults


RED: str = '\033[1;38;5;196m'
RESET: str = '\033[0m'


def _print_error(message: str) -> None:
    """Print an error message in red."""
    print(f"{RED}[ERROR]{RESET}: {message}")


def _is_positive_int(value: object, name: str) -> bool:
    """Return True if value is an int > 0, else print an error.

    Args:
        value: The value received from the command line.
        name: The option name, used in the error message.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        _print_error(f"'{name}' should be an int")
        return False
    if value <= 0:
        _print_error(f"'{name}' should be > 0")
        return False
    return True


def _is_non_empty_str(value: object, name: str) -> bool:
    """Return True if value is a non-blank str, else print an error.

    Args:
        value: The value received from the command line.
        name: The argument name, used in the error message.
    """
    if not isinstance(value, str):
        _print_error(f"'{name}' should be a str")
        return False
    if not value.strip():
        _print_error(f"'{name}' can't be empty")
        return False
    return True


class RagCLI():
    """Index the vLLM corpus, search it and answer questions about it."""

    def index(
            self,
            max_chunk_size: int = 2000,
            source_path: str = "data/raw",
            processed_path: str = "data/processed"
            ) -> None:
        """Chunk the raw corpus and save the BM25 index.

        Args:
            max_chunk_size: Maximum length of a chunk, in characters.
            source_path: Directory holding the raw corpus to index.
            processed_path: Directory where the index is saved.
        """
        if not _is_positive_int(max_chunk_size, "max_chunk_size"):
            return
        build_index(source_path, processed_path, max_chunk_size)

    def search(
            self,
            query: str,
            k: int = 10,
            processed_path: str = "data/processed"
    ) -> None:
        """Print the top-k sources retrieved for a single query.

        Args:
            query: The question to search for.
            k: Number of sources to return.
            processed_path: Directory holding the index built by 'index'.
        """
        if not _is_positive_int(k, "k"):
            return
        if not _is_non_empty_str(query, "query"):
            return
        try:
            retriever = Retriever(processed_path)
        except OSError as e:
            _print_error(str(e))
            return
        result = retriever.search(query, k)
        ms_result = MinimalSearchResults(
            question_id=str(uuid4()),
            question=query,
            retrieved_sources=result
        )
        print(ms_result.model_dump_json(indent=2))

    def search_dataset(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def answer_dataset(self) -> None:
        pass


def main() -> None:
    """Run the CLI."""
    fire.Fire(RagCLI())
