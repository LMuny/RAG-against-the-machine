"""Command-line interface of the RAG system, built with Python Fire."""

import os
from typing import TypeVar
import fire
from pydantic import BaseModel, ValidationError
from tqdm import tqdm
from uuid import uuid4
from src.evaluate import dataset_recall
from src.generator import Generator
from src.indexer import build_index
from src.retriever import Retriever
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


RED: str = '\033[1;38;5;196m'
RESET: str = '\033[0m'
RECALL_KS: tuple[int, ...] = (1, 3, 5, 10)

ModelT = TypeVar("ModelT", bound=BaseModel)


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


def _load_json(path: str, model: type[ModelT]) -> ModelT | None:
    """Read a JSON file and validate it, or print an error and return None.

    Args:
        path: Path of the JSON file to read.
        model: The pydantic model the file must follow.
    """
    try:
        with open(path, encoding="utf-8") as file:
            return model.model_validate_json(file.read())
    except (OSError, UnicodeDecodeError) as error:
        _print_error(f"Can't read {path}: {error}")
    except ValidationError as error:
        _print_error(f"{path} is not a valid {model.__name__} file:\n"
                     f"{error}")
    return None


def _save_json(data: BaseModel, save_directory: str, file_name: str) -> None:
    """Write a model as JSON in save_directory, or print an error.

    Args:
        data: The model to save.
        save_directory: Directory where the file is written.
        file_name: Name of the file to write.
    """
    path = os.path.join(save_directory, file_name)
    try:
        os.makedirs(save_directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(data.model_dump_json(indent=2))
    except OSError as error:
        _print_error(f"Can't write {path}: {error}")
        return
    print(f"Saved {type(data).__name__} to {path}")


def _load_retriever(processed_path: str) -> Retriever | None:
    """Load the index, or print an error and return None."""
    try:
        return Retriever(processed_path)
    except OSError as error:
        _print_error(str(error))
        return None


def _load_generator(model_name: str) -> Generator | None:
    """Load the model, or print an error and return None."""
    try:
        return Generator(model_name)
    except Exception as error:
        _print_error(f"Can't load model {model_name}: {error}")
        return None


def _read_source(source: MinimalSource) -> tuple[MinimalSource, str]:
    """Return the source with the text it points to ('' if unreadable)."""
    try:
        with open(source.file_path, encoding="utf-8") as file:
            text = file.read()
    except (OSError, UnicodeDecodeError):
        return source, ""
    return source, text[source.first_character_index:
                        source.last_character_index]


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
        try:
            build_index(source_path, processed_path, max_chunk_size)
        except OSError as e:
            _print_error(str(e))

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
        retriever = _load_retriever(processed_path)
        if retriever is None:
            return
        result = retriever.search(query, k)
        ms_result = MinimalSearchResults(
            question_id=str(uuid4()),
            question=query,
            retrieved_sources=result
        )
        print(ms_result.model_dump_json(indent=2))

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results",
            processed_path: str = "data/processed"
    ) -> None:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: JSON file of questions (RagDataset).
            k: Number of sources to retrieve per question.
            save_directory: Directory where the results are saved.
            processed_path: Directory holding the index built by 'index'.
        """
        if not _is_positive_int(k, "k"):
            return
        if not (_is_non_empty_str(dataset_path, "dataset_path")
                and _is_non_empty_str(save_directory, "save_directory")):
            return
        dataset = _load_json(dataset_path, RagDataset)
        if dataset is None:
            return
        retriever = _load_retriever(processed_path)
        if retriever is None:
            return
        search_results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            search_results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=retriever.search(question.question, k)
            ))
        results = StudentSearchResults(search_results=search_results, k=k)
        _save_json(results, save_directory, os.path.basename(dataset_path))

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str
    ) -> None:
        """Print the recall@k of search results against a dataset.

        Args:
            student_search_results_path: Output file of 'search_dataset'.
            dataset_path: JSON file of answered questions (RagDataset).
        """
        if not (_is_non_empty_str(student_search_results_path,
                                  "student_search_results_path")
                and _is_non_empty_str(dataset_path, "dataset_path")):
            return
        results = _load_json(student_search_results_path,
                             StudentSearchResults)
        dataset = _load_json(dataset_path, RagDataset)
        if results is None or dataset is None:
            return
        for k in RECALL_KS:
            if k > results.k:
                break
            recall = dataset_recall(results, dataset, k)
            print(f"Recall@{k}: {recall:.3f}")

    def answer(
            self,
            query: str,
            k: int = 10,
            processed_path: str = "data/processed",
            model_name: str = "Qwen/Qwen3-0.6B"
    ) -> None:
        """Answer a single query from the top-k retrieved sources.

        Args:
            query: The question to answer.
            k: Number of sources given to the model.
            processed_path: Directory holding the index built by 'index'.
            model_name: Hugging Face name of the model to use.
        """
        if not _is_positive_int(k, "k"):
            return
        if not _is_non_empty_str(query, "query"):
            return
        retriever = _load_retriever(processed_path)
        if retriever is None:
            return
        sources = retriever.search(query, k)
        generator = _load_generator(model_name)
        if generator is None:
            return
        contexts = [_read_source(source) for source in sources]
        prompt = generator.build_prompt(query, contexts)
        answer = MinimalAnswer(
            question_id=str(uuid4()),
            question=query,
            retrieved_sources=sources,
            answer=generator.generate([prompt])[0]
        )
        print(answer.model_dump_json(indent=2))

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str = "data/output/search_results_and_answer",
            model_name: str = "Qwen/Qwen3-0.6B"
    ) -> None:
        """Answer every question of a 'search_dataset' output and save it.

        Args:
            student_search_results_path: Output file of 'search_dataset'.
            save_directory: Directory where the answers are saved.
            model_name: Hugging Face name of the model to use.
        """
        if not (_is_non_empty_str(student_search_results_path,
                                  "student_search_results_path")
                and _is_non_empty_str(save_directory, "save_directory")):
            return
        results = _load_json(student_search_results_path,
                             StudentSearchResults)
        if results is None:
            return
        print(f"Loaded {len(results.search_results)} questions")
        generator = _load_generator(model_name)
        if generator is None:
            return
        prompts: list[str] = []
        for result in tqdm(results.search_results, desc="Building prompts"):
            contexts = [_read_source(source)
                        for source in result.retrieved_sources]
            prompts.append(generator.build_prompt(result.question, contexts))
        answers = generator.generate(prompts)
        answered: list[MinimalAnswer] = []
        for result, answer in zip(results.search_results, answers):
            answered.append(MinimalAnswer(**result.model_dump(),
                                          answer=answer))
        _save_json(StudentSearchResultsAndAnswer(search_results=answered,
                                                 k=results.k),
                   save_directory,
                   os.path.basename(student_search_results_path))


def main() -> None:
    """Run the CLI."""
    fire.Fire(RagCLI())
