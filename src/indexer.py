"""Build and save the BM25 index of the corpus."""

import os
import bm25s
from tqdm import tqdm
from src.chunker_md import chunk_markdown, chunk_text
from src.chunker_py import chunk_python
from src.models import MinimalSource

EXTENSIONS: tuple[str, ...] = (".md", ".txt", ".py")


def find_files(
        raw_dir: str,
        extension: str | tuple[str, ...] = EXTENSIONS
) -> list[str]:
    file_list: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(raw_dir):
        for file in filenames:
            if file.endswith(extension):
                file_list.append(os.path.join(dirpath, file))
    return sorted(file_list)


def chunk_file(path: str, max_size: int) -> list[tuple[MinimalSource, str]]:
    if path.endswith(".md"):
        return chunk_markdown(path, max_size)
    if path.endswith(".txt"):
        return chunk_text(path, max_size)
    if path.endswith(".py"):
        return chunk_python(path, max_size)
    return []


def build_index(raw_dir: str, save_dir: str, max_chunk_size: int) -> None:
    texts: list[str] = []
    sources: list[MinimalSource] = []
    for path in tqdm(find_files(raw_dir, EXTENSIONS), desc="Chunking"):
        try:
            chunks = chunk_file(path, max_chunk_size)
        except (OSError, UnicodeDecodeError) as error:
            print(f"Skipped {path}: {error}")
            continue
        for source, text in chunks:
            sources.append(source)
            readable_path = source.file_path.replace("/", " ")
            readable_path = readable_path.replace("_", " ")
            readable_path = readable_path.replace(".", " ")
            texts.append(readable_path + "\n" + text)

    tokens = bm25s.tokenize(texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokens)
    os.makedirs(save_dir, exist_ok=True)
    corpus: list[dict[str, str | int]] = []
    for source in sources:
        corpus.append(source.model_dump())
    retriever.save(save_dir, corpus=corpus)
    print(f"Ingestion complete! Indices saved under {save_dir}")
