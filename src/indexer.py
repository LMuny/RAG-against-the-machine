"""Build and save the BM25 index of the corpus."""

import os
import bm25s
from tqdm import tqdm
from src.chunker_md import chunk_markdown
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


def build_index(raw_dir: str, save_dir: str, max_chunk_size: int) -> None:
    texts: list[str] = []
    sources: list[MinimalSource] = []
    for path in tqdm(find_files(raw_dir, (".md", ".txt")), desc="Chunking"):
        try:
            chunks = chunk_markdown(path, max_chunk_size)
        except (OSError, UnicodeDecodeError) as error:
            print(f"Skipped {path}: {error}")
            continue
        for source, text in chunks:
            sources.append(source)
            texts.append(text)
    tokens = bm25s.tokenize(texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokens)
    os.makedirs(save_dir, exist_ok=True)
    corpus: list[dict[str, str | int]] = []
    for source in sources:
        corpus.append(source.model_dump())
    retriever.save(save_dir, corpus=corpus)
    print(f"Ingestion complete! Indices saved under {save_dir}")
