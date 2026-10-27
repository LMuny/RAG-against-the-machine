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
    """Return the sorted paths of the files under raw_dir with an extension.

    Args:
        raw_dir: Directory to walk recursively.
        extension: Extension, or tuple of extensions, to keep.
    """
    file_list: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(raw_dir):
        for file in filenames:
            if file.endswith(extension):
                file_list.append(os.path.join(dirpath, file))
    return sorted(file_list)


def chunk_file(path: str, max_size: int) -> list[tuple[MinimalSource, str]]:
    """Chunk a file with the strategy matching its extension.

    Returns an empty list for an unsupported extension.
    """
    if path.endswith(".md"):
        return chunk_markdown(path, max_size)
    if path.endswith(".txt"):
        return chunk_text(path, max_size)
    if path.endswith(".py"):
        return chunk_python(path, max_size)
    return []


def build_index(raw_dir: str, save_dir: str, max_chunk_size: int) -> None:
    """Chunk every file of raw_dir and save a BM25 index in save_dir.

    Each chunk is indexed with its file path, made readable, in front of its
    text. Only the source locations are saved, not the texts.

    Args:
        raw_dir: Directory holding the corpus.
        save_dir: Directory where the index is saved.
        max_chunk_size: Maximum length of a chunk, in characters.

    Raises:
        OSError: If raw_dir holds no file to index, or save_dir can't be
            written.
    """
    texts: list[str] = []
    sources: list[MinimalSource] = []
    files = find_files(raw_dir, EXTENSIONS)
    if not files:
        raise OSError(f"No .py, .md or .txt file found in {raw_dir}")
    for path in tqdm(files, desc="Chunking"):
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
