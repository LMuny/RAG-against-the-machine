import ast
from src.models import MinimalSource
from src.chunker_md import read_file, split_text

INSTANCES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def line_start_offsets(text: str) -> list[int]:
    offsets: list[int] = [0]
    position = text.find('\n')
    while position != -1:
        offsets.append(position + 1)
        position = text.find('\n', position + 1)
    offsets.append(len(text))
    return offsets


def block_line_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, INSTANCES):
            first_line = node.lineno
            if node.decorator_list:
                first_line = node.decorator_list[0].lineno
            if node.end_lineno is None:
                continue
            last_line = node.end_lineno
            ranges.append((first_line, last_line))
    return ranges


def find_blocks(text: str, tree: ast.Module) -> list[tuple[int, int]]:
    offsets = line_start_offsets(text)
    boundaries: set[int] = {0, len(text)}
    for first_line, last_line in block_line_ranges(tree):
        boundaries.add(offsets[first_line - 1])
        boundaries.add(offsets[last_line])
    sorted_boundaries = sorted(boundaries)
    blocks: list[tuple[int, int]] = []
    for index in range(len(sorted_boundaries) - 1):
        blocks.append((sorted_boundaries[index], sorted_boundaries[index + 1]))
    return blocks


def chunk_python(path: str, max_size: int
                 ) -> list[tuple[MinimalSource, str]]:
    text = read_file(path)
    chunks: list[tuple[MinimalSource, str]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        blocks = [(0, len(text))]
    else:
        blocks = find_blocks(text, tree)
    for block_start, block_end in blocks:
        for chunk_start, chunk_end in split_text(
                text, block_start, block_end, max_size):
            source = MinimalSource(file_path=path,
                                   first_character_index=chunk_start,
                                   last_character_index=chunk_end)
            chunks.append((source, text[chunk_start:chunk_end]))
    return chunks
