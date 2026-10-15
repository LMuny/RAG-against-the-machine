"""Split Markdown files into chunks along their sections."""


def read_file(path: str) -> str:
    """Return the content of a text file decoded as UTF-8."""
    try:
        with open(path, 'r', encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise OSError(f"{e}, {path} can't be reached")


def find_sections(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) positions of each section of a Markdown text.

    A section starts at a heading line and ends at the next one.
    Lines starting with '#' inside fenced code blocks are not headings.
    """
    pos: int = 0
    in_code = False
    starts: list[int] = [0]
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            in_code = not in_code
        elif not in_code and line.startswith('#'):
            starts.append(pos)
        pos += len(line)
    ends = starts[1:] + [len(text)]
    return [(s, e) for s, e in zip(starts, ends) if s < e]


def find_cuts(text: str, start: int, end: int, sep: str) -> list[int]:
    """Return the positions right after each sep in [start, end), plus end."""
    cuts: list[int] = []
    pos = text.find(sep, start, end)
    while pos != -1:
        cuts.append(pos + len(sep))
        pos = text.find(sep, pos + len(sep), end)
    cuts.append(end)
    return cuts


def split_section(text: str, start: int, end: int,
                  max_size: int, sep: str) -> list[tuple[int, int]]:
    """Split text[start:end] into spans of at most max_size characters."""
    chunks: list[tuple[int, int]] = []
    chunk_start = start
    prev = start
    for cut in find_cuts(text, start, end, sep):
        if cut - chunk_start > max_size and prev > chunk_start:
            chunks.append((chunk_start, prev))
            chunk_start = prev
        prev = cut
    if chunk_start < end:
        chunks.append((chunk_start, end))
    return chunks
