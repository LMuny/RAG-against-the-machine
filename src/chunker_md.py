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
    sections: list[tuple[int, int]] = []
    for section_start, section_end in zip(starts, ends):
        if section_start < section_end:
            sections.append((section_start, section_end))
    return sections


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


def split_text(text: str, start: int, end: int,
               max_size: int) -> list[tuple[int, int]]:
    """Split text[start:end] into spans of at most max_size characters.

    Cut on blank lines first, then on line breaks, then every max_size
    characters as a last resort.
    """
    result: list[tuple[int, int]] = []
    for para_start, para_end in split_section(text, start, end,
                                              max_size, "\n\n"):
        if para_end - para_start <= max_size:
            result.append((para_start, para_end))
            continue
        for line_start, line_end in split_section(text, para_start, para_end,
                                                  max_size, "\n"):
            if line_end - line_start <= max_size:
                result.append((line_start, line_end))
                continue
            for piece_start in range(line_start, line_end, max_size):
                piece_end = min(piece_start + max_size, line_end)
                result.append((piece_start, piece_end))
    return result
