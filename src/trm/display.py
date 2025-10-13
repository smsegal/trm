"""Shared rendering utilities for sudoku boards."""

from __future__ import annotations

from typing import Iterable, Set

from rich.console import Console
from rich.panel import Panel

console = Console()


def _stylize_cell(char: str, *, style: str | None) -> str:
    if style is None:
        return char
    return f"[{style}]{char}[/]"


def format_sudoku_ascii(
    board: str,
    *,
    highlight: Iterable[int] | None = None,
    puzzle: bool = False,
) -> str:
    """Format a flattened sudoku board as a bordered grid with optional highlighting."""
    highlight_set: Set[int] = set(highlight or [])
    rows = [board[i : i + 9] for i in range(0, len(board), 9)]

    top = "┌───────┬───────┬───────┐"
    inner = "├───────┼───────┼───────┤"
    bold_inner = "├═══════╪═══════╪═══════┤"
    bottom = "└───────┴───────┴───────┘"

    lines: list[str] = [top]
    for row_idx, row in enumerate(rows):
        cells: list[str] = []
        for col_idx, char in enumerate(row):
            flat_idx = row_idx * 9 + col_idx
            base = "·" if char == "." else char
            style: str | None = None
            if char == ".":
                style = "dim"
            elif puzzle:
                style = "bold"
            if flat_idx in highlight_set:
                style = "bold red"
            cells.append(_stylize_cell(base, style=style))
        left = " ".join(cells[:3])
        middle = " ".join(cells[3:6])
        right = " ".join(cells[6:])
        lines.append(f"│ {left} │ {middle} │ {right} │")

        if row_idx == len(rows) - 1:
            continue
        if (row_idx + 1) % 3 == 0:
            lines.append(bold_inner)
        else:
            lines.append(inner)
    lines.append(bottom)
    return "\n".join(lines)


def render_sudoku(
    board: str,
    *,
    title: str,
    highlight: Iterable[int] | None = None,
    puzzle: bool = False,
    border_style: str = "blue",
) -> None:
    """Render a sudoku board in the terminal using Rich panels."""
    ascii_board = format_sudoku_ascii(board, highlight=highlight, puzzle=puzzle)
    panel = Panel.fit(
        ascii_board,
        title=title,
        border_style=border_style,
        padding=(0, 2),
    )
    console.print(panel)
