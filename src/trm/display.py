"""Shared rendering utilities for sudoku boards."""

from __future__ import annotations

from typing import Iterable, Mapping

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
    cell_styles: Mapping[int, str] | None = None,
) -> str:
    """Format a flattened sudoku board as a bordered grid with optional highlighting."""
    highlight_set = set(highlight or [])
    effective_styles: dict[int, str] = dict(cell_styles or {})
    for idx in highlight_set:
        effective_styles.setdefault(idx, "bold red")
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
            if flat_idx in effective_styles:
                style = effective_styles[flat_idx]
            cells.append(_stylize_cell(base, style=style))
        def join_with_cell_separators(chunk: list[str]) -> str:
            if not chunk:
                return ""
            joint: list[str] = []
            for idx, cell in enumerate(chunk):
                joint.append(cell)
                if idx < len(chunk) - 1:
                    joint.append("[dim]│[/]")
            return "".join(joint)

        left = join_with_cell_separators(cells[:3])
        middle = join_with_cell_separators(cells[3:6])
        right = join_with_cell_separators(cells[6:])
        lines.append(f"│ {left} ║ {middle} ║ {right} │")

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
    cell_styles: Mapping[int, str] | None = None,
) -> None:
    """Render a sudoku board in the terminal using Rich panels."""
    ascii_board = format_sudoku_ascii(
        board,
        highlight=highlight,
        puzzle=puzzle,
        cell_styles=cell_styles,
    )
    panel = Panel.fit(
        ascii_board,
        title=title,
        border_style=border_style,
        padding=(0, 2),
    )
    console.print(panel)
