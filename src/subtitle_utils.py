from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


SRT_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")


@dataclass(frozen=True)
class SubtitleRow:
    sequence: int
    start_ms: int
    end_ms: int
    text: str


def ms_to_srt(milliseconds: int) -> str:
    milliseconds = max(0, int(milliseconds))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def srt_to_ms(value: str) -> int:
    match = SRT_TIME_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT time: {value}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return (((hours * 60 + minutes) * 60) + seconds) * 1000 + millis


def parse_srt(content: str) -> List[SubtitleRow]:
    blocks = re.split(r"\n\s*\n", content.strip()) if content.strip() else []
    rows: List[SubtitleRow] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        try:
            sequence = int(lines[0].strip())
            start_text, end_text = [part.strip() for part in lines[1].split("-->", 1)]
            start_ms = srt_to_ms(start_text)
            end_ms = srt_to_ms(end_text)
        except (TypeError, ValueError):
            continue
        rows.append(SubtitleRow(sequence, start_ms, end_ms, "\n".join(lines[2:]).strip()))
    return rows


def render_srt(rows: Iterable[SubtitleRow]) -> str:
    blocks = []
    for index, row in enumerate(rows, 1):
        blocks.append(
            f"{index}\n{ms_to_srt(row.start_ms)} --> {ms_to_srt(row.end_ms)}\n{row.text.strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")
