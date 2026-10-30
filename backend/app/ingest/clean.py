"""Text cleaning (spec §3.4).

Operates on the plain-text extract (explaintext) which uses `== Section ==`
headers. Splits into sections, drops boilerplate sections (References, External
links, ...), removes LaTeX/math artifacts, and collapses whitespace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Sections to drop (casefolded, matched exactly against the heading text).
DROP_SECTIONS = {
    "references",
    "external links",
    "further reading",
    "see also",
    "notes",
    "bibliography",
    "selected publications",
    "notes and references",
    "citations",
    "footnotes",
    "sources",
    "publications",
    "works",
}

_HEADER = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
_DISPLAYSTYLE = re.compile(r"\{\\displaystyle")
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\s?|\\[,!;:]")
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAIL_WS = re.compile(r"[ \t]+(\n)")


@dataclass
class Section:
    title: str  # "" for the lead/intro
    text: str


def _strip_math(text: str) -> str:
    """Remove {\\displaystyle ...} wrappers (keeping inner text) and stray
    LaTeX control sequences."""
    out = []
    i = 0
    while i < len(text):
        m = _DISPLAYSTYLE.match(text, i)
        if m:
            # find matching closing brace for the '{' at m.start()
            depth = 0
            j = m.start()
            inner_start = m.end()
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = text[inner_start:j]
            out.append(_LATEX_CMD.sub(" ", inner))
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _tidy(text: str) -> str:
    text = _strip_math(text)
    text = _TRAIL_WS.sub(r"\1", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def parse_sections(extract: str) -> list[Section]:
    sections: list[Section] = []
    matches = list(_HEADER.finditer(extract))
    lead = extract[: matches[0].start()] if matches else extract
    sections.append(Section("", lead.strip()))
    for idx, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(extract)
        sections.append(Section(title, extract[start:end].strip()))
    return sections


def clean_extract(extract: str) -> list[Section]:
    """Return cleaned, non-empty sections with boilerplate dropped."""
    cleaned: list[Section] = []
    for sec in parse_sections(extract or ""):
        if sec.title.casefold() in DROP_SECTIONS:
            continue
        body = _tidy(sec.text)
        if not body and sec.title == "":
            continue  # empty lead
        if not body:
            continue  # empty section
        cleaned.append(Section(sec.title, body))
    return cleaned


def plain_text(sections: list[Section]) -> str:
    parts = []
    for sec in sections:
        if sec.title:
            parts.append(f"== {sec.title} ==")
        parts.append(sec.text)
    return "\n\n".join(parts).strip()


def char_count(sections: list[Section]) -> int:
    return sum(len(s.text) for s in sections)
