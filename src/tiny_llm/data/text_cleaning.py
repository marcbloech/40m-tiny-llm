"""Shared text cleaning and content filters for training corpora."""

from __future__ import annotations

import re
import unicodedata

# Characters to strip: U+FFFD replacement char, BOM, zero-width chars, soft hyphen.
_JUNK_CHARS = str.maketrans("", "", "\ufffd\ufeff\u200b\u200c\u200d\u00ad\ufffe")

_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_FENCED_CODE = re.compile(r"```[\s\S]*?```")
_RE_INDENTED_CODE = re.compile(r"(?:(?:^|\n)[ \t]{4,}\S[^\n]*){3,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

_RE_HTML_MARKUP = re.compile(
    r"(?is)<\s*/?\s*(html|head|body|div|span|script|style|p|a|img|table|tr|td|ul|ol|li|h[1-6]|form|input|button|section|article|br|meta)\b[^>]*>"
)
_RE_CODE_MARKERS = re.compile(
    r"(?im)"
    r"(```|</?\s*(script|style|html|body|div|span|p|a|img|table|form|input)\b|"
    r"^\s*(#\s*In\[[^\]]*\]|In\s*\[[0-9 ]*\]:|%%[A-Za-z_]+)|"
    r"^\s*(def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\w+(?:\.\w+)*\s+import|"
    r"print\s*\(|for\s+.+:|while\s+.+:|if\s+.+:|@\w+)|"
    r"\b(function\s+\w+\s*\(|console\.log\s*\(|var\s+\w+\s*=|let\s+\w+\s*=|const\s+\w+\s*=|"
    r"public\s+static\s+void|#include\s*<|SELECT\s+.+\s+FROM|CREATE\s+TABLE|"
    r"<html\b|<body\b|<script\b|<style\b))"
)
_RE_CODE_LINE = re.compile(
    r"(?im)^\s*(def\s+\w+\s*\(.*|class\s+\w+.*|import\s+\w+.*|"
    r"from\s+\w+(?:\.\w+)*\s+import.*|print\s*\(.*|for\s+.+:|while\s+.+:|"
    r"if\s+.+:|elif\s+.+:|else:|return\b.*|@\w+.*|"
    r"function\s+\w+\s*\(.*|console\.log\s*\(.*|"
    r"var\s+\w+\s*=.*|let\s+\w+\s*=.*|const\s+\w+\s*=.*|"
    r"public\s+static\s+void.*|#include\s*<.*)\s*$"
)
_RE_INLINE_CODE_SNIPPET = re.compile(
    r"(?i)\b(def\s+\w+\s*\([^)]*\):?|class\s+\w+\s*:|"
    r"print\s*\([^)]*\)|console\.log\s*\([^)]*\)|"
    r"return\s+[-+\w'\"()[\].]+)"
)
_RE_CODE_TOPIC = re.compile(
    r"(?i)\b("
    r"python|javascript|java|c\+\+|c#|typescript|html|css|sql|bash|shell|"
    r"programming|programmer|source\s+code|code\s+snippet|write\s+code|"
    r"debug|syntax\s+error|compiler|interpreter|software\s+library|api\s+endpoint|"
    r"(?:write|generate|create|implement|fix|modify|refactor)\s+(?:a\s+)?"
    r"(?:program|function|class|script|algorithm)"
    r")\b"
)
_RE_LATEX_MATH = re.compile(
    r"(?s)(\$\$.*?\$\$|\$[^$\n]{1,120}\$|\\\(|\\\)|\\\[|\\\]|"
    r"\\begin\{(?:equation|align|math|matrix|cases)\}|"
    r"\\(?:frac|sum|int|sqrt|lim|alpha|beta|gamma|theta|pi)\b)"
)
_RE_MATH_EXPRESSION = re.compile(
    r"(?m)(^|\s)([A-Za-z]\s*=\s*[-+*/^(). A-Za-z0-9]+|"
    r"\d+(?:\.\d+)?\s*[-+*/^=<>]\s*\d+(?:\.\d+)?|"
    r"[A-Za-z0-9)\]]\s*[\^*/]\s*[A-Za-z0-9([])"
)
_RE_MATH_TOPIC = re.compile(
    r"(?i)\b("
    r"math|mathematics|arithmetic|algebra|calculus|geometry|trigonometry|"
    r"equation|formula|solve for|derivative|integral|matrix|polynomial|"
    r"probability|statistics|theorem|proof|calculate|compute|factorial|fibonacci"
    r")\b"
)

_ENGLISH_STOPWORDS = frozenset(
    (
        "the be to of and a in that have i it for not on with he as you do at this "
        "but his by from they we say her she or an will my one all would there their "
        "what so up out if about who get which go me when make can like time no just "
        "him know take people into year your good some could them see other than then "
        "now look only come its over think also back after use two how our work first "
        "well way even new want because any these give day most us"
    ).split()
)

_BOILERPLATE_PREFIXES = [
    "Media playback is unsupported on your device",
    "Media playback is not supported on this device",
    "These are external links and will open in a new window",
    "Image copyright",
    "Image caption",
    "Share this with",
    "Copy this link",
    "These are external links",
    "Close share panel",
    "Sign up for our newsletter",
]


def clean_text(text: str, *, strip_code_and_math: bool = True) -> str:
    """Apply standard normalization and web-text cleanup."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(_JUNK_CHARS)
    text = _RE_URL.sub("", text)
    if strip_code_and_math:
        text = _RE_HTML.sub("", text)
        text = _RE_FENCED_CODE.sub("", text)
        text = _RE_INDENTED_CODE.sub("", text)
        text = _RE_CODE_LINE.sub("", text)
        text = _RE_INLINE_CODE_SNIPPET.sub("", text)
        text = _RE_LATEX_MATH.sub("", text)
        text = _RE_MATH_EXPRESSION.sub(" ", text)

    for prefix in _BOILERPLATE_PREFIXES:
        if text[:200].startswith(prefix):
            text = text[len(prefix) :].lstrip()
            break

    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    if strip_code_and_math:
        text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def is_probably_english(text: str, threshold: float = 0.65) -> bool:
    """Return whether text appears to be English.

    Uses fast-langdetect when available and falls back to a conservative
    stopword/ASCII heuristic for tests or minimal environments.
    """
    sample = (text or "")[:800].replace("\n", " ")
    words = re.findall(r"[A-Za-z']+", sample.lower())
    if len(words) < 5:
        return False

    try:
        from fast_langdetect import detect

        result = detect(sample[:500], model="auto", k=1)
        return result[0]["lang"] == "en" and result[0]["score"] > threshold
    except Exception:
        ascii_ratio = sum(ord(ch) < 128 for ch in sample) / max(len(sample), 1)
        stopword_hits = sum(word in _ENGLISH_STOPWORDS for word in words)
        return ascii_ratio > 0.85 and stopword_hits / len(words) >= 0.12


def has_code_or_markup(text: str) -> bool:
    """Return whether text contains code, notebook cells, or HTML markup."""
    text = text or ""
    return bool(_RE_HTML_MARKUP.search(text) or _RE_CODE_MARKERS.search(text))


def has_math_content(text: str) -> bool:
    """Return whether text contains mathematical notation or math-task wording."""
    text = text or ""
    return bool(_RE_LATEX_MATH.search(text) or _RE_MATH_EXPRESSION.search(text))


def is_code_or_math_task(text: str) -> bool:
    """Return whether an instruction asks for code/HTML or math behavior."""
    text = text or ""
    return bool(
        _RE_CODE_TOPIC.search(text)
        or _RE_MATH_TOPIC.search(text)
        or has_code_or_markup(text)
        or has_math_content(text)
    )


def is_clean_instruction_example(
    instruction: str,
    input_text: str,
    output: str,
    *,
    require_english: bool = True,
    filter_code_and_math: bool = True,
) -> bool:
    """Return whether an SFT example should be kept for non-code instruction following."""
    parts = [instruction or "", input_text or "", output or ""]
    combined = "\n".join(parts)
    if not all(part.strip() for part in (instruction, output)):
        return False
    if filter_code_and_math:
        if any(has_code_or_markup(part) or has_math_content(part) for part in parts):
            return False
        if is_code_or_math_task(combined):
            return False
    english_probe = combined
    if not filter_code_and_math:
        english_probe = clean_text(combined, strip_code_and_math=True) or combined
    english_threshold = 0.35 if not filter_code_and_math else 0.65
    if require_english and not is_probably_english(english_probe, threshold=english_threshold):
        return False
    return True


__all__ = [
    "clean_text",
    "has_code_or_markup",
    "has_math_content",
    "is_clean_instruction_example",
    "is_code_or_math_task",
    "is_probably_english",
]
