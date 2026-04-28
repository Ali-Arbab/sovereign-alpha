"""Cleaning, chunking, and content-addressed tokenization caching.

Per master directive section 6.3: every pipeline stage is content-addressed.
A persona prompt change re-triggers ONLY the inference stage, not
re-tokenization. Likewise, a tokenizer-version bump re-triggers
tokenization, not ingestion.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

_SKIP_TAGS = frozenset({"script", "style", "noscript"})


class _TextExtractor(HTMLParser):
    """Strip HTML tags, return text. Drops content of <script>, <style>,
    and <noscript> tags entirely (they are noise for downstream NLP)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._depth_skip: int = 0

    def handle_starttag(self, tag: str, _attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._depth_skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._depth_skip > 0:
            self._depth_skip -= 1

    def handle_data(self, data: str) -> None:
        if self._depth_skip == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def clean_html(html: str | bytes) -> str:
    """Strip HTML tags + script/style content, normalize whitespace."""
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return re.sub(r"\s+", " ", parser.text()).strip()


def chunk_text(text: str, *, max_chars: int = 8000, overlap: int = 200) -> list[str]:
    """Split `text` into chunks of up to `max_chars`, sliding by
    `max_chars - overlap`.

    Character-based, not token-based -- LLM tokenization is roughly 4
    chars per BPE token, so 8000 chars ~= 2000 tokens, comfortably under
    every modern context window. Overlap preserves context across chunk
    boundaries so a sentence is never cleanly bisected by a chunk edge.

    Returns [] for empty input.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")
    if not text:
        return []

    chunks: list[str] = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + max_chars, n)
        chunks.append(text[pos:end])
        if end == n:
            break
        pos = end - overlap
    return chunks


class Tokenizer(Protocol):
    """Tokenizer interface; bind to a concrete backend (tiktoken / HF /
    llama.cpp) at the call site. `name` MUST change between
    incompatible versions so the cache invalidates."""

    @property
    def name(self) -> str: ...

    def encode(self, text: str) -> list[int]: ...


class WhitespaceTokenizer:
    """Stub tokenizer: split on whitespace, hash each token to a 32-bit int.

    NOT a research tokenizer. It exists so the pipeline plumbing can be
    exercised in tests and the bootstrap-test before a real tokenizer
    is wired in (post-hardware, with vLLM / llama.cpp). Tag every
    artifact produced with this tokenizer as bootstrap-only.
    """

    name = "whitespace_v1"

    def encode(self, text: str) -> list[int]:
        return [int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16) for t in text.split()]


def cache_key(text: str, tokenizer_name: str) -> str:
    """Content-addressed cache key for tokenization output."""
    h = hashlib.sha256()
    h.update(tokenizer_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def tokenize_with_cache(
    text: str,
    tokenizer: Tokenizer,
    cache_dir: Path,
) -> list[int]:
    """Tokenize `text`, caching the result on disk by content hash.

    Cache layout: `{cache_dir}/{cache_key}.json` containing the token-id list.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(text, tokenizer.name)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return list(json.load(f))

    tokens = list(tokenizer.encode(text))
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(tokens, f)
    return tokens
