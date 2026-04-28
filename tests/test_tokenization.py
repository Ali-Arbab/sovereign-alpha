"""Tests for the Module I tokenization pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.module_1_extraction.tokenization.pipeline import (
    WhitespaceTokenizer,
    cache_key,
    chunk_text,
    clean_html,
    tokenize_with_cache,
)


def test_clean_html_strips_tags() -> None:
    html = "<html><body><p>Hello <b>world</b></p></body></html>"
    assert clean_html(html) == "Hello world"


def test_clean_html_drops_script_and_style_content() -> None:
    html = """
    <html>
      <head>
        <style>body { color: red; }</style>
        <script>var x = 'should-not-appear';</script>
      </head>
      <body><p>Visible</p></body>
    </html>
    """
    cleaned = clean_html(html)
    assert "Visible" in cleaned
    assert "should-not-appear" not in cleaned
    assert "color: red" not in cleaned


def test_clean_html_normalizes_whitespace() -> None:
    html = "<p>   foo\n\n\tbar   baz   </p>"
    assert clean_html(html) == "foo bar baz"


def test_clean_html_handles_bytes() -> None:
    assert clean_html(b"<p>hello</p>") == "hello"


def test_clean_html_decodes_entities() -> None:
    assert clean_html("<p>caf&eacute;</p>") == "café"


def test_chunk_text_basic_split() -> None:
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, max_chars=30, overlap=5)
    assert chunks[0] == text[:30]
    # Each subsequent chunk starts at end - overlap of the previous
    assert chunks[1].startswith(text[25:30])


def test_chunk_text_empty_returns_empty() -> None:
    assert chunk_text("") == []


def test_chunk_text_single_chunk_when_text_fits() -> None:
    text = "short"
    chunks = chunk_text(text, max_chars=100, overlap=10)
    assert chunks == ["short"]


def test_chunk_text_full_coverage_of_input() -> None:
    """Every character of input must appear in at least one chunk."""
    text = "x" * 250
    chunks = chunk_text(text, max_chars=100, overlap=20)
    assert "".join(chunks).count("x") >= 250


def test_chunk_text_invalid_args() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_text("abc", max_chars=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("abc", max_chars=10, overlap=-1)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("abc", max_chars=10, overlap=10)


def test_whitespace_tokenizer_splits_on_whitespace() -> None:
    tok = WhitespaceTokenizer()
    ids = tok.encode("the quick brown fox")
    assert len(ids) == 4


def test_whitespace_tokenizer_deterministic_per_token() -> None:
    tok = WhitespaceTokenizer()
    a = tok.encode("the cat")
    b = tok.encode("the cat")
    assert a == b


def test_whitespace_tokenizer_name_is_versioned() -> None:
    assert WhitespaceTokenizer().name == "whitespace_v1"


def test_cache_key_changes_with_text() -> None:
    a = cache_key("foo", "tok_v1")
    b = cache_key("bar", "tok_v1")
    assert a != b


def test_cache_key_changes_with_tokenizer_name() -> None:
    a = cache_key("foo", "tok_v1")
    b = cache_key("foo", "tok_v2")
    assert a != b


def test_cache_key_deterministic() -> None:
    assert cache_key("foo", "tok") == cache_key("foo", "tok")


def test_tokenize_with_cache_writes_then_reads(tmp_path: Path) -> None:
    tok = WhitespaceTokenizer()
    cache = tmp_path / "tok-cache"
    tokens_a = tokenize_with_cache("hello world", tok, cache)
    # File should now exist
    files = list(cache.glob("*.json"))
    assert len(files) == 1
    # Re-call hits cache and returns the same result
    tokens_b = tokenize_with_cache("hello world", tok, cache)
    assert tokens_a == tokens_b


def test_tokenize_with_cache_different_tokenizers_separate_cache_slots(tmp_path: Path) -> None:
    cache = tmp_path / "tok-cache"

    class _OtherTok:
        name = "other_v1"

        def encode(self, text: str) -> list[int]:
            return [42 for _ in text.split()]

    tokenize_with_cache("hello world", WhitespaceTokenizer(), cache)
    tokenize_with_cache("hello world", _OtherTok(), cache)
    assert len(list(cache.glob("*.json"))) == 2


def test_tokenize_with_cache_round_trip_returns_list_not_tuple(tmp_path: Path) -> None:
    """JSON deserialization yields lists; the cache contract must too."""
    tok = WhitespaceTokenizer()
    cache = tmp_path / "tok-cache"
    tokenize_with_cache("a b c", tok, cache)
    second = tokenize_with_cache("a b c", tok, cache)
    assert isinstance(second, list)
