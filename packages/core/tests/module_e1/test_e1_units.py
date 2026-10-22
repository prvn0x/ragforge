"""Unit guarantees of the streaming sanitizer: port conformance, latency (what is held and what
is not), code fidelity, fence rules, allow-list normalization and lifecycle."""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from module_e1.e1_support import ALLOWED, links, sanitize, split_at, stream, violations
from ragforge_core.agent.sanitize import HOLD_LIMIT, MarkdownStreamSanitizer, normalize_url
from ragforge_core.domain.ports import OutputSanitizer

CANONICAL = ALLOWED[0]

# 5000+ chars of code full of things that would be hostile outside a fence.
_CODE_LINE = (
    "x = '[a](http://evil.example) ![i](http://e.example/p.png) <img src=x> `t` www.e.co'\n"
)
CODE_BODY = (_CODE_LINE * (5000 // len(_CODE_LINE) + 1))[:5000]
FENCED = f"```python\n{CODE_BODY}\n```\n"


def test_e1_implements_the_port() -> None:
    sanitizer: OutputSanitizer = MarkdownStreamSanitizer()
    assert isinstance(sanitizer, OutputSanitizer)


# --------------------------------------------------------------------------- code fidelity
def test_e1_5000_char_fenced_block_round_trips_byte_identical() -> None:
    assert len(CODE_BODY) == 5000
    assert sanitize(FENCED) == FENCED
    assert "".join(stream(list(FENCED))) == FENCED


@settings(max_examples=200, deadline=None)
@given(cuts=st.lists(st.integers(min_value=0, max_value=len(FENCED)), max_size=60))
def test_e1_5000_char_fence_any_split_byte_identical(cuts: list[int]) -> None:
    assert "".join(stream(split_at(FENCED, cuts))) == FENCED


def test_e1_fence_content_is_not_held() -> None:
    """Code passes unheld: TTFT inside long code blocks is preserved (no 512-char hold)."""
    sanitizer = MarkdownStreamSanitizer()
    assert sanitizer.feed("```py\n") == "```py\n"
    assert sanitizer.feed("[a](http://e") == "[a](http://e"
    assert sanitizer.feed("vil.example) `") == "vil.example) `"
    assert sanitizer.feed("\n```\nafter") == "\n```\n"
    assert sanitizer.finish() == "after"


def test_e1_tilde_fence_and_longer_closer() -> None:
    text = "~~~~ info `with` backticks\n```\n![x](http://e)\n~~~~~~   \n![y](http://e)"
    out = sanitize(text)
    assert out.startswith("~~~~ info `with` backticks\n```\n![x](http://e)\n~~~~~~   \n")
    assert out.endswith("\\[image removed\\]")


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("``` a`b\n![x](http://e)\n", "backtick in a backtick fence's info string"),
        ("- item\n```\n![x](http://e)\n", "inside a list item"),
        ("> q\n```\n![x](http://e)\n", "after a blockquote line"),
        ("- item\n\n```\n![x](http://e)\n", "after a list item (container still open)"),
        ("| a |\n|---|\n```\n![x](http://e)\n", "after a table row (markdown-it swallows it)"),
        (" ```\n![x](http://e)\n", "not at column 0"),
        ("x ```\n![x](http://e)\n", "not at line start"),
        ("``\n![x](http://e)\n", "fewer than 3 backticks"),
    ],
)
def test_e1_ambiguous_fences_stay_text(text: str, why: str) -> None:
    out = sanitize(text)
    assert "\\[image removed\\]" in out, why
    assert not violations(out), why


def test_e1_container_closes_after_blank_and_unindented_paragraph() -> None:
    text = "- item\n\npara\n```\n![x](http://e)\n```\n"
    assert sanitize(text) == "- item\n\npara\n```\n![x](http://e)\n```\n"


@pytest.mark.parametrize(
    "closer",
    ["``", "~~~", "``` x", "    ```", "\t```"],
)
def test_e1_non_closers_keep_the_fence_open(closer: str) -> None:
    text = f"```\ncode\n{closer}\n![x](http://e)\n"
    assert sanitize(text) == text  # still code: verbatim, and the renderer agrees
    assert not violations(sanitize(text))


# --------------------------------------------------------------------------- latency / holding
def test_e1_plain_text_streams_with_one_word_of_latency() -> None:
    sanitizer = MarkdownStreamSanitizer()
    assert sanitizer.feed("Hello wor") == "Hello "
    assert sanitizer.feed("ld, all") == "world, "
    assert sanitizer.finish() == "all"


def test_e1_bare_url_is_held_until_a_delimiter() -> None:
    sanitizer = MarkdownStreamSanitizer()
    assert sanitizer.feed("see htt") == "see "
    assert sanitizer.feed("p://evil.exa") == ""
    assert sanitizer.feed("mple/x now") == "`http://evil[.]example/x` "
    assert sanitizer.finish() == "now"


def test_e1_constructs_are_held_until_resolved() -> None:
    sanitizer = MarkdownStreamSanitizer([CANONICAL])
    assert sanitizer.feed("a [doc") == "a "
    assert sanitizer.feed("s](https://docs.example.com/gu") == ""
    assert sanitizer.feed("ide) b") == f"[docs](<{CANONICAL}>) "
    assert sanitizer.finish() == "b"


def test_e1_held_construct_is_bounded_by_hold_limit() -> None:
    sanitizer = MarkdownStreamSanitizer()
    out = sanitizer.feed("[" + " " * (HOLD_LIMIT - 2))
    assert out == ""  # still within the window: held
    out = sanitizer.feed("  ")
    assert out.startswith("\\[")  # window exhausted: flushed inert, without waiting for more


def test_e1_allow_applies_to_text_still_held() -> None:
    """E calls allow() on TurnSources, which may arrive while a link is mid-stream."""
    sanitizer = MarkdownStreamSanitizer()
    assert sanitizer.feed("[a](https://docs.example.com/guide") == ""
    sanitizer.allow([CANONICAL])
    assert sanitizer.feed(")") == f"[a](<{CANONICAL}>)"


# --------------------------------------------------------------------------- allow-list
@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://Docs.Example.com/guide", "https://docs.example.com/guide"),
        ("https://u:p@docs.example.com:443/guide#x", "https://docs.example.com/guide"),
        ("http://docs.example.com:80", "http://docs.example.com/"),
        ("https://docs.example.com:8443/a?b=1", "https://docs.example.com:8443/a?b=1"),
        ("https://[::1]/a", "https://[::1]/a"),
    ],
)
def test_e1_normalize_url(url: str, key: str) -> None:
    normalized = normalize_url(url)
    assert normalized is not None
    assert normalized[0] == key


@pytest.mark.parametrize(
    "url",
    ["javascript:alert(1)", "data:x", "ftp://h/x", "/relative", "https://", "http://h:99999/"],
)
def test_e1_normalize_rejects_unlinkable(url: str) -> None:
    assert normalize_url(url) is None


def test_e1_emitted_href_is_the_stored_canonical_never_the_models_spelling() -> None:
    stored = "https://Docs.Example.com/guide"
    out = sanitize("[x](https://user:secret@DOCS.example.com:443/guide)", [stored])
    assert links(out) == [stored]
    assert "secret" not in out


def test_e1_unsafe_fragments_are_not_carried_over() -> None:
    out = sanitize(f"[x]({CANONICAL}#a?b=c) [y]({CANONICAL}#ok-1)")
    assert links(out) == [CANONICAL, f"{CANONICAL}#ok-1"]


# --------------------------------------------------------------------------- lifecycle
def test_e1_feed_after_finish_is_an_error_and_finish_is_idempotent() -> None:
    sanitizer = MarkdownStreamSanitizer()
    sanitizer.feed("a")
    assert sanitizer.finish() == "a"
    assert sanitizer.finish() == ""
    with pytest.raises(RuntimeError):
        sanitizer.feed("b")


def test_e1_crlf_split_across_chunks() -> None:
    assert "".join(stream(["a\r", "\n```\r", "\ncode\r\n```\r"])) == "a\n```\ncode\n```\n"


def test_e1_instances_are_independent() -> None:
    """One instance per answer; interleaved streams must not share state."""
    first, second = MarkdownStreamSanitizer([CANONICAL]), MarkdownStreamSanitizer()
    out_first = first.feed("```\n[a](") + second.feed("[a](") + first.feed(CANONICAL + ")")
    out_second = second.feed(CANONICAL + ")") + second.finish()
    assert out_first == f"```\n[a]({CANONICAL})"  # first is inside a fence: verbatim
    assert out_second == "a"  # second never allowed the URL: link dropped
    assert first.finish() == ""


def test_e1_empty_and_whitespace_input() -> None:
    assert sanitize("") == ""
    assert sanitize(" \n\t\n") == " \n\t\n"
    assert "".join(stream(["", "", "x", ""])) == "x"


def test_e1_very_long_lines_stay_conservative() -> None:
    """Line tracking is capped; past the cap, line-start decisions take the safe branch."""
    indent = " " * 1100
    assert sanitize(indent + "~~~\nx") == indent + "\\~~~\nx"
    long_line = "word " * 300 + "| [r]: http://evil.example\n```\n![x](http://e)\n"
    out = "".join(stream(split_at(long_line, range(0, len(long_line), 5))))
    assert "\\[r\\]" in out  # not at line start: plain text, not a removed definition
    assert "\\[image removed\\]" in out  # the pipe kept the container open: no fence
    assert not violations(out)


# --------------------------------------------------------------------------- CPU bounds
# The sanitizer runs on the API event loop shared by every tenant's stream, so hostile input must
# cost bounded CPU per char. Before the link-scan guards a 200 KB `[` flood took ~22 s; the budgets
# below are ~10x the measured cost, loose enough for slow CI yet far below the old quadratic.
FLOODS = {
    "brackets": "[" * 200_000,
    "bracket_link_blocks": ("[" * 60 + "](x)") * 3_000,
    "nested_rewrites": ("[" * 200 + "a" + "](x)" * 200 + " ") * 100,
    "bracket_lines": ("[" * 512 + "\n") * 400,
    "images": "![" * 100_000,
    # Each dropped link rebuilds the working buffer; one ~1 MB feed must not copy it per link.
    "dropped_links_one_feed": "[a](x) " * 150_000,
}


@pytest.mark.parametrize("name", sorted(FLOODS))
def test_e1_bracket_floods_cost_linear_cpu(name: str) -> None:
    text = FLOODS[name]
    started = time.perf_counter()
    out = sanitize(text)
    assert time.perf_counter() - started < 5.0, name
    assert not violations(out[:20_000])


def test_e1_large_feed_window_is_invisible() -> None:
    """`feed()` moves big inputs into the buffer in windows; constructs straddling a window
    edge (links, fences, held words) must come out exactly as with any other split."""
    unit = f"[a](x) [docs]({CANONICAL}) see www.evil.example `code` ![i](y)\n"
    text = "```\n" + "c" * 3000 + "\n```\n" + unit * 400
    whole = sanitize(text)
    assert "".join(stream(split_at(text, range(0, len(text), 101)))) == whole
    assert links(whole) == [CANONICAL] * 400
    assert not violations(whole)


def test_e1_scan_credit_is_chunking_independent() -> None:
    """The credit only charges completed scans, so even once it runs out the output is the same
    for any split (and stays safe: an unscanned `[` is plain text)."""
    text = ("[" * 60 + "](http://evil.example)") * 400 + f" [docs]({CANONICAL})"
    whole = sanitize(text)
    assert "".join(stream(split_at(text, range(0, len(text), 97)))) == whole
    assert "".join(stream(split_at(text, range(5, len(text), 1009)))) == whole
    assert not violations(whole)


def test_e1_scan_credit_never_starves_normal_prose() -> None:
    """Citations and ordinary links never exhaust the credit: every allowed link survives."""
    para = f"See [1] and [2], then [the guide]({CANONICAL}) and [x](http://evil.example). "
    out = sanitize(para * 2_000)
    assert links(out) == [CANONICAL] * 2_000
