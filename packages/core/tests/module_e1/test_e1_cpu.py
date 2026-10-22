"""CPU bounds of bare-URL detection and held runs (the sanitizer shares the API event loop).

Searching `BARE_URL_PATTERN` unanchored is quadratic per search (its e-mail and scheme
alternatives can start at every char and run to the end of the word), and the joined word grows
while constructs emit nothing, so `"a[](x)" * 34_000` (200 KB) took ~15 s and
`"a<javascript:x>" * 15_000` ~6.7 s. `_find_bare_url` must return exactly what the pattern
would (proved here against the pattern itself) in linear time. Budgets are ~10x the measured
cost and far below the old quadratic cost.
"""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from module_e1.e1_support import sanitize, stream, violations
from ragforge_core.agent.sanitize import (
    BARE_URL_PATTERN,
    MarkdownStreamSanitizer,
    _find_bare_url,
    _split_trailing,
)

# --------------------------------------------------------------------------- equivalence
_FRAGMENTS = [
    *"aAwW1+.-:/@ \t\n()_x;,!?*'\"~",
    *("www.", "WwW.", "mailto:", "MAILTO:", "xmpp:", "://", "\u00e9", "\u0440\u0444"),
    *("\u0130", "\u0131", "\u00a0"),
]
_URLISH = st.lists(st.sampled_from(_FRAGMENTS), max_size=40).map("".join)


def _reference(text: str) -> int:
    match = BARE_URL_PATTERN.search(text)
    return -1 if match is None else match.start()


@settings(max_examples=3_000, deadline=None)
@given(pieces=st.lists(_URLISH, max_size=6))
def test_e1_find_bare_url_matches_the_pattern(pieces: list[str]) -> None:
    text = "".join(pieces)
    assert _find_bare_url(text) == _reference(text), text


@settings(max_examples=3_000, deadline=None)
@given(tail=_URLISH, new=_URLISH)
def test_e1_find_bare_url_skips_an_already_searched_tail(tail: str, new: str) -> None:
    """With `searched` = a prefix known to hold no match, the result is still the pattern's."""
    if _reference(tail) != -1:
        tail = tail[: BARE_URL_PATTERN.search(tail).start()]  # type: ignore[union-attr]
    assert _reference(tail) == -1
    assert _find_bare_url(tail + new, len(tail)) == _reference(tail + new), (tail, new)


def _split_trailing_reference(region: str) -> tuple[str, str]:
    """The original quadratic `_split_trailing`, kept as the specification."""
    end = len(region)
    while end:
        ch = region[end - 1]
        if ch in ".,:;!?*_~'\"" or (
            ch == ")" and region.count("(", 0, end) < region.count(")", 0, end)
        ):
            end -= 1
            continue
        break
    if not BARE_URL_PATTERN.match(region[:end]):
        return region, ""
    return region[:end], region[end:]


@settings(max_examples=2_000, deadline=None)
@given(text=_URLISH, trail=st.text(alphabet="().,:;!?*_~'\"x", max_size=12))
def test_e1_split_trailing_matches_the_reference(text: str, trail: str) -> None:
    region = text + trail
    assert _split_trailing(region) == _split_trailing_reference(region)


# --------------------------------------------------------------------------- CPU budgets
BUDGET_S = 2.0

FLOODS = {
    # The measured payloads: constructs that emit nothing grow the joined word.
    "empty_links_200kb": "a[](x)" * 34_000,
    "dropped_autolinks": "a<javascript:x>" * 15_000,
    # Other constructs that emit nothing or are re-scanned into the same word.
    "dropped_links_with_text": "x[x](y)" * 30_000,
    "data_autolinks": "a<data:x>" * 20_000,
    "rewritten_autolinks": "a<http://zzz>" * 20_000,
    "rewritten_emails": "a<a@b.c>" * 20_000,
    "at_then_empty_links": "a@" + "b[](x)" * 34_000,
    "scheme_chars_then_drops": "a1.+-" + "b<javascript:x>" * 15_000,
    "at_flood": "@" * 200_000,
    "at_word_flood": "a@" * 100_000,
    "dotted_word_flood": "a." * 100_000,
    "scheme_word_flood": "a1" * 100_000,
    "slash_free_colons": "a:" * 100_000,
    "lone_backslashes": "a\\" * 100_000,
    "close_parens_after_url": ("www.e" + ")" * 500 + " ") * 400,
    "trailing_punct_after_url": ("http://e" + ".," * 250 + " ") * 400,
}


def _feed_in_chunks(text: str, size: int) -> str:
    sanitizer = MarkdownStreamSanitizer()
    out = [sanitizer.feed(text[i : i + size]) for i in range(0, len(text), size)]
    out.append(sanitizer.finish())
    return "".join(out)


@pytest.mark.parametrize("name", sorted(FLOODS))
def test_e1_bare_url_floods_cost_linear_cpu(name: str) -> None:
    text = FLOODS[name]
    started = time.perf_counter()
    whole = sanitize(text)
    chunked = _feed_in_chunks(text, 7)
    assert time.perf_counter() - started < BUDGET_S, name
    assert chunked == whole, name  # chunking still cannot matter
    assert not violations(whole[:20_000])


def test_e1_find_bare_url_is_linear_on_a_huge_word() -> None:
    """The unanchored pattern needs ~n^2/2 steps here (hours at 1 MB)."""
    for word in ("a" * 1_000_000, "a1" * 500_000, "a@" * 500_000, "@" + "a" * 1_000_000):
        started = time.perf_counter()
        assert _find_bare_url(word) == -1
        assert time.perf_counter() - started < BUDGET_S


@pytest.mark.parametrize(
    "text",
    ["a" * 60_000, ("x" * 511 + " ") * 120, "a@" * 30_000, "a\\" * 30_000, "w.w" * 20_000],
)
def test_e1_words_fed_one_char_at_a_time_cost_linear_cpu(text: str) -> None:
    """A word held across feeds resumes its scan; restarting it cost HOLD_LIMIT per feed."""
    started = time.perf_counter()
    events = stream(list(text))
    assert time.perf_counter() - started < BUDGET_S
    assert "".join(events) == sanitize(text)


def test_e1_joined_word_detection_survives_the_fast_path() -> None:
    """The tail skip must still see URLs that start in text emitted before the drop."""
    for text in (
        "ww<javascript:x>w.evil.example",
        "a" * 300 + "@" + "<javascript:x>" * 3 + "evil.example",
        "htt[](x)p://evil.example",
        "mail[](x)to:a@b",
        "x/[](x)/evil.example",
        "a1.b+c-d" + "[](x)" * 10 + "://evil.example",
    ):
        out = sanitize(text)
        assert not violations(out), (text, out)
