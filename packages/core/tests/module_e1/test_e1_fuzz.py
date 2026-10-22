"""Differential, grammar-based fuzzing of the streaming sanitizer (ADR-0011 "Tests").

Generated hostile markdown is split at random token boundaries (and, separately, at random
character offsets), streamed through the sanitizer, and every emitted prefix is rendered by
markdown-it-py. The invariant: no prefix of the stream renders into an image, raw HTML, a link
whose href is not allow-listed, or autolinkable bare-URL text.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from module_e1.e1_support import (
    ALLOWED,
    document,
    group_tokens,
    inline,
    sanitize,
    split_at,
    stream,
    violations,
)

FUZZ = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _assert_every_prefix_safe(events: list[str]) -> None:
    """The client renders the ACCUMULATED stream, so every prefix must be safe. (A single event
    rendered in isolation is not meaningful: fenced code passes unheld, so a mid-fence event is
    code only in the context of the opener that preceded it.)"""
    rendered = ""
    for event in events:
        if not event:
            continue
        rendered += event
        assert not violations(rendered), (rendered, violations(rendered))


@FUZZ
@given(tokens=document(), data=st.data())
def test_e1_token_boundary_splits_never_render_exfiltration(
    tokens: list[str], data: st.DataObject
) -> None:
    cut_after = data.draw(st.lists(st.booleans(), min_size=len(tokens), max_size=len(tokens)))
    events = stream(group_tokens(tokens, cut_after))
    _assert_every_prefix_safe(events)
    # Determinism: the split never changes the output, so boundaries cannot smuggle anything.
    assert "".join(events) == sanitize("".join(tokens))


@FUZZ
@given(tokens=document(), cuts=st.lists(st.integers(min_value=0, max_value=4000), max_size=40))
def test_e1_arbitrary_char_splits_match_whole_feed(tokens: list[str], cuts: list[int]) -> None:
    text = "".join(tokens)
    events = stream(split_at(text, cuts))
    _assert_every_prefix_safe(events)
    assert "".join(events) == sanitize(text)


@FUZZ
@given(tokens=document())
def test_e1_double_feeding_never_produces_a_live_link(tokens: list[str]) -> None:
    """Sanitizing is not idempotent (escapes get escaped), but it must stay SAFE when applied
    to its own output, e.g. a replayed answer."""
    once = sanitize("".join(tokens))
    twice = sanitize(once)
    assert not violations(twice), (twice, violations(twice))
    thrice = "".join(stream(list(once)))  # and one char at a time
    assert not violations(thrice)


@FUZZ
@given(text=inline, prefix=st.sampled_from(["", "> ", "- ", "| ", "\n", "```\n"]))
def test_e1_inline_constructs_fed_one_char_at_a_time(text: str, prefix: str) -> None:
    events = stream(list(prefix + text))
    _assert_every_prefix_safe(events)
    assert "".join(events) == sanitize(prefix + text)


def test_e1_oracle_is_not_vacuous() -> None:
    """The oracle must flag every vector it is supposed to catch, or the fuzz proves nothing."""
    hostile = [
        "![x](http://evil.example/a.png)",
        "[x](http://evil.example)",
        "<http://evil.example>",
        "<me@evil.example>",
        "see www.evil.example now",
        "see http://evil.example now",
        "mail me@evil.example",
        "write a@évil.com",
        "write a@evil.рф",
        "mail me@例え.jp",
        "mailto:;@evil.example?body=SECRET",
        "mailto:a@b",
        "see //evil.example/x now",
        '<img src="http://evil.example">',
        "<div>\nx\n</div>",
        "[x][r]\n\n[r]: http://evil.example",
        "![x][r]\n\n[r]: http://evil.example",
        "| a |\n|---|\n| [x](http://evil.example) |",
    ]
    for markdown in hostile:
        assert violations(markdown), markdown
    assert not violations(f"[ok]({ALLOWED[0]}) and `http://in.code` and\n```\n<img>\n```")
