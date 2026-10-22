"""Pure presentation helpers: Markdown safety, badges, citation chips, extras, timings."""

from __future__ import annotations

from uuid import uuid4

import pytest
from ui_support import NOW, citation

from ragforge_contracts.api import Collection
from ragforge_ui.render import (
    EXTRAS_HEADING,
    WORD_JOINER,
    answer_markdown,
    badge_for,
    call_summary,
    citation_line,
    collection_labels,
    extra_view,
    for_streamlit,
    md_literal,
    preview_text,
    safe_href,
    timings_text,
)
from ragforge_ui.state import Extra, PendingCall

WJ = WORD_JOINER


# ------------------------------------------------------------------ text safety
def test_md_literal_makes_server_text_inert() -> None:
    hostile = "[click](https://evil.example) **bold** <b>x</b> :red[a] $x$\n# heading"
    out = md_literal(hostile)
    for active in ("[", "]", "(", ")", "*", "<", ">", ":", "$", "#"):
        assert f"\\{active}" in out
        assert out.replace(f"\\{active}", "").count(active) == 0, active
    assert "\n" not in out  # one line: cannot start a heading or list item


def test_md_literal_strips_control_and_bidi_characters() -> None:
    assert md_literal("a\x1b[31mb\u202ec") == "a\\[31mbc"


def test_for_streamlit_escapes_streamlit_only_syntax_outside_code() -> None:
    text = "Costs $5 and $6 :red[alert] :material/home: `a:$b` done"
    out = for_streamlit(text)
    assert out == f"Costs \\$5 and \\$6 \\:red[alert] \\:{WJ}material/home\\: `a:$b` done"


def test_shortcodes_get_a_word_joiner_because_escapes_do_not_stop_them() -> None:
    # Streamlit's frontend rewrites `:material/` in the raw source and runs remark-emoji on
    # text nodes AFTER escapes resolve, so `\\:smile\\:` alone would still become an emoji.
    out = for_streamlit("ok :smile: and \\:tada\\: and :+1: and :material/home:")
    assert out == (
        f"ok \\:{WJ}smile\\: and \\:{WJ}tada\\: and \\:{WJ}+1\\: and \\:{WJ}material/home\\:"
    )
    assert ":material/" not in out.replace("\\", "")  # the raw-source rewrite can't fire


def test_plain_colons_get_no_word_joiner() -> None:
    assert WJ not in for_streamlit("Note: a b. Time 12: 30. See https://x.example/a")
    assert WJ not in md_literal("Kafka: basics")


def test_md_literal_breaks_shortcodes_too() -> None:
    assert md_literal("Deploy :rocket: now") == f"Deploy \\:{WJ}rocket\\: now"


def test_for_streamlit_never_doubles_an_existing_escape() -> None:
    assert for_streamlit("a\\: b \\$ c \\\\") == "a\\: b \\$ c \\\\"


def test_for_streamlit_leaves_fenced_code_verbatim_even_unclosed() -> None:
    closed = "see:\n```python\nx = {'a': 1}  # $HOME\n```\nafter: y"
    assert for_streamlit(closed) == "see\\:\n```python\nx = {'a': 1}  # $HOME\n```\nafter\\: y"
    unclosed = "intro:\n~~~\nk: v\n"
    assert for_streamlit(unclosed) == "intro\\:\n~~~\nk: v\n"


def test_for_streamlit_unmatched_backtick_is_not_a_code_span() -> None:
    assert for_streamlit("a ` b: c") == "a ` b\\: c"


def test_answer_sections_each_start_a_paragraph() -> None:
    answer = (
        "**In simple terms:** A log \\[S1\\]. **Technically:** Partitions \\[S2\\].\n"
        "**In production:** It broke \\[S3\\]."
    )
    out = answer_markdown(answer)
    assert out.split("\n\n") == [
        "**In simple terms\\:** A log \\[S1\\].",
        "**Technically\\:** Partitions \\[S2\\].",
        "**In production\\:** It broke \\[S3\\].",
    ]


def test_a_bold_section_word_inside_a_sentence_is_not_a_section() -> None:
    answer = "**In simple terms:** It is **Technically** a log, **In production** too."
    assert "\n" not in answer_markdown(answer)


def test_bold_label_with_colon_outside_also_starts_a_section() -> None:
    assert answer_markdown("a. **Technically**: b") == "a.\n\n**Technically**\\: b"


def test_collection_labels_disambiguate_duplicate_names() -> None:
    a = Collection(id=uuid4(), name="notes", active_space_id=1, created_at=NOW)
    b = Collection(id=uuid4(), name="notes", active_space_id=1, created_at=NOW)
    c = Collection(id=uuid4(), name="work", active_space_id=1, created_at=NOW)
    labels = collection_labels([a, b, c])
    assert list(labels) == [a.id, b.id, c.id]  # keyed by id: both "notes" stay selectable
    assert labels[c.id] == "work"
    assert labels[a.id] == f"notes · {str(a.id)[:8]}"
    assert len(set(labels.values())) == 3


def test_answer_sections_are_not_split_inside_code() -> None:
    answer = "**Technically:** see\n```\n**Technically** stays\n```"
    assert "\n```\n**Technically** stays\n```" in answer_markdown(answer)


def test_preview_is_plain_text_without_escapes_or_markers() -> None:
    assert preview_text("**In simple terms:** a \\[S1\\]\x1b") == "In simple terms: a [S1]"


# ------------------------------------------------------------------ badges
@pytest.mark.parametrize(
    ("status", "label", "color"),
    [
        ("verified", "Verified", "green"),
        ("partial", "Partly verified", "orange"),
        ("unsupported", "Not supported", "red"),
        ("unverified", "Not checked", "gray"),
    ],
)
def test_badge_for_each_known_status(status: str, label: str, color: str) -> None:
    badge = badge_for(status)
    assert badge is not None
    assert (badge.label, badge.color) == (label, color)
    assert badge.meaning.endswith(".")


def test_unknown_status_is_neutral_never_green_and_escaped() -> None:
    badge = badge_for("[great](x)")
    assert badge is not None
    assert badge.color == "gray"
    assert badge.label == "\\[great\\]\\(x\\)"
    assert "unchecked" in badge.meaning


def test_no_verdict_no_badge() -> None:
    assert badge_for(None) is None


# ------------------------------------------------------------------ citations
def test_citation_chip_uses_the_answer_label_and_links_the_paragraph() -> None:
    line = citation_line(citation(3), index=1)
    assert line == (
        "**\\[S3\\]** [Kafka basics › Consumers](https://www.notion.so/Kafka-basics-abc#block1)"
    )
    assert "block-3" not in line  # the anchor is a source id, never shown


def test_inferred_citation_is_starred_and_attributed_to_the_server() -> None:
    line = citation_line(citation(2, inferred=True), index=1)
    assert line.startswith("**\\[S2\\*\\]**")
    assert "matched by the server" in line


def test_citation_without_label_falls_back_to_index() -> None:
    plain = citation(1).model_copy(update={"label": None})
    assert citation_line(plain, index=4).startswith("**\\[4\\]**")


def test_citation_title_cannot_inject_a_link() -> None:
    evil = citation(1, title="x](https://evil.example) [y")
    line = citation_line(evil, index=1)
    assert line.count("](") == 1  # only the chip's own link
    assert "https://www.notion.so" in line


def test_citation_without_url_is_plain_text() -> None:
    bare = citation(1).model_copy(update={"url": None})
    assert "](" not in citation_line(bare, index=1)


def test_safe_href_only_http_and_encodes_link_breakers() -> None:
    assert safe_href("javascript:alert(1)") is None
    assert safe_href("data:text/html,x") is None
    assert safe_href("https://n.so/a b(c)<d>") == "https://n.so/a%20b%28c%29%3Cd%3E"
    assert safe_href("https://n.so/p?x=1#blk") == "https://n.so/p?x=1#blk"


# ------------------------------------------------------------------ extras
def test_code_extra_splits_prose_and_code_with_language() -> None:
    view = extra_view(
        Extra("code", "```python\ndef f(**kw):\n    return r'\\.'\n```\nThen call f\\.")
    )
    assert view.title == "A code example"
    assert view.code == "def f(**kw):\n    return r'\\.'"  # verbatim: nothing unescaped
    assert view.language == "python"
    assert view.after == "Then call f\\."
    assert view.before == ""


def test_code_extra_with_odd_info_string_gets_no_language() -> None:
    view = extra_view(Extra("code", "```<script> x\nprint()\n```"))
    assert (view.code, view.language) == ("print()", None)


def test_incident_extra_is_markdown_prose_only() -> None:
    view = extra_view(Extra("incident", "Common failure pattern\\: a rebalance storm $1\\."))
    assert view.title == "A common production failure pattern"
    assert view.code is None
    assert view.before == "Common failure pattern\\: a rebalance storm \\$1\\."


def test_unknown_extra_kind_is_still_general_knowledge() -> None:
    assert extra_view(Extra("trivia", "x")).title == "General knowledge"
    assert "Not from your notes" in EXTRAS_HEADING
    assert "not verified" in EXTRAS_HEADING


def test_timings() -> None:
    assert timings_text(1.234, 5.0) == "first token 1.2 s · total 5.0 s"
    assert timings_text(None, 0.5) == "first token - · total 0.5 s"


def test_section_right_after_a_fence_keeps_the_fence_closed() -> None:
    out = answer_markdown("```\nx\n```\n**Technically:** y")
    assert out == "```\nx\n```\n\n**Technically\\:** y"


# ------------------------------------------------------------------ approvals
def test_call_summary_names_the_document_a_removal_would_touch() -> None:
    doc = str(uuid4())
    call = PendingCall("t1", "remove_from_index", {"document_id": doc})
    line = call_summary(call, {doc: "Kafka [basics](https://evil.example)"})
    assert line is not None
    assert line.startswith("Remove “Kafka \\[basics\\]\\(https")  # a title can't be a link
    assert f"document {doc[:8]}…" in line


def test_call_summary_says_plainly_when_the_document_is_unknown() -> None:
    doc = str(uuid4())
    line = call_summary(PendingCall("t1", "remove_from_index", {"document_id": doc}), {})
    assert line is not None
    assert "unknown document" in line
    assert doc[:8] in line


def test_call_summary_other_tools_and_no_document() -> None:
    doc = str(uuid4())
    line = call_summary(PendingCall("t1", "some_new_tool", {"document_id": doc}), {doc: "Notes"})
    assert line == f"Acts on “Notes” (document {doc[:8]}…)"
    assert call_summary(PendingCall("t2", "forget_fact", {"fact_id": "f1"}), {}) is None


def test_a_longer_closing_fence_ends_the_block_so_text_after_it_stays_escaped() -> None:
    """Integrated review P2: a closer longer than the opener was read as "unclosed", so model
    text after it ($LaTeX$, :material/verified: icons) rendered live."""
    from ragforge_ui.render import answer_markdown

    out = answer_markdown("```\nx\n````\nAfter $a$ :material/verified:")
    after = out.split("````", 1)[1]
    assert "$a$" not in after
    assert ":material/verified:" not in after
