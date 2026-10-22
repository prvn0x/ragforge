"""Token counting: the ONE place the retrieval pipeline measures text length.

Why an approximation: the embedding model (nomic-embed-text, BERT WordPiece, uncased) and the
reranker (FlashRank MiniLM/TinyBERT, WordPiece) need a `tokenizer.json` that is not shipped with
any installed package, and fetching it at runtime would be a network call. Phase 2 bakes the
tokenizer into the server image (docs/handoff/B.md) and swaps the body of `count_tokens` only.

The approximation is deliberately CONSERVATIVE (it over-counts relative to WordPiece), because
the chunker's hard limit (512) exists so the reranker never truncates a chunk: over-counting
makes chunks slightly smaller, under-counting silently truncates them.

Rules (mirroring how BERT's BasicTokenizer + WordPiece behave):
  * Scripts written without spaces cost one token PER CHARACTER. BERT splits every CJK
    ideograph into its own token, and Kana / Thai / Lao / Khmer / Myanmar runs fall back to
    per-character pieces because the vocabulary has almost no multi-character entries for them.
    Treating such a run as one "word" (the `\\w+` rule below) under-counts it about 5x.
  * A Hangul syllable costs as many tokens as its NFD decomposition has jamo (2 or 3): the
    uncased tokenizer strips accents via NFD, which decomposes each syllable into jamo before
    WordPiece sees it.
  * Every other punctuation / symbol character is its own token.
  * A word containing a digit costs 1 + (n - 1) // 2 tokens (hex, ids and numbers split into
    ~2-character pieces; measured against the real model).
  * Any other word of n characters costs 1 + (n - 1) // 6 tokens (rare words split into
    sub-word pieces; WordPiece averages ~4-6 characters per piece for unknown words).
"""

from __future__ import annotations

import re
import unicodedata

# Per-character scripts (see module docstring). Kept in one class so the word rule can exclude it.
_PER_CHAR = (
    "฀-໿"  # Thai, Lao
    "က-႟"  # Myanmar
    "ក-៿"  # Khmer
    "ᄀ-ᇿ㄰-㆏"  # Hangul jamo
    "぀-ヿㇰ-ㇿｦ-ﾟ"  # Hiragana, Katakana (+ extensions, half-width)
    "㐀-䶿一-鿿豈-﫿"  # CJK ideographs (+ ext. A, compatibility)
    "\U00020000-\U0002fa1f"  # CJK ideographs ext. B-F + compatibility supplement
)
_HANGUL_SYLLABLES = "가-힯"
_PIECES = re.compile(
    rf"([{_HANGUL_SYLLABLES}]+)|([{_PER_CHAR}]+)|([^\W{_PER_CHAR}{_HANGUL_SYLLABLES}]+)|[^\w\s]",
    re.UNICODE,
)
_SUBWORD_CHARS = 6


def count_tokens(text: str) -> int:
    """Approximate the number of model tokens in `text` (conservative; see module docstring).

    Additive across whitespace: count(a + " " + b) == count(a) + count(b). The chunker relies on
    that to sum unit costs instead of re-counting whole chunks."""
    total = 0
    for match in _PIECES.finditer(text):
        hangul, per_char, word = match.groups()
        if hangul:
            total += len(unicodedata.normalize("NFD", hangul))
        elif per_char:
            total += len(per_char)
        elif word:
            if any(ch.isdigit() for ch in word):
                # Digit-bearing words (hex, ids, versions, numbers) shatter into ~2-char WordPiece
                # pieces: measured on real nomic-embed-text, the 6-char rule under-counted
                # "0x0000abcd"-style text by more than 2x and overflowed the model context.
                total += 1 + (len(word) - 1) // 2
            else:
                total += 1 + (len(word) - 1) // _SUBWORD_CHARS
        else:
            total += 1
    return total
