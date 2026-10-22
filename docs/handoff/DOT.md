# DOT handoff: source-backed dotted names are not sentence ends

Module: DOT (Phase 3c). Owner report 2026-10-19: two answer sentences that the notes back were cut
at the LAST dot of a dotted name, and the first fragment was flagged "Not found in your notes":

- "You can accomplish this by making a POST request to /actuator/loggers/{logger.name} with a JSON body ... [S1]." was cut after `logger.`
- "Setting max.in.flight.requests.per.connection to 1 provides strictest ordering [S2]." was cut after `per.`

Cause: `_identifier_dot` only looks AFTER a lower.lower dot (a known extension, or a `_CHAIN`
character next). So the last dot of a dotted name followed by a space was a sentence end.

## What I built (the binding design-review rule)

`packages/core/src/ragforge_core/agent/coverage.py`

- **`dotted_names(texts: Iterable[str]) -> frozenset[str]`** (exported). It parses each page as Markdown and reads what the reader sees (`_shown_pieces`): each inline part with emphasis marks dropped and escapes/entities resolved (the same view `_view` gives the answer side), plus each code span, URL and code block as its own piece; HTML blocks give nothing, and any non-text atom bounds a run. In that text it takes every maximal ASCII `[A-Za-z0-9_.]` run and strips its leading and trailing dots. A run that still holds a dot is a name. Matching is verbatim: case-sensitive, no NFKC.
  - WHY the parsed view (review round 2, P2): on raw Markdown, `max.in.**flight**.requests` gave the partial names `max.in` / `requests` and `max\_poll.ms` gave `poll.ms`, names the reader never sees as a whole run, while the name the reader does see was missing. Code is kept because config keys are usually in code spans (the live `{logger.name}` page is).
- **`_NamedRuns`**, the exemption used inside `sentence_ends`. An interior ASCII `.` that is followed by `[A-Za-z0-9_]` is in-token ONLY when the maximal run around it (dots stripped) is in `names`.
  - It is OR'd after the existing `_dot_in_token`. It only runs when the old code would cut at a single `.` (`j - i == 1`), so every existing fail-closed rule stays ahead of it.
  - Dot runs, homoglyph or look-alike dots, `_dot_like`, `_foreign_tail`, bidi, RTL and blank glyphs are untouched.
  - A dot followed by whitespace, a closer, an emphasis mark, a code span or the end of the text is never exempt.
  - **Never a capital after the dot (review rounds 1 and 2, P1).** Only a dot followed by `[a-z0-9_]` is ever exempt, whatever the run's dot count. `fast.It`, `2.8.It`, `U.S.The` and `kafka.streams.It` stay cut even when a source holds them verbatim. WHY: a missing space in pasted or PDF-extracted notes ("Redis is fast.It keeps data", "Kafka 2.8.It added KRaft mode") put the run into `names`, and an answer carrying an unbacked clause before it became one covered unit. A reader sees a period then a capital as a sentence end, so that was a merge. Round 1 kept capitals exempt in 2+-dot runs (`java.util.List`); round 2 showed the same merge there, so that exception is gone. `java.util.List` keeps its cut before `List`, exactly as before DOT. Both live cases are all lowercase, so they are unaffected.
  - It caches one run at a time, so the scan stays linear.
- **Keyword-only `names: frozenset[str] = frozenset()`** on `sentence_ends`, `segment`, `visible_runs`, `residue`, `refusal_only` and `source_sentences`. It also reaches `_view` and `_Builder` internally (positional there, default empty).
  - `visible_runs` accepts `names` only so the seam has one signature. Its runs and accepted labels do not depend on cuts.
- **Empty `names` behaves exactly as before.** I checked this differentially against the pre-change module on 35,446 texts (every string literal in `packages/core/tests/module_*` plus 20,000 random texts). `sentence_ends`, the `segment` unit spans and labels, and `residue` all matched, with 0 differences.

`packages/core/src/ragforge_core/agent/attribution.py` (needed so one answer is cut the same way everywhere)

- `attribute(..., *, names=frozenset())` passes `names` to `segment`, `refusal_only` and each source's `source_sentences` (through `_source`). A dotted name is then one token on both the answer side and the source side.
  - Source-side merging (review round 1, P2), a deliberate choice: the source is cut with the same `names`, so a no-space dot in a page also stops cutting THAT page's sentences. That only happens where the page shows one token: the capital rule above applies here too, so "fast.It keeps" in a page is still two source sentences. A lowercase no-space run ("not durable.it is fast") is one source sentence for `_negation_ok` / `_neighbours`, exactly as a reader sees that page. Keeping sources on the old cuts would split "max.in.flight.requests.per.connection to 1" from its number in the source and lose number parity with the answer side. The inferred citation is still ground-checked.
- `settle(..., names=frozenset())` passes `names` to `residue`. Without it, a unit that attribution kept whole would be residue again in two pieces. That fails closed (a test pins it), but it is a false flag.

Tests: `packages/core/tests/module_ag/test_ag_dot_names.py` (78 tests, hypothesis included). I changed no existing module_ag assertion.

- Both live sentences are each one unit with the names of their source pages. With valid labels they have no residue, and `attribute` followed by `settle` gives `verified` with no flagged claims. Without names they are cut exactly as before; the old false flag is pinned.
- An uncited live sentence is still residue. An uncited one is inferred as ONE unit.
- Source typos (round 1, P1/P2): with a page holding "Redis is fast.It keeps data ..." (so `fast.It` is in `names`, plus `fast.And`, `data.The`, `Kafka.It`), the texts "..., fast.It keeps ...", "fast.And it", "data.The broker", "Kafka.It is", "*fast*.It" are each cut exactly as without names. End to end, "Redis loses nothing on crash, fast.It keeps data in memory [S1]." leaves "Redis loses nothing on crash, fast." as residue and as an uncited clause. `java.util.List` keeps its cut before `List` with or without names.
- Capitals in multi-dot runs (round 2, P1), end to end through `residue`, `attribute` and `settle` with every dotted run of the answer backed by the page: "Kafka 2.8.It added KRaft mode.", "Offices in the U.S.The company grew." and "Use kafka.streams.It keeps state." as pages; the answers "..., as 2.8.It added KRaft mode [S1].", "... in the U.S.The company grew [S1]." and "..., use kafka.streams.It keeps state [S1]." leave the unbacked first clause as residue and uncited, and a "verified" verifier verdict settles to `partial`.
- Markdown sources (round 2, P2): `dotted_names` of `max.in.**flight**.requests`, `max\_poll.ms`, a code span, a code block, an HTML block, a code span next to a dot (`x.y`.z gives only `x.y`) and inline HTML between runs; a page with emphasis inside the name keeps the plain answer sentence as one unit.
- Merge attacks are cut exactly as without names, even with near-miss names present (`fast.and`, `and.it`, `nasa.in`, `a.b`, `js.it`): fast.and.it, nasa.in.fact, /dev/null.it, (nasa.it, done.it.works, node.js.it, `fast.And it`, "see a.b. Then", "e.g.", "3.5", the name followed by ". Then", a different case, one character more, a longer run.
- Terminators: ". Then", ". then", ".\n" and a final dot always cut.
- Homoglyph dots (U+2024, U+FF0E, U+00B7, U+2027, U+0701), ".." and ".\u200b" still cut. Bidi or RTL anywhere still makes the whole answer residue.
- Emphasis and code:
  - Parsed emphasis is not visible text, so `**max.in.flight.requests.per**.connection` is the name: the reader sees no space.
  - Literal `**.**` between letters is NOT emphasis (CommonMark flanking). Its asterisks bound the run, so the old cuts stay, which is the safe side.
  - `fast**.**and.it` and `**fast**.and.it` stay cut.
  - A code span next to the dot keeps the old cuts. A name inside a code span is atomic either way.
- Properties (hypothesis):
  1. The cuts with `names` equal the cuts without `names` minus ONLY the ends of interior dots of runs in `names`, never one before a capital. An independent regex oracle computes those, so dot runs are never exempt. The word pool includes capitals after dots ("It", "Kafka", "Then").
  2. Cores that are not in `names` keep every cut.
  3. Even with EVERY dotted run of the text backed, no single dot followed by whitespace is ever merged.
  4. Even with every dotted run backed, a dot before a capital is never exempt, in any run.
  5. `segment` with names never yields more units than without.
- CPU: 20,000-character degenerate dotted runs stay linear.

## Integration steps (outside my paths)

1. **`packages/core/src/ragforge_core/agent/citations.py`** (required; IV3-CORE currently works around it):
   - Add keyword-only `names: frozenset[str] = frozenset()` to `check_citations` and pass it to `segment(answer, names=names)` (line ~284).
   - Thread it through `_stray_spans(text, names)` to `visible_runs(text, names=names)` (line ~255). That second one is a no-op today; it keeps the seam consistent.
   - WHY: strict mode demands a label on every prose unit. Without `names`, the fragment "...{logger." has no label and the turn is rejected or regenerated. That is the same false flag, through the strict path.
2. **Chat path, `packages/core/src/ragforge_core/agent/nodes.py`**:
   - Build `names = dotted_names(state["hits"][n - 1]["display_text"] for n in _shown_labels(state))`. Use ONLY the shown `S` sources, never tool results: a tool result is text the model steers. Compute it even for action replies, so every reader of one draft takes the same set.
   - Pass the same set to `check_citations(...)` (~line 553), `attribute(...)` in `_attribute` (~line 523), `settle(...)` (~line 619) and, for consistency, `refusal_only(draft, names=names)` (~line 869).
   - A small `_names(state)` helper next to `_shown_labels` keeps the calls identical. Build it from the shown S hits' `display_text` ONLY, and pass the same set to all five call sites. A mismatched set fails closed (pinned by `test_settle_without_the_same_names_still_fails_closed`), but it is a false flag.
   - **DOT alone does not fix the live chat-path false flag** (review round 1, P2). Until this step lands, `check_citations` (strict) and the chat path's `attribute` / `settle` / `refusal_only` calls pass no names, so the `{logger.` and `per.` fragments are still unlabelled and flagged or regenerated.
   - **Required gate before pasting the ADR paragraph as unconditional:** add an end-to-end chat-path test (in the integrator's own test dir) with both live sentences, cited, on hits holding `/actuator/loggers/{logger.name}` and `max.in.flight.requests.per.connection`: expect `verified`, no flagged claims.
3. Interview path: IV3-CORE already passes `names=known.names` (from the notes' `display_text`) to `attribute`, `settle` and `segment` in `agent/interview/verify.py`. Once step 1 lands, it can go back to `check_citations(strict=True, names=known.names)` if it wants.
4. Nothing in contracts, domain, the client or config changes.

## Test live (integrator)

- Ask the two live questions ("how do I change a Spring Boot log level at runtime?" and "how do I get strict ordering from a Kafka producer?") on the notes that hold `/actuator/loggers/{logger.name}` and `max.in.flight.requests.per.connection`. Expect `verified`, or at least no "Not found in your notes" fragment ending in `logger.` or `per.`.
- Regression probe: an answer with "Kafka is fast.and.it never loses data" uncited still flags.
- Typo probe: a note with "Redis is fast.It keeps data in memory"; an answer "Redis loses nothing on crash, fast.It keeps data in memory [S1]." must flag "Redis loses nothing on crash, fast.". Same with a note "Kafka 2.8.It added KRaft mode." and the answer "Kafka was built by NASA in 1850, as 2.8.It added KRaft mode [S1]." (must flag the first clause).
- Then run `uv run pytest packages/core/tests/module_ag packages/core/tests/module_iv packages/core/tests/module_e1 -q`.

## Status of checks at hand-off

- Clean on my paths: `ruff format --check`, `ruff check`, `mypy` on coverage.py and attribution.py, and `MYPYPATH=packages/core/tests/module_ag mypy packages/core/tests/module_ag/test_*.py`. `lint-imports` passes (6 kept, 0 broken).
- Round 3 (after the round-2 review fixes): `uv run pytest packages/core/tests/module_ag packages/core/tests/module_iv packages/core/tests/module_e1 -q`: 1938 passed, 0 failed.

## Paragraph for ADR-0015 "Current state" (append after the code-coverage paragraph)

**Dotted names (owner report 2026-10-19, DOT).**
- *What a dot does.* A single ASCII "." that is followed by `[a-z0-9_]` (never a capital) is not a sentence end when the maximal `[A-Za-z0-9_.]` run around it (leading/trailing dots stripped) is a dotted name found verbatim in what the reader sees of the SHOWN `S` sources (`coverage.dotted_names`: the parsed Markdown view, code spans and code blocks included; case-sensitive, never from `T` tool results). `segment`, `residue`, `visible_runs`, `attribute` and `settle` take the set (`names=`), and the interview path passes it. The chat path (`nodes.py`) and the strict citation check (`check_citations`) pass it once DOT's integration steps land; until then they keep the old cuts, so the two live false flags remain on the chat path. Without it, the last dot of "max.in.flight.requests.per.connection to 1 [S2]." and of "/actuator/loggers/{logger.name} with ... [S1]." cut the sentence, and the backed first fragment was flagged.
- *Why source-backed.* A shape rule ("two or more dots between identifier characters is a token") was rejected in design review: it merges "Kafka is fast.and.it never loses data", which hides a sentence end exactly as "node.js.it" once did.
- *What it never does.* It only removes cuts at interior dots of runs the page shows as one token. A dot followed by whitespace, a closer, an emphasis mark, a code span or the end is never exempt. A dot followed by a capital is never exempt, in any run ("fast.It", "2.8.It", "U.S.The", "kafka.streams.It", and so "java.util.List" keeps its old cut before "List"), even when a page holds it: a reader sees a period then a capital as a sentence end, and a missing space is a common typo in pasted notes. Dot runs, look-alike dots, `_dot_like`, `_foreign_tail`, bidi, RTL and blank-glyph rules stay ahead of it.
- *How it is held.* Hypothesis tests show the cuts with names equal the cuts without names minus exactly those dots, and that no dot followed by whitespace or by a capital is ever merged, even when every dotted run of the answer is backed. With no names the segmentation is unchanged (differential check on 35k texts).
- *Sources.* Sources are cut with the same names, so a page's own lowercase no-space run ("not durable.it is fast") is one source sentence for the negation and number evidence, as the reader sees that page. The inferred citation is still ground-checked.
- *Residual.* A source that holds a lowercase no-space typo ("durable.it", "fast.and.it") verbatim exempts that run in the answer too. This is accepted: the reader sees neither a space nor a capital there, so it is not two sentences to a reader. A capitalised typo ("fast.It", "2.8.It") is never exempt.
