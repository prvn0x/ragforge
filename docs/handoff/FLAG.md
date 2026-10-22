# FLAG handoff: false "Not verified" flags (identifier rule + claim reasons)

Module FLAG. Only owned paths were edited. No shared contract changed; none was insufficient.

## What was built

### (a) Single-identifier rule (`agent/attribution.py`)
- `identifier_name(span)`: the binding grammar, ASCII only, no normalisation:
  `[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*(\(\))?`, plus an identifier-shape signal
  (`_shaped`): contains `_` or `.`, ends in `()`, or has a lower/digit -> upper camel hump.
  `createMemento()`, `ConfigurationManager`, `max.poll.interval.ms` and `MAX_RETRIES` qualify.
  `Memento`, `acks`, `true`, `3`, `null`, `id`, `acks=all`, `foo(x)` and anything non-ASCII do not.
- `_code_names(code)`: the whole tokens of one source code element. Each maximal dotted run whose
  neighbours are ASCII and do not continue it (`[A-Za-z0-9_$\-/\\]`) gives each of its segments
  (a simple answer name matches any segment, so `this.createMemento()` backs `createMemento()`).
  It also gives the whole run, but only when no `.` continues it on either side (a dotted answer
  name must equal a whole maximal source name, so `interval.ms` and `max.poll` do NOT match
  `max.poll.interval.ms`). When `(` follows the run, it also gives `name()` / `whole()`.
  Code holding a bidi control, an RTL character or a blank glyph yields nothing.
- `_token_backed` is used ONLY in `_model_code_ok` (code the model labelled), S labels only
  (`_label_source` returns None for T labels). It reads the code of the cited label's SHOWN chunk
  (`_Source` is built from `display_text`), never the page's prose, never another page.
  `code_matches` (used by inferred citations and `verbatim_source`) is unchanged.
- Order of the rules: the caller rejects `joined_code` units before `_model_code_ok` runs, and
  segmentation (`_ends_inside`, `_dot_like`, `_foreign_tail`, the bidi/RTL/blank-glyph
  whole-answer rules) runs before attribution. The token rule only sees a span that stands alone.
- Decisions:
  - Inline code spans of the source count, as well as fenced and indented blocks. The page's
    author marked the name as code just as a block does, and the parser gives its exact content.
  - `foo()` IS backed by a source `foo(x)`. In prose, `name()` means "the function", not "called
    with no argument". What the sentence says about the call is the verifier's.
  - Case-sensitive. NFC/NFKC: no normalisation at all. Any non-ASCII character fails the
    grammar, so the span falls back to the whole-element rule (fail closed). This is stricter
    than coverage's NFKC reading, which is deliberate.

### (b) Claim reasons
- `attribute` now also returns `code_inexact`: the code-unbacked units whose code is an
  UNALTERED EXCERPT of a cited S source's code (`code_reason`/`_tokens_in_source`, review round
  1 P1 replaced the earlier bag-of-words check). Each code element of the unit, tokenized as
  words, numbers and single punctuation characters with every whitespace run read as one gap
  (`_code_run`), must be a CONTIGUOUS run of ONE source code element's tokens that starts and
  ends at a whitespace boundary of the source and, unless it reaches the element's end, does not
  stop with an argument open: after one of `/ \ = : - , | & ( [ { < + * @ $ %`, or on a flag
  word starting with `-` (`_excerpt_of`, `_open_end`). Every identifier-shaped name must also be
  a whole token of that code. Case-sensitive; code with bidi/RTL/blank glyphs is never muted.
  So `acks=all` (page: `acks=1`, `retries=all`), `enable.idempotence=false` (page: `=true`
  and `enable.auto.commit=false`), `rm -rf /` (page: `rm -rf /tmp/cache`), reordered options,
  a dropped middle line, a typo (`exactly_one_v2`) and a cut inside a dotted key are all
  `not_found` (red). A text that also occurs as a plain uncited unit is never muted.
- Unplaceable objections (review round 1, P2): `settle` passes `placed=False` to
  `flag_reasons` when the gate is closed (inconclusive or not located verdict) or the verdict is
  `partial` with no claim listed; nothing is muted then.
- `flag_reasons` (settle path) builds (text, reason) pairs in the existing claim order
  (verifier claims, residue, code-unbacked, uncited). Verifier claims and residue units are
  `not_found`. A code-unbacked unit is `inexact_code` (this takes precedence over its own
  `uncited` membership) unless it is residue on its own account or a verifier claim overlaps it
  by POSITION (`_overlaps`). `merge_reasons` then dedupes, with not_found winning, and only then
  splits into the two tuples, so they stay aligned after the dedupe.
- `uncited_reasons` (no-provenance path in `verify_citations`, answer order) and `all_not_found`
  (invalid-label path; unsupported verdict) cover the other producers.
- `state.claim_reasons` + `claim_reasons_from_state`: a missing key (old checkpoint), a
  misaligned list or an unknown value all read as `not_found`. `nodes._flagged` dedupes by text,
  with not_found winning. Rule-7 inferred sentences (untrusted verifier) are `not_found`.
  `finalize` emits `TurnVerification(status, claims, reasons)`. The runner initialises
  `claim_reasons: []`.
- `chat_stream._verification`: each claim is sanitized ONCE and the same text goes into both
  `unsupported_claims` and `claims[i].text`. Missing or misaligned reasons become `not_found`,
  and an unknown reason string maps to `not_found` (`_claim_reason`) instead of raising.
- The status semantics are unchanged: `verified` still needs empty residue, and inexact_code
  claims still make the answer `partial`.

### Eval-ready
- Pure, injectable functions for an offline harness (no HTTP, no model): `attribute`, `settle`,
  `flag_reasons`, `merge_reasons`, `uncited_reasons`, `identifier_name`, `code_reason`.
- `nodes.ground_check` now logs one `answer_settled` line per verifier call. It carries `prompt`
  (`ground_check@<v>`), `model`, `prompt_tokens`/`completion_tokens`, `latency_ms`, `chunk_ids`
  (the context chunks shown), `outcome` (`parsed`|`malformed`), `located`, `verdict`, `status`
  and the counts (`verifier_claims`, `residue`, `code_unbacked`, `inexact_code`,
  `inferred_kept`). It never logs text.

## Files
- src: `agent/attribution.py`, `agent/nodes.py`, `agent/state.py`, `agent/runner.py`,
  `apps/api/chat_stream.py` (`coverage.py` unchanged)
- tests (new): `tests/module_ag/test_ag_flag_identifiers.py` (69, incl. one-character-mutation
  property + hypothesis), `tests/module_ag/test_ag_flag_reasons.py` (22, incl. runner end to
  end and the log line), `tests/module_ag/test_ag_flag_excerpts.py` (21: the reviewer's three
  counterexamples and more altered/truncated forms, unaltered excerpts that stay muted, a
  hypothesis property swapping one token of a backed line for another token of the same page
  -> not_found, unplaceable objections never mute, a name found only in a source comment)
- tests (edited): `tests/module_e2/test_e2_chat_sse.py`
  - The existing full-turn test gains one assertion: `claims == [UnsupportedClaim("<s>bad
    claim</s>", "not_found")]`. A producer without reasons now also fills `claims`; this is
    additive, and nothing existing changed.
  - New test `test_verification_claims_carry_reasons_with_the_same_sanitized_text`: reasons are
    mapped, the texts are identical to `unsupported_claims`, and a verified event has no claims.
- No existing module_ag assertion needed changing.

## Integration steps
1. Nothing outside my paths is required to build or test.
2. ADR-0015 "Current state": replace the sentence "**Code is covered only as a whole fenced block
   or a whole inline code span of a cited source, exactly. There is no exception.**" with the
   paragraph below (the integrator writes the ADR):

   > **Code is covered only as a whole fenced block or a whole inline code span of a cited
   > source, exactly, with ONE narrow exception (owner decision 2026-10-18, FLAG):** an inline
   > code span of a model-labelled sentence whose content is a single ASCII identifier
   > (`[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*(\(\))?`) that also LOOKS like one (has
   > `_` or `.`, ends in `()`, or has a camel hump) is backed when it is a whole token of a code
   > element (fenced/indented block or inline span) of the SHOWN chunk of an `S` label the
   > sentence cites. A simple name may match any segment of a dotted source name. A dotted name
   > must equal a whole maximal one. `name()` matches `name(` with any arguments. The match is
   > case-sensitive and does no normalisation: any non-ASCII character, a neighbouring character
   > that continues the token (`[A-Za-z0-9_$-/\]` or non-ASCII), or bidi/RTL/blank glyphs in the
   > source code mean no match. It never applies to inferred citations, `T` labels, prose of the
   > page, uncited pages, multi-token spans, or spans with `=`, arguments or spaces; those keep
   > the whole-element rule. `joined_code`, `_ends_inside` and every other fail-closed rule run
   > first. **Limits:** the rule backs the NAME only. What the sentence says about it ("`x()`
   > deletes the database") is verifier-dependent like any prose (section 4.1). A name may
   > match a same-named identifier in an unrelated part of the cited chunk's code, including
   > one that appears only in a comment or a string literal of it (`// never call
   > dropAllTables()` backs the name `dropAllTables()`; whether the sentence about it is right
   > is the verifier's). Every flagged claim also carries a reason: `inexact_code` (muted on
   > clients) only when the labelled unit's code is unbacked but each of its code elements is a
   > contiguous, unaltered excerpt of ONE code element of a cited `S` source (same tokens, same
   > punctuation, same order, gaps where the source has whitespace; it starts and ends at a
   > whitespace boundary of the source and does not stop on an open argument such as `rm -rf /`
   > or `--topic`), every identifier-shaped name in it is a whole token there, and the
   > verifier's objections could all be placed in the answer. Everything else is `not_found`:
   > typos, changed values (even when the new value appears elsewhere in the page's code),
   > reordered or truncated commands, dropped lines, verifier claims, residue and uncited
   > units; not_found wins by positional overlap with a verifier claim. Reasons never change
   > the status: an `inexact_code` claim still makes the answer `partial`. Limit: an excerpt
   > that ends at a whitespace boundary can still drop meaning that follows it (a trailing
   > `--force` left off); the muted flag still marks it as not verified.

3. Update the ADR's status line / amendment history and NEXT.md item 1(a)/(b) as done-pending
   blind review (this is a guarantee change: the owner asked for a blind review).

## What the integrator must test live
- Ask "explain Memento" (the owner's case). Expected: `createMemento()` and
  `ServiceConfiguration` are not flagged. If the answer is otherwise clean, the status is
  `verified`.
- An answer that excerpts part of a command from a page (e.g. a shortened `kafka-topics ...`
  line): `partial`, with `claims[0].reason == "inexact_code"` (muted in the UI).
- The `processing.guarantee` question (dev runs C4/C5): if the model writes `exactly_one_v2`,
  the claim must be RED (`not_found`).
- `curl` the SSE stream and check that `verification.claims[i].text ==
  verification.unsupported_claims[i]`.
- Resume an approval turn started before this build (old checkpoint): no crash, and the claims
  are `not_found`.
- Check the api log: one `answer_settled` line per turn with the prompt tag, model, chunk_ids and
  latency, and no note text.

## Known gaps / notes for the reviewer
- A unit citing two S pages whose code spans are backed by DIFFERENT pages is still unbacked (as
  before). Its reason is `not_found`, because `_tokens_in_source` needs one source to hold every
  token. This fails toward the prominent flag.
- `grounding.py` (not mine) still has no log line of its own. The eval line is emitted at its
  only call site (`nodes.ground_check`). Other model calls (generate, strict, extras) are outside
  this module.
- Round 1 P2 "name only in a comment": chose the documented-limit option (comment/string
  stripping per language is a parser of its own, and fragile); the limit is in the ADR paragraph
  and `test_a_name_only_in_a_source_comment_backs_the_name_not_the_claim` shows the verdict
  decides.
- The planted-error tests prove that the verdict, not the token rule, decides claims about a
  backed name. With a lenient verifier such a claim reaches `verified` (the documented residual).
