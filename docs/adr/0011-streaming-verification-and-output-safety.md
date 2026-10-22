# ADR-0011: Streaming with post-hoc verification; incremental output sanitization

**Status:** Accepted · 2026-10-16 · Revised in rounds 3, 4 and 5

## Context
Clients render tokens while streaming, and retrieved content is untrusted. Exfiltration vectors:
- inline images, reference-style images/links, autolinks;
- **bare URLs, which GFM renderers autolink**;
- raw HTML;
- padding past a hold-back limit.

Also, a naive "hold on backtick" corrupts long code blocks.

## Decision
1. **Incremental sanitizer** (server side, before any `token` event, for every client). It's a state machine with these states:
   - **Text:** passes through, except the triggers below.
   - **Fenced code:** entered **only for an unambiguous top-level CommonMark fence**:
     - the sanitizer **tracks open containers** (list items, including their continuation indentation, and blockquotes). Fenced state is entered only at **column 0 with no open container**. A fence-looking line inside or after a list item or blockquote stays in Text state;
     - ≥ 3 backticks or tildes;
     - for backtick fences, **no backtick in the info string**.

     It closes only on a line with ≤ 3 spaces of indent, the same character, a run length ≥ the opener's, and **no info string** (trailing whitespace only). Inside, content passes through unheld. **Anything ambiguous stays in Text state** (fail safe: treated as text, sanitized).
   - **Inline code** (single backtick): held until the closing backtick or end of line; then emitted as code, or escaped if unclosed.
   - **Construct** (`!`, `[`, `<`, or a line starting with `[`): held until resolved.
     - **Overflow** past 512 chars → flushed as **inert, markdown-escaped** text that can never complete later.
   - **Bare URL** (`scheme://` or `www.` in text): held until a delimiter.
     - If its normalized form exactly matches an allow-listed hit URL, emit that **canonical stored URL** as a link.
     - Otherwise emit it **defanged inside inline code**: backticks are stripped from the URL, the delimiter run is chosen longer than any backtick run inside it, and dots are replaced (`example[.]com`). It can't be autolinked or fetched, and it can't break out of the code span.
2. **Link and image rules:**
   - Remote inline images are dropped (`[image removed]`).
   - A link survives only if its normalized URL (scheme, lower-cased host, non-default port, path, **query**; credentials stripped, fragment ignored) **exactly equals** a hit's normalized `external_url`. The query is part of the key so that two hits differing only by query (`/view?id=1` vs `?id=2`) cannot link to each other (amended in Phase 2, handoff E1). A model fragment is kept only if it matches `[A-Za-z0-9_-]{1,128}` and the stored URL has none. The emitted href is the **stored canonical URL**, never the model's text.
   - Reference definitions (`[x]: url`) are removed; reference-style images and links become plain text.
   - Raw HTML is escaped. `javascript:` and `data:` are dropped.
3. Clients render pending tokens as plain text until `verification` arrives (defense in depth).
4. **Deterministic citation check:** cited anchors must be in `hits`; strict mode requires one per sentence.
5. **Ground check:** one LLM call with JSON-schema output; malformed after a repair retry → `partial`. Its own precision and recall are measured on the verifier eval set ([ADR-0012](0012-evaluation-strategy.md)). Models below the SLO show "unverified".
6. On `unsupported` → one strict regeneration → `answer.replace`; a second failure → abstain.
7. Sources are fenced as untrusted; destructive tools need HITL with a UI-only scope; memory writes are quote-gated.

## Tests
- **Differential, grammar-based fuzzing:** generated inputs, not just random splits of a fixed corpus.
  - The generator covers container nesting (blockquote/list + fences), fence length and character mismatches, info strings with backticks, padded overflow, reference/inline/autolink/bare URLs and HTML.
  - Inputs are split at random token boundaries, sanitized, and the output is rendered by a GFM library.
- The invariant: **no emitted event, and no concatenation of them, renders (GFM) into an external fetch or a non-allow-listed link.** This is checked by rendering with a GFM library and inspecting the AST.
- A separate test: a 5k-char fenced code block round-trips byte-identical.

## Consequences
TTFT is preserved; no mid-stream exfiltration; code blocks are intact; bare non-cited URLs appear defanged.
