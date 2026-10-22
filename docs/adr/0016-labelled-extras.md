# ADR-0016: Interview-prep answer shape and labelled "Not from your notes" extras

**Status:** Accepted · 2026-10-18 · Builds on [ADR-0015](0015-retrieve-then-answer.md) (the verification guarantee is unchanged)

## Context
The owner uses RAGForge to prepare for interviews. After Phase 2b the answers were correct but bare. The owner asked for:
- a plain-language explanation first;
- then the technical one;
- then one real production incident from a large company (what broke, why, the fix), told crisply rather than as a story;
- code on request.

Two constraints pull against each other:
- **ADR-0015's guarantee:** a `verified` answer contains only text backed by the owner's notes.
- **The owner's notes don't always have** an incident or code for a topic. (Some topics do have incident-style pages.)

## Decision
1. **Answer shape (`answer@10`)**, all of it from the notes and verified as before:
   - **In simple terms:** 2–3 sentences;
   - **Technically:** at most 5 sentences;
   - **In production:** at most 3 sentences, and only when a source describes a real incident. Otherwise the section is left out and never invented.
   - Each section name is bold and starts its first cited sentence, so the names add no uncited unit and the allowlist is unchanged.
   - Code appears only when asked for, copied whole from a source.
2. **Extras, outside verification.** When the answer has no notes-backed "In production" part, or the user asked for code and the answer has none, one more model call (`extras_incident@5` / `extras_code@2`) adds general-knowledge content. The question and the answer already shown go to it as fenced DATA in the user message, never into its system prompt; seeing the answer keeps the extra consistent with it:
   - **when:** it runs after `finalize` (verification and the committed answer are final by then);
   - **how it travels:** as its own `TurnExtra` → SSE `extra` event. The agent node only EXTRACTS structure and never rewrites characters: an incident is the first paragraph as one line, without control or bidi characters or `[S<n>]` labels; code is the first fenced block as CommonMark parses it (markdown-it, the verifier's parser), with its language tag and the first line of prose after it. The API layer then renders extras as **literal text** (`sanitize.literal`: every ASCII punctuation escaped, URLs defanged as `https[:]//` / `www[.]`), so no emphasis, heading, label or link can render in any Markdown client. Code goes in one fence longer than any backtick run inside it (`sanitize.fenced`), and identifiers like `max_poll_interval_ms` survive exactly. (An r3 fix that stripped every `_`/`*` corrupted identifiers. The final review caught it, and it was replaced by this split.) A URL inside code stays as code text, and a terminal may still autolink it;
   - **how clients show it:** apart from the answer, under "Not from your notes". `make ask` holds it back and prints it LAST, after the citations and verdict. Every line inside is prefixed "│ " and hard-wrapped to the terminal's DISPLAY width (wide CJK and emoji count as 2, tabs are expanded), and the block is closed by "── end (general knowledge, not verified) ──". Text inside an extra can't pass for the footer;
   - **what it never does:** it is never verified, cited or committed to `messages`, so later turns never treat it as the user's notes;
   - **when it doesn't run:** abstains, refusals ("Your notes don't cover this."), answers citing nothing, and actions get none. Code is made only for an explicit code request: a code/snippet noun after show/give/write/share/see (not "status codes", "code of conduct" and the like), "example code", "with code", "code it up", "write … implementation", or implement/write … in <language> ("Go" only capitalised). "Give me an example of X" is a concept question, and the trigger table in the tests pins both directions. A code extra is also made after a refusal when the user explicitly asked for code (answer@10 refuses rather than write code; this is the case the extra exists for). Any failure of the extras call is logged (`extras_failed`) and never fails the turn, except a lost run lease, which stops it like any node. A failure in the graph machinery around the node after `finalize` (a lease check or a checkpoint write) can still surface an error event AFTER the committed answer was shown: the answer is not lost. The graph routes to the node only when an extra will be made, so other turns pay nothing.
   - **Extras are told never to name a company** (`extras_incident@3`, kept through `@5`; `@5` adds a `NONE` reply for non-engineering topics, which the server drops). They describe the common failure pattern and open with "Common failure pattern:". Real MNC incidents come only from the owner's notes, verified. A live test with `@1`/`@2` ("name a company only if publicly attributed") produced an invented company outage. This is a prompt rule, not a server check: an extra that names a company anyway is still shown, labelled as unverified general knowledge. The server keeps only the first paragraph, requires the opener (and text after it), and cuts at 700 characters at a sentence or word. Code extras are capped at 40 lines.
3. **Code as a follow-up.** `make ask` ends a cited answer with `→ ask "show the code"`. The offer is printed by the client, not written by the model, so it is no unverified text.
4. **Switches:**
   - `agent_extras` (env `RAGFORGE_AGENT_EXTRAS`), default **true** in the app;
   - `AgentSettings.extras` defaults to false, so library users and tests opt in.

## Amendment 2026-10-19: general-knowledge interview mode (Phase 3c)
- **What it is:** a second surface outside verification, following the same rule as extras. `source=general` on `/v1/interview/question` and `/grade` makes model general-knowledge questions and grades.
- **When it runs:** only when the user opts in with a click. Clients offer it only after 409 `interview-no-notes` / `interview-exhausted`, never in a mixed session, and the server never chooses it.
- **What it never does:** it is never verified or cited and reads no notes. MCP does not expose it, because an AI caller would pick it on its own.
- **How it is labelled:** "General knowledge · not verified · graded by the model against its own answer key".
- **Binding:** the signed question token is required, and it binds the question, the key, the level and the source.
- **Scores:** kept apart from notes-based scores in the session summary.
- **Abuse cost:** 1–2 model calls per question, under the same per-principal rate limit as notes mode.

## Options considered
- *Let the model mix general knowledge into the answer.* Rejected: it breaks the guarantee, and the reader couldn't tell notes from model.
- *A "Not from your notes" section inside the answer, allowlisted by its heading.* Rejected: the model controls where the heading goes, so claims could be routed around verification. A separate event controlled by the server can't be.
- *Notes only, no extras.* Simplest, but it leaves the incident section empty for most topics, which was the owner's main ask.

## Consequences
- **The conversation lease is held while the extra is generated** (it runs inside the turn). A client that re-enables input when `verification` arrives would get `TurnInProgress` for the 1–2 s the extra takes, so clients must wait for `done` before sending the next message (DESIGN SSE contract). Releasing the lease before extras is possible, since extras writes no state that matters, but it is not done yet.
- One extra model call on most question turns. TTFT is unchanged (the extra comes after the answer). Total time, which the eval measures up to `done` and therefore including the extra, and free-tier request use both grow. Evals record extras (`Outcome.extras`) so their content can be sampled and labelled; how often they name invented incidents is not yet measured.
- An extra can be wrong. It is labelled as unverified general knowledge, and that is the honest limit.
- Eval grading reads only the answer text (`TokenEvent`s), so extras don't affect correct% or verified%.

## Tests
- `module_ag/test_ag_extras.py`: when an extra runs, its order after verification, that it is not committed, the code trigger (a positive/negative table), the abstain/refusal/action exclusions, structure-only extraction (a one-line incident with labels removed and NONE/empty dropped; code from the first CommonMark fence, kept verbatim and capped) and literal rendering at the API, failure isolation, lease loss, usage in TurnDone, the data fences, and a real second turn.
- `module_e2/test_e2_chat_sse.py`: its own event, and no source URLs allowed.
- `test_client_ask.py`: it is printed apart and labelled, and the code offer appears.
