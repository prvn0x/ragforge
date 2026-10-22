# Handoff IV-UI: interview mode in the clients + flag reasons everywhere

Built against the LOCKED contracts (`api.py` "flagged claims" / "interview mode (Phase 3b)",
`sse.py` `VerificationEvent.claims`) and the SDK methods `interview_question` /
`interview_grade`. No server code was touched; every test fakes the API.

## What was built

### `packages/client/src/ragforge_client/interview.py` (pure, no I/O, shared by UI and terminal)
- `detect_start(message) -> Start | None`: deterministic phrases only (owner decision):
  "interview me [on|about X]", "quiz me [on X]", "ask me [some|a few|N] [interview] questions
  [on|about X]", `/interview [X]`; polite prefixes ("please", "can you"), trailing punctuation
  (including the full-width U+FF1F) ignored. X may be a LIST ("Kafka, Redis", "Kafka and Redis",
  "Kafka & Redis"); a fully quoted X is one topic; "anything"/"all topics" = mixed. At most 8
  topics, each clipped to 200 chars.
- `route(message, pending=...)`: precedence (design review, binding) session command (whole
  message) > `/answer [text]` > start phrase (restart with new topics, same session) > leave rule
  > answer. Leave rule: ends in "?" (or U+FF1F) AND first word is an interrogative opener
  (what/why/how/when/where/which/who/whom/whose/explain/can/could/should/would/will/is/are/was/
  were/does/do/did) or starts with "tell me". Hedged answers ("Is it because the ISR shrinks…?")
  match and SUSPEND (pinned in the table test); `/answer` or the UI's "↩ That was my answer —
  grade it" grades that very message. With no question pending, free text leaves too.
- `route_suspended(message)`: only `/answer`, `/resume`, `resume interview`, `/stop` and start
  phrases touch a suspended interview; "stop"/"again" typed there are ordinary chat.
- `InterviewSession`: topics (alternated one per request), difficulty (easier/standard/harder;
  auto-steps up at 9+, down at 3-), kind rotation concept → scenario → design → code (all kinds
  sent, the due one FIRST; the rotation follows the kind the server actually asked),
  `asked_chunk_ids` → `exclude_chunk_ids` (dedup, cap 500, OLDEST dropped when full),
  `question_token` echoed into the grade request, history → `summary()` (asked/answered/skipped,
  average, weak topics ≤ 6 with deduplicated pages + urls, exhausted topics).
  "next topic": list → next topic; one fixed topic → cleared to MIXED (documented: the client
  has no other topic to pick); already mixed → just moves on.
  `exhausted()`: drops the exhausted topic and continues; ends when none is left (or mixed).
- `is_exhausted(problem)`: `problem.type == INTERVIEW_EXHAUSTED`, never the status alone.
- `apply(route) -> Step(do=ask|grade|show|end|chat|nothing)`: the one state machine both
  clients drive.

### `ask.py` (`make ask`)
- Flag reasons in the footer: red "Not found in your notes:" (`!`) then dim "Couldn't match
  exactly (code/format, usually fine):" (`~`); off a TTY the headings tell them apart. Old server
  (no `claims`) and unknown reasons → not_found. `split_claims` / `FlaggedClaims` are the shared
  splitter (UI and MCP use them too).
- REPL interview: same phrases start it; question card `── Question n · kind · level · topic ──`
  (no sources before answering); next input graded (grade card with labelled lists, model answer,
  citations, verdict, split flags), then the next question is fetched; commands work; a normal
  question pauses (Q&A answers it) with `/answer` / `/resume` / `/stop`; "stop" prints the
  summary. Errors (API down, 401, 429, 404 = no interview endpoint, other) are one friendly line
  and the session is kept. Q&A behaviour outside the mode is unchanged.

### UI (`packages/ui/src/ragforge_ui/`)
- `interview.py` (new, no Streamlit): `InterviewUi` state + `advance(ui, route, backend, …)`
  which drives the session and appends cards (question / answer / grade / notice / error /
  summary). Cards carry `position` (number of chat turns when made) so the page interleaves
  them with Q&A turns in the order they happened.
- `interview_views.py` (new): cards, controls, start form, paused bar, and `draw_flags` (red
  `st.error` list "Not found in your notes"; muted collapsed `st.expander` "Couldn't match
  exactly (code/format)"), used by chat answers AND grade cards.
- `views.py`: sidebar "Interview mode" toggle (mirrors the state; on → topic box + Start; off →
  stop with summary), phrase detection in the normal chat box (same `detect_start`), typed
  messages routed by the shared rules, buttons queued like chat (busy flag, no races).
  Question card: kind badge, level, topic, question; sources hidden until graded. Grade card:
  big score (`st.metric`), summary + "Right" labelled "the grader's judgement", "Missed" /
  "Misconceptions" labelled "checked against your notes", model answer (Markdown), verdict badge,
  split flags, sources, "This question came from". Buttons: Skip / Next question, Harder,
  Easier, Next topic, Stop; "Grade again" after a failed grade; paused bar with "↩ That was my
  answer — grade it", "Back to the question", "End interview". "New conversation" also clears
  the interview (a clean page; no summary).
- `api.py`: `Backend.interview_question/interview_grade`; `SdkBackend` runs them with a 120 s
  timeout (`INTERVIEW_TIMEOUT_S`; a grade is several model calls server-side).
- `state.py`: `Turn.claims` stores `VerificationEvent.claims`.
- `render.py`: `bullet_markdown`, `kind_label` (unknown kind → escaped text), `level_label`,
  `weak_topics_markdown` (escaped titles, safe links), `summary_line`, flag titles.

### MCP (`packages/mcp/src/ragforge_mcp/`)
- `AskResult.flagged_claims: list[{text, reason: "not_found" | "inexact_code"}]` (default `[]`,
  placed before `extras` so field order still puts notes-derived content first);
  `unsupported_claims` kept unchanged for compatibility. Unknown/missing reasons → `not_found`.
  Tool description updated.

## Round-1 review fixes (blind principal review, 8.2/10)
- P1 timeout: `INTERVIEW_TIMEOUT_S = 120.0` now lives in `ragforge_client.interview` (UI's
  `api.py` re-exports it). `make ask` opens a second client for the interview calls
  (`ask.interview_client`, `repl(..., interview_api=...)`) with that read timeout; Q&A keeps the
  SDK default. A read timeout (not a connect timeout) is now reported as "The server took longer
  than 120 s (it may still be working): wait a moment, then try again" in both clients, not
  "Cannot reach the API". Fake-transport tests pin the 120 s read timeout on both calls.
- P2 commands: the start phrases' polite frame now wraps commands too ("ok stop", "Stop it",
  "Next question please", "can you repeat that?", "stop the quiz"); the opposite direction is
  pinned (answers containing these words still grade).
- P2 escaping: server topics are unescaped once before display (`ask.plain_line`,
  `render.topic_markdown`) in the question header, the UI card and the summary.
- P2 terminal "answer kept": `InterviewSession.grade_failed(answer)` keeps a failed answer
  (`unsent`), so a bare `/answer` resends it, including a paused hedged message; skipping or a
  new question forgets it.
- P2 `split_claims`: a bare `unsupported_claims` text absent from `claims` is shown as
  not_found (only an explicit `inexact_code` mutes a claim).
- P2 no source selected: interview controls/start form are disabled with "Pick at least one
  source in the sidebar to use interview mode" (no silent drop).
- P2 UI vs terminal after a grade: documented (module docstrings): the terminal fetches the next
  question at once; the UI waits for "Next question" so the grade card can be read, and free
  text typed there goes to Q&A (suspending; "Back to the interview" resumes).
- P2 heading: "Model answer" (neutral) in both clients; the verdict badge/line says how well the
  notes back it.

## Integration steps
1. Nothing outside my paths is required for the code to run. No new dependencies.
2. Run `make check` (my scope passed: see below).
3. DESIGN.md section 10 / the UI and MCP docs could mention interview mode, `flagged_claims`
   and the split flags (I don't own docs; suggested text: the "What was built" bullets above).

## Contract notes (no change requested; for the integrator's awareness)
- `InterviewGradeRequest.kind` / `.difficulty` are strict enums while the question response is
  open. A question with a kind/difficulty this client doesn't know (newer server) is graded with
  the kind the client REQUESTED first and the session's difficulty; the `question_token` carries
  the server's own binding. If IV-CORE enforces the token against the echoed kind, it should
  trust the token's kind over the request's.
- `exclude_chunk_ids` cap 500 is respected by dropping the oldest ids.

## Checks run (all green)
- `uv run pytest packages/core/tests/module_ui packages/core/tests/module_mcp packages/core/tests/test_client_ask.py packages/core/tests/test_client_interview.py -q` → 499 passed
  (also `test_client_evalrun.py`, which imports ask.py: 30 passed).
- `uv run ruff check` / `ruff format --check` on my paths: clean.
- `uv run mypy packages/client/src packages/mcp/src packages/ui/src`: clean; test files (root
  tests, and `MYPYPATH=<dir>` for module_ui / module_mcp): clean.
- `uv run lint-imports`: 6 kept, 0 broken.

## What the integrator must test LIVE (needs IV-CORE's endpoints)
1. `make ui`: toggle Interview mode → Start with "Kafka, Redis" → question card (no source
   titles) → answer → grade card (score, lists, model answer with Verified/Partly badge, sources
   revealed) → Next question asks Redis → Harder → Stop shows the summary with page links.
2. In the chat box: "interview me on Kafka" enters the mode; "what is ISR?" pauses it and is
   answered by Q&A; "↩ That was my answer — grade it" grades it; "where's the question?" while
   in the mode re-shows the question (no search).
3. A topic with few pages: keep skipping until 409 INTERVIEW_EXHAUSTED → "No new questions are
   left on …" and, when all topics are done, the summary.
4. A chat answer with an `inexact_code` claim (e.g. the Memento `createMemento()` case): it must
   appear only in the collapsed muted expander, not in red.
5. `make ask`: same flow in the terminal (`quiz me on Kafka`, answer, `harder`, a normal
   question, `/answer`, `stop`).
6. MCP `ask`: `flagged_claims` present with reasons; `unsupported_claims` unchanged.
7. Grade latency vs the shared 120 s interview timeout (UI and `make ask`) and the server's real
   timings. If real grades approach 120 s, raise `INTERVIEW_TIMEOUT_S` in
   `ragforge_client/interview.py` (one place).

## Eval-readiness
Clients make no model calls, so there is nothing to log client-side. The decision steps an
eval harness may want offline are pure: `detect_start`, `route`, `route_suspended`,
`InterviewSession.question_request/grade_request/apply/summary`, `split_claims`, and the UI's
`advance` (takes any object with the two sync interview methods).
