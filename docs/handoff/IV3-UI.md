# Handoff IV3-UI: interview 3c in the clients (levels, opt-in general fallback, labels, summary split, start phrases, refusal badge)

Built against the LOCKED contracts (`api.py` interview section: `InterviewLevel`,
`InterviewSource`, `level`/`source`/`exclude_questions`, `INTERVIEW_NO_NOTES`, the grade
`chunk_ids` rule) and the SDK's `interview_question` / `interview_grade`. No contract, SDK,
server, pyproject or docs file was touched (except this handoff). Every test fakes the API.

## What was built

### Shared rules: `packages/client/src/ragforge_client/interview.py` (pure, no I/O)
- **Levels.** `parse_level(text)`: ONE deterministic table. It accepts "Staff", "staff engineer",
  "SDE1", "sde-2", "SDE II", "software engineer 2", "senior" (= Senior SDE-1), "Sr. SDE-1",
  "senior sde ii", the enum values, an article before it and "level" after it. "sde" alone, a
  bare number, "principal" and "junior" are rejected: no guessing.
  - `detect_start` takes off a trailing level clause: "... at [the] X [level]", "... as a X",
    "... for X", `--level X` / `--level=X`. It also takes a level BEFORE the topic ("interview
    me as a staff engineer on Kafka").
  - A clause counts only when ALL of X is a level, so "data at rest" and "Kafka as a log" stay
    whole topics. The leftmost level clause wins.
  - An unknown `--level` value is kept as `Start.unknown_level`, and the clients print "Unknown
    level ...: using <level>. Levels: ...". So is a trailing clause that LOOKS like a level the
    table doesn't know (`_LEVEL_LIKE`: a level-ish head word such as principal / junior / L5 /
    sde, or one or two words then "level", never "the ... level" and never across "on/about"):
    "Kafka at principal level" gives topic "Kafka" plus the notice, while "caching at the
    application level" stays a topic (review round 1).
  - A trailing "please" (", please.") is taken off BEFORE the level clause is looked for, so
    "interview me on Kafka at staff level please" is Staff on "Kafka" (review round 1).
  - `LEVEL_NAMES` / `level_name`; `question_line(q)` gives "Staff" or "Staff · harder".
    Difficulty is now a step WITHIN the level: `difficulty_label(standard) == "standard"`, and
    the notice reads "difficulty: harder (within Staff)".
- **The level is fixed per session.** `InterviewSession.level` defaults to `DEFAULT_LEVEL`
  (Senior SDE-1) and is ALWAYS sent on the question request.
  - The grade request echoes the QUESTION's `level`, `source` and `question_token`. If the level
    is unknown (a newer server), it falls back to the session level.
  - A start naming ANOTHER level returns `Do.restart` with "Level changed to X: that starts a new
    interview ...". The pending question is recorded as skipped (as on stop), and the client
    shows this session's summary, then begins a new session from the same `Start`.
  - A start with the same level, or no level, is a topic switch in the same session.
- **Opt-in general fallback.**
  - `fallback_reason(problem)` reads the Problem TYPE: `INTERVIEW_NO_NOTES` gives no_notes,
    `INTERVIEW_EXHAUSTED` gives exhausted. Never the status.
  - `exhausted(problem)` drops the topic and opens an `Offer(topic, reason)`, with the notice
    'Your notes have nothing on "X".' or 'All your "X" notes are asked.' It never switches.
  - When no topic is left but offers are open, the session is `waiting`: "next" asks nothing
    and says so (`WAITING_NOTICE`).
  - A mixed session is never offered the fallback and ends as before. Going mixed ("next topic"
    on a single topic) drops open offers.
  - `General(topic|None)` routes come from a UI button, `/general [topic]`, or "y"/"yes". "y"
    counts only while `session.offering` = the session WAITS on its offers (no topic left, no
    question pending), the same rule the terminal's offer line prints. While other topics
    remain, an offer is taken by its button or `/general <topic>`: the UI idles after every
    grade, so the old "offers open and nothing pending" rule let a stray "y" typed questions
    later take an old offer (review round 1).
  - A general question that arrived WITHOUT its `question_token` (a server bug) makes
    `grade_request` raise a plain ValueError ("this general-knowledge question came without its
    server token, so it can't be graded; skip it"), checked first, instead of a pydantic
    validation dump (review round 1).
  - `accept_general` moves an OFFERED topic (never one the notes still cover) back into the
    rotation as the next topic, marked general. A pending notes question is skipped for it.
  - General requests send `source=general` plus `exclude_questions`: the session's earlier
    general questions FOR THAT TOPIC, newest 20, each clipped to 1000 characters (the contract
    caps).
  - A general grade request sends `chunk_ids=[]` (the contract rule), the token and the key.
  - "next topic" alternates notes and general topics; every question keeps its own label.
  - A start phrase clears offers and general topics, so general is always a fresh opt-in.
  - A general topic that 409s is dropped without a new offer.
- **Labels.** One wording, shared by both clients:
  - `GENERAL_LABEL` = "General knowledge · not verified";
  - `GENERAL_NOTE`;
  - `GENERAL_GRADE_LABEL` = the design-review text;
  - `NO_KEY_NOTE`;
  - `offer_label(topic)` = 'Interview me on "X" from general knowledge (not from your notes, not
    verified)'. The topic is named so a topic list's offers can be told apart.
- **Source safety.**
  - `is_general(x)`: anything but an explicit `notes` source counts as general, so an unknown
    source is never shown as notes.
  - `general_grade(grade, question)`: a grade is drawn as general if the grade OR its question
    says so. A general question's grade can never be drawn as notes, whatever the server labels
    it.
- **Summary split.**
  - `Summary.average` is now NOTES-ONLY. The new fields are `general_average`, `notes_answered`,
    `general_answered`, `weak_general` (`WeakGeneral(topic, kind, score)`, keyed by the SESSION
    topic the owner typed), `level` and `topics`.
  - `summary_headline` gives "Staff · Kafka · avg 6.5". With both sources: "... · avg 6.5 from
    your notes · general knowledge avg 4.0 (not verified)". The two are never averaged together.
- **Start phrases (live failure).** New phrases:
  - "interview mode", "INTERVIEW MODE", "interview mode on", "interview mode [on|about|for] X";
  - "start|begin|enter [an|the] interview [mode] [on X]".
  - Every existing negative still holds. New negatives: "what is interview mode?", "interview
    mode is broken", "start the interviewer", "start interview prep notes", "interview me as a
    friend", "explain Kafka as a senior SDE 2 would".

### `make ask`: `packages/client/src/ragforge_client/ask.py`
- **Refusal badge.** `nothing_to_check(status, *, cited, flagged)` and `NOTHING_TO_CHECK` =
  "Nothing to check: no answer came from your notes."
  - The footer shows "(nothing to check: no answer came from your notes · ...)" only for an
    UNVERIFIED turn with NO citations AND NO flagged claims (a pure refusal). A cited
    unverified turn, or an uncited one with flagged claims, still prints "unverified".
  - Why `flagged` (review round 1): under an untrusted verifier the server also sends
    `unverified` with no citations for a SUBSTANTIVE answer (chat finalize rule 7, attribution
    settle rule 4a, interview verify with no ground check): inferred citations are dropped and
    every sentence is flagged not_found. That answer made claims, so "below the quality bar" is
    the true reason there. A pure refusal is the only case with no flagged claims.
  - A notes grade's verdict line follows the same rule.
- **Question card.** It shows the level (plus the difficulty when not standard). A general
  question prints inside the ADR-0016 extras box (`boxed(...)`, refactored out of
  `format_footer`): "── General knowledge · not verified ──", then "│ "-prefixed lines, then
  "── end (general knowledge, not verified) ──".
- **Grade card.**
  - A notes grade adds "Answer key (checked against your notes):" (the key is no longer a hint
    once the answer is in) or `NO_KEY_NOTE`.
  - A general grade prints entirely in the box under `GENERAL_GRADE_LABEL`, with no verdict,
    citations or flags. Missed and misconceptions are labelled "against the model's own key".
- **Offers.** A one-line offer: 'Type "/general" or "y"' while the session waits on its offers,
  else 'Type "/general X"'.
- **Session.** The start line names the level ("fixed for this interview"). A restart prints the
  summary, then the new session.
- **Summary.** The split summary: weak notes topics with their pages to revisit, then weak
  general topics with "consider adding notes on X".

### UI: `packages/ui/src/ragforge_ui/`
- **Start form.** A Level dropdown next to the topic box (SDE-1, SDE-2, Senior SDE-1 (default),
  Senior SDE-2, Staff).
- **Running session.** A Level dropdown that mirrors the session, with the caption "fixed for
  this interview: changing it starts a new one". A change queues
  `Start(named_topics, level=new)`, which shows the old summary card, then the new session.
- **Offer buttons.** One button per open offer (per topic). Only that click (a `General` step)
  switches. Typing "y" in the chat box takes the offer only while the session waits on its
  offers (no topic left); otherwise "y" is an answer (question pending) or a normal message.
- **Question card.** It shows "Level: <level>[ · difficulty]".
- **General cards.**
  - Each card opens with **General knowledge · not verified** and a caption, under its own
    avatar (`:material/public:`), with a gray kind badge. This is the extras look.
  - A general grade card has `GENERAL_GRADE_LABEL`, lists labelled "against the model's own
    key", "The model's answer key · not verified", and "**Model answer** · *General knowledge ·
    not verified*". It has NO verdict badge, flags or source chips.
- **Notes grade card.** A new "Answer key · checked against your notes" list, or the
  `NO_KEY_NOTE` caption when the key is empty.
- **Summary card.**
  - The bold headline ("Staff · Kafka · avg 6.5") and the counts.
  - "Weak topics from your notes" with page links.
  - A separate bordered **General knowledge · not verified** block with "consider adding notes
    on X".
- **Refusal badge.** `render.badge_for(status, cited=, flagged=)`: an unverified, uncited AND
  unflagged chat turn (or notes grade) gets the "Nothing to check" badge with "No answer came
  from your notes.". A cited unverified turn, or an uncited one with flagged claims
  (`turn.claims or turn.unsupported_claims`; `grade.unsupported_claims`), keeps "Not checked"
  and the quality-bar text.

### MCP: `packages/mcp/src/ragforge_mcp/turn.py`
- `_verdict_warning`: an unverified turn with no citations and no unsupported claims returns
  `NOTHING_TO_CHECK_WARNING` = "Nothing to check: no answer came from the user's notes.". A
  cited unverified turn, or an uncited one with unsupported claims, keeps "verifier off for
  this model".
- MCP has NO interview tools today, so nothing exposes `source=general` (design review: never
  in MCP). `level` was not added either, because there is no interview tool to add it to.

## Tests (all fakes, no network)
- `test_client_interview.py` (new 3c section):
  - the level table (positives and negatives);
  - start phrases with levels;
  - new start phrases and negatives;
  - level sent and echoed; the default; unknown level fallback; restart vs switch; unknown
    `--level` mid-session; difficulty within the level;
  - both 409 types open an offer and never switch; no_notes is not "fully covered"; other
    problems are not fallbacks; mixed never offers;
  - accepting sends general with `exclude_questions` per topic; caps and clipping; a general
    grade sends no chunks and echoes the token, level and source; an unknown source is never
    notes; an empty key still grades;
  - "y" only when nothing is pending; `/general` with no open offer is refused;
  - next topic moves between notes and general; a start clears offers; going mixed drops
    offers; an exhausted general topic is dropped without a new offer;
  - the summary split and the headlines.
- `test_client_ask.py` (new 3c section):
  - footer verdict both ways (table);
  - the REPL with a level phrase; another level gives summary then a new session; unknown
    level;
  - the fallback on both 409s with "y", the boxed general cards and the split summary;
    declining never switches; no offer when mixed; "y" is an answer while a question is
    pending;
  - answer key or `NO_KEY_NOTE`; "nothing to check" on a grade; no label on notes cards; a
    mislabelled grade for a general question is still boxed.
- `module_ui/test_iv3_ui_app.py` (AppTest):
  - level dropdown default and choice, sent and shown; mid-session change gives a new session
    and the summary; level phrase;
  - fallback on both 409 types, ONLY after the click (request sources asserted before and after;
    "Next question" while waiting asks nothing); labels on question, grade and model answer; no
    verifier texts or sources on general cards; `exclude_questions` sent on the next general
    question;
  - one offer per topic, no label on notes cards; mixed never offered;
  - the summary split; the new start phrases;
  - refusal badge both ways; empty answer key;
  - a mislabelled grade is never drawn as notes; "y" in the chat box.
- `module_ui/test_iv3_ui_controller.py`: `badge_for` both ways; `advance` restart card order and
  the old summary's level; unknown level notice; a stale `General`; waiting then click; escaping
  in level, headline and weak-general helpers.
- `module_mcp/test_iv3_mcp_verdict.py`: the MCP warning both ways, and partial is unaffected.
- Review round 1 additions: unverified + uncited + flagged keeps the "not checked" text in all
  three clients (footer table, grade verdict line, `badge_for`, UI AppTest, MCP); "please" and
  unknown natural-language levels in the level phrase table (plus a terminal notice test); "y"
  never takes an old offer while topics remain (session + UI AppTest) and does while waiting;
  a tokenless general question explains itself and can be skipped (session + UI AppTest).
- Updated pinned assertions in my own tests, each a deliberate 3c behaviour change:
  - the difficulty notice and label;
  - the terminal header with level (unknown difficulty now follows the level);
  - the summary counts line;
  - exhaustion now OFFERS instead of ending (UI app, controller, terminal, session tests);
  - `test_mcp_ask.py::test_unverified_and_missing_verdicts_are_flagged`: its first stream now
    carries a citation, so it still tests the "not checked" text; the uncited case is the new
    test.

## Checks run (all green)
- `uv run pytest packages/core/tests/module_ui packages/core/tests/module_mcp packages/core/tests/test_client_ask.py packages/core/tests/test_client_interview.py packages/core/tests/test_client_evalrun.py -q`: 698 passed (after review round 1).
- `uv run ruff check` / `ruff format --check` on my paths: clean.
- `uv run mypy packages/client/src packages/ui/src packages/mcp/src`: clean. Test files:
  `MYPYPATH=<dir>` for module_ui and module_mcp, and the two root test files: clean.
- `uv run lint-imports`: 6 kept, 0 broken.

## Integration steps
1. Nothing outside my paths is needed for the code to run. No new dependencies.
2. Run `make check` (it includes the other two modules).
3. Optional docs (I don't own docs): DESIGN.md section 10 and the UI/MCP docs could mention:
   - the level dropdown and phrases;
   - the opt-in general fallback and its label;
   - `Summary.average` is notes-only;
   - the "Nothing to check" badge.

## Contract notes (no change requested)
- `InterviewQuestion.level`/`source` default to `senior_sde_1`/`notes` when a server omits them.
  So a 3b server shows "Senior SDE-1" on a Staff session's questions (honest: it ignored the
  level). The grade echoes that.
- The contract caps `exclude_questions` at 20, and the server design review says the server uses
  the last 15. The client sends the newest 20 and the server trims.
- For `source=general` the contract requires `question_token`. If a server omits it, the
  client refuses to send the grade with "this general-knowledge question came without its
  server token, so it can't be graded; skip it" (UI "Answer not sent: Because ...", terminal
  "Not sent: ..."). The question stays pending until skipped.

## What the integrator must test LIVE (needs IV3-CORE's server)
1. `make ui`:
   - toggle on, set Level to Staff, Start on "Kafka". The question card shows "Level: Staff",
     and the grade is at Staff;
   - change the session Level to SDE-2: the summary card appears, then a new Question 1 at SDE-2.
2. A topic with no notes (e.g. "interview me on Quasars"):
   - "Your notes have nothing on ..." plus the offer button; nothing is asked until the click;
   - click: a general card under "General knowledge · not verified", with no source chips and
     no badge;
   - answer: the general grade card with the long label;
   - "Next question": no repeat (`exclude_questions`);
   - "Stop": a separate general block with "consider adding notes on Quasars".
3. A small topic (Memento): skip until `INTERVIEW_EXHAUSTED` gives "All your ... notes are
   asked." plus the offer. A topic list ("Kafka, Quasars") offers per topic while Kafka
   continues.
4. In the chat box, type "INTERVIEW MODE", "start an interview on Redis" and "interview me on
   Kafka at staff level": each enters the mode (no Q&A refusal).
5. A refusal ("what is a quasar?" with no notes) shows "Nothing to check / No answer came from
   your notes." and never the quality-bar text. A cited answer on an untrusted verifier still
   shows "Not checked", and so does a substantive answer whose inferred citations the server
   dropped (no chips, red "Not found in your notes" list): confirm the server sends a refusal
   with NO flagged claims, which the rule relies on.
6. A notes question with empty `expected_points` (IV3-CORE's weak-points path): the grade works,
   and the grade card says "No checked answer key for this question; graded against your notes."
7. `make ask`: `/interview kafka --level staff`; a no-notes topic gives the offer line, then "y"
   gives the boxed general question; `stop` shows the split summary.
8. MCP `ask` on a refusal: warning "Nothing to check: no answer came from the user's notes."

## Eval-readiness
The clients make no model calls, so nothing is logged client-side. The decision steps are pure
and offline-callable:
- `parse_level`, `detect_start`, `detect_general`, `route`, `route_suspended`;
- `fallback_reason`, `InterviewSession.question_request/grade_request/exhausted/accept_general/apply/summary`;
- `summary_headline`, `is_general`/`general_grade`;
- `nothing_to_check(status, cited=, flagged=)`, `badge_for`;
- the UI's `advance` (it takes any object with the two sync interview methods).

## Known gaps
- No live run: it depends on IV3-CORE's server.
- The level dropdown in a running session restarts on the ORIGINAL named topics (`named_topics`).
  If topics were dropped or cleared to mixed meanwhile, the restart asks the named ones again
  (documented in the code; deliberate: "a new interview on the same topics").
- While paused (a normal question suspended the interview), the offer buttons are hidden;
  `/general` typed in the chat box still works then.
