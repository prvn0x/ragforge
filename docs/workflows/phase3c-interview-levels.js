// Paths scrubbed for publishing: <REPO> is the checkout, <SCRATCH> a session scratch folder.
export const meta = {
  name: 'ragforge-interview-3c',
  description: 'Interview 3c (levels, general-knowledge fallback, acronym topics) + bare dotted-name false flag, parallel builders with blind review loops',
  phases: [
    { title: 'Build', detail: '3 builders: IV3-CORE, IV3-UI, DOT in owned paths' },
    { title: 'Review', detail: 'blind principal review per module' },
    { title: 'Fix', detail: 'fix P0/P1, then re-review (max 2 rounds)' },
  ],
}

const ROOT = '<REPO>'
const COMMON = `
You are a senior engineer building ONE module of RAGForge (multi-tenant RAG over the owner's Notion notes, used for INTERVIEW PREP) in a shared folder where 2 other engineers build other modules AT THE SAME TIME.
Repository: ${ROOT}  (NOT a git repo: never run git commands. Do not commit.)

READ FIRST: ${ROOT}/CLAUDE.md, ${ROOT}/docs/NEXT.md section "TO DO NEXT" item 1 (the owner's decisions: they are binding), ${ROOT}/docs/DESIGN.md (sections 1, 8 "SSE contract"), ${ROOT}/docs/adr/0015-retrieve-then-answer.md ("Current state"), ${ROOT}/docs/adr/0016-labelled-extras.md, the contracts ${ROOT}/packages/contracts/src/ragforge_contracts/api.py (see the NEW "flagged claims" and "interview mode (Phase 3b)" sections at the end) and sse.py (VerificationEvent.claims is NEW), the SDK ${ROOT}/packages/client/src/ragforge_client/client.py (interview_question / interview_grade are NEW), the domain ${ROOT}/packages/core/src/ragforge_core/domain/models.py (TurnVerification.claim_reasons is NEW).
These shared interfaces are LOCKED by the integrator: code against them, never change them. If one is truly insufficient, say so in your handoff file.
PHASE 3c: interview mode (Phase 3b) is SHIPPED and live; you extend it. The locked contracts gained (api.py, interview section): InterviewLevel, InterviewSource, level/source/exclude_questions on InterviewQuestionRequest, level/source on InterviewQuestion/InterviewGradeRequest/InterviewGrade, INTERVIEW_NO_NOTES, and InterviewGradeRequest's chunk_ids rule (notes: 1..10, general: none). Read docs/NEXT.md "Current state" (Phase 3b) for what exists and its open residuals.

HARD RULES:
1. Create/edit ONLY the paths you own (listed below). Never edit pyproject.toml, uv.lock, compose.yaml, Makefile, Dockerfile, .env, packages/contracts/**, packages/client/src/ragforge_client/client.py, docs/** (except your handoff file). Never run uv add / uv lock / uv sync.
2. Anything you need outside your paths goes in your handoff file ${ROOT}/docs/handoff/<MODULE_ID>.md. The integrator applies it.
3. Format/lint ONLY your paths: \`uv run ruff format <your paths>\`, \`uv run ruff check --fix <your paths>\`. Never run make fmt or ruff on all of packages/.
4. Tests go in files prefixed with your module id (e.g. test_iv_question.py) in your owned test dir; no __init__.py in test dirs. No real network or LLM: fake the chat model / API (existing tests show the fakes: look at packages/core/tests/module_ag and module_e2 conftest/fixtures; the DB-backed tests use the test Postgres from packages/core/tests/conftest.py the same way existing tests do). Do NOT run make up / make nuke / docker compose (the live stack is shared and in use). Do NOT touch the real .env.
5. Layering (import-linter, \`uv run lint-imports\`): keep it passing. ragforge_mcp / ragforge_ui / ragforge_client import only ragforge_client + ragforge_contracts, never ragforge_core.
6. Secrets: never print, log or put API keys in exceptions, URLs or reprs.
7. Text from the API is already-sanitized CommonMark; render as Markdown, never raw HTML.
8. Principal-engineer quality: typed (mypy --strict clean), small focused functions, docstrings explaining WHY for non-obvious decisions, prefer framework/library primitives over hand-rolled code, tests that PROVE behaviour (including failure paths), clean failure handling.

EVAL-READY (the eval layer is built next, by someone else, against your code): keep the decision steps as pure, injectable functions (retrieval/selection, prompt building, reply parsing, verification, reason mapping) that an offline eval harness can call with a fake or real chat model and fixed chunks, without HTTP. Every model call logs one structured line with: prompt name@version, model, usage, latency, the chunk ids given to the model, and the outcome (parsed / malformed / verdict). Never log note text, answers or keys.

SHARED SEAM (locked, DOT ships it): agent/coverage.py \`segment\`, \`residue\`, \`visible_runs\` and agent/attribution.py \`attribute\` (and whatever they call internally) gain a keyword-only parameter \`names: frozenset[str] = frozenset()\`: the set of dotted names (maximal [A-Za-z0-9_.] runs containing an interior dot, leading/trailing dots stripped) found verbatim in the SHOWN S sources (never T tool results). A helper \`dotted_names(texts: Iterable[str]) -> frozenset[str]\` in coverage.py builds it. Default empty = today's behaviour exactly. DOT owns these files; IV3-CORE passes \`names\` in its own callers (interview verify.py, lexically_backed); the integrator wires the chat path.
LOGGING: every interview_model_call line also carries level, kind, difficulty, source (no text).

DEFINITION OF DONE: your tests pass; \`uv run ruff check <your paths>\` and \`uv run ruff format --check <your paths>\` clean; \`uv run mypy\` clean on your source files and (MYPYPATH=<your test dir>) your test files; \`uv run lint-imports\` passes. Before finishing ALSO run the existing suites closest to what you touched (listed in your module) and make sure they still pass. Write your handoff file: what you built, exact integration steps, anything you needed outside your paths, and what the integrator must test live.
`

const MODULES = [
  {
    id: 'IV3-CORE', title: 'Interview 3c server: levels, general-knowledge fallback, acronym topics, no-notes signal, robust question replies',
    owns: 'packages/core/src/ragforge_core/agent/interview/**, packages/core/src/ragforge_core/apps/api/routes/interview.py, packages/core/src/ragforge_core/prompts/templates/interview_*.md (edit + new), packages/core/src/ragforge_core/config.py and bootstrap.py and apps/api/deps.py (ADDITIVE only), packages/core/tests/module_iv/**, docs/handoff/IV3-CORE.md',
    tests: 'uv run pytest packages/core/tests/module_iv packages/core/tests/module_e2 -q',
    spec: `Extend the Phase 3b interview server (owner decisions 2026-10-19, binding):
1. LEVELS: InterviewLevel sde_1 | sde_2 | senior_sde_1 | senior_sde_2 | staff. request.level None -> the configured default (senior_sde_1). The level changes BOTH what is asked and how it is graded: a per-level rubric in the prompts (generic, not tied to any company): SDE-1 = what/why/basic usage; SDE-2 = how it works inside, common failure modes; Senior SDE-1 = trade-offs, failure modes, the why, production judgement; Senior SDE-2 = cross-system design, operating at scale, migrations; Staff = architecture across teams, cost, risk, long-term evolution, when NOT to use it. Each level states what 3/6/8/10 look like. Difficulty (easier/standard/harder) stays a fine adjustment WITHIN the level. Bind level and source into question_token (bump the token version; old tokens -> 400 as before). Echo level/source in InterviewQuestion and InterviewGrade.
2. GENERAL-KNOWLEDGE FALLBACK (source=general), ONLY when the client asks for it (never chosen by the server): no retrieval, no chunks. New prompts interview_general_question@1 / interview_general_grade@1. The model writes ONE question on the topic at the level and kind, plus its own expected points (no labels). exclude_questions (client-sent, untrusted DATA, fenced) avoid repeats. Grading: an LLM judge against the question + its key (both from the client: fence as data, bind through the token); score/summary/right/missed/misconceptions/model_answer; citations []; verification "unverified" ALWAYS; unsupported_claims []. A topic is REQUIRED for source=general (400 otherwise). Same rate limit, same one-repair JSON rule, same structured log line plus source=general. Never reads notes in this mode (a test proves no DB note read happens). The model must not name companies in incidents (same rule as extras_incident).
3. ACRONYM TOPICS: "HLD" must find the page titled "High Level Design" (live failure). Generic rule, no hardcoded list: a topic token of 2-6 letters matches a title or a heading-path segment whose words' initials spell it (case-insensitive; ignore stop words like of/and/the; e.g. HLD, LLD, OOP, DSA, CAP). Do it under RLS + the same collection/active-space/live-document filters, with bound parameters (no regex/LIKE built from user text in SQL; doing the initials match in Python over the visible titles/heading paths is fine if bounded and indexed reasonably). Keep the existing FTS + hybrid search paths.
4. NO-NOTES vs EXHAUSTED: 409 with type INTERVIEW_NO_NOTES when nothing matches the topic at all (and nothing was excluded), INTERVIEW_EXHAUSTED when everything matching was asked. Detail text is plain; clients branch on type.
5. QUESTION REPLY ROBUSTNESS (live failure: "the model's QuestionReply did not match the schema twice" on Spring Boot notes): a reply whose question is fine but whose expected points are not lexically backed must NOT fail the request. Keep the repair attempt, then accept the question with only the backed points (possibly none); log the validation reason category per attempt (our validator's message, never note text) in interview_model_call, so evals can count it. Only an unusable question (schema-invalid, too short) after the repair is a 502.
DESIGN REVIEW DECISIONS (binding; override the above where they differ):
- Token v3 binds the RESOLVED level and the source (and topic for general). For source=general the contract now REQUIRES question_token (validator) and a topic. Old tokens -> 400.
- Level resolution: the config value becomes an InterviewLevel (map the old "Senior SDE-1" string; fail at startup on an unknown value); /question resolves level=None to it and ALWAYS sets level and source explicitly on responses; /grade uses the request's echoed level (verified by the token).
- Level semantics in NOTES mode: the level sets the REASONING demanded over the notes' facts; never penalise aspects the notes do not support (a Staff rubric must not punish the notes lacking cost/cross-team material). Per-level 3/6/8/10 anchors; difficulty is one step within a level, defined so harder-SDE-2 and easier-Senior-1 do not overlap.
- General mode: exclude_questions -> use only the last 15, each cut to 300 chars, fenced as DATA; log excluded_given/excluded_used. The grade prompt treats the client-echoed key as a HINT (its own knowledge may override a wrong key; the summary says so). Misconceptions render "(general knowledge: ...)" never "(your notes: ...)". strip_labels on the general question, key, model answer and every item; empty link allow-list; a test proves no general output carries an [S/T<n>] label or a link. Never call verified_items/check_answer with empty notes. General-mode kind rules of its own (KIND_RULES say "from the notes"). A test proves a source=notes request never returns source=general.
- NO_NOTES is "the topic matches nothing BEFORE exclusion" (one cheap existence query or the topic match run without exclude): with a topic list a session that already excluded Kafka chunks asking Redis with no notes must get NO_NOTES, not EXHAUSTED.
- Reply robustness: QuestionReply validates question and points separately: points 0..6 in the parse model (the prompt still asks for 3-6) so 2 points is no longer schema-invalid; optionally recover an unlabelled point by attributing the label when exactly one shown note lexically backs it. Log points_given/points_backed/points_kept and the validator reason category per attempt. LIKELY ROOT CAUSE of the live Spring Boot 502: lexically_backed -> attribute -> segment cut dotted property names (DOT bug), so pass \`names\` there; add a dotted-property point case to module_iv.
- Acronyms: apply only when the token has no FTS/title hit of its own, is 3-6 ASCII letters, or 2 letters typed in CAPITALS. Try initials with AND without stop words; split hyphens ("High-Level"); match a contiguous word run of a title or heading segment; ASCII/Latin only. EXPAND the acronym to the matched phrase(s) and feed them to the existing FTS and _search (embedding "HLD" is weak); expansion hits rank below direct hits. Fetch DISTINCT title + heading segments under RLS and the visible filters with a LIMIT (~5000), logging truncation.
6. Tests (module_iv): per-level prompt content and token binding (a level/source swap with a valid token -> rejected), general mode never touches notes and is always unverified with no citations, topic required for general, acronym matching positives/negatives (HLD -> High Level Design; "HL" or "HLDX" not; stop-word handling; a title in another tenant/collection never matches), NO_NOTES vs EXHAUSTED types, weak-points reply accepted with 0..1 points and the reason logged, still 502 for a too-short question.`,
  },
  {
    id: 'IV3-UI', title: 'Interview 3c clients: level dropdown, opt-in general-knowledge fallback, labels, summary split, start phrases, refusal badge text',
    owns: 'packages/client/src/ragforge_client/interview.py, packages/client/src/ragforge_client/ask.py, packages/ui/src/ragforge_ui/**, packages/mcp/src/ragforge_mcp/** (only if needed for the new fields), packages/core/tests/module_ui/**, packages/core/tests/test_client_interview.py, packages/core/tests/test_client_ask.py, packages/core/tests/module_mcp/** (only if you touch mcp), docs/handoff/IV3-UI.md',
    tests: 'uv run pytest packages/core/tests/module_ui packages/core/tests/module_mcp packages/core/tests/test_client_ask.py packages/core/tests/test_client_interview.py -q',
    spec: `Extend the Phase 3b clients (owner decisions 2026-10-19, binding). Server work is concurrent (IV3-CORE): code against the locked contracts and SDK, test with fakes.
1. LEVEL: a dropdown next to the topic when starting in the UI (SDE-1, SDE-2, Senior SDE-1 (default), Senior SDE-2, Staff); in make ask and the chat phrases: "interview me on Kafka at staff level", "... as a senior SDE 2", "/interview kafka --level staff" (deterministic table; tolerate "SDE2", "sde-2", "senior sde ii", "staff engineer"). The level is FIXED for a session: changing it starts a new session (say so). Show the level on every question card and in the summary ("Staff · Kafka · avg 6.5"). Send level in both requests; echo the question's level/source on grade.
2. OPT-IN FALLBACK: on 409 INTERVIEW_NO_NOTES ("Your notes have nothing on X") or INTERVIEW_EXHAUSTED ("All your X notes are asked"), OFFER a one-click "Interview me from general knowledge (not from your notes, not verified)" (UI button; terminal: a prompt line accepting "/general" or "y"); never switch automatically. With a topic list, the offer is per topic. General-knowledge questions/grades carry a clear, consistent "General knowledge · not verified" label everywhere (question card, grade card, model answer), no source chips, no verified badge, never styled like notes content (reuse the ADR-0016 extras look). Send exclude_questions (the session's earlier general questions on that topic). "next topic" may move between notes and general topics; each question keeps its own label.
3. SUMMARY SPLIT: notes-based scores vs general-knowledge scores; weak notes topics list pages to revisit; weak general topics say "consider adding notes on <topic>".
4. START PHRASES: add "interview mode [on X]", "start interview [on X]", "start an interview [on X]" (live failure: the owner typed "INTERVIEW MODE" and got a Q&A refusal). Keep every existing negative.
5. REFUSAL BADGE (live failure): a chat answer "Your notes don't cover this." showed "Not checked: The model in use is below the verifier's quality bar, so nothing was checked." That text is only true when the verifier is untrusted. When the turn is unverified AND the answer cited nothing (no citations), say "Nothing to check: no answer came from your notes." Keep the old text for an unverified answer that has citations. Same in make ask and MCP if they show such text.
6. A notes question may now come with expected_points EMPTY: the grade still works; the UI shows "No checked answer key for this question; graded against your notes" only where the key would be shown.
DESIGN REVIEW DECISIONS (binding):
- Always echo question.level and question.source (and the token) on grade. Branch on source FIRST: a general grade never shows the verifier badge texts; label: "General knowledge · not verified · graded by the model against its own answer key; not checked against your notes or any source".
- Never offer the general fallback when there is no topic (mixed sessions). General and notes scores are never averaged together.
- MCP: do NOT expose source=general in MCP tools (an MCP caller is a model and would pick it on its own; the owner decided it is always a human opt-in). Expose level only if trivial.
Tests: level phrase table + dropdown AppTest + level fixed per session, fallback offered on both 409 types and only after a click (no automatic switch; AppTest), labels present on every general card and absent on notes cards, exclude_questions sent, summary split, new start phrases (and negatives), refusal badge text both ways, empty expected points.`,
  },
  {
    id: 'DOT', title: 'False flag: bare dotted names (config keys, paths) cut as sentence ends',
    owns: 'packages/core/src/ragforge_core/agent/coverage.py, packages/core/src/ragforge_core/agent/attribution.py (only if needed), packages/core/tests/module_ag/test_ag_dot_*.py (new), packages/core/tests/module_ag/** (only assertions this change legitimately alters; justify each), docs/handoff/DOT.md',
    tests: 'uv run pytest packages/core/tests/module_ag packages/core/tests/module_iv packages/core/tests/module_e1 -q',
    spec: `Live false flags (owner reports, 2026-10-19): the answer sentence "You can accomplish this by making a POST request to /actuator/loggers/{logger.name} with a JSON body ... [S1]." was split at "logger." and the fragment "You can accomplish this by making a POST request to /actuator/loggers/{logger." flagged "Not found in your notes"; likewise "Setting max.in.flight.requests.per.connection to 1 provides strictest ordering [S2]." split at "per.". Both are backed by the notes. Cause: coverage.sentence_ends / _dot_in_token / _identifier_dot treat a lower.lower dot as a sentence end unless the next word is a known extension or the token continues with a _CHAIN character ("._/(\`"), so the LAST dot of a dotted name followed by a space cuts.
This is a GUARANTEE change (ADR-0015): read its "Current state" and the history of _CHAIN / the identifier-dot rule (a chain BEFORE the dot was rejected once: "node.js.it runs ..." hid a sentence end; \`=\` in _CHAIN reopened the sentence-merge class). The invariant: never MERGE two sentences a reader sees as two (an extra cut is safe: it only adds residue / a false flag; a missed cut can let an uncited sentence ride on a cited one).
Design the narrowest rule that fixes both live cases generically, for example (judge, don't just copy): the dot is in-token when the whole no-space run around it is a dotted name with at least 2 dots between identifier characters (a.b.c), or the dot is inside an unclosed {...}, [...] or (...) group in the same no-space run, or the run is a path (contains "/" before the dot in the same run); and in every case the character after the dot is a lowercase/identifier character (a reader does not see a sentence end without a following space). Consider and pin with tests the merge attacks: "Kafka is fast.and.it never loses data", "done.it.works", "see a.b. Then", "e.g.", "3.5", "node.js.it runs", homoglyph dots (_dot_like), bidi, emphasis around the dot, code spans adjacent. Property-test with hypothesis that no visible "word. Word" (dot + space + capital) boundary is ever merged.
DESIGN REVIEW DECISION (binding; replaces the example rule above, which merges "fast.and.it"): the rule is SOURCE-BACKED. An interior ASCII "." followed by [A-Za-z0-9_] is not a sentence end ONLY when the maximal [A-Za-z0-9_.] run around it (leading/trailing dots stripped) is in \`names\` (the SHARED SEAM parameter: dotted names found verbatim in the shown S sources; case-sensitive, [A-Za-z0-9_.] boundaries, no normalisation). A dot followed by whitespace or the end of text is never exempted ("see a.b. Then" still cuts). _dot_like, non-ASCII, bidi rules untouched; every existing fail-closed rule stays ahead. Implement \`dotted_names\` and the keyword-only \`names\` parameter (default empty = today's behaviour exactly) through segment/residue/visible_runs/attribute and their internals. Property tests (hypothesis): cuts with \`names\` == cuts without \`names\` minus ONLY the interior dots of runs in \`names\`; for random cores not in \`names\` every boundary stays cut; the terminator rule holds for ". lower", a newline after the dot and a dot with no space. Pin: fast.and.it, nasa.in.fact, /dev/null.it, (nasa.it, homoglyph dots, emphasis/code around the dot, both live sentences (covered when cited and names come from the source).
Deliver: the rule, tests (both live sentences now one unit each and covered when cited; merge attacks still cut), and an exact paragraph for ADR-0015 "Current state" in your handoff.`,
  },
]

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    tests_passing: { type: 'boolean' },
    test_command: { type: 'string' },
    checks: { type: 'string' },
    handoff: { type: 'string' },
    known_gaps: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files', 'tests_passing', 'test_command', 'checks', 'handoff', 'known_gaps'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    score: { type: 'number' },
    verdict: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['P0', 'P1', 'P2'] },
          title: { type: 'string' },
          location: { type: 'string' },
          why: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['severity', 'title', 'location', 'why', 'fix'],
      },
    },
    verified: { type: 'array', items: { type: 'string' } },
  },
  required: ['score', 'verdict', 'findings', 'verified'],
}

function moduleBlock(m) {
  return `
MODULE ${m.id}: ${m.title}
Paths you OWN (create/edit only these): ${m.owns}
Handoff file: ${ROOT}/docs/handoff/${m.id}.md
Existing suites you must keep green: ${m.tests}

WHAT TO BUILD:
${m.spec}`
}

function reviewPrompt(m, round) {
  return `You are a principal engineer with 25 years at a large tech company (distributed systems, security, search/RAG, LLM products). Review ONE module of RAGForge BLIND (round ${round}). Be adversarial but calibrated: 9+ = approve with minor comments; 8 = approve after short fixes; 7 = needs another revision.
Repository: ${ROOT} (NOT a git repo: never run git). Do NOT modify any file. Do NOT run make up/nuke or docker compose. Do NOT open docs/REVIEW.md.
Two other modules are being built concurrently in the same folder; judge ONLY module ${m.id} (failures clearly caused by another module's in-progress files are not this module's defects, but say so).

Read CLAUDE.md, docs/NEXT.md "TO DO NEXT" item 1 (owner decisions), docs/DESIGN.md section 8, docs/adr/0015 "Current state", docs/adr/0016, the contracts packages/contracts/src/ragforge_contracts/api.py and sse.py, the SDK client.py. Then read ALL of module ${m.id}'s files and changes: ${m.owns}, and its handoff.

Module ${m.id} was supposed to build:
${m.spec}

Run read-only verification: ${m.tests}; uv run mypy on its source files and tests; uv run ruff check on its paths; uv run lint-imports. Write any scratch probes ONLY under <SCRATCH>/review-${m.id}/ and run them with uv run python (never name a probe after a library, e.g. h2.py).

Check: correctness against the owner decisions, security (tenant isolation/RLS, prompt injection through notes or user answers, key leakage, unsafe rendering), the verification guarantee (can anything now be marked verified that is not backed? can a reason be wrong so a real problem looks muted?), failure handling, test quality (do tests PROVE behaviour; try to construct a counterexample the tests miss), code quality, ownership violations. Return P0/P1 only for real defects; style nits are P2.`
}

function fixPrompt(m, review, round) {
  return `${COMMON}
${moduleBlock(m)}

You are continuing module ${m.id}, already implemented in the owned paths. A blind principal-engineer review (round ${round}) scored it ${review.score}/10 with these findings. Fix EVERY P0 and P1 (and cheap P2s inside your paths). If you believe a finding is wrong, do not change code; explain why in your summary. Stay strictly inside your owned paths; anything else goes into your handoff file. Re-run the definition-of-done checks.

FINDINGS:
${JSON.stringify(review.findings, null, 2)}`
}

const blocking = r => !r || r.findings.some(f => f.severity === 'P0' || f.severity === 'P1')

const results = await pipeline(
  MODULES,
  async m => {
    const build = await agent(`${COMMON}\n${moduleBlock(m)}`, { label: `build:${m.id}`, phase: 'Build', schema: BUILD_SCHEMA })
    log(`${m.id} built (tests passing: ${build ? build.tests_passing : 'unknown'})`)
    return build
  },
  async (build, m) => {
    const history = []
    let review = await agent(reviewPrompt(m, 1), { label: `review:${m.id}#1`, phase: 'Review', schema: REVIEW_SCHEMA })
    history.push(review)
    log(`${m.id} review 1: ${review ? review.score : '?'}/10`)
    let round = 1
    while (blocking(review) && review && round <= 2) {
      const fix = await agent(fixPrompt(m, review, round), { label: `fix:${m.id}#${round}`, phase: 'Fix', schema: BUILD_SCHEMA })
      round += 1
      review = await agent(reviewPrompt(m, round), { label: `review:${m.id}#${round}`, phase: 'Review', schema: REVIEW_SCHEMA })
      history.push(review)
      log(`${m.id} review ${round}: ${review ? review.score : '?'}/10`)
      if (fix) build = fix
    }
    return {
      module: m.id,
      final_score: review ? review.score : null,
      open_blockers: review ? review.findings.filter(f => f.severity !== 'P2') : [],
      open_p2: review ? review.findings.filter(f => f.severity === 'P2') : [],
      scores: history.map(h => (h ? h.score : null)),
      build,
    }
  },
)

return results.filter(Boolean)
