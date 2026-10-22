// Paths scrubbed for publishing: <REPO> is the checkout, <SCRATCH> a session scratch folder.
export const meta = {
  name: 'ragforge-phase3-build',
  description: 'Phase 3: MCP server + Streamlit UI built in parallel, each with blind principal review and fix rounds',
  phases: [
    { title: 'Build', detail: '2 builders implement MCP and UI in owned paths' },
    { title: 'Review', detail: 'blind principal review per module' },
    { title: 'Fix', detail: 'fix P0/P1, then re-review (max 2 rounds)' },
  ],
}

const ROOT = '<REPO>'

const COMMON = `
You are a senior engineer building ONE module of RAGForge (multi-tenant RAG over the owner's Notion notes) in a shared folder where 1 other engineer builds another module AT THE SAME TIME.
Repository: ${ROOT}  (NOT a git repo: never run git commands. Do not commit.)

READ FIRST: ${ROOT}/CLAUDE.md, ${ROOT}/docs/DESIGN.md (sections 1, 8 "SSE contract", 10 "MCP server"), ${ROOT}/docs/adr/0005-mcp-as-thin-api-client.md, ${ROOT}/docs/adr/0015-retrieve-then-answer.md ("Current state" section only), ${ROOT}/docs/adr/0016-labelled-extras.md, the SDK ${ROOT}/packages/client/src/ragforge_client/client.py and ${ROOT}/packages/client/src/ragforge_client/ask.py (the terminal client: reuse its key-file lookup read_key/resolve_key_file, clean, unescape_outside_fences, TerminalText ideas), the contracts ${ROOT}/packages/contracts/src/ragforge_contracts/api.py and sse.py.

HARD RULES:
1. Create/edit ONLY the paths you own (listed below). Never edit packages/core/**, packages/contracts/**, packages/client/** (except reading), any pyproject.toml, uv.lock, compose.yaml, Makefile, Dockerfile, docs/** (except your handoff file). Never run uv add / uv lock / uv sync. Dependencies are already installed: mcp 2.3.0 (NOTE: v2 API — FastMCP was renamed: \`from mcp.server.mcpserver import MCPServer\`; check the installed package source under .venv/lib/python3.12/site-packages/mcp for the real API before coding) and streamlit 1.66.0 (use streamlit.testing.v1.AppTest for tests).
2. Anything you need outside your paths (a Makefile target, a compose service, a script entry, an SDK method) goes in your handoff file ${ROOT}/docs/handoff/<MODULE_ID>.md. The integrator applies it.
3. Format/lint ONLY your paths: \`uv run ruff format <your paths>\`, \`uv run ruff check --fix <your paths>\`. Never run make fmt or ruff on packages/.
4. Tests go under your test dir with file names prefixed by your module id (e.g. test_mcp_tools.py); no __init__.py in test dirs. No real network or LLM: fake the API with respx (httpx) or a fake RagforgeClient. Do NOT run make up / make nuke / docker compose (the live stack is shared and in use).
5. Layering (import-linter): ragforge_mcp and ragforge_ui may import ragforge_client and ragforge_contracts, NEVER ragforge_core. Thin clients of the public API only (ADR-0005).
6. Secrets: the API key is read from a key file (reuse ragforge_client.ask.resolve_key_file/read_key semantics: --key-file > RAGFORGE_KEY_FILE > the single ~/.config/ragforge/*.key; refuse group/other-readable) or an env var where noted; never print, log or put it in exceptions, URLs or reprs.
7. Rendering: answer text from the API is already-sanitized CommonMark (escaped); render it as Markdown, never as raw HTML (no unsafe_allow_html with server text). Extras (SSE \`extra\`) are general knowledge, NOT the user's notes: always shown APART from the answer, AFTER citations/verdict, under a clear "Not from your notes" label, never styled as verified.
8. Principal-engineer quality: typed (mypy --strict clean), small focused functions, docstrings explaining WHY for non-obvious decisions, tests that PROVE the behaviour (not just happy paths), clean failure handling (API down, 401, 429, stream error mid-answer).

DEFINITION OF DONE: your tests pass; \`uv run ruff check <your paths>\` and \`uv run ruff format --check <your paths>\` clean; \`uv run mypy <your source dir>\` clean; \`MYPYPATH=<your test dir> uv run mypy <your test dir>/test_*.py\` clean; \`uv run lint-imports\` passes. Before finishing run \`uv run pytest packages/core/tests/test_client_ask.py -q\` (must still pass). Write your handoff file with exact integration steps (Makefile targets, how to run, how to add to Claude Code).
`

const MODULES = [
  {
    id: 'MCP', title: 'MCP server (thin API client)',
    owns: 'packages/mcp/src/ragforge_mcp/**, packages/core/tests/module_mcp/**, docs/handoff/MCP.md',
    spec: `Build the RAGForge MCP server per DESIGN section 10 and ADR-0005, with the mcp 2.x SDK (MCPServer).
- Entry point ragforge_mcp.server:main (already registered as console script \`ragforge-mcp\`). Transports: stdio (default) and streamable HTTP (--http --port N).
- Config: API URL (--url / RAGFORGE_URL, default http://localhost:8000), key via the key-file rules above or RAGFORGE_API_KEY env var; collections (--collection, repeatable; default: all collections visible to the key, max 20).
- Tools (typed inputs/outputs, clear descriptions written FOR an AI agent):
  * ask(question, conversation_id?): runs a chat turn via SDK chat(); returns answer text (Markdown escapes undone for agent readability, keep code verbatim), citations [{label "S<n>", title, heading_path, url, inferred}], verification status, unsupported_claims, extras [{kind, text}] clearly marked "not_from_notes": true, conversation_id (so the agent can ask follow-ups). On an interrupt (destructive tool pending) return needs_approval {run_id, ui_url} and NEVER approve (there is deliberately no approve tool).
  * search(query, k<=20): SDK query(): ranked passages with citation info, no LLM.
  * list_sources(): collections + connectors with sync status.
  * sync_now(connector_id?): trigger a connector sync (default: all Notion connectors of the key) and return job ids.
- Resource ragforge://documents/{id}: document metadata + its chunks' text if the API exposes it (else metadata only; note any SDK/API gap in the handoff).
- Errors: map ApiError/HTTP errors to clear tool errors (401 bad key, 403 scope, 429 retry-after, API unreachable), never leak the key.
- Tests (respx or a fake client + the mcp in-memory client/session utilities from the SDK): every tool's happy path and error paths, interrupt -> needs_approval (and no approve tool exists in list_tools), extras are separated and marked, key never appears in any tool output/log/exception, stdio server starts (smoke via in-memory transport).
- Handoff: a Makefile target \`mcp\` (stdio run) and \`mcp-install\` that prints (does NOT run) the exact \`claude mcp add ...\` command for the owner, plus how to create an MCP-scoped key (no approvals:write) via the SDK create_api_key.`,
  },
  {
    id: 'UI', title: 'Streamlit UI (thin API client)',
    owns: 'packages/ui/src/ragforge_ui/**, packages/core/tests/module_ui/**, docs/handoff/UI.md',
    spec: `Build the RAGForge Streamlit UI: the owner's daily interview-prep surface.
- Entry: packages/ui/src/ragforge_ui/app.py (run with \`streamlit run\`), logic split into small testable modules (e.g. state.py, render.py, api.py) so most behaviour is unit-tested without Streamlit.
- Config: API URL (RAGFORGE_URL, default http://localhost:8000), key via the key-file rules above; never show the key.
- Chat page (default):
  * st.chat_input + streamed answer (SDK chat() events) with st.write_stream or incremental placeholder; conversation persists in session state with follow-ups (conversation_id); "New conversation" button.
  * The answer rendered as Markdown (the API text is escaped CommonMark; render as Markdown, never unsafe HTML). Answer sections ("In simple terms / Technically / In production") must look clean.
  * Below the answer: a verification badge (verified / partial / unsupported / unverified with plain-language meaning), unsupported claims listed as highlighted "not verified" items, citation chips [S1] Title › heading with a link to the Notion page/paragraph (deep link url), inferred citations marked "S2*  matched by the server".
  * THEN, separately, extras (SSE \`extra\`) in a visibly distinct container titled "Not from your notes — general knowledge (not verified)": incident text, or code in st.code with its language.
  * A "Show the code" button after a cited answer without code, which sends "show the code" as the next turn.
  * Timings (first token, total) small and muted; errors (API down, 401, 429, mid-stream error) shown as friendly messages without losing the conversation.
  * On an interrupt (pending destructive tool): show the pending calls and Approve/Reject buttons that call SDK resume(run_id, decisions) (the UI key has approvals:write).
- Sources page (sidebar): collections, connectors with last sync/status, a "Sync now" button per connector (SDK sync_connector), and recent job status if cheap.
- Tests: unit tests of the pure render/state helpers (citation chips, badge mapping, extras separation, interrupt decisions, error mapping, key never rendered), plus AppTest smoke tests with a fake client (chat turn renders answer, badge, citations, extras block AFTER citations, Show-the-code button behaviour, sources page renders).
- Handoff: a Makefile target \`ui\` (uv run streamlit run packages/ui/src/ragforge_ui/app.py on port 8501, headless) and any compose/Dockerfile suggestion (optional).`,
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

WHAT TO BUILD:
${m.spec}`
}

function reviewPrompt(m, round) {
  return `You are a principal engineer with 25 years at a large tech company (distributed systems, security, developer tools, LLM products). Review ONE module of RAGForge BLIND (round ${round}). Be adversarial but calibrated: 9+ = approve with minor comments; 8 = approve after short fixes; 7 = needs another revision.
Repository: ${ROOT} (NOT a git repo: never run git). Do NOT modify any file. Do NOT run make up/nuke or docker compose. Do NOT open docs/REVIEW.md.
Another module is being built concurrently in the same folder; judge ONLY module ${m.id}.

Read CLAUDE.md, docs/DESIGN.md sections 8 (SSE contract) and 10 (MCP), docs/adr/0005, docs/adr/0016, the SDK packages/client/src/ragforge_client/client.py and ask.py, the contracts packages/contracts/src/ragforge_contracts/api.py and sse.py. Then read ALL of module ${m.id}'s files: ${m.owns}.

Module ${m.id} was supposed to build:
${m.spec}

Run read-only verification: its tests (uv run pytest packages/core/tests/module_${m.id.toLowerCase()} -q), uv run mypy on its source dir and (MYPYPATH=<test dir>) its tests, uv run ruff check on its paths, uv run lint-imports.

Check: correctness (SSE event handling incl. extras/interrupts/errors), security (key handling and leakage, unsafe HTML rendering of server text, MCP having no approve path, prompt-injection surface of tool outputs), whether extras can ever look like verified notes content, conformance to ADR-0005 (thin client, no ragforge_core import), failure handling, test quality (do tests PROVE behaviour), code quality, ownership violations. Return P0/P1 only for real defects; style nits are P2.`
}

function fixPrompt(m, review, round) {
  return `${COMMON}
${moduleBlock(m)}

You are continuing module ${m.id}, already implemented in the owned paths. A blind principal-engineer review (round ${round}) scored it ${review.score}/10 with these findings. Fix EVERY P0 and P1 (and cheap P2s inside your paths). If you believe a finding is wrong, do not change code; explain why. Stay strictly inside your owned paths; anything else goes into your handoff file. Re-run the definition-of-done checks.

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