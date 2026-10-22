# Handoff: EV-API (server eval seams, ADR-0017 §1)

Status: built and green on the module suites. Nothing outside my paths was edited; the items the
integrator must apply are in "Needed outside my paths".

## What was built
| ADR-0017 §1 row | Where | Notes |
|---|---|---|
| eval scope + switch | `apps/api/deps.py` (`get_eval_api_enabled`, `eval_surface`, `EvalKey`, `EvalQuotaKey`, `require_eval_seam`, `eval_seam_allowed`, `get_eval_limiter`), `config.py` (`eval_api`, `eval_rate_per_minute`, `eval_burst`, `eval_concurrency`) | `/v1/eval/*`: switch first (404 while off, BEFORE auth, so a probe can't tell disabled from missing), then key (401), then `eval` scope (403). Additive seams on shared routes: 403 when the switch is off or the scope is missing. Fails closed: no settings means off. |
| key rules | `services/auth/scopes.py` (`EVAL_KEY_SCOPES`, `MCP_EXCLUDED_SCOPES`, `check_eval_scopes`, `effective_scopes`) | Creation refuses `eval` on an MCP key (name contains `mcp`) and `eval` beside anything except `chat`/`read` (422). An MCP key row that somehow holds `eval` loses it when it authenticates. `eval` never makes a key a key manager. |
| `POST /v1/eval/verify` | `routes/eval.py` + `agent/eval_verify.py` | The chat pipeline as shipped (normalize, check_citations strict as configured with dotted names, attribute, short-circuits, ONE ground check on the annotated text, settle), and it stops at settle: `finalize`'s rule 7 (the untrusted-verifier downgrade) is NOT applied, because E-V is what decides `agent_verifier_trusted` (a client that wants an untrusted server's chat badge applies rule 7 to `status` itself). Built only from calls into the public agent functions (read-only). Returns the evidence from each layer BEFORE the merge: `coverage_units` (kind + span), `verifier_claims` (span, or null when unplaced, including paraphrases `parse_verdict` drops, taken from the raw reply), `verifier_ran=False` on every short-circuit, verdict/conclusive, and the model the provider REPORTED (None when no reply came back) plus `ground_check@v`. Spans are mapped back onto the answer AS SUPPLIED through a log of the marker rewrites themselves (linear; checked against `normalize_citations`, `NormalizationDrift` if it ever disagrees). Notes are re-read through RLS in a short tx; the model runs after it closes. Per-principal limiter with its own budget (same token-bucket class as interview). Provider errors propagate (502/503), never a fake `unverified`. |
| `GET /v1/eval/chunks?ids=` | `routes/eval.py` | Request order, de-duplicated, max 200, 422 on a malformed id; ids that are invisible or tombstoned are simply absent. |
| `POST /v1/eval/resolve` | `routes/eval.py` (`resolve_chunk` is pure) | PageKey file=(file, metadata.filename, falling back to external_id), notion=(notion, external_id). Chunk key matched by (heading_path, display_sha256, occurrence in ordinal order). Stale: candidates under the same (page, heading_path), never re-mapped. A page key naming two or more LIVE documents (a file uploaded twice) is `ambiguous` (open-enum value) for the page and its chunk keys, with the chunks under the same heading_path of every such document as candidates (newest first), and a counts-only warning log; the client counts only `ok`, so the item goes stale. A foreign collection gives 404. |
| `/v1/query stages=true` | `routes/ask.py`, `agent/retrieval.py` (`StageRecorder`, `recording`, `stage_score`) | Runs the same four steps over the INJECTED retriever's own backend, per-list k and RRF k (`recording`), recorded, with the injected reranker. `hits` are identical to the plain call (tested with non-default k's). Every stage is sorted by (-score, chunk_id). Several vector spaces interleave by rank. Checked before any read or model call. Without the flag the old path runs unchanged. |
| chat `?trace=shown` + `prompt_versions` | `routes/ask.py`, `apps/api/chat_stream.py`, `agent/runner.py` (`TurnTrace`), `prompts/__init__.py` (`SURFACES`, `surface_of`, `surface_versions`, `PROMPT_VERSIONS`) | `DoneEvent.shown` (label, chunk_id, document_id, display_sha256 hex) is sent only with the trace AND the eval scope (an unknown trace value gives 422). `prompt_versions` is sent to every caller as {chat, verifier, interview}: "name@v,..."; a turn resumed by a newer build reports both sets. `judge` is absent on the server because the judge runs in the client (§4). `prompt_version` is kept. |
| interview `chunk_ids` | `routes/interview.py` (`question_seams`), `agent/interview/service.py` (`targeted_question`) | Eval seam only, checked in a dependency declared before `InterviewKey`, so a refused call spends no interview quota. Notes mode only (general gives 422). Every id must be visible in `collection_ids` (404 otherwise), and this is checked before any model call. The seed is S1 and the neighbours fill the context as for a sampled seed. The token is a normal v3 token. |
| eval-scope frozen grade | `routes/interview.py` (`token_rule`, `TokenRule`, `ENFORCE_NOTES_TOKEN`) | An eval-scope NOTES grade needs no token and may use any level. The eval check runs first, so the exemption survives `ENFORCE_NOTES_TOKEN=True`. General mode always needs its token. A token that IS sent is always verified, for every key. Real keys keep v3. |

## Round-1 review fixes (this session)
- **P1 span map:** the difflib alignment (roughly cubic, blocked the event loop) is replaced by
  a log of the rewrites `normalize_citations` makes: each fixpoint pass is replayed over
  `structure(text).label_spans` as its two regex substitutions (`_RANGE` then `_LENIENT`), the
  changed matches are recorded, and positions map back through the composed logs with `bisect`.
  The replay is compared with `normalize_citations(answer)`; a mismatch raises
  `NormalizationDrift` (500) rather than mis-mapping. 20,000 lenient characters: ~0.07 s.
- **P1 status:** `status` is settle's. `verifier_trusted` is no longer an input of verify, and
  `VerifyConfig` no longer reads `agent_verifier_trusted`. Tested at the route with the setting off.
- **P2s:** `verifier_model` is None when no usage was recorded; the interview seam is checked
  before the quota; `stages` uses the injected retriever's configuration (the
  `get_search_backend` factory and its conftest override are removed); duplicate filenames are
  `ambiguous`; the span property test now uses a positional oracle (the canonical answer's
  spans moved by the test's own record of the respelled markers), plus range and stacked-label
  cases and a marker-end edge case. Mutation-checked by hand (each one was killed and then
  reverted): end edge off by one, start edge off by one, seam after the quota, and a
  default-built retriever.
- **P2 eval-tenant check (not changed, by decision):** `require_eval_key` does not check
  `tenants.kind = 'eval'`. Migration 0012 is EV-SNAP's and is still in flight in this folder, and
  the module tests mint eval keys in ordinary tenants. Today the guard is the key-minting rules:
  no owner key can hold `eval` (no escalation), and only EV-SNAP's provisioning mints eval keys,
  in eval tenants. Item 5 below adds the defence in depth once 0012 lands.

## Needed outside my paths (integrator)
1. **Dev switch:** set `RAGFORGE_EVAL_API=true` on the `api` service in `compose.yaml` (dev only)
   and add a commented `# RAGFORGE_EVAL_API=false` line to `.env.example`. The default stays off.
2. **Public accessors on `rag/retriever.py` `HybridRetriever`** (recommended): read-only
   `backend`, `per_list_k` and `rrf_k` properties. `agent/retrieval.recording` currently reads
   `_backend`, `_per_list_k` and `_rrf_k` (documented there). Switch it to the properties once
   they exist.
3. **Optional contract values (OPEN enums, so they already validate):** `CoverageUnitKind` has
   no `invalid_citation`, which is what verify reports for rule-1 offenders. `ResolveStatus` has
   no `ambiguous`, which resolve reports for a page key with several live documents. Add both
   members if you want them typed.
4. Migration 0012 (`tenants.kind = 'eval'`, `eval_of_tenant_id`) and the snapshot are EV-SNAP's,
   not mine. The routes here rely only on RLS of the caller's tenant.
5. **After 0012 lands (defence in depth, review P2):** in `deps.require_eval_key` (or in
   `authenticate`, for a key that holds `eval`), confirm `rf.tenants.kind = 'eval'` for the
   caller's tenant, and give 403 otherwise. At the same time, change the module_eval_api
   `new_tenant` fixture to mint `eval_key` in an eval tenant.
6. **Public rewrite log in `agent/citations.py`** (recommended): a
   `normalize_citations_with_edits(answer) -> (draft, edits)` (or public `_RANGE`/`_LENIENT`
   rewrite helpers). `agent/eval_verify._normalize_pass` imports `_LENIENT`, `_RANGE`,
   `_canonical`, `_expand_range` and `_MAX_PASSES` to replay normalization. The drift check makes a
   divergence loud, but a public API would remove the coupling.

## Verification (run in this session)
- `uv run pytest packages/core/tests/module_eval_api packages/core/tests/module_e2 packages/core/tests/module_ag packages/core/tests/module_iv packages/core/tests/module_d1 -q`: **2115 passed** (round-1 fixes). module_eval_api on its own: 109 tests.
- ruff format --check + ruff check on all owned paths: clean. `uv run mypy packages/core/src`:
  clean (168 files). `MYPYPATH=packages/core/tests mypy --explicit-package-bases .../test_ev_api_*.py conftest.py`: clean.
  `uv run lint-imports`: 6 kept, 0 broken.
- Mutation checks, each applied by hand, killed, then reverted (and diffed to confirm the revert):
  stage tie-break removed, `shown` sent without the trace, general-mode token exemption, and
  unplaced verifier claims dropped.
- The first test attempt hit a testcontainers Postgres start timeout (120 s, likely load from
  parallel agents). The re-run was clean.

## The integrator must test live (`make up`, eval tenant + eval key)
- `RAGFORGE_EVAL_API` off gives 404 on `/v1/eval/*` and 403 on `stages` / `?trace=shown` /
  `chunk_ids`. On, it works with the eval key, and the owner's key gets 403.
- `/v1/eval/verify` with `RAGFORGE_AGENT_VERIFIER_TRUSTED=false`: a clean D0 item still returns
  `verified`, with no rule-7 downgrade.
- `/v1/eval/verify` on a real D0 answer with the real verifier model: check that
  `verifier_model` is the model the provider reports and `verifier_prompt` is `ground_check@<v>`.
- `/v1/query stages=true` on D0: `reranked` equals `hits` order, and two identical calls give
  identical stages (the A/A precondition with host Ollama).
- Chat `?trace=shown`: the `shown` digests equal `/v1/eval/chunks` `display_sha256` for the same ids.
- Interview `chunk_ids` + a tokenless notes grade at all five levels with the eval key. The same
  tokenless grade with the owner key still follows v3 (optional today).
