# W4-DECISIONS handoff

## What I built
- `docs/DECISIONS.md` (one file, about 140 lines), per PHASE5 W4 + R2-6 + R2-8 items 1 and 5:
  1. **Decision index:** one row per ADR-0001..0018 (link, decision, main alternative rejected, why, status). Status was checked against the code, the Makefile and the tests, not copied from the ADRs. Rule used: the status covers the ADR's core scope; items the ADR tags [A]/[S] are named in the note when not built.
     - Implemented: 0001, 0002, 0003, 0005, 0007, 0008, 0010, 0011, 0013, 0015, 0016, 0017, 0018.
     - Partial: 0004 (no `compose.gpu.yml`, no `make eval-models`, no openai/bedrock adapters), 0006 (τ not tuned, default marked provisional), 0009 (no orphan repair job), 0012 (largely replaced by 0017; no recorded corpus, CI gate, `--calibrate`), 0014 (no nightly `pg_dump`, no `make reindex`, [A] no restore drill, no `audit_log`/`usage_events` retention jobs).
     - Not built: none as a whole ADR.
  2. **Review history** for Phase 0 (design and code), 1, 2a, 2b, 2c, 3, 3b, 3c, ADR-0017 and ADR-0018. The scores are quoted exactly from REVIEW.md, NEXT.md, PHASE2A.md and the ADRs. There is one line per phase on what the reviews caught that the tests did not.
  3. How the work was done, with links to all 7 `docs/workflows/*.js`.
  4. Parked work: CI, Helm/kind/KEDA/NetworkPolicies/multi-arch, the backup/restore drill, the load test, DeepEval, the token rotation (described only as an "accepted risk", with no detail of the exposure), and the blog post.
  5. Known open issues: the measured limits (hosted p95 about 6.6 s against the 5 s SLO, held-out set not sealed, private verifier and judge waiting for labels, the identifier-typo class), the full ADR-0017 P2 and deferred list, the ADR-0018 MCP log P2, and two gaps found while checking (no ADR for Gemini, PDF uploads rejected).
  6. DESIGN.md sections replaced by later ADRs (section → ADR table).
- Privacy: no question text, note titles, private topic list, tenant/collection ids, key prefixes or home paths. The only examples are synthetic ("fast.and.it", "(S3)").

## Checks run
- A link checker over every relative link in DECISIONS.md: 0 broken.
- Every `docs/adr/*.md` path appears in DECISIONS.md (the R2-8 item 4 check).
- No banned words ("blazing", "state of the art", "seamless", "robust", "powerful"), no absolute home paths, no UUID-shaped strings.
- `uv run pytest packages/core/tests/test_docs_consistency.py -q`: 45 passed (after the round-1 fixes).
- **Seam with W1 on not-built make targets (round-1 P1).** DECISIONS.md must name `make reindex`, `make eval-models` and `make label-verdicts` as not built. W1 chose option (b): `NOT_BUILT_TARGETS` in `test_docs_consistency.py` allows exactly these three, in DECISIONS.md only, and `test_real_not_built_targets_are_still_not_built_and_come_from_an_adr` fails if one of them is built or is no longer named in an ADR. So DECISIONS.md keeps the `make X` wording. If a new not-built target is ever named here, W1's list must grow with it.
- No Python was written, so ruff and mypy do not apply.

## Round-1 review fixes
- P1 test conflict: resolved by W1's allow-list (see Checks); no DECISIONS change needed; test green.
- P1 design-first overclaim: section 3 now says Phase 0's design had 7 rounds (5 blind), Phases 1-3 built on it with no own design review, and 3b, 3c, ADR-0017, ADR-0018 each had a blind design review before code.
- P2: added a DESIGN §11 Clone-and-run row (no `mcp`/`ui`/`seed-demo` services in compose.yaml, `make ui`/`make mcp`/`make eval-demo`/`make models`, `scripts/bootstrap.sh` does not pull models); section 6 now also covers drift with no ADR.
- P2: self-score wording now covers "self" and "self-claimed" (the 9.4 is marked "self-claimed").
- P2: ADR-0014 restore drill moved to an "[A] ... not built" clause (the ADR tags it Advanced (M8)); ADR-0013 note adds "[A] team tenants: not built".
- P2: ADR-0014 "Why" cell now gives the ADR's own reasoning (Derived row, RTO line). I did not add "size and restore time", because the ADR does not state it.

## Integration steps for the integrator
1. Run the W1 consistency test and fix any anchor or link it flags. I used no `#anchor` links to other files, and none to my own headings either.
2. README doc map: link `docs/DECISIONS.md` as "decisions, review history, parked work, open issues" (W1 owns README).
3. RESULTS.md (W3) and GUIDE.md (W2) should link here for "not done" instead of repeating it (R2-8 item 1).
4. If W0's `make tenant` / `make mcp-key` change any ADR-described behaviour, nothing here needs a change: DECISIONS.md does not describe those commands.
5. Re-check the status column if anything lands before publishing: the ADR-0018 MCP logging fix, a Gemini ADR, the τ calibration, or the backups.

## Open questions
1. **Gemini has no ADR.** ADR-0004 still says `LLM_PROVIDER=openai|bedrock`. Should ADR-0004 get an amendment (or a new ADR-0019) for the Gemini provider and the provider menu? DECISIONS.md lists it as a gap for now.
2. **ADR-0017 final re-review score.** ADR-0017 records the final integrated review as 7.0 with all P0/P1 fixed, but no re-review score. DECISIONS.md says "no re-review score recorded". If a score exists elsewhere, add it.
3. **Design reviews before Phase 1, 2a, 2b, 2c and 3.** None are recorded, so the table says "Not recorded" and section 3 says these phases were built on the Phase 0 design. Confirm that none took place.
4. **Orphan repair job (ADR-0009).** I found the metric but no repair job, and marked 0009 Partial. Confirm that it was never built, rather than living under another name.
5. **Retention jobs (ADR-0014).** I found no deletion job for `audit_log` (1 y) or `usage_events` (90 d). Only `jobs_history` has a 30-day drop. Confirm.
6. **Workflow scripts contain absolute home paths** (for example `ROOT` in `docs/workflows/phase5-writeup.js`). DECISIONS.md links them, and R2-4 says they ship after scrubbing. The pre-publish gate (its home-path pattern) will catch this; it is noted here so it is not a surprise.
7. **Status rule for [A]/[S] items.** I chose "status covers core scope; [A]/[S] items are named in the note". If the owner prefers that any unbuilt item makes an ADR Partial, then 0007 (webhooks), 0010 (per-tenant fairness) and 0013 (`document_acl`) become Partial.
