# Handoff: module G1 (Gemini ChatModel adapter + provider selection)

## What's in place
| Path | Contents |
|---|---|
| `adapters/llm/chat_gemini.py` | `GeminiChatModel` (the `ChatModel` port) over REST `streamGenerateContent?alt=sse` with httpx; `GeminiChatModel.from_settings(settings, client=None)`; `to_gemini_schema()`; errors `ModelConfigurationError` (and its `QuotaExhausted`), `ModelRequestRejected`, `ResponseBlocked`, `StreamInterrupted`, `OutputLimitReached` (an `OutputTruncated`), `UnsupportedSchema` (a `RagforgeError`, not a `ValueError`); output ceilings `ANSWER_MAX_OUTPUT_TOKENS` / `STRUCTURED_MAX_OUTPUT_TOKENS`. |
| `adapters/llm/provider.py` | `build_chat_model(settings, client=None) -> ManagedChatModel` (a `ChatModel` with `aclose()`), picking Ollama or Gemini by `RAGFORGE_LLM_PROVIDER`. |
| `config.py` | `LLMProvider` is now exactly `ollama` / `gemini` (the unimplemented `openai` / `bedrock` values were removed: before, selecting them silently ran Ollama). New: `gemini_api_key: SecretStr`, `gemini_model: str` (default `gemini-3.5-flash-lite`), `gemini_thinking_budget: int` (default 0, range -1..32768). Selecting gemini without a key fails at startup with `RAGFORGE_GEMINI_API_KEY must be set when RAGFORGE_LLM_PROVIDER=gemini`. |
| `bootstrap.py` | Only the chat construction changed: `build_chat_model(settings)` replaces `OllamaChatModel.from_settings(settings)`; it is still closed via the exit stack. |
| `tests/module_g1/**` | 210 tests (respx) + 2 live smoke tests (a two-turn tool round trip replayed through a fresh adapter, and a `thinking_budget=-1` variant that replays a real cached signature), which are skipped unless `RAGFORGE_GEMINI_API_KEY` is set in the shell environment (it never reads `.env`). |

## Error contract (for D1, the agent, and E, the API)
- Every Gemini error except rate limits is a `ChatModelError` (the same class the Ollama adapter raises), so `except ChatModelError` covers both providers. Retry semantics are carried by the class (there is no `retryable` attribute; round 1 had one that nothing read):
  - retryable after a delay: `UpstreamRateLimited`;
  - transient (retry with backoff): plain `ChatModelError` for 5xx, timeouts, transport errors and broken streams;
  - permanent (do not retry): `ModelConfigurationError` (incl. `QuotaExhausted`), `ModelRequestRejected`, `ResponseBlocked`, `OutputTruncated` (Gemini raises the subclass `OutputLimitReached`), `StreamInterrupted`.
  - **Retry classes hold only before output.** In `stream()`, a rate limit or a transient failure (5xx error event, timeout, dropped connection, a stream cut off without a finish reason, a malformed event) that strikes AFTER any `TextDelta` or `ToolCall` was yielded raises `StreamInterrupted(cause)` instead, with `cause` set to `"rate_limited"` or `"transient"`. Replaying would duplicate text the user saw or tool calls the agent already ran; a caller may restart the turn from scratch only after discarding what it showed. `complete_json` yields nothing early, so its errors keep their retry class. The Ollama adapter has the same gap (not my path): D2 should mirror this.
  - If D1/E want a uniform flag, it belongs on the shared `ChatModelError` base in `chat_ollama.py` (not my path).
  - `UpstreamRateLimited(retry_after_s)` covers 429 and 503, plus a 429 reported mid-stream. The retry delay is taken from the Retry-After header first (delta-seconds or an HTTP date), then from `google.rpc.RetryInfo.retryDelay` in the body. `retry_after_s` is `None` when there is no usable hint (no header and no RetryInfo, an unparseable value, or NaN); otherwise it is a finite float between 0 and `MAX_RETRY_AFTER_S` (300 s), with negative values raised to 0 and larger ones clamped.
  - `ModelConfigurationError` (502) covers 401 and 403, a 400 with `ErrorInfo.reason == API_KEY_INVALID` (this is how Gemini actually reports a bad key), 404 (unknown model), and a 400 with `FAILED_PRECONDITION` (unsupported region or billing). Retrying cannot fix these; an operator has to.
  - `ModelRequestRejected(ChatModelError, ValidationFailed)` covers 400 `INVALID_ARGUMENT` and 413. It is permanent. Its status stays 502 because our request to the provider was wrong, not the user's input.
  - `ResponseBlocked(stage, reason, usage)` (422) has `stage` set to `"prompt"` (from `promptFeedback.blockReason`) or `"response"` (from finishReason SAFETY, RECITATION, LANGUAGE, OTHER, BLOCKLIST, PROHIBITED_CONTENT, SPII, or one of the IMAGE_* reasons). `reason` is a fixed lowercase category. It replaces the final `UsageReport`, the same way `OutputTruncated` does.
  - `OutputLimitReached(usage)`, a subclass of the shared `OutputTruncated`, is raised on `MAX_TOKENS`. Its message says "output token limit reached" (on Gemini that is the `maxOutputTokens` ceiling, not the context window). Every request carries `generationConfig.maxOutputTokens`: `ANSWER_MAX_OUTPUT_TOKENS` (2048) for `stream()`, `STRUCTURED_MAX_OUTPUT_TOKENS` (4096) for `complete_json`, plus thinking headroom (the configured positive budget; 24 576 for `-1`/model default; 2048 after budget 0 was refused; 0 when budget 0 is sent), because thought tokens count as output. These are module constants, not settings (no frozen name exists); a future `RAGFORGE_GEMINI_MAX_OUTPUT_TOKENS` would replace them.
  - `QuotaExhausted` (a `ModelConfigurationError`, so permanent) replaces `UpstreamRateLimited` when a 429 carries a `google.rpc.QuotaFailure` whose `violations[].quotaId` names a per-day quota (matched on `PerDay` in an identifier-shaped id only; the id is never echoed). A per-minute quota stays `UpstreamRateLimited` with its RetryInfo delay. Without this, a spent free-tier daily quota would be retried every 300 s for hours and reported as "rate limited".
  - No exception raised by the adapter has a reachable `__context__` holding an httpx exception (whose `.request.headers` contain the key) or a `JSONDecodeError` (whose `.doc` holds the event / model text): those errors are raised after the `except` block, not with `from None`, which only suppresses display.
  - A `STOP` with no text and no tool call (thought parts do not count) raises plain `ChatModelError("model returned an empty answer")` instead of yielding only a `UsageReport` (round 5). Gemini does this occasionally, most often right after a `functionResponse` turn or when thinking consumed the output; passing it on would let the agent commit an empty draft. Nothing was emitted, so it is transient: D1 may retry the turn once. `complete_json` already raised on empty output. The Ollama adapter does not check this (not my path); D2 may want parity.
  - A `Role.tool` message whose `tool_call_id` answers no assistant tool call in the same history raises `ModelRequestRejected("tool result does not answer a tool call in this history")` before any request is sent (round 4; before that it was a bare `ValueError`, which fell outside the `ChatModelError` contract). The Ollama adapter tolerates such a result. Port-doc request for the integrator: `domain/ports.py` should state that every tool result must answer a call in the same history (D1 already guarantees it: only user/assistant text is committed and tool exchanges are kept whole).
  - An SSE line or event over 4 MiB raises plain `ChatModelError("oversized event from model provider")`. Since round 4 the cap is enforced on every network read (the adapter splits lines itself from `aiter_bytes()`), so an endless line without a newline cannot grow memory past the cap.
  - `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, `TOO_MANY_TOOL_CALLS`, an unknown finish reason, `FINISH_REASON_UNSPECIFIED` (the proto default, "no reason given": fails closed since round 3), a stream with no finish reason, and malformed events all raise a plain `ChatModelError` (or `StreamInterrupted` after output).
- Exception text contains only fixed categories: the HTTP code and Google's `error.status` enum. It never contains `error.message` or body text, because Google echoes prompt fragments there.

## Requests for the integrator / other modules
1. **D1 (agent): turn `ResponseBlocked` into an abstain.** The agent does not catch chat-model errors today, so a blocked answer currently reaches the API as a 422 problem.
   - Proposal: in the draft and regenerate nodes, catch `ResponseBlocked` and route to the existing abstain path, discarding any `TurnText` already streamed.
   - D1 can't import adapters (import-linter), so this needs either the error moved to `domain/errors.py` (for example `ModelRefused(RagforgeError)`, frozen, needs approval) or a match on `exc.status == 422` plus `type(exc).__name__`. I recommend moving it to domain, together with D2's old proposal for a domain `UpstreamError` (502).
   - Concrete frozen-change proposal: `domain/errors.py` gains `class ModelRefused(RagforgeError): status = 422; title = "Model response blocked"` with `stage: str`, `reason: str`, `usage: ChatTurnUsage | None`; `ResponseBlocked` then subclasses it (one-line change in my file). D1 catches `ModelRefused` in draft/regenerate and routes to abstain; the agent-level test ("a blocked draft becomes an abstain, streamed text discarded") belongs to D1 because only D1 owns the agent. Round-2 review asked for that test here; it cannot be written in my paths without editing the agent or domain.
   - Same treatment for `StreamInterrupted` (a domain `ModelStreamInterrupted`) if D1 wants to restart a turn after discarding streamed text.
2. **D1 (grounding): resolved inside G1 (round 6).** `UnsupportedSchema` is now a `RagforgeError` (500), not a `ValueError`, so `grounding.py`'s `except ValueError` no longer swallows a schema programming error into the malformed-verdict path (`test_g1_review_round6.py` proves it escapes that handler shape before any request). Narrowing that `except` is still good hygiene but no longer required.
3. **Docs/compose (integrator):**
   - ADR-0004 says `LLM_PROVIDER=openai|bedrock`; it should now say `ollama|gemini`.
   - `compose.yaml` must pass `RAGFORGE_LLM_PROVIDER`, `RAGFORGE_GEMINI_API_KEY`, `RAGFORGE_GEMINI_MODEL` and `RAGFORGE_GEMINI_THINKING_BUDGET` to the api service.
   - `.env.example` should list them, with an empty key.
   - Note that embeddings stay on Ollama, so the Ollama host is still required when gemini is selected.
4. **Config hardening (outside my scope):** pydantic echoes the raw input dict into `ValidationError` text for model-level validators. I checked this: `_worker_pool_fits` failing prints a truncated `{'db_api_password': 'supe...`. My Gemini check is a field validator so it echoes only the empty key, but the worker-pool check still leaks a prefix of secrets into startup logs. Fix: add `hide_input_in_errors=True` to `Settings.model_config`.
5. **Agent call order:** when Google sends a `functionCall.id`, the adapter keeps it in-process (with the thought signature, keyed by our minted id) and echoes it as `functionCall.id` and `functionResponse.id` on replay, so results pair by id. When Google sent none, or the entry was evicted / lives on another replica, pairing falls back to name and order: D1 must still append `Role.tool` messages in call order (same rule as Ollama). Our uuid ids never leave the process. Consecutive tool results are merged into one `user` turn.
6. **Read timeout and thinking:** thinking is silent on the wire, so a fixed 60 s would time out a large budget before the first byte. The per-request read timeout is the client's read timeout (60 s) plus an allowance sized from the CONFIGURED budget, capped at `MAX_THINKING_READ_S` (300 s) and never below the client's own value:
   - budget 0 (the default), sent: no allowance (60 s);
   - budget 0 refused by the model (the fallback dropped `thinkingConfig`): +30 s (90 s), for this retry and every later request. The model then thinks at its default level, which is "minimal" on the default model. Before round 4 this case got the full dynamic allowance (300 s) for the life of the process. An operator who runs a model that thinks heavily by default should set an explicit budget;
   - a positive budget: + `budget / 100` s, kept after a fallback;
   - -1 (dynamic) or `None` (model default by choice): +240 s, capped at 300 s.
   The API's own request timeout (if E sets one) must exceed this for thinking budgets.

## Things I could NOT verify against Google's docs (WebFetch, 2026-10-17)
The public docs now lead with the Interactions API (`/v1beta/interactions`), and the `generateContent` reference pages I fetched were truncated before the GenerationConfig, Part and Schema sections.
- **Confirmed from the docs:**
  - endpoint `:streamGenerateContent` with `alt=sse`;
  - the `x-goog-api-key` header;
  - request fields `contents`, `systemInstruction`, `tools[].functionDeclarations`, `generationConfig.responseMimeType` / `responseSchema` / `temperature`;
  - the `FinishReason` and `BlockReason` enums;
  - the `usageMetadata` field names;
  - `parameters` (an OpenAPI Schema subset) vs `parametersJsonSchema` being mutually exclusive.
- **From prior knowledge, not re-verified:**
  - `functionCall {name, args}` and `functionResponse {name, response: {output}}` parts sent in a `user` turn;
  - `thoughtSignature` on `functionCall` parts;
  - `generationConfig.thinkingConfig.thinkingBudget`;
  - the Schema `Type` enum spelled in upper case (`OBJECT`, `STRING`, ...);
  - `nullable`, `propertyOrdering`, and `format` support limited to date-time/enum for strings;
  - that an OBJECT with empty `properties` is rejected (so tools without arguments omit `parameters`).
- **Thinking on Gemini 3.x:** the docs list `thinkingLevel` (minimal/low/medium/high), and `gemini-3.5-flash-lite` defaults to "minimal". I could not confirm that `thinkingBudget: 0` is accepted there.
  - The adapter retries once without `thinkingConfig` only when a 400 is recognisably about it: a `google.rpc.BadRequest` field violation whose field path names `thinking_config`, or an anchored phrase (`thinking budget/config/level`, `only works in thinking mode`, `thinking is not supported`), never a bare "think".
  - The instance stops sending the budget ONLY when that retry is accepted (HTTP 200). A 400 caused by echoed user text fails the retry the same way, so the budget survives; end users cannot flip it.
  - If 3.x silently ignores the budget, the latency is whatever the model's default level gives.
  - A `RAGFORGE_GEMINI_THINKING_LEVEL` setting would be the follow-up, but it is not frozen, so I did not add it.
- **Thought signatures:** these are cached in-process (an LRU of 1024, keyed by our minted tool-call id; since round 5 a replay refreshes an entry's recency, so an active conversation is not evicted by insertion age alone. Before that it was FIFO in practice, although documented as LRU). When a replayed model turn's first `functionCall` has no cached signature (eviction, restart, another replica, a delayed tool approval), the adapter sends Google's placeholder `skip_thought_signature_validator`, on the first call of each model turn only (parallel calls carry one signature, on the first part).
  - The placeholder value is from prior knowledge of Google's Gemini 3 thought-signature guide. The current docs pages (thinking, function-calling) lead with the Interactions API and did not show it on 2026-10-17. The extended live smoke test checks it: its second turn runs through a fresh adapter with no cached signature.
  - A durable signature still needs provider metadata on `ToolCall` (a frozen domain change). The placeholder only removes the hard failure; the model loses its prior reasoning context for that turn.
- **Default model `gemini-3.5-flash-lite`:** the docs' models page recommends it for low latency and lists `gemini-2.5-*` as legacy-access only. Override it with `RAGFORGE_GEMINI_MODEL`.

**Live check: NOT yet run (still open after round 4, 2026-10-17). This is the main remaining risk and an owner action.** `RAGFORGE_GEMINI_API_KEY` was again not set in the agent's shell environment. I deliberately did not lift the key out of `.env`: the test is designed to run only when the owner opts in by exporting the key, it calls a billed external API, and the working agreement keeps the owner's secrets out of agent hands. The owner runs it once: `RAGFORGE_GEMINI_API_KEY=... uv run pytest packages/core/tests/module_g1/test_g1_live_smoke.py -v` (prefer typing the key via a hidden prompt, e.g. `read -s RAGFORGE_GEMINI_API_KEY; export RAGFORGE_GEMINI_API_KEY`). Two tests: (a) text turn, tool call, the tool result replayed through a FRESH adapter (settles `functionResponse {name, response: {output}}` and the `skip_thought_signature_validator` placeholder), and a JSON call (settles upper-case Schema types and string enums); (b) `thinking_budget=-1`, tool call and replay through the SAME adapter (settles that a real signature round-trips and that a thinking model answers within the stretched read timeout). Running with the default settings also settles whether `thinkingBudget: 0` is accepted on `gemini-3.5-flash-lite` (if not, the one-shot fallback drops it and the run still passes). Record the date and result here.

## Schema conversion (`to_gemini_schema`)
- **Kept:**
  - `type`, mapped to upper case; a type list containing `null`, or pydantic's `anyOf [X, null]`, becomes `nullable: true`;
  - `properties`, `required` (filtered to existing properties), `items`, `anyOf` with several concrete branches;
  - `enum` (string-only) and `const` (as a one-value enum);
  - `description`, `minItems`/`maxItems`, `minLength`/`maxLength`, `pattern`, `minimum`/`maximum`, `minProperties`/`maxProperties`;
  - supported `format`s;
  - `propertyOrdering`, which is added so output order is deterministic.
- **Inlined:** local, non-recursive `$ref` into `$defs` / `definitions`.
- **Dropped (this only loosens what the model is told; callers validate):** `$schema`, `title`, `additionalProperties`, `default`, unsupported `format`s (for example `uuid`), non-string enums, exclusive bounds.
- **Rejected with `UnsupportedSchema`:** an `enum` with no `type` whose values are not all strings (inferring STRING would then drop the enum and tell the model "string" for an integer field), `oneOf`, `allOf`, `not`, conditionals, tuple `items`, recursive or missing `$ref`, several non-null types, a node with no type, and an OBJECT with no properties at any depth (map / dict-like fields, `additionalProperties` only). Gemini rejects those with a permanent 400, so they fail at conversion instead. The one exception is a tool's top-level parameters with no properties: the tool is declared without `parameters`.

## API key format
The adapter accepts only printable ASCII with no whitespace, after stripping surrounding whitespace. Anything else raises `ModelConfigurationError("model provider API key has invalid characters")` at construction, which `build_chat_model` runs during bootstrap, so the failure still happens at startup. Before this check, a key pasted with a smart quote made httpx raise `UnicodeEncodeError`, whose args held the whole key.
- This check is deliberately NOT in `Settings`. Pydantic puts the rejected input value into `ValidationError` text (verified: `input_value='<the key>'`), so a field validator would print the key into startup logs. It can move into Settings once `hide_input_in_errors=True` is set on `Settings.model_config` (request 4 above, outside my scope).

## Needs live evidence (owner / integrator action; not faked)
Nothing below can be settled by respx tests; each needs a real key or the live stack.
1. **Live Gemini smoke test, default settings** (settles `thinkingBudget: 0` on `gemini-3.5-flash-lite`, `functionResponse.response.output`, the `skip_thought_signature_validator` placeholder, upper-case Schema types, and that the second turn after a tool result is not an empty STOP). Run from the repo root, typing the key at a hidden prompt:
   `read -s RAGFORGE_GEMINI_API_KEY && export RAGFORGE_GEMINI_API_KEY && uv run pytest packages/core/tests/module_g1/test_g1_live_smoke.py -v; unset RAGFORGE_GEMINI_API_KEY`
   Both tests (default variant and `thinking_budget=-1` variant) run in that one command. Record the date and pass/fail in the "Live check" paragraph above. If the default variant fails with a 400 about `thinking_config` even after the fallback, add the unfrozen `RAGFORGE_GEMINI_THINKING_LEVEL` setting as a follow-up.
2. **End-to-end with the stack** (after the integrator lands request 3 in compose / `.env.example`): with `RAGFORGE_LLM_PROVIDER=gemini` and the key provided through the G2 setup flow, `make up && curl -fsS localhost:8000/readyz`, then one question through the API that triggers a tool call.
3. **Blocked answer becomes an abstain** (request 1): needs the domain `ModelRefused` change and a D1 agent-level test; not testable inside G1's paths. Once `domain.errors.ModelRefused` exists, G1's change is one line: `class ResponseBlocked(ChatModelError, ModelRefused)`.
4. **Output ceiling vs. thinking (round 6):** that thought tokens count toward `maxOutputTokens` (the headroom assumes so; if they do not, the headroom only loosens the ceiling) and that 2048 tokens never cut a normal answer. Covered by run 1 (no `OutputLimitReached` in either variant); also grep the eval run for `output token limit`.
5. **Per-day quota shape (round 6):** the `quotaId` naming (`...PerDay...`) is from Google's documented free-tier errors, not observed here. Settled only when a real daily-quota 429 is seen; no command forces one safely.
6. **Latency of the Gemini arm:** p95 < 10 s with budget 0, measured by the eval runner against the live stack (after item 2): `uv run ragforge-eval run --label gemini-flash-lite-b0 --split dev --notes "provider=gemini budget=0"` then `uv run ragforge-eval compare evals/private/runs/*.jsonl`.

## Round 5 (2026-10-17)
- Fixed: empty `STOP` is now a transient `ChatModelError` (`test_g1_review_round5.py`, three tests incl. the after-a-tool-result shape and the thought-only shape); the signature cache is a true LRU (`_recall` refreshes on replay; a test proves a replayed entry survives 1023 newer inserts at the real size of 1024 while the never-replayed one is evicted, and a mutation check confirmed the test fails without the refresh).
- Tests that only inspect the request now return a minimal text answer (`answered()` in `g1_support.py`), since a bare STOP is no longer a success.
- Not changed in G1 (integrator-owned, see requests 1, 3 and 4): domain `ModelRefused`, ADR-0004 wording, compose / `.env.example` variables, `hide_input_in_errors=True`.

## Round 6 (2026-10-17)
- `UnsupportedSchema` is a `RagforgeError` (500), not a `ValueError` or `ChatModelError`: a schema bug can no longer be caught by grounding's `except ValueError`.
- `maxOutputTokens` on every request (2048 answer / 4096 structured, plus thinking headroom); `MAX_TOKENS` raises `OutputLimitReached(OutputTruncated)` with an output-limit message.
- Per-day `QuotaFailure` 429 -> `QuotaExhausted` (permanent), before or mid-stream; per-minute stays a retryable rate limit; a non-identifier `quotaId` is ignored.
- Transport errors, malformed events and invalid structured output are raised outside the `except` block: `__context__` is `None`, and tests walk the full cause/context chain (before output, and through `StreamInterrupted` after output) asserting no httpx exception and no key.
- Thought signatures over 64 KiB are dropped (replayed as the placeholder, nothing cached).
- An untyped `enum` is inferred as STRING only when every value is a string; otherwise `UnsupportedSchema`.
- Tests: `test_g1_review_round6.py` (24 tests). Not changed (integrator-owned): domain `ModelRefused`, ADR-0004, compose / `.env.example`, `hide_input_in_errors=True`.

## Live evidence (2026-10-17, integrator)
`test_g1_live_smoke.py` against the real Gemini API with the owner's free-tier key (model `gemini-3.5-flash`, thinking budget 0; key read from `.env`, never printed): **2/2 passed**. That covers `test_live_stream_tool_call_and_json` (streaming text, a function-call round trip, JSON-schema mode) and `test_live_dynamic_thinking_replays_a_real_signature` (budget -1, a real thoughtSignature replayed). This settles the wire-format doubts the reviewers raised (functionResponse shape, Schema types, thinking config).
Also verified live: `make setup` accepted the newer `AQ.`-format key after the charset fix; `/readyz` 200; the `RAGFORGE_GEMINI_*` variables are present only in api and worker (0 in postgres).
