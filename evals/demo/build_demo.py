"""Build the D0 demo datasets (ADR-0017 §2 D0) from the committed pages.

    uv run python evals/demo/build_demo.py

Runs the SERVER's own file parser and chunker over evals/demo/pages, so every chunk key in the
generated files is exactly what ingestion will produce. Writes:
  chunks.json     - the chunk-key manifest (page key, heading path, digest, occurrence, text)
  verifier.json   - D3 VerifierSet: clean items + planted mutations, each with its span and check
  interview.json  - D4 InterviewSet: 4 seeds, each with the four frozen answer tiers
Re-run after a chunker change: changed keys show up as a diff (the harness marks them stale).
This is an integrator tool (it imports ragforge_core); the eval package itself never does.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ragforge_client.evals.schema import (
    AnswerTier,
    InterviewSeed,
    InterviewSet,
    Mutation,
    Span,
    VerifierItem,
    VerifierSet,
)
from ragforge_contracts.api import ChunkKey, PageKey
from ragforge_core.adapters.connectors.files.parser import SandboxedDocumentParser
from ragforge_core.bootstrap import _chunker  # the exact chunker the server builds

HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"


@dataclass(frozen=True)
class DemoChunk:
    key: ChunkKey
    text: str


async def chunk_pages() -> list[DemoChunk]:
    parser, chunker = SandboxedDocumentParser(), _chunker()
    out: list[DemoChunk] = []
    for path in sorted(PAGES.glob("*.md")):
        doc = await parser.parse(path.name, path.read_bytes())
        seen: Counter[tuple[tuple[str, ...], str]] = Counter()
        for chunk in sorted(chunker.chunk(doc), key=lambda c: c.ordinal):
            sha = chunk.display_sha256.hex()
            occurrence = seen[(chunk.heading_path, sha)]
            seen[(chunk.heading_path, sha)] += 1
            key = ChunkKey(
                page=PageKey(source_kind="file", key=path.name),
                heading_path=list(chunk.heading_path),
                display_sha256=sha,
                occurrence=occurrence,
            )
            out.append(DemoChunk(key, chunk.display_text))
    return out


def find(chunks: list[DemoChunk], page: str, needle: str) -> DemoChunk:
    hits = [c for c in chunks if c.key.page.key == page and needle in c.text]
    if len(hits) != 1:
        raise SystemExit(f"{page}: {needle!r} matched {len(hits)} chunks, expected 1")
    return hits[0]


# ------------------------------------------------------------------ D3 spec
# (id, question, [(page, needle-in-chunk)], clean answer). Answers cite S<n> in `shown` order.
CLEAN: list[tuple[str, str, list[tuple[str, str]], str]] = [
    (
        "c-durability",
        "Which settings stop Acme losing acknowledged Kafka events?",
        [("kafka-producer-config.md", "acks=all")],
        "Acme producers set `acks=all`, so a write is acknowledged only after every in-sync "
        "replica has it [S1]. With `min.insync.replicas=2` on the topic, a single broker loss "
        "never loses an acknowledged event [S1].",
    ),
    (
        "c-ordering",
        "How do you keep Kafka producer ordering with retries?",
        [("kafka-producer-config.md", "Retries can reorder")],
        "Retries can reorder records when more than one request is in flight [S1]. Setting "
        "`max.in.flight.requests.per.connection` to 1 gives strict ordering at a throughput "
        "cost [S1].",
    ),
    (
        "c-aof",
        "How much data can Redis AOF lose?",
        [("redis-persistence.md", "at most about one second")],
        "With `appendfsync everysec`, at most about one second of writes is lost on a crash [S1].",
    ),
    (
        "c-stampede",
        "How does Acme avoid Redis stampedes?",
        [("redis-caching.md", "jitter")],
        "Every key gets a TTL of 10 minutes plus random jitter of up to 60 seconds, so many "
        "keys never expire at the same moment [S1]. Acme guards hot keys with a short lock so "
        "only one request reloads it [S1].",
    ),
    (
        "c-breaker",
        "What are Acme's circuit breaker settings?",
        [("circuit-breaker.md", "carrier-tracking")],
        "The carrier-tracking client opens after 50 percent of calls fail within a 30 second "
        "window [S1]. It stays open for 20 seconds, then moves to half-open with 5 trial calls "
        "[S1].",
    ),
    (
        "c-lag",
        "What caused the tracking lag incident?",
        [("incident-kafka-lag.md", "geocoding API")],
        "A new enrichment step called an external geocoding API inside the poll loop [S1]. Slow "
        "calls pushed the time between polls past `max.poll.interval.ms`, so the group kept "
        "rebalancing [S1].",
    ),
    (
        "c-idempotency",
        "How do idempotency keys prevent double charges?",
        [("idempotency-keys.md", "Idempotency-Key"), ("idempotency-keys.md", "def charge")],
        "The server stores the key with the response for 24 hours [S1]. A retry with the same "
        "key returns the stored response instead of charging again [S1].",
    ),
    (
        "c-deploy",
        "How does Acme guard the topic delete command?",
        [("deploy-runbook.md", "DANGER")],
        "The runbook guards it with a check on the environment:\n\n"
        '```bash\nif [ "$ENV" != "dev" ]; then exit 1; fi\n'
        "kafka-topics --delete --topic tracking-events\n```\n\n[S1]",
    ),
]

# (id, source id, mutation, old, new, validity check). `old` must occur once in the clean answer.
PLANTED: list[tuple[str, str, Mutation, str, str, str]] = [
    (
        "p-typo-1",
        "c-ordering",
        Mutation.identifier_typo,
        "max.in.flight.requests.per.connection",
        "max.in.flight.request.per.connection",
        "token_absent_from_shown",
    ),
    (
        "p-typo-2",
        "c-lag",
        Mutation.identifier_typo,
        "max.poll.interval.ms",
        "max.poll.interval.sec",
        "token_absent_from_shown",
    ),
    (
        "p-number-1",
        "c-aof",
        Mutation.number_drift,
        "about one second",
        "about ten seconds",
        "number_absent_from_shown",
    ),
    (
        "p-number-2",
        "c-breaker",
        Mutation.number_drift,
        "50 percent",
        "80 percent",
        "number_absent_from_shown",
    ),
    (
        "p-entity-1",
        "c-durability",
        Mutation.entity_swap,
        "in-sync replica",
        "Redis replica",
        "entity_absent_from_cited",
    ),
    (
        "p-entity-2",
        "c-lag",
        Mutation.entity_swap,
        "geocoding API",
        "payments API",
        "entity_absent_from_cited",
    ),
    (
        "p-negation-1",
        "c-stampede",
        Mutation.negation_flip,
        "so many keys never expire",
        "so many keys always expire",
        "owner_review",
    ),
    (
        "p-fabricated-1",
        "c-aof",
        Mutation.fabricated_sentence,
        "on a crash [S1].",
        "on a crash [S1]. Acme replicates AOF files to three regions.",
        "no_label_no_attribution",
    ),
    (
        "p-fabricated-2",
        "c-idempotency",
        Mutation.fabricated_sentence,
        "instead of charging again [S1].",
        "instead of charging again [S1]. The keys are signed with HMAC-SHA512.",
        "no_label_no_attribution",
    ),
    (
        "p-moved-1",
        "c-idempotency",
        Mutation.citation_moved,
        "instead of charging again [S1].",
        "instead of charging again [S2].",
        "new_page_lacks_content",
    ),
    (
        "p-guard-1",
        "c-deploy",
        Mutation.guard_dropped,
        'if [ "$ENV" != "dev" ]; then exit 1; fi\n',
        "",
        "line_in_source_block",
    ),
    (
        "p-typo-3",
        "c-durability",
        Mutation.identifier_typo,
        "min.insync.replicas=2",
        "min.insync.replica=2",
        "token_absent_from_shown",
    ),
]


_TOKEN_CHARS = re.compile(r"[A-Za-z0-9_.\-]")


def _widen(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) to whole tokens (identifier characters, dots, dashes), so a one-letter
    edit inside a name spans the whole name, which is what a reader sees as wrong."""
    while start > 0 and _TOKEN_CHARS.match(text[start - 1]):
        start -= 1
    while end < len(text) and _TOKEN_CHARS.match(text[end]):
        end += 1
    return start, end


def planted_span(clean: str, old: str, new: str) -> tuple[str, Span]:
    if clean.count(old) != 1:
        raise SystemExit(f"{old!r} must occur exactly once in the clean answer")
    start = clean.index(old)
    mutated = clean.replace(old, new)
    if not new:  # a deletion (guard_dropped): the line that now follows the cut
        line_end = mutated.find("\n", start)
        return mutated, Span(start=start, end=line_end if line_end > start else start + 1)
    # what changed: drop the common prefix and suffix of old and new, then widen to tokens
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    q = 0
    while q < min(len(old), len(new)) - p and old[-1 - q] == new[-1 - q]:
        q += 1
    lo, hi = _widen(mutated, start + p, max(start + p, start + len(new) - q))
    while lo < hi and mutated[lo] in ". ":  # an inserted sentence starts at its first word
        lo += 1
    if hi <= lo:
        hi = lo + 1
    return mutated, Span(start=lo, end=hi)


def validity_ok(check: str, item_new: str, shown_texts: list[str]) -> bool:
    """The mechanical checks of ADR-0017 §2 D3. owner_review is confirmed by the owner later."""
    new_tokens = set(re.findall(r"[A-Za-z0-9_.]+", item_new))
    if check in ("token_absent_from_shown", "number_absent_from_shown"):
        # whole tokens: "min.insync.replica" is NOT in a page that says "min.insync.replicas"
        source_tokens = {t for s in shown_texts for t in re.findall(r"[A-Za-z0-9_.]+", s)}
        source_tokens |= {t.strip(".") for t in source_tokens}
        return any(t.strip(".") not in source_tokens for t in new_tokens)
    if check == "entity_absent_from_cited":
        return not any(item_new in s for s in shown_texts)
    return True  # coverage-layer checks are structural (no label / moved label / dropped line)


def build_verifier(chunks: list[DemoChunk]) -> VerifierSet:
    items: list[VerifierItem] = []
    clean_by_id: dict[str, tuple[str, list[DemoChunk], str]] = {}
    for cid, question, shown_spec, answer in CLEAN:
        shown = [find(chunks, page, needle) for page, needle in shown_spec]
        if cid == "c-idempotency" and len(shown) == 2:
            pass  # S2 exists so the "moved citation" mutation has a wrong page to move to
        clean_by_id[cid] = (question, shown, answer)
        items.append(
            VerifierItem(
                id=cid,
                source_id=cid,
                question=question,
                answer=answer,
                shown=[c.key for c in shown],
            )
        )
    for pid, source, mutation, old, new, check in PLANTED:
        question, shown, clean = clean_by_id[source]
        mutated, span = planted_span(clean, old, new)
        if not validity_ok(check, new, [c.text for c in shown]):
            raise SystemExit(f"{pid}: the planted text is in the notes; not an error")
        items.append(
            VerifierItem(
                id=pid,
                source_id=source,
                question=question,
                answer=mutated,
                shown=[c.key for c in shown],
                mutation=mutation,
                span=span,
                validity_check=check,
                owner_reviewed=mutation is not Mutation.negation_flip,
            )
        )
    return VerifierSet(name="demo", items=items)


# ------------------------------------------------------------------ D4 spec
SEEDS = [
    dict(
        id="s-rebalance",
        topic="Kafka consumer rebalance",
        kind="concept",
        seed_level="senior_sde_1",
        question="Why is the cooperative sticky assignor better than eager rebalancing, and "
        "how does static membership help during deploys?",
        expected_points=[
            "Eager revokes every partition from every consumer and pauses the group",
            "Cooperative sticky moves only the partitions that change owner",
            "group.instance.id gives a stable identity so a quick restart does not rebalance",
        ],
        notes=[
            ("kafka-consumer-rebalance.md", "Eager vs cooperative"),
            ("kafka-consumer-rebalance.md", "group.instance.id"),
        ],
        gold="Eager rebalancing revokes every partition from every consumer, so the whole "
        "group pauses. The cooperative sticky assignor only moves the partitions that change "
        "owner, so most consumers keep working. Static membership with group.instance.id gives "
        "each consumer a stable identity, so a quick restart in a deploy doesn't rebalance.",
        partial="Cooperative only moves the partitions that change owner, eager stops everyone.",
        wrong="Cooperative rebalancing is faster because it compresses the partitions before "
        "sending them to the new consumer.",
        misconception_sentence="Static membership makes Kafka keep a consumer's partitions "
        "forever, even after it crashes for good.",
    ),
    dict(
        id="s-outbox",
        topic="outbox pattern",
        kind="design",
        seed_level="senior_sde_2",
        question="A service writes to Postgres and publishes to Kafka. What goes wrong, how "
        "does the outbox pattern fix it, and what must consumers do?",
        expected_points=[
            "A crash between the two writes leaves the database and the stream disagreeing",
            "The business row and an outbox row are written in one transaction; a relay publishes",
            "Delivery is at least once, so consumers must be idempotent",
        ],
        notes=[("outbox-pattern.md", "Dual writes"), ("outbox-pattern.md", "A separate relay")],
        gold="It's a dual write: if the service crashes between the Postgres write and the "
        "Kafka publish, the two disagree. The outbox writes the business row and an outbox row "
        "in one transaction, and a relay publishes outbox rows to Kafka. The relay can publish "
        "twice after a crash, so it's at least once and consumers must dedupe on the event id.",
        partial="Write an outbox row in the same transaction and have a relay publish it.",
        wrong="Use a distributed lock around both writes so they happen together.",
        misconception_sentence="With the outbox pattern every event is delivered exactly once, "
        "so consumers never see duplicates.",
    ),
    dict(
        id="s-ratelimit",
        topic="API rate limiting",
        kind="concept",
        seed_level="sde_2",
        question="How does Acme rate limit its API and why per key?",
        expected_points=[
            "A token bucket of 100 tokens refilling at 10 per second; empty returns 429",
            "Per key because many customers share one NAT address",
            "Buckets in Redis, updated atomically by a Lua script",
        ],
        notes=[("api-rate-limiting.md", "Token bucket"), ("api-rate-limiting.md", "Lua")],
        gold="Each API key has a token bucket of 100 tokens refilling at 10 per second; an "
        "empty bucket returns 429 with Retry-After. It's per key, not per IP, because many "
        "customers share one corporate NAT. The buckets live in Redis and a Lua script does the "
        "read and decrement atomically so every replica agrees.",
        partial="Token bucket per API key, and a 429 when it's empty.",
        wrong="Acme limits each IP address to one request per second using nginx.",
        misconception_sentence="The buckets are kept in each API replica's memory, so no "
        "shared store is needed.",
    ),
    dict(
        id="s-idempotency",
        topic="idempotency keys",
        kind="code",
        seed_level="sde_1",
        question="Write the server-side logic that stops a retried payment from being charged "
        "twice.",
        expected_points=[
            "Look the Idempotency-Key up first and return the saved response if present",
            "Otherwise charge, then store the result under the key for 24 hours",
        ],
        notes=[("idempotency-keys.md", "def charge")],
        gold="Check the store for the idempotency key first; if there's a saved response, "
        "return it. Otherwise call payments.charge, store the result under the key with a "
        "24 hour TTL, and return it.",
        partial="Store the key so a retry is ignored.",
        wrong="Retry the payment until it succeeds and refund any duplicates later.",
        misconception_sentence="The key must be stored before charging, with no response, so "
        "a retry is simply rejected with an error.",
    ),
]


def build_interview(chunks: list[DemoChunk]) -> InterviewSet:
    seeds: list[InterviewSeed] = []
    for spec in SEEDS:
        keys = [find(chunks, page, needle).key for page, needle in spec["notes"]]
        gold, bad = str(spec["gold"]), str(spec["misconception_sentence"])
        seeds.append(
            InterviewSeed(
                id=str(spec["id"]),
                topic=str(spec["topic"]),
                kind=spec["kind"],
                seed_level=spec["seed_level"],
                question=str(spec["question"]),
                expected_points=list(spec["expected_points"]),
                chunk_keys=list({k.model_dump_json(): k for k in keys}.values()),
                answers={
                    AnswerTier.gold: gold,
                    AnswerTier.partial: str(spec["partial"]),
                    AnswerTier.wrong: str(spec["wrong"]),
                    AnswerTier.misconception: f"{gold} {bad}",
                },
                misconception_sentence=bad,
            )
        )
    return InterviewSet(name="demo", seeds=seeds)


def write(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


async def main() -> None:
    chunks = await chunk_pages()
    write(
        "chunks.json",
        [{"key": c.key.model_dump(mode="json"), "text": c.text} for c in chunks],
    )
    write("verifier.json", build_verifier(chunks).model_dump(mode="json"))
    write("interview.json", build_interview(chunks).model_dump(mode="json"))
    print(f"{len(chunks)} chunks; verifier and interview sets written", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
