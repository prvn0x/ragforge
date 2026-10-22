"""Shared source pages for module_ag tests (not a test module, so mypy sees it once: the
Makefile type-checks each module's test_*.py files in isolation, and a test importing another
test module made mypy find that file under two module names)."""

from __future__ import annotations

from module_d1.d1_fakes import make_hit
from ragforge_core.agent.coverage import valid_labels
from ragforge_core.agent.state import HitDict, hit_to_state

HITS: list[HitDict] = [
    hit_to_state(
        make_hit(
            1,
            title="Kafka",
            text="Kafka stores records in partitions. The cluster has 3 brokers. "
            "The default port is 9092. Replication factor is 3 for topics.",
        )
    ),
    hit_to_state(
        make_hit(
            2,
            title="Postgres",
            text="Postgres uses MVCC. Vacuum reclaims dead tuples. "
            "Autovacuum runs every 60 seconds by default.",
        )
    ),
    hit_to_state(
        make_hit(
            3,
            title="Kafka CLI",
            # Code as the Notion normalizer emits it: a fenced block and an inline span.
            text="Create a topic with:\n```bash\n"
            "kafka-topics --create --topic logs --partitions 3\n```\n"
            "List topics with `kafka-topics --list`.",
        )
    ),
    hit_to_state(
        make_hit(
            4,
            title="Build",
            # A two-line block (review of the parser rewrite, round 1): code joined by a Unicode
            # line separator must not match it.
            text="Clear the build cache with:\n```sh\ncd /tmp/build\nrm -rf cache\n```",
        )
    ),
    hit_to_state(
        make_hit(
            5,
            title="Deploy",
            # Relative indentation that matters (blind review round 2, P2): re-indenting
            # cleanup() under the `if` with mixed tabs and spaces must not match this block.
            text="Deploy guarded by the check:\n```python\nif ok:\n    deploy()\ncleanup()\n```",
        )
    ),
    hit_to_state(
        make_hit(
            6,
            title="Ops",
            # Prose next to hidden elements (review rounds 2-8). Since the prose-quote exception
            # was removed (round 9) none of it backs code; kept as a regression page.
            text="Set **retention.ms** to 0 for compacted topics. "
            "Clean with rm -rf /`var/cache` or rm -rf /[tmp](http://x) or ~~rm -rf /home~~ "
            "nightly. Undo with git reset --hard HEAD~1 first.",
        )
    ),
]
SHOWN = (1, 2, 3, 4, 5, 6)
VALID = valid_labels(SHOWN, 0)
