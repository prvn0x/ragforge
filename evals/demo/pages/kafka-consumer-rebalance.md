# Kafka Consumer Rebalance

## What triggers a rebalance
A rebalance happens when a consumer joins, leaves, or misses heartbeats for longer than `session.timeout.ms`. A consumer that takes longer than `max.poll.interval.ms` between polls is also removed from the group.

## Eager vs cooperative
The eager protocol revokes every partition from every consumer, which pauses the whole group. The cooperative sticky assignor moves only the partitions that change owner, so most consumers keep working during a rebalance.

## Static membership
Setting `group.instance.id` gives a consumer a stable identity, so a quick restart during a deploy does not trigger a rebalance at all.
