# Kafka Basics at Acme Shipping

## Topics and partitions
A topic is split into partitions. Each partition is an ordered, append-only log stored on a broker. Ordering is guaranteed only within one partition, never across partitions.

## Keys and routing
The producer hashes the record key to pick a partition, so records with the same key always land in the same partition. Acme keys shipment events by `shipment_id` so every event of one shipment stays in order.

## Consumer groups
Each partition is read by exactly one consumer in a group. Adding consumers beyond the partition count leaves the extra consumers idle.
