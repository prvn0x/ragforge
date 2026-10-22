# Outbox Pattern

## Dual writes
Writing to Postgres and then publishing to Kafka is a dual write: if the service crashes between the two, the database and the event stream disagree.

## How the outbox works
The service writes the business row and an outbox row in one database transaction. A separate relay reads the outbox table and publishes each row to Kafka, marking it sent.

## Delivery guarantee
The relay may publish a row twice after a crash, so delivery is at least once. Consumers must be idempotent, for example by deduplicating on the event id.
