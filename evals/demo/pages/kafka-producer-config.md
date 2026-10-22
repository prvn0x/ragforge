# Kafka Producer Configuration

## Durability settings
Acme producers set `acks=all` so a write is acknowledged only after every in-sync replica has it. Together with `min.insync.replicas=2` on the topic, a single broker loss never loses an acknowledged event.

## Ordering and retries
Retries can reorder records when more than one request is in flight. Setting `max.in.flight.requests.per.connection` to 1 gives strict ordering at a throughput cost. With `enable.idempotence=true`, ordering is kept with up to 5 in-flight requests.

## Example
```java
Properties props = new Properties();
props.put("acks", "all");
props.put("enable.idempotence", "true");
props.put("max.in.flight.requests.per.connection", "5");
KafkaProducer<String, String> producer = new KafkaProducer<>(props);
```
