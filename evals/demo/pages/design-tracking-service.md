# Design: Shipment Tracking Service

## Requirements
The service ingests about 20,000 position updates per second from trucks and answers "where is my shipment" queries with a p99 under 200 milliseconds.

## High level design
Trucks send updates to an ingest API, which publishes them to Kafka keyed by `shipment_id`. A consumer keeps the latest position per shipment in Redis. Queries read Redis; history goes to a columnar store for analytics.

## Trade-offs
Keying by shipment keeps each shipment's updates in order but can create hot partitions for very busy shipments. Redis holds only the latest position, so a Redis loss is rebuilt by replaying the last hour of the topic.
