# Postgres Indexing

## B-tree indexes
The default B-tree index serves equality and range queries. A composite index on `(tenant_id, created_at)` serves queries that filter by tenant and sort by time, but not queries that filter by time alone.

## Partial indexes
A partial index covers only the rows matching a condition. Acme indexes `shipments(status) WHERE status = 'in_transit'` because only about 3 percent of rows are in transit.

## Checking a plan
`EXPLAIN ANALYZE` runs the query and shows the real plan with timings. A sequential scan on a large table in a hot path usually means a missing index.
