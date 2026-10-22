# API Rate Limiting

## Token bucket
Each API key has a token bucket of 100 tokens that refills at 10 tokens per second. A request spends one token; an empty bucket returns HTTP 429 with a `Retry-After` header.

## Why per key
Limits are per API key, not per IP address, because many customers share one corporate NAT address.

## Storage
The buckets live in Redis so every API replica sees the same counts. The update is a single Lua script, so the read and the decrement are atomic.
