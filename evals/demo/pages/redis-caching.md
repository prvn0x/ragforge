# Redis Caching

## Cache-aside
Acme uses cache-aside: the service reads Redis first, and on a miss it loads the row from Postgres and writes it to Redis with a TTL. Writes go to Postgres and then delete the cache key.

## TTL and stampedes
Every key gets a TTL of 10 minutes plus random jitter of up to 60 seconds, so many keys never expire at the same moment. A hot key that expires can cause a stampede; Acme guards hot keys with a short lock so only one request reloads it.

## Eviction
The cluster runs with `maxmemory-policy allkeys-lru`, so the least recently used keys are evicted when memory is full.
