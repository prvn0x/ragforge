# Idempotency Keys

## The problem
A client that times out cannot know whether its payment request was applied, so it retries. Without protection the payment is charged twice.

## The Acme design
The client sends an `Idempotency-Key` header. The server stores the key with the response for 24 hours. A retry with the same key returns the stored response instead of charging again.

## Code
```python
def charge(request, key):
    saved = store.get(key)
    if saved is not None:
        return saved
    result = payments.charge(request)
    store.put(key, result, ttl_hours=24)
    return result
```
