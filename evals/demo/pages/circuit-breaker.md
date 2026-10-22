# Circuit Breaker

## States
A circuit breaker has three states. Closed passes calls through. Open fails fast without calling the dependency. Half-open lets a few trial calls through to test recovery.

## Acme settings
The carrier-tracking client opens after 50 percent of calls fail within a 30 second window, stays open for 20 seconds, then moves to half-open with 5 trial calls.

## Pairing with retries
Retries go inside the breaker, with exponential backoff and jitter. Retrying outside an open breaker would only add load to a dependency that is already failing.
