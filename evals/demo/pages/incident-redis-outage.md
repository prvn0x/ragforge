# Incident: Session Store Outage

## What happened
The Redis primary of the session store ran out of memory and stopped accepting writes for 12 minutes. Users were logged out.

## Root cause
A bug stored a full shipment manifest in every session instead of an id. Memory filled within two hours of the deploy, and the session keys had no TTL, so nothing was evicted.

## Fix
Sessions now store only ids, every session key has a TTL, and an alert fires when memory use passes 80 percent.
