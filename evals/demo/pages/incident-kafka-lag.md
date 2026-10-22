# Incident: Shipment Tracking Lag (2025)

## What happened
Consumer lag on the tracking topic grew to 40 minutes during peak season. Customers saw stale shipment positions.

## Root cause
A new enrichment step called an external geocoding API inside the poll loop. Slow calls pushed the time between polls past `max.poll.interval.ms`, so consumers were kicked out of the group and the group kept rebalancing.

## Fix
The geocoding calls moved to a bounded worker pool, the poll loop only hands records off, and a timeout of 2 seconds was set on the external call.
