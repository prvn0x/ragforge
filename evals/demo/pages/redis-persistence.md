# Redis Persistence

## RDB snapshots
RDB writes a point-in-time snapshot of the dataset to disk at intervals. It is compact and fast to restore, but writes since the last snapshot are lost on a crash.

## AOF
The append-only file logs every write. With `appendfsync everysec`, at most about one second of writes is lost on a crash. AOF files grow, so Redis rewrites them in the background.

## What Acme runs
The session store uses AOF with `appendfsync everysec`. The rate-limit counters use no persistence at all, because losing them only resets limits.
