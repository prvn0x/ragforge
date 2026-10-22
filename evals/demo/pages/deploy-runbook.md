# Deploy Runbook

## Rolling deploys
Services deploy with a rolling update: a quarter of the pods at a time, each must pass its readiness probe before the next batch starts.

## Rollback
A deploy is rolled back if the error rate rises above 2 percent for 5 minutes. Rollback redeploys the previous image tag; database migrations must therefore stay backward compatible for one release.

## Dangerous commands
```bash
# DANGER: never run this against production
if [ "$ENV" != "dev" ]; then exit 1; fi
kafka-topics --delete --topic tracking-events
```
