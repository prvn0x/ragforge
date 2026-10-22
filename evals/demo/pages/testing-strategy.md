# Testing Strategy

## The test pyramid
Most tests are fast unit tests, fewer are integration tests against real Postgres and Kafka in containers, and a handful are end-to-end tests of the main user journeys.

## Contract tests
Services that call each other share consumer-driven contract tests, so a provider change that breaks a consumer fails in the provider's build.

## Flaky tests
A test that fails without a code change is quarantined within a day and fixed or deleted within a week. Quarantined tests still run but do not block merges.
