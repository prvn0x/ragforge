"""The eval layer (ADR-0017). Pure metric and statistics functions, dataset and run-file schemas,
the judge port and the runners. Imports only ragforge_client and ragforge_contracts (ADR-0005);
the Streamlit Eval page reads run and label files through this package.

Module ownership (Phase 4a build):
    schema   - LOCKED by the integrator: run file v2, datasets, labels, judge results, summaries
    metrics  - B2: pure metric functions (E-R, E-G, E-V, E-I, E-A)
    stats    - B2: pure statistics (bootstrap, exact tests, kappa, Page's L, MDE)
    gate     - B2: the regression gate
    judge    - B3: the judge port, Ollama/Gemini adapters, cache
    datasets - B3: D0/D1/D3/D4/D5 loading, the D3 mutation generator, sealing
    runners  - B3: the E-R/E-G/E-V/E-I/E-A runners (write run files)
    cli      - B3: the `ragforge-eval2` subcommands

The Phase 2b harness (`ragforge_client.evalrun`) stays as is and keeps reading v1 runs.
"""
