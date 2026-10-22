"""The Eval page (ADR-0017 §6): pure view models, local file access and the Streamlit views.

    config  - where the run, dataset and label files live (flags after `--`, or env)
    data    - read-only loading of runs, summaries, gate verdicts, judge results, calibrations,
              datasets; the label store factory (the page's ONLY write goes through LabelWriter)
    model   - pure: formatting, held-out filtering, compare rules, flips, the blind queue items
    notes   - the shown notes, fetched on demand through the SDK and digest-checked, never stored
    runs, compare, drill, labels, quality - the Streamlit views (thin: they only draw)

Everything is read locally through `ragforge_client.evals` (ADR-0005: the UI may import the
client, never the core). Note text is never written anywhere by this page.
"""
