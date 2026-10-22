"""Judge prompt files (ADR-0017 §4). A file is `<name>_v<version>.md`, its id `<name>@<version>`:
the id goes into every JudgeKey, so editing a file's wording means adding a new version file, never
changing an old one (the cache and calibration of the old version would silently lie)."""
