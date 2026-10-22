"""Static guarantees on the obs stack's files (ADR-0018 R2-6/R2-7): what is exposed on the host,
what Grafana may phone home, how images are pinned, what Prometheus scrapes, and that the
provisioned dashboard is a well-formed board with the four ADR rows."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ragforge_core.observability import catalog, db_collector
from test_obs_stack_support import (
    COMPOSE_SNIPPET,
    COMPOSE_YAML,
    OBS_DIR,
    OBS_SERVICES,
    PROMETHEUS_YML,
    PROVISIONING,
    REPO_ROOT,
    dashboard,
    load_yaml,
    merged_into_compose_yaml,
    obs_compose,
    panels,
)

DIGEST = re.compile(r"^[a-z0-9./-]+:[\w.-]+@sha256:[0-9a-f]{64}$")
ADR_ROWS = ["Answers", "Models", "Retrieval & jobs", "Freshness & health"]


@pytest.fixture(scope="module")
def snippet() -> dict[str, Any]:
    return obs_compose()


def _service(snippet: dict[str, Any], name: str) -> dict[str, Any]:
    service: dict[str, Any] = snippet["services"][name]
    return service


def _mounts(service: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Container path -> (source, mode) for the short `source:target[:mode]` syntax."""
    mounts = {}
    for mount in service["volumes"]:
        source, target, *mode = mount.split(":")
        mounts[target] = (source, mode[0] if mode else "rw")
    return mounts


# --------------------------------------------------------------------------- compose snippet


def test_the_stack_is_opt_in_behind_the_obs_profile(snippet: dict[str, Any]) -> None:
    assert set(snippet["services"]) == OBS_SERVICES
    for name in OBS_SERVICES:
        assert _service(snippet, name)["profiles"] == ["obs"], name


def test_prometheus_publishes_no_port(snippet: dict[str, Any]) -> None:
    prometheus = _service(snippet, "prometheus")

    assert "ports" not in prometheus
    assert "network_mode" not in prometheus  # `host` would expose 9090 without a ports entry


def test_grafana_is_published_on_loopback_only(snippet: dict[str, Any]) -> None:
    assert _service(snippet, "grafana")["ports"] == ["127.0.0.1:3000:3000"]


def test_prometheus_has_no_admin_or_lifecycle_api(snippet: dict[str, Any]) -> None:
    command = _service(snippet, "prometheus")["command"]

    assert not [flag for flag in command if "admin-api" in flag or "lifecycle" in flag]
    assert "--config.file=/etc/prometheus/prometheus.yml" in command  # nothing inherited


def test_prometheus_retention_is_bounded_by_time_and_size(snippet: dict[str, Any]) -> None:
    command = _service(snippet, "prometheus")["command"]

    assert "--storage.tsdb.retention.time=15d" in command
    assert "--storage.tsdb.retention.size=512MB" in command


@pytest.mark.parametrize(("name", "limit"), [("prometheus", "256m"), ("grafana", "512m")])
def test_memory_is_capped(snippet: dict[str, Any], name: str, limit: str) -> None:
    assert _service(snippet, name)["mem_limit"] == limit


@pytest.mark.parametrize("name", sorted(OBS_SERVICES))
def test_images_are_pinned_by_digest(snippet: dict[str, Any], name: str) -> None:
    assert DIGEST.match(_service(snippet, name)["image"]), _service(snippet, name)["image"]


@pytest.mark.parametrize("name", sorted(OBS_SERVICES))
def test_containers_are_hardened(snippet: dict[str, Any], name: str) -> None:
    service = _service(snippet, name)

    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert "env_file" not in service  # it would hand the service every secret in .env


@pytest.mark.parametrize("name", sorted(OBS_SERVICES))
def test_data_lives_in_named_volumes_and_config_is_read_only(
    snippet: dict[str, Any], name: str
) -> None:
    for target, (source, mode) in _mounts(_service(snippet, name)).items():
        if source.startswith("./"):
            assert mode == "ro", target
            assert (REPO_ROOT / source).exists(), f"{source}: no such path in the repo"
            assert (REPO_ROOT / source).resolve().is_relative_to(OBS_DIR), source
        else:
            assert source in snippet["volumes"], f"{source}: not a declared named volume"


def test_grafana_phones_nothing_home_and_admits_only_the_admin(snippet: dict[str, Any]) -> None:
    env = _service(snippet, "grafana")["environment"]

    assert {
        "GF_ANALYTICS_REPORTING_ENABLED": "false",
        "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
        "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES": "false",
        "GF_NEWS_NEWS_FEED_ENABLED": "false",
        "GF_AUTH_ANONYMOUS_ENABLED": "false",
        "GF_USERS_ALLOW_SIGN_UP": "false",
        "GF_SERVER_ENFORCE_DOMAIN": "true",
        "GF_SNAPSHOTS_EXTERNAL_ENABLED": "false",
        "GF_SECURITY_DISABLE_GRAVATAR": "true",
        # Background plugin signing-key sync (GETs grafana.com at start, then every 10 days):
        # no analytics flag covers it.
        "GF_PLUGINS_PUBLIC_KEY_RETRIEVAL_DISABLED": "true",
        "GF_PLUGINS_PREINSTALL_DISABLED": "true",
        # Grafana's own /metrics is registered before auth: nobody scrapes it, so not served.
        "GF_METRICS_ENABLED": "false",
    }.items() <= env.items()


def test_every_grafana_com_endpoint_fails_closed_on_loopback(snippet: dict[str, Any]) -> None:
    # A call that no flag above disables (or that an upgrade adds) is refused inside the
    # container instead of leaving the machine.
    env = _service(snippet, "grafana")["environment"]
    endpoints = ("GF_GRAFANA_COM_URL", "GF_GRAFANA_COM_API_URL", "GF_GRAFANA_NET_URL")

    for name in endpoints:
        assert env[name].startswith("http://127.0.0.1:9"), name


def test_grafana_admin_password_comes_only_from_the_generated_env_var(
    snippet: dict[str, Any],
) -> None:
    env = _service(snippet, "grafana")["environment"]

    assert env["GF_SECURITY_ADMIN_PASSWORD"] == "${RAGFORGE_GRAFANA_ADMIN_PASSWORD:?run make env}"
    # No other RAGForge secret reaches either obs service.
    text = json.dumps(snippet["services"])
    assert set(re.findall(r"\$\{(\w+)", text)) == {"RAGFORGE_GRAFANA_ADMIN_PASSWORD"}


def test_prometheus_mounts_config_where_its_relative_rule_glob_resolves(
    snippet: dict[str, Any],
) -> None:
    mounts = _mounts(_service(snippet, "prometheus"))

    assert mounts["/etc/prometheus/prometheus.yml"] == (
        "./docker/observability/prometheus.yml",
        "ro",
    )
    assert mounts["/etc/prometheus/rules"] == ("./docker/observability/rules", "ro")
    assert load_yaml(PROMETHEUS_YML)["rule_files"] == ["rules/*.yml"]


# --------------------------------------------------------------------------- prometheus.yml


def _jobs() -> dict[str, dict[str, Any]]:
    return {job["job_name"]: job for job in load_yaml(PROMETHEUS_YML)["scrape_configs"]}


def test_scrapes_every_15_seconds() -> None:
    assert load_yaml(PROMETHEUS_YML)["global"]["scrape_interval"] == "15s"


def _seconds(duration: str) -> int:
    assert re.fullmatch(r"\d+s", duration), duration
    return int(duration[:-1])


def test_scrape_timeout_outlasts_both_db_reads_and_fits_the_interval() -> None:
    # The DB collector reads queue_stats then connector_freshness, each bounded by TIMEOUT_S.
    # A timeout below their sum turns a hung Postgres into `up == 0` (the wrong alert).
    settings = load_yaml(PROMETHEUS_YML)["global"]
    timeout = _seconds(settings["scrape_timeout"])

    assert timeout > 2 * db_collector.TIMEOUT_S
    assert timeout <= _seconds(settings["scrape_interval"])


def test_scrapes_the_api_on_its_internal_metrics_port() -> None:
    api = _jobs()["ragforge-api"]

    assert api["static_configs"] == [{"targets": [f"api:{catalog.API_METRICS_PORT}"]}]


def test_discovers_every_worker_replica_by_dns() -> None:
    worker = _jobs()["ragforge-worker"]

    assert worker["dns_sd_configs"] == [
        {"names": ["worker"], "type": "A", "port": catalog.WORKER_METRICS_PORT,
         "refresh_interval": "30s"}
    ]  # fmt: skip
    assert "static_configs" not in worker  # a static target would hit one replica only


def test_scrapes_nothing_else() -> None:
    assert set(_jobs()) == {"ragforge-api", "ragforge-worker"}


# --------------------------------------------------------------------------- grafana


def test_datasource_is_the_uid_every_panel_uses() -> None:
    (source,) = load_yaml(PROVISIONING / "datasources" / "prometheus.yml")["datasources"]
    used = {
        json.dumps(panel["datasource"], sort_keys=True)
        for panel in panels(dashboard())
        if "datasource" in panel
    } | {
        json.dumps(target["datasource"], sort_keys=True)
        for panel in panels(dashboard())
        for target in panel.get("targets", [])
    }

    assert source["url"] == "http://prometheus:9090"
    assert source["access"] == "proxy"  # browsers reach only Grafana
    assert used == {json.dumps({"type": "prometheus", "uid": source["uid"]}, sort_keys=True)}


def test_dashboards_are_provisioned_read_only_from_the_mounted_folder(
    snippet: dict[str, Any],
) -> None:
    (provider,) = load_yaml(PROVISIONING / "dashboards" / "ragforge.yml")["providers"]
    mounts = _mounts(_service(snippet, "grafana"))

    assert provider["type"] == "file"
    assert provider["allowUiUpdates"] is False
    assert mounts[provider["options"]["path"]] == (
        "./docker/observability/grafana/dashboards",
        "ro",
    )
    assert mounts["/etc/grafana/provisioning"] == (
        "./docker/observability/grafana/provisioning",
        "ro",
    )


def test_dashboard_has_the_four_adr_rows_in_order() -> None:
    rows = [p["title"] for p in dashboard()["panels"] if p["type"] == "row"]

    assert rows == ADR_ROWS


def test_every_row_has_panels() -> None:
    current, counts = None, dict.fromkeys(ADR_ROWS, 0)
    for panel in dashboard()["panels"]:
        if panel["type"] == "row":
            current = panel["title"]
        else:
            assert current is not None, f"{panel['title']} sits above the first row"
            counts[current] += 1

    assert min(counts.values()) >= 3, counts


def test_dashboard_panels_have_unique_ids_and_ref_ids() -> None:
    all_panels = list(panels(dashboard()))
    ids = [p["id"] for p in all_panels]

    assert len(ids) == len(set(ids))
    for panel in all_panels:
        refs = [t["refId"] for t in panel.get("targets", [])]
        assert len(refs) == len(set(refs)), panel["title"]
        assert panel["type"] in {"row", "alertlist"} or refs, f"{panel['title']}: no query"


def test_dashboard_panels_fit_the_grid_without_overlapping() -> None:
    cells: dict[tuple[int, int], str] = {}
    for panel in dashboard()["panels"]:
        pos = panel["gridPos"]
        assert pos["x"] >= 0
        assert pos["x"] + pos["w"] <= 24, panel["title"]
        for x in range(pos["x"], pos["x"] + pos["w"]):
            for y in range(pos["y"], pos["y"] + pos["h"]):
                assert (x, y) not in cells, f"{panel['title']} overlaps {cells[(x, y)]}"
                cells[(x, y)] = panel["title"]


def test_dashboard_lists_the_prometheus_alerts() -> None:
    (alert_list,) = [p for p in dashboard()["panels"] if p["type"] == "alertlist"]
    (source,) = load_yaml(PROVISIONING / "datasources" / "prometheus.yml")["datasources"]

    assert alert_list["options"]["datasource"] == source["name"]
    assert source["jsonData"]["manageAlerts"] is True  # lists data-source rules (read-only)
    assert alert_list["options"]["stateFilter"]["firing"] is True


def test_dashboard_links_to_grafanas_alert_rules_page() -> None:
    # Prometheus has no host port, so if the alert list panel's datasource option is wrong the
    # only other alert UI is Grafana's own rules page: one click away, same origin.
    urls = [link["url"] for link in dashboard()["links"]]

    assert urls == ["/alerting/list"]


def test_panels_eval_traffic_can_move_say_so() -> None:
    # Model and retrieval metrics carry no surface label (a catalogue gap), so an eval run moves
    # these panels even with surface=chat selected; the panel must not pretend otherwise.
    unfiltered = re.compile(r"ragforge_(llm|embed|retrieval)_")
    flagged = [
        panel
        for panel in panels(dashboard())
        if any(unfiltered.search(t["expr"]) for t in panel.get("targets", []))
    ]

    assert flagged
    for panel in flagged:
        assert "eval and demo traffic" in panel.get("description", ""), panel["title"]


def test_dashboard_is_stable_and_identified() -> None:
    board = dashboard()

    assert board["uid"] == "ragforge-overview"
    assert board["editable"] is False
    assert isinstance(board["schemaVersion"], int)
    assert "id" not in board  # an id in a provisioned file collides with Grafana's own


# --------------------------------------------------------------------------- merged compose


def test_compose_yaml_matches_the_snippet_once_merged() -> None:
    # After the integrator pastes the services, the snippet must not drift from what runs.
    if not merged_into_compose_yaml():
        pytest.skip("obs services not merged into compose.yaml yet (the snippet is checked)")
    snippet = load_yaml(COMPOSE_SNIPPET)

    assert obs_compose()["services"] == snippet["services"]
    assert snippet["volumes"].items() <= obs_compose()["volumes"].items()


def _fake_env(tmp_path: Path) -> Path:
    """An env file with a fake value for every variable either file references (never .env)."""
    names: set[str] = set()
    for path in (COMPOSE_YAML, COMPOSE_SNIPPET):
        names |= set(re.findall(r"\$\{(\w+)", path.read_text(encoding="utf-8")))
    env_file = tmp_path / "fake.env"
    env_file.write_text("".join(f"{name}=fake-{name.lower()}\n" for name in sorted(names)))
    return env_file


@pytest.mark.integration
def test_snippet_merges_into_compose_yaml(tmp_path: Path) -> None:
    # `docker compose config` resolves the merge read-only: no daemon, nothing created. The
    # shell environment is reduced so an exported RAGFORGE_* never leaks in, and --env-file
    # replaces the real .env, which is never read.
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker CLI not installed")
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    files = ["-f", str(COMPOSE_YAML)]
    if not merged_into_compose_yaml():
        files += ["-f", str(COMPOSE_SNIPPET)]
    args = ["compose", "--env-file", str(_fake_env(tmp_path)), *files, "--profile", "obs"]
    proc = subprocess.run(  # noqa: S603 - the resolved docker binary, read-only `config`
        [docker, *args, "config", "--format", "json"], capture_output=True, env=env, timeout=60
    )
    assert proc.returncode == 0, proc.stderr.decode()[-2000:]
    services = json.loads(proc.stdout)["services"]

    assert set(services) >= OBS_SERVICES
    published = {
        (name, port.get("host_ip"), port.get("published"))
        for name, service in services.items()
        for port in service.get("ports", [])
        if name in OBS_SERVICES
    }
    assert published == {("grafana", "127.0.0.1", "3000")}
    grafana_env = services["grafana"]["environment"]
    assert grafana_env["GF_SECURITY_ADMIN_PASSWORD"] == "fake-ragforge_grafana_admin_password"
    assert not any(k.startswith("RAGFORGE_") for k in grafana_env)
