"""One source of truth with ragforge_core.config.Settings: the default model and the validation
rules that setup and `--check` (doctor) apply must equal what api and worker enforce at startup.
Otherwise a doctor "ok" could still be followed by a crash at startup, and the compose default
would silently override the project default."""

from __future__ import annotations

import re
from typing import Annotated, Any

import pytest
import yaml  # type: ignore[import-untyped]  # pyyaml is locked (transitively); no stubs
from g2_support import REPO_ROOT, load_setup_provider
from pydantic import TypeAdapter, ValidationError

from ragforge_core.config import Settings

sp = load_setup_provider()


def _field_adapter(name: str) -> TypeAdapter[Any]:
    """Validate a single value with exactly the type and constraints Settings declares."""
    field = Settings.model_fields[name]
    return TypeAdapter(Annotated[(field.annotation, *field.metadata)])  # type: ignore[arg-type]


def _settings_accepts(name: str, raw: str) -> bool:
    try:
        _field_adapter(name).validate_python(raw)
    except ValidationError:
        return False
    return True


def _settings_default_model() -> str:
    default = Settings.model_fields["gemini_model"].default
    assert isinstance(default, str)
    return default


def test_compose_default_model_is_the_settings_default() -> None:
    loaded = yaml.safe_load((REPO_ROOT / "compose.yaml").read_text())
    value = loaded["services"]["api"]["environment"]["RAGFORGE_GEMINI_MODEL"]

    assert value == f"${{RAGFORGE_GEMINI_MODEL:-{_settings_default_model()}}}"


def test_env_example_and_setup_suggestion_use_the_settings_default() -> None:
    example = (REPO_ROOT / ".env.example").read_text()

    assert f"# RAGFORGE_GEMINI_MODEL={_settings_default_model()}\n" in example
    assert _settings_default_model() == sp.DEFAULT_GEMINI_MODEL
    assert sp._PREFERRED_MODELS[0] == _settings_default_model()


def test_default_model_is_suggested_when_the_key_offers_it() -> None:
    names = ["gemini-2.5-flash", _settings_default_model(), "gemini-2.5-pro"]
    models = [sp.GeminiModel(name=n, display_name=n) for n in names]

    assert sp.suggest_model(models) == _settings_default_model()


@pytest.mark.parametrize(
    "raw",
    ["-2", "-1", "0", "128", "32768", "32769", "99999", "1.5", "abc", "-", "007", "+5", " 5"],
)
def test_budget_rule_matches_settings_on_every_value_setup_could_keep(raw: str) -> None:
    # setup keeps or rejects exactly what Settings would; anything it keeps must start the api.
    if sp.valid_budget(raw):
        assert _settings_accepts("gemini_thinking_budget", raw), raw
    for bound in (str(sp.BUDGET_MIN), str(sp.BUDGET_MAX)):
        assert sp.valid_budget(bound)
        assert _settings_accepts("gemini_thinking_budget", bound)
    for outside in (str(sp.BUDGET_MIN - 1), str(sp.BUDGET_MAX + 1)):
        assert not sp.valid_budget(outside)
        assert not _settings_accepts("gemini_thinking_budget", outside)


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3.5-flash-lite",
        "models/gemini-2.5-flash",
        ".hidden",
        "-flag",
        "_x",
        "a" * 128,
        "a" * 129,
        "gemini 2.5",
        "gemini/2.5",
        "",
    ],
)
def test_model_rule_matches_settings(model: str) -> None:
    script_ok = bool(sp._SETTINGS_MODEL_RE.fullmatch(model))

    assert script_ok == _settings_accepts("gemini_model", model)


def test_listed_model_ids_are_always_valid_settings_values() -> None:
    # Anything _parse_model keeps is written to .env as-is; Settings must accept it.
    for name in (".evil", "-x", "ok-model.1"):
        parsed = sp._parse_model(
            {"name": f"models/{name}", "supportedGenerationMethods": ["generateContent"]}
        )
        if parsed is not None:
            assert _settings_accepts("gemini_model", parsed.name)
    assert re.fullmatch(sp._MODEL_RE, ".evil") is None
