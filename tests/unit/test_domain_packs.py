from __future__ import annotations

import json
from pathlib import Path

import pytest

from user_signal_mining_agents import domain_packs


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_domain_packs_rejects_duplicate_domain_ids(tmp_settings, tmp_path: Path) -> None:
    domain_packs_path = _write_json(
        tmp_path / "domain_packs.json",
        [
            {
                "domain_id": "restaurants",
                "title": "Restaurants",
                "founder_prompts_path": str(tmp_settings.founder_prompts_path),
            },
            {
                "domain_id": "restaurants",
                "title": "Duplicate",
                "founder_prompts_path": str(tmp_settings.founder_prompts_path),
            },
        ],
    )
    settings = tmp_settings.model_copy(update={"domain_packs_path": domain_packs_path})

    with pytest.raises(ValueError, match="Duplicate domain_id"):
        domain_packs.load_domain_packs(settings)


def test_resolve_domain_packs_uses_enabled_by_default(tmp_settings, tmp_path: Path) -> None:
    saas_prompts = _write_json(
        tmp_path / "saas_prompts.json",
        [{"id": "s1", "statement": "Q", "domain": "saas"}],
    )
    domain_packs_path = _write_json(
        tmp_path / "domain_packs.json",
        [
            {
                "domain_id": "restaurants",
                "title": "Restaurants",
                "founder_prompts_path": str(tmp_settings.founder_prompts_path),
                "enabled": True,
            },
            {
                "domain_id": "saas",
                "title": "SaaS",
                "founder_prompts_path": str(saas_prompts),
                "enabled": False,
            },
        ],
    )
    settings = tmp_settings.model_copy(update={"domain_packs_path": domain_packs_path})

    selected = domain_packs.resolve_domain_packs(settings)
    assert [pack.domain_id for pack in selected] == ["restaurants"]

    explicit = domain_packs.resolve_domain_packs(settings, domain_ids=["saas"])
    assert [pack.domain_id for pack in explicit] == ["saas"]


def test_load_founder_prompts_filters_selected_domains(tmp_settings, tmp_path: Path) -> None:
    saas_prompts = _write_json(
        tmp_path / "saas_prompts.json",
        [{"id": "s1", "statement": "Q", "domain": "saas"}],
    )
    domain_packs_path = _write_json(
        tmp_path / "domain_packs.json",
        [
            {
                "domain_id": "restaurants",
                "title": "Restaurants",
                "founder_prompts_path": str(tmp_settings.founder_prompts_path),
                "enabled": True,
            },
            {
                "domain_id": "saas",
                "title": "SaaS",
                "founder_prompts_path": str(saas_prompts),
                "enabled": True,
            },
        ],
    )
    settings = tmp_settings.model_copy(update={"domain_packs_path": domain_packs_path})

    prompts = domain_packs.load_founder_prompts(settings, domain_ids=["saas"])

    assert len(prompts) == 1
    assert prompts[0].id == "s1"
    assert prompts[0].domain == "saas"


def test_load_founder_prompts_rejects_domain_mismatch(tmp_settings, tmp_path: Path) -> None:
    mismatched_prompts = _write_json(
        tmp_path / "mismatch.json",
        [{"id": "m1", "statement": "Q", "domain": "restaurants"}],
    )
    domain_packs_path = _write_json(
        tmp_path / "domain_packs.json",
        [
            {
                "domain_id": "saas",
                "title": "SaaS",
                "founder_prompts_path": str(mismatched_prompts),
                "enabled": True,
            }
        ],
    )
    settings = tmp_settings.model_copy(update={"domain_packs_path": domain_packs_path})

    with pytest.raises(ValueError, match="domain mismatch"):
        domain_packs.load_founder_prompts(settings)


def test_load_founder_prompts_rejects_duplicate_prompt_ids(tmp_settings, tmp_path: Path) -> None:
    saas_prompts = _write_json(
        tmp_path / "saas_prompts.json",
        [{"id": "p1", "statement": "Q", "domain": "saas"}],
    )
    domain_packs_path = _write_json(
        tmp_path / "domain_packs.json",
        [
            {
                "domain_id": "restaurants",
                "title": "Restaurants",
                "founder_prompts_path": str(tmp_settings.founder_prompts_path),
                "enabled": True,
            },
            {
                "domain_id": "saas",
                "title": "SaaS",
                "founder_prompts_path": str(saas_prompts),
                "enabled": True,
            },
        ],
    )
    settings = tmp_settings.model_copy(update={"domain_packs_path": domain_packs_path})

    with pytest.raises(ValueError, match="Duplicate founder prompt id"):
        domain_packs.load_founder_prompts(settings)
