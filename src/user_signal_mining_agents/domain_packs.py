"""Domain pack loading and founder prompt selection."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from .config import ROOT_DIR, Settings
from .schemas import DomainPack, FounderPrompt


def parse_domain_ids(raw: str | None) -> list[str] | None:
    """Parse a comma-separated domain list into a normalized id list."""

    if raw is None:
        return None

    ids = [part.strip() for part in raw.split(",") if part.strip()]
    return ids or None


def _default_pack(settings: Settings) -> DomainPack:
    return DomainPack(
        domain_id="restaurants",
        title="Restaurants",
        founder_prompts_path=str(settings.founder_prompts_path),
        evaluation_notes="Backwards-compatible fallback when no domain pack file is configured.",
        enabled=True,
    )


def load_domain_packs(settings: Settings) -> list[DomainPack]:
    """Load domain pack declarations, falling back to legacy single-pack mode."""

    if not settings.domain_packs_path.exists():
        return [_default_pack(settings)]

    data = json.loads(settings.domain_packs_path.read_text(encoding="utf-8"))
    packs = TypeAdapter(list[DomainPack]).validate_python(data)

    seen: set[str] = set()
    duplicates: set[str] = set()
    for pack in packs:
        if pack.domain_id in seen:
            duplicates.add(pack.domain_id)
        seen.add(pack.domain_id)

    if duplicates:
        duplicate_ids = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate domain_id values in {settings.domain_packs_path}: {duplicate_ids}")

    return packs


def resolve_domain_packs(
    settings: Settings,
    *,
    domain_ids: list[str] | None = None,
) -> list[DomainPack]:
    """Resolve the active/selected domain packs."""

    packs = load_domain_packs(settings)
    requested_ids = domain_ids or parse_domain_ids(settings.active_domains)

    if requested_ids:
        unique_requested = list(dict.fromkeys(requested_ids))
        by_id = {pack.domain_id: pack for pack in packs}
        missing = [domain_id for domain_id in unique_requested if domain_id not in by_id]
        if missing:
            missing_ids = ", ".join(missing)
            raise ValueError(f"Unknown domain id(s): {missing_ids}")
        selected = [by_id[domain_id] for domain_id in unique_requested]
    else:
        selected = [pack for pack in packs if pack.enabled]

    if not selected:
        raise ValueError("No domain packs selected. Enable packs or pass --domain.")

    return selected


def _resolve_pack_prompt_path(settings: Settings, pack: DomainPack) -> Path:
    raw_path = Path(pack.founder_prompts_path)
    if raw_path.is_absolute():
        return raw_path

    root_candidate = ROOT_DIR / raw_path
    if root_candidate.exists():
        return root_candidate

    pack_relative_candidate = settings.domain_packs_path.parent / raw_path
    if pack_relative_candidate.exists():
        return pack_relative_candidate

    return root_candidate


def load_founder_prompts(
    settings: Settings,
    *,
    domain_ids: list[str] | None = None,
) -> list[FounderPrompt]:
    """Load founder prompts from selected domain packs and validate consistency."""

    selected_packs = resolve_domain_packs(settings, domain_ids=domain_ids)
    prompts: list[FounderPrompt] = []
    seen_prompt_ids: set[str] = set()

    for pack in selected_packs:
        prompt_path = _resolve_pack_prompt_path(settings, pack)
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Founder prompt file not found for domain {pack.domain_id!r}: {prompt_path}"
            )

        data = json.loads(prompt_path.read_text(encoding="utf-8"))
        domain_prompts = TypeAdapter(list[FounderPrompt]).validate_python(data)

        for prompt in domain_prompts:
            if prompt.domain != pack.domain_id:
                raise ValueError(
                    "Founder prompt domain mismatch: "
                    f"prompt {prompt.id!r} declares {prompt.domain!r} but is in pack {pack.domain_id!r}"
                )
            if prompt.id in seen_prompt_ids:
                raise ValueError(f"Duplicate founder prompt id across selected packs: {prompt.id!r}")
            seen_prompt_ids.add(prompt.id)

        prompts.extend(domain_prompts)

    return prompts
