from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pydantic import TypeAdapter

from .config import ROOT_DIR, ensure_scaffold_directories, get_settings
from .schemas import FounderPrompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="usm",
        description="Utilities for the founder-grounded review mining project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "show-config",
        help="Print the resolved application settings.",
    )

    subparsers.add_parser(
        "bootstrap",
        help="Create the default data, prompt, and artifact directories.",
    )

    validate_parser = subparsers.add_parser(
        "validate-founder-prompts",
        help="Validate a founder prompt JSON file against the shared schema.",
    )
    validate_parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Optional path to the founder prompt JSON file.",
    )

    return parser


def cmd_show_config() -> int:
    settings = get_settings()
    print(json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def cmd_bootstrap() -> int:
    settings = get_settings()
    directories = ensure_scaffold_directories(settings)
    print("Created or verified directories:")
    for directory in directories:
        try:
            display_path = directory.relative_to(ROOT_DIR)
        except ValueError:
            display_path = directory
        print(f"- {display_path}")
    return 0


def cmd_validate_founder_prompts(path: Path | None) -> int:
    settings = get_settings()
    prompt_path = path or settings.founder_prompts_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Founder prompt file not found: {prompt_path}")
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    prompts = TypeAdapter(list[FounderPrompt]).validate_python(data)
    print(f"Validated {len(prompts)} founder prompts from {prompt_path}.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "show-config":
        return cmd_show_config()
    if args.command == "bootstrap":
        return cmd_bootstrap()
    if args.command == "validate-founder-prompts":
        return cmd_validate_founder_prompts(args.path)

    parser.error(f"Unknown command: {args.command}")
    return 2
