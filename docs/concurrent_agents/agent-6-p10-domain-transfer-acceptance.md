# Branch Acceptance Note: codex/p10-domain-transfer

## Scope
- Implement declarative `DomainPack` loading and validation.
- Add non-restaurant founder prompt packs (`saas`, `ecommerce`).
- Extend evaluation reporting with per-domain breakdown and transfer deltas.
- Add domain selection via config/CLI without code edits.

## Delivered Outputs
- `src/user_signal_mining_agents/domain_packs.py` for pack parsing, selection, and prompt loading.
- New founder prompt assets:
  - `founder_prompts/domain_packs.json`
  - `founder_prompts/saas.json`
  - `founder_prompts/ecommerce.json`
- Domain-aware runner/report/CLI wiring.
- Unit tests for domain pack parsing, selection, and domain-transfer report behavior.

## Validation
- Unit tests cover:
  - Duplicate/invalid domain packs.
  - Domain selection behavior.
  - Prompt-domain consistency checks.
  - Domain filter behavior in evaluation.
  - Report sections for domain quality and transfer deltas.

## Exit Criteria Check
- Domain selection configurable without code edits: satisfied (`USM_DOMAIN_PACKS_PATH`, `USM_ACTIVE_DOMAINS`, `--domain`).
- Reports include per-domain quality breakdown and transfer deltas: satisfied.
- Tests validate domain pack parsing and selection behavior: satisfied.
