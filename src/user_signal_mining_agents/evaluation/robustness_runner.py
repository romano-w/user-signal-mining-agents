"""Robustness evaluation runner with perturbation suites and release gates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .. import console as con
from ..agents.judge import judge_named_pair
from ..agents.variant_pipeline import run_variant_pipeline
from ..config import Settings, get_settings
from ..schemas import FounderPrompt, JudgeResult, JudgeScores, RobustnessCase, SynthesisResult


RUBRIC_DIMS = [
    "relevance",
    "contradiction",
    "coverage",
    "distinctiveness",
]

DEFAULT_PROMPT_SAMPLE_SIZE = 3


class RobustnessGateThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_overall_drop: float = Field(ge=0.0)
    max_dimension_drop: float = Field(ge=0.0)
    min_case_pass_rate: float = Field(ge=0.0, le=1.0)


class RobustnessCaseOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    case_id: str
    family: str
    description: str
    expected_behavior: str
    perturbed_prompt_id: str
    perturbed_statement: str
    control_scores: JudgeScores
    perturbed_scores: JudgeScores
    dimension_deltas: dict[str, float] = Field(default_factory=dict)
    delta_overall: float
    passed: bool
    failure_reasons: list[str] = Field(default_factory=list)


class RobustnessSuiteSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    suite_description: str
    prompt_ids: list[str] = Field(default_factory=list)
    thresholds: RobustnessGateThresholds
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    gate_passed: bool
    failed_case_keys: list[str] = Field(default_factory=list)
    gate_failure_reasons: list[str] = Field(default_factory=list)
    outcomes: list[RobustnessCaseOutcome] = Field(default_factory=list)


@dataclass(frozen=True)
class RobustnessSuiteSpec:
    suite_id: str
    description: str
    thresholds: RobustnessGateThresholds
    cases: tuple[RobustnessCase, ...]


_ADVERSARIAL_CORE_CASES = (
    RobustnessCase(
        case_id="rb_negation_flip",
        family="negation",
        description="Flip one polarity marker in the founder prompt statement.",
        transform_spec={"strategy": "negation_flip"},
        expected_behavior="Maintain grounded analysis while preserving contradiction awareness.",
    ),
    RobustnessCase(
        case_id="rb_noise_injection",
        family="noise",
        description="Inject deterministic lexical noise tokens into the prompt statement.",
        transform_spec={
            "strategy": "token_noise",
            "seed": 17,
            "injection_rate": 0.2,
            "token": "[noise]",
        },
        expected_behavior="Keep quality stable with only minor score movement under noisy wording.",
    ),
    RobustnessCase(
        case_id="rb_context_shift",
        family="context_shift",
        description="Apply deterministic context shifts to adjacent-domain wording.",
        transform_spec={
            "strategy": "domain_shift",
            "context_prefix": "Assume this founder asks the same question for a SaaS product.",
            "replacements": [
                ["restaurant", "product"],
                ["diners", "users"],
                ["takeout", "self-serve onboarding"],
                ["menu", "feature set"],
                ["staff", "support team"],
            ],
        },
        expected_behavior="Retain structured, evidence-driven reasoning under contextual reframing.",
    ),
)

_SUITE_REGISTRY: dict[str, RobustnessSuiteSpec] = {
    "adversarial_core": RobustnessSuiteSpec(
        suite_id="adversarial_core",
        description="Core perturbation suite: negation, lexical noise, and context shifts.",
        thresholds=RobustnessGateThresholds(
            max_overall_drop=0.5,
            max_dimension_drop=1.0,
            min_case_pass_rate=1.0,
        ),
        cases=_ADVERSARIAL_CORE_CASES,
    ),
    "default": RobustnessSuiteSpec(
        suite_id="default",
        description="Default robustness suite alias for adversarial_core.",
        thresholds=RobustnessGateThresholds(
            max_overall_drop=0.5,
            max_dimension_drop=1.0,
            min_case_pass_rate=1.0,
        ),
        cases=_ADVERSARIAL_CORE_CASES,
    ),
}


def list_suite_ids() -> list[str]:
    return sorted(_SUITE_REGISTRY)


def get_suite_spec(suite_id: str) -> RobustnessSuiteSpec:
    spec = _SUITE_REGISTRY.get(suite_id)
    if spec is None:
        valid = ", ".join(sorted(_SUITE_REGISTRY))
        raise ValueError(f"Unknown robustness suite {suite_id!r}. Choose one of: {valid}")
    return spec


def suite_output_dir(settings: Settings, suite_id: str) -> Path:
    return settings.run_artifacts_dir.parent / "robustness_runs" / suite_id


def _load_founder_prompts(settings: Settings) -> list[FounderPrompt]:
    data = json.loads(settings.founder_prompts_path.read_text(encoding="utf-8"))
    return TypeAdapter(list[FounderPrompt]).validate_python(data)


def _select_prompts(
    prompts: list[FounderPrompt],
    prompt_ids: list[str] | None,
) -> list[FounderPrompt]:
    if prompt_ids:
        allowed = set(prompt_ids)
        selected = [prompt for prompt in prompts if prompt.id in allowed]
        if not selected:
            raise ValueError(f"None of the requested prompt_ids were found: {prompt_ids}")
        return selected
    return prompts[:DEFAULT_PROMPT_SAMPLE_SIZE]


def _normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    compact = re.sub(r"\s+([,.;:!?])", r"\1", compact)
    return compact


def _perturb_negation(statement: str, transform_spec: dict[str, object]) -> str:
    rewrites = (
        (r"\bnot\b", ""),
        (r"\bnever\b", "always"),
        (r"\bno\b", "some"),
        (r"\bcan\b", "cannot"),
        (r"\bshould\b", "should not"),
        (r"\bis\b", "is not"),
        (r"\bare\b", "are not"),
    )
    for pattern, replacement in rewrites:
        updated, count = re.subn(pattern, replacement, statement, count=1, flags=re.IGNORECASE)
        if count:
            return _normalize_text(updated)

    fallback_prefix = str(transform_spec.get("fallback_prefix", "not")).strip()
    return _normalize_text(f"{fallback_prefix} {statement}")


def _stable_noise_rank(seed: int, token: str, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{index}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _perturb_noise(statement: str, transform_spec: dict[str, object]) -> str:
    token = str(transform_spec.get("token", "[noise]")).strip() or "[noise]"
    injection_rate = float(transform_spec.get("injection_rate", 0.2))
    injection_rate = max(0.0, min(1.0, injection_rate))
    seed = int(transform_spec.get("seed", 17))

    words = statement.split()
    if not words:
        return token

    insertion_count = max(1, int(round(len(words) * injection_rate)))
    insertion_count = min(insertion_count, len(words))

    ranked = sorted(
        (_stable_noise_rank(seed, word, index), index)
        for index, word in enumerate(words)
    )
    selected_indexes = sorted(index for _, index in ranked[:insertion_count])

    noisy_words = words[:]
    for offset, index in enumerate(selected_indexes):
        noisy_words.insert(index + offset, token)

    return _normalize_text(" ".join(noisy_words))


def _context_replacements(transform_spec: dict[str, object]) -> list[tuple[str, str]]:
    raw = transform_spec.get("replacements")
    if not isinstance(raw, list):
        return [
            ("restaurant", "product"),
            ("diners", "users"),
            ("takeout", "onboarding"),
            ("menu", "feature set"),
            ("staff", "support team"),
        ]

    replacements: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        source = str(item[0]).strip()
        target = str(item[1]).strip()
        if not source or not target:
            continue
        replacements.append((source, target))
    return replacements


def _perturb_context_shift(statement: str, transform_spec: dict[str, object]) -> str:
    rewritten = statement
    replacement_count = 0

    for source, target in _context_replacements(transform_spec):
        pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
        rewritten, count = pattern.subn(target, rewritten)
        replacement_count += count

    context_prefix = str(
        transform_spec.get(
            "context_prefix",
            "Assume the same user problem appears in an adjacent domain.",
        )
    ).strip()

    if replacement_count == 0 and context_prefix:
        rewritten = f"{context_prefix} {statement}"
    elif context_prefix:
        rewritten = f"{context_prefix} {rewritten}"

    return _normalize_text(rewritten)


def apply_perturbation(statement: str, case: RobustnessCase) -> str:
    family = case.family.strip().lower()
    transform_spec = dict(case.transform_spec)

    if family == "negation":
        return _perturb_negation(statement, transform_spec)
    if family == "noise":
        return _perturb_noise(statement, transform_spec)
    if family == "context_shift":
        return _perturb_context_shift(statement, transform_spec)

    raise ValueError(f"Unsupported robustness case family: {case.family!r}")


def _try_load_synthesis(path: Path) -> SynthesisResult | None:
    if not path.exists():
        return None
    return SynthesisResult.model_validate_json(path.read_text(encoding="utf-8"))


def _try_load_judge(path: Path) -> JudgeResult | None:
    if not path.exists():
        return None
    return JudgeResult.model_validate_json(path.read_text(encoding="utf-8"))


def _run_or_load_control(
    prompt: FounderPrompt,
    settings: Settings,
    variants_root: Path,
    *,
    skip_cached: bool,
) -> SynthesisResult:
    synthesis_path = variants_root / "control" / prompt.id / "synthesis.json"
    if skip_cached:
        cached = _try_load_synthesis(synthesis_path)
        if cached:
            con.cached("robustness", f"[control] Using cached synthesis for {prompt.id}")
            return cached

    return run_variant_pipeline(
        prompt,
        "control",
        settings,
        output_root=variants_root,
        persist=True,
    )


def _judge_or_load(
    prompt: FounderPrompt,
    control_result: SynthesisResult,
    perturbed_result: SynthesisResult,
    case: RobustnessCase,
    settings: Settings,
    root: Path,
    *,
    skip_cached: bool,
) -> tuple[JudgeResult, JudgeResult]:
    run_dir = root / "judges" / case.case_id / prompt.id
    judge_control_path = run_dir / "judge_control.json"
    judge_perturbed_path = run_dir / "judge_perturbed.json"

    if skip_cached:
        cached_control = _try_load_judge(judge_control_path)
        cached_perturbed = _try_load_judge(judge_perturbed_path)
        if cached_control and cached_perturbed:
            con.cached("robustness", f"[{case.case_id}] Using cached judge scores for {prompt.id}")
            return cached_control, cached_perturbed

    control_judge, perturbed_judge = judge_named_pair(
        prompt,
        control_result,
        perturbed_result,
        left_variant="control",
        right_variant=f"perturbed::{case.case_id}",
        settings=settings,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    judge_control_path.write_text(control_judge.model_dump_json(indent=2), encoding="utf-8")
    judge_perturbed_path.write_text(perturbed_judge.model_dump_json(indent=2), encoding="utf-8")
    return control_judge, perturbed_judge


def evaluate_case_thresholds(
    control_scores: JudgeScores,
    perturbed_scores: JudgeScores,
    thresholds: RobustnessGateThresholds,
) -> tuple[bool, dict[str, float], float, list[str]]:
    dimension_deltas = {
        dim: getattr(perturbed_scores, dim) - getattr(control_scores, dim)
        for dim in RUBRIC_DIMS
    }
    delta_overall = perturbed_scores.overall_preference - control_scores.overall_preference

    failures: list[str] = []
    if delta_overall < -thresholds.max_overall_drop:
        failures.append(
            f"overall drop {delta_overall:.2f} exceeded max {-thresholds.max_overall_drop:.2f}"
        )

    for dim, delta in dimension_deltas.items():
        if delta < -thresholds.max_dimension_drop:
            failures.append(
                f"{dim} drop {delta:.2f} exceeded max {-thresholds.max_dimension_drop:.2f}"
            )

    return (not failures), dimension_deltas, delta_overall, failures


def run_robustness_suite(
    settings: Settings | None = None,
    *,
    suite_id: str = "default",
    prompt_ids: list[str] | None = None,
    skip_cached: bool = True,
) -> RobustnessSuiteSummary:
    s = settings or get_settings()
    suite = get_suite_spec(suite_id)
    prompts = _select_prompts(_load_founder_prompts(s), prompt_ids)

    con.header(
        "Robustness Evaluation",
        f"suite: {suite.suite_id} | prompts: {len(prompts)} | model: {s.llm_model}",
    )

    root = suite_output_dir(s, suite.suite_id)
    variants_root = root / "variants"

    outcomes: list[RobustnessCaseOutcome] = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        con.prompt_table(prompt.id, prompt_index, len(prompts))
        control_result = _run_or_load_control(
            prompt,
            s,
            variants_root,
            skip_cached=skip_cached,
        )

        for case in suite.cases:
            perturbed_statement = apply_perturbation(prompt.statement, case)
            perturbed_prompt = prompt.model_copy(
                update={
                    "id": f"{prompt.id}__{case.case_id}",
                    "statement": perturbed_statement,
                }
            )

            perturbed_result = _run_or_load_control(
                perturbed_prompt,
                s,
                variants_root,
                skip_cached=skip_cached,
            )
            control_judge, perturbed_judge = _judge_or_load(
                prompt,
                control_result,
                perturbed_result,
                case,
                s,
                root,
                skip_cached=skip_cached,
            )

            passed, dim_deltas, delta_overall, failures = evaluate_case_thresholds(
                control_judge.scores,
                perturbed_judge.scores,
                suite.thresholds,
            )
            con.step(
                "robustness",
                f"{prompt.id}/{case.case_id} delta={delta_overall:+.2f} => {'PASS' if passed else 'FAIL'}",
            )

            outcomes.append(
                RobustnessCaseOutcome(
                    prompt_id=prompt.id,
                    case_id=case.case_id,
                    family=case.family,
                    description=case.description,
                    expected_behavior=case.expected_behavior,
                    perturbed_prompt_id=perturbed_prompt.id,
                    perturbed_statement=perturbed_statement,
                    control_scores=control_judge.scores,
                    perturbed_scores=perturbed_judge.scores,
                    dimension_deltas=dim_deltas,
                    delta_overall=delta_overall,
                    passed=passed,
                    failure_reasons=failures,
                )
            )

    total_cases = len(outcomes)
    passed_cases = sum(1 for outcome in outcomes if outcome.passed)
    failed_cases = total_cases - passed_cases
    pass_rate = (passed_cases / total_cases) if total_cases else 0.0

    gate_passed = pass_rate >= suite.thresholds.min_case_pass_rate
    failed_case_keys = [
        f"{outcome.prompt_id}:{outcome.case_id}"
        for outcome in outcomes
        if not outcome.passed
    ]
    gate_failure_reasons: list[str] = []
    if not gate_passed:
        gate_failure_reasons.append(
            f"Case pass rate {pass_rate:.2%} is below required {suite.thresholds.min_case_pass_rate:.2%}."
        )
        if failed_case_keys:
            gate_failure_reasons.append(
                "Failed cases: " + ", ".join(failed_case_keys)
            )

    return RobustnessSuiteSummary(
        suite_id=suite.suite_id,
        suite_description=suite.description,
        prompt_ids=[prompt.id for prompt in prompts],
        thresholds=suite.thresholds,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=pass_rate,
        gate_passed=gate_passed,
        failed_case_keys=failed_case_keys,
        gate_failure_reasons=gate_failure_reasons,
        outcomes=outcomes,
    )

