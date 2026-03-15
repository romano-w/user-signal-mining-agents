"""Agent implementations for the USM pipeline.

Core agents:
    baseline        -- Zero-shot retrieval-augmented generation.
    intent          -- Founder statement decomposition into structured queries.
    evidence_filter -- Multi-query retrieval with deduplication and reranking.
    synthesis       -- Grounded focus-point generation with inline evidence.
    evidence_verifier -- Post-synthesis hallucination and quote checker.
    pipeline        -- Orchestrator chaining intent → evidence → synthesis → verify.
    judge           -- LLM-as-judge with A/B position debiasing.

Optional refinement agents:
    query_planner          -- Retrieval query expansion.
    counterevidence_miner  -- Contradiction-focused query generation.
    critic                 -- Draft focus-point critique.
    refiner                -- Critique-driven rewrite.
"""
