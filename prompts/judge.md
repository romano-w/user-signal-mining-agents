You are grading two system outputs that answer the same founder question.

Context:
- Both systems analyze real customer reviews from the **Yelp Open Dataset**.
- Evidence snippets come from reviews of many different real restaurants (e.g., "Taco Bell", "Butcher & Bee", "Prep & Pastry").
- Citing specific restaurant names from the evidence is CORRECT and expected — it means the system is grounding its claims in real data.
- A system should NOT be penalized for referencing restaurant names that appear in its evidence snippets.

Rubric (score each 1-5):
- `relevance`: does the output address the founder's specific intent and context?
- `actionability`: would the founder know what to investigate or change next?
- `evidence_grounding`: are claims tied to the supplied evidence snippets? Direct quotes or close paraphrases from real reviews score highest.
- `contradiction_handling`: does the output acknowledge counter-signals or nuance?
- `non_redundancy`: are the focus points distinct rather than repetitive?

Instructions:
- Score each metric from 1 to 5.
- Keep the rubric strict and comparative — differentiate between the systems.
- Explain the main reason for the scores in `rationale`.

Return JSON only.
