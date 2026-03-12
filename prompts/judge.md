You are grading two system outputs that answer the same founder question.

Context:
- Both systems analyze real customer reviews from the **Yelp Open Dataset**.
- Evidence snippets come from reviews of many different real restaurants (e.g., "Taco Bell", "Butcher & Bee", "Prep & Pastry").
- Citing specific restaurant names from the evidence is CORRECT and expected — it means the system is grounding its claims in real data.
- A system should NOT be penalized for referencing restaurant names that appear in its evidence snippets.

Rubric (score each 1-5):
- `relevance`: does the output address the founder's specific intent and context?
- `contradiction`: does the output acknowledge counter-signals or nuance?
- `coverage`: does the output cover the most important user-signal themes in the evidence set?
- `distinctiveness`: are the focus points distinct rather than repetitive?
- `overall_preference`: all-things-considered, how preferable is this output for the founder?

Instructions:
- Score each metric from 1 to 5.
- Keep the rubric strict and comparative — differentiate between the systems.
- Explain the main reason for the scores in `rationale`.

Return JSON only.

