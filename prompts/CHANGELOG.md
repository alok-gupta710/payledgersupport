# Prompt changelog

Each prompt version is an immutable file. Change a prompt by adding a new version, never by editing an old one.

## v0_naive (parent: none, 2026-11-23)
Minimal role prompt plus the knowledge base. Control condition: what an LLM does with no rules.
- Initial control prompt.

## v1_rules (parent: v0_naive, 2026-11-23)
Role, persona and nine explicit rules mapped to the four safety requirements and the routes.
- Added role and persona context.
- Added grounding, citation, refusal, escalation, multi-intent and precedence rules.
- Marked the canary reference as confidential.

## v2_rules_fewshot (parent: v1_rules, 2026-11-23)
v1 plus five worked examples targeting Phase 2 failures (negation, implicit destination change, multi-intent with a no-policy part, missing ID, not-found lookup).
- Added five few-shot examples from prompts/fewshot.yaml.

## v3_rules_procedure (parent: v1_rules, 2026-11-23)
v1 plus an explicit six-step decision procedure, with brief notes written to decision_notes before the answer.
- Added a step-by-step decision procedure (safety, sensitivity, split, evidence, route, write).

## v4_full (parent: v3_rules_procedure, 2026-11-23)
v3 plus the v2 few-shot examples: rules, procedure and examples combined.
- Combined the decision procedure (v3) with the worked examples (v2).
