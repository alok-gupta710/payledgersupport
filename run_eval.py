"""CI entry point: run the evaluation suite against the deployed service and enforce the release gates.

    python run_eval.py                 # full suite, EVAL_REPEATS runs, red team, exit 1 if any gate fails
    python run_eval.py --quick         # one repeat, no red team (smoke test before a full run)
    python run_eval.py --baseline      # the Phase 2 baseline instead of the agent

Writes results/ci_metrics.json. Without an OpenAI key the agent is a placeholder, so gates are
reported but never enforced (exit 0) and the output says so.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import evaluation as E
import llm_agent as p3
import safety_review as SR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    settings = p3.load_settings()
    kn = E.Knowledge.load()
    suite = E.load_suite(kb_ids=kn.all_ids)
    red = [] if args.quick else SR.red_team_mutations(suite)
    repeats = 1 if args.quick else int(os.getenv("EVAL_REPEATS", "3"))
    workers = int(os.getenv("EVAL_WORKERS", "3"))
    svc = None
    if args.baseline:
        system = E.BaselineSystem()
    else:
        svc = E.make_eval_service()
        system = E.AgentSystem(svc)
    rows = E.run_suite(system, suite, kn, repeats=repeats, workers=workers)
    rt = E.run_suite(system, red, kn, workers=workers) if red else []
    q, s, c = E.quality_metrics(rows), E.safety_metrics(rows + rt), E.consistency_metrics(rows)
    gates = E.check_gates(q, s, c)
    Path("results").mkdir(exist_ok=True)
    Path("results/ci_metrics.json").write_text(json.dumps({"system": system.name, "quality": q, "safety": s,
                                                          "gates": gates}, indent=2, default=str))
    for g in gates:
        print(f"[{g['status']:>4}] {g['gate']:<36} {g['value']!s:<8} target {g['target']}")
    if svc:
        svc.shutdown()
    failed = [g for g in gates if g["status"] == "FAIL"]
    if settings.dry_run and not args.baseline:
        print("DRY RUN: agent replies are placeholders, gates not enforced.")
        return 0
    print("RESULT:", "BLOCK" if failed else "SHIP")
    return 1 if failed and not args.baseline else 0


if __name__ == "__main__":
    sys.exit(main())
