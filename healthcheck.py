"""
healthcheck.py: run before deploying (and in CI).

    python healthcheck.py            # offline checks
    python healthcheck.py --online   # also calls OpenAI and Pinecone once
    python healthcheck.py --service  # also starts the service and sends one test message

Exit code 0 = ready (possibly degraded), 1 = a blocking problem was found.
"""
import argparse
import json
import sys

import config as cfgmod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deployment readiness checks")
    ap.add_argument("--online", action="store_true", help="call OpenAI and Pinecone once each")
    ap.add_argument("--service", action="store_true", help="start the service and send one test message")
    args = ap.parse_args(argv)
    try:
        cfg = cfgmod.load_config()
    except cfgmod.ConfigError as exc:
        print(f"[FAIL] configuration: {exc}")
        return 1
    checks = cfgmod.doctor(cfg, online=args.online)
    icon = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
    for c in checks:
        print(f"{icon[c.status]} {c.name:<28} {c.detail}")
    failed = [c for c in checks if c.status == "fail"]
    if args.service and not failed:
        from service import SupportService
        svc = SupportService(cfg)
        sid = svc.new_session("healthcheck")
        r = svc.handle(sid, "healthcheck", "What's the fee for GBP payouts?")
        print(f"[{' OK ' if r.outcome != 'exception' else 'FAIL'}] {'service round trip':<28} outcome={r.outcome} "
              f"route={r.route} latency={r.latency_ms:.0f}ms")
        print(json.dumps(svc.health(), indent=2))
        svc.shutdown()
        if r.outcome == "exception":
            failed.append(r)
    print(f"\n{'NOT READY' if failed else 'READY'}: {len(failed)} blocking, "
          f"{sum(c.status == 'warn' for c in checks)} warnings")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
