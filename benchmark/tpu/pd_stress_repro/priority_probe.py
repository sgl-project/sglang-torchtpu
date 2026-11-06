#!/usr/bin/env python3
"""Check whether a request's `priority` reaches the prefill engine through a router.

Sends one long request carrying a unique priority value and, while it runs,
looks for that value as a `priority="<v>"` label in the prefill `/metrics`.

    python3 priority_probe.py --router http://localhost:8001 --prefill http://P:30000 \
        --priority 4242 --endpoint chat --model gpt-oss-120b

Use a fresh priority value per run: a label can stay in `/metrics` after its
request finishes.
"""

import argparse
import re
import threading
import time

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--priority", type=int, required=True)
    ap.add_argument("--endpoint", choices=["chat", "gen"], default="chat")
    ap.add_argument("--model", default="default")
    args = ap.parse_args()

    if args.endpoint == "chat":
        url = args.router + "/v1/chat/completions"
        body = {
            "model": args.model,
            "max_tokens": 600,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Count from 1 to 300."}],
            "priority": args.priority,
        }
    else:
        url = args.router + "/generate"
        body = {
            "text": "Count from 1 to 300:",
            "priority": args.priority,
            "sampling_params": {"max_new_tokens": 600, "temperature": 0},
        }

    out = {}
    t = threading.Thread(
        target=lambda: out.update(r=requests.post(url, json=body, timeout=600))
    )
    t.start()
    time.sleep(1.5)
    labels = set(
        re.findall(r'priority="([^"]*)"', requests.get(args.prefill + "/metrics").text)
    )
    t.join()
    print(
        f"{url} status={out['r'].status_code} "
        f"priority {args.priority} seen by prefill: {str(args.priority) in labels}"
    )


if __name__ == "__main__":
    main()
