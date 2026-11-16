"""Measure local llama-server throughput and inspect message shape."""

from __future__ import annotations

import json
import time

import requests

BASE = "http://127.0.0.1:8080/v1"


def timed(payload: dict, label: str) -> dict:
    t0 = time.time()
    r = requests.post(f"{BASE}/chat/completions", json=payload, timeout=900)
    dt = time.time() - t0
    if r.status_code != 200:
        print(f"{label}: HTTP {r.status_code} {r.text[:500]}")
        return {}
    data = r.json()
    usage = data.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    rate = (ct / dt) if dt else 0
    print(f"{label}: {dt:.1f}s, {ct} completion tokens -> {rate:.2f} tok/s")
    return data


def dump(label: str, data: dict) -> None:
    print(f"\n--- {label} ---")
    if not data:
        return
    ch = data["choices"][0]
    msg = ch["message"]
    print("message keys:", list(msg.keys()))
    print("finish_reason:", ch.get("finish_reason"))
    content = msg.get("content") or ""
    print("has <think>:", "<think>" in content)
    print("content head:", repr(content[:400]))
    for k, v in msg.items():
        if k != "content":
            print(f"  {k}: {str(v)[:300]}")


def main() -> None:
    p = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Say the single word: hello"}],
        "temperature": 0.2,
        "max_tokens": 32,
    }
    dump("plain-32", timed(p, "plain-32"))

    p2 = {
        "model": "local-model",
        "messages": [
            {"role": "user", "content": 'Output JSON only: {"ok": true, "n": 3}'}
        ],
        "temperature": 0.1,
        "max_tokens": 64,
        "response_format": {"type": "json_object"},
    }
    dump("json_object-64", timed(p2, "json_object-64"))

    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "prompt": {"type": "string"}},
        "required": ["id", "prompt"],
    }
    p3 = {
        "model": "local-model",
        "messages": [
            {"role": "user", "content": "Return one storyboard scene about a lighthouse."}
        ],
        "temperature": 0.3,
        "max_tokens": 200,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "scene", "schema": schema, "strict": True},
        },
    }
    dump("json_schema-200", timed(p3, "json_schema-200"))

    print("\n--- /models ---")
    r = requests.get(f"{BASE}/models", timeout=30)
    print(r.status_code, json.dumps(r.json(), indent=2)[:800])


if __name__ == "__main__":
    main()
