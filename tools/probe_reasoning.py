"""Verify reasoning suppression, throughput and JSON mode on the live server."""

from __future__ import annotations

import json
import time

import requests

BASE = "http://127.0.0.1:8080/v1"


def run(label: str, messages: list[dict], max_tokens: int = 200, **extra) -> dict:
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        **extra,
    }
    t0 = time.time()
    r = requests.post(f"{BASE}/chat/completions", json=payload, timeout=900)
    dt = time.time() - t0
    if r.status_code != 200:
        print(f"{label}: HTTP {r.status_code} {r.text[:300]}", flush=True)
        return {}
    data = r.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    ct = usage.get("completion_tokens", 0)
    print(
        f"{label}: {dt:.1f}s | out={ct} tok | {ct/dt if dt else 0:.2f} tok/s | "
        f"content={len(content)}c reasoning={len(reasoning)}c | "
        f"finish={data['choices'][0].get('finish_reason')}",
        flush=True,
    )
    print("   content:", repr(content[:500]), flush=True)
    return data


if __name__ == "__main__":
    run(
        "plain",
        [{"role": "user", "content": "Answer in one short sentence: what colour is a clear daytime sky?"}],
        max_tokens=80,
    )

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["id", "prompt"],
                },
            },
        },
        "required": ["title", "scenes"],
    }
    run(
        "json_schema",
        [{"role": "user", "content": "Two-scene storyboard about a lighthouse in a storm."}],
        max_tokens=350,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "storyboard", "schema": schema, "strict": True},
        },
    )

    run(
        "json_object",
        [{"role": "user", "content": 'Return JSON: {"ok": true, "n": 3}'}],
        max_tokens=120,
        response_format={"type": "json_object"},
    )
