#!/usr/bin/env python3
"""
example_client.py – Demonstrate interaction with the multi-model LLM manager.

The client talks to the manager's REST API (not directly to Triton), so you
only need the manager service to be running.

Usage
-----
    python client/example_client.py --prompt "What is the capital of France?"
    python client/example_client.py --model mistral-7b --prompt "Explain quantum entanglement."
    python client/example_client.py --list-models
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode()
        print(f"HTTP {exc.code} {exc.reason}: {error_body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        sys.exit(1)


def list_models(base_url: str) -> None:
    data = _request("GET", f"{base_url}/v1/models")
    print(f"{'Model':<20} {'Loaded':<8} {'VRAM (GiB)':<12} Description")
    print("-" * 70)
    for m in data["models"]:
        print(
            f"{m['name']:<20} {str(m['loaded']):<8} {m['vram_gb']:<12.1f} {m['description']}"
        )
    print()
    print(
        f"VRAM budget: {data['used_vram_gb']:.1f} / "
        f"{data['total_vram_budget_gb']:.1f} GiB used, "
        f"{data['free_vram_gb']:.1f} GiB free."
    )


def load_model(base_url: str, model_name: str) -> None:
    print(f"Loading model '{model_name}' ...")
    data = _request("POST", f"{base_url}/v1/models/{model_name}/load")
    print(f"Status: {data['status']}")


def unload_model(base_url: str, model_name: str) -> None:
    print(f"Unloading model '{model_name}' ...")
    data = _request("POST", f"{base_url}/v1/models/{model_name}/unload")
    print(f"Status: {data['status']}")


def generate(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> None:
    print(f"Generating with '{model_name}' ...\n")
    payload = {
        "text_input": prompt,
        "parameters": {
            "temperature": str(temperature),
            "max_tokens": str(max_tokens),
        },
    }
    data = _request("POST", f"{base_url}/v1/models/{model_name}/generate", payload)
    print("=" * 60)
    print(data.get("text_output", ""))
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Client for the multi-model LLM manager."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Manager service base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--model",
        default="llama3-8b",
        help="Model name to use for generation (default: llama3-8b)",
    )
    parser.add_argument("--prompt", help="Text prompt for generation.")
    parser.add_argument(
        "--list-models", action="store_true", help="List all available/loaded models."
    )
    parser.add_argument(
        "--load", action="store_true", help="Load the specified model."
    )
    parser.add_argument(
        "--unload", action="store_true", help="Unload the specified model."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Maximum tokens to generate."
    )
    args = parser.parse_args()

    if args.list_models:
        list_models(args.url)
    elif args.load:
        load_model(args.url, args.model)
    elif args.unload:
        unload_model(args.url, args.model)
    elif args.prompt:
        generate(args.url, args.model, args.prompt, args.temperature, args.max_tokens)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
