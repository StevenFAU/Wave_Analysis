#!/usr/bin/env python
"""Validate the dataset registry and print a verification-status summary.

uv run python scripts/build_registry.py
"""

from collections import Counter
from pathlib import Path

import yaml

from wave_analysis.registry import validate_registry

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    data = yaml.safe_load((ROOT / "data" / "registry" / "datasets.yaml").read_text())
    problems = validate_registry(data)
    for p in problems:
        print("ERROR:", p)
    status = Counter(d["verification"]["status"] for d in data["datasets"])
    phases = Counter(d["phase"] for d in data["datasets"])
    print(
        f"{len(data['datasets'])} datasets; verification {dict(status)}; phases {dict(sorted(phases.items()))}"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
