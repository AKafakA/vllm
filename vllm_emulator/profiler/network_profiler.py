"""Optional profiler for local network collective latency samples.

Note: This script currently emits a template profile. Replace with NCCL/MPI-based
measurement in multi-node environments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional network profiler template")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    template = {
        "all_reduce": {
            "nvlink": [
                {"bytes": 4096, "world_size": 2, "latency_us": 50},
                {"bytes": 1048576, "world_size": 2, "latency_us": 200},
            ],
            "pcie": [],
            "ib": [],
        },
        "send_recv": {},
        "kv_transfer": {},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
