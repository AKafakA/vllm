"""Inject a sched_overhead_table into a profile pack (in-place).

Usage:
    python3 tools/merge_ipc_table.py <profile_pack.json> <ipc_table.json>

The IPC table may be either:
  (a) A JSON array of cells: [{"num_reqs": N, "overhead_us": ..., ...}, ...]
  (b) A dict with "sched_overhead_table" key (from a prior merge) OR
      a wrapper {"metadata": ..., "cells": [...]} (v3 output format).

The profile pack MUST exist; this script replaces its `sched_overhead_table`
key with the provided cells. A `.bak` copy of the pack is written first.
"""
import json
import shutil
import sys
from pathlib import Path


def extract_cells(ipc_json):
    """Accept any of the three shapes the IPC sweep scripts produce."""
    if isinstance(ipc_json, list):
        return ipc_json
    if isinstance(ipc_json, dict):
        if "sched_overhead_table" in ipc_json:
            return ipc_json["sched_overhead_table"]
        if "cells" in ipc_json:
            return ipc_json["cells"]
    raise ValueError(
        f"Unexpected IPC table JSON shape: {type(ipc_json).__name__}. "
        f"Expected list of cells, or dict with 'sched_overhead_table' or 'cells'."
    )


def main():
    if len(sys.argv) != 3:
        print("Usage: merge_ipc_table.py <profile_pack.json> <ipc_table.json>",
              file=sys.stderr)
        sys.exit(2)

    profile_path = Path(sys.argv[1])
    ipc_path = Path(sys.argv[2])

    if not profile_path.exists():
        sys.exit(f"profile pack not found: {profile_path}")
    if not ipc_path.exists():
        sys.exit(f"ipc table not found: {ipc_path}")

    pack = json.load(open(profile_path))
    ipc_json = json.load(open(ipc_path))
    cells = extract_cells(ipc_json)

    if not cells:
        sys.exit(f"IPC table at {ipc_path} has no cells; refusing to merge "
                 "an empty sched_overhead_table.")

    # Validate + normalize: accept v1 schema ('overhead_us') and v3
    # schema ('overhead_median_us'/'overhead_mean_us'). Canonicalize by
    # filling 'overhead_us' from 'overhead_median_us' if missing. Mutates
    # cells in place so the downstream oracle (which reads 'overhead_us')
    # works for both sweep versions.
    for i, c in enumerate(cells):
        if "num_reqs" not in c:
            sys.exit(f"cell {i} missing 'num_reqs': {c}")
        if "overhead_us" not in c:
            if "overhead_median_us" in c:
                c["overhead_us"] = c["overhead_median_us"]
            else:
                sys.exit(
                    f"cell {i} missing both 'overhead_us' and "
                    f"'overhead_median_us': {c}")

    # Backup before overwriting.
    backup = profile_path.with_suffix(profile_path.suffix + ".bak")
    shutil.copy2(profile_path, backup)

    pack["sched_overhead_table"] = cells
    pack.setdefault("sched_overhead_table_meta", {})
    pack["sched_overhead_table_meta"]["source"] = str(ipc_path)
    pack["sched_overhead_table_meta"]["num_cells"] = len(cells)

    with open(profile_path, "w") as f:
        json.dump(pack, f, indent=2)

    print(f"merged {len(cells)} cells from {ipc_path} into {profile_path}")
    print(f"backup at {backup}")


if __name__ == "__main__":
    main()
