"""Profile the extracted units and pull unbiased samples out for inspection.

Reading only the head of a JSONL file describes whichever shard happened to be
written first, so this streams every record to build the real distributions, and
uses reservoir sampling to pick examples that are representative rather than
merely early.

Samples are written to disk as actual directories of files, because the point of
extracting whole units is that a human (or `helm lint`) can look at one.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import re
import sys
from collections import Counter

SAFE = re.compile(r"[^A-Za-z0-9._-]")

# Per-class quality fields worth summarising, as (field, kind).
QUALITY_FIELDS = {
    "helm_chart": [("templates", "num"), ("templated_templates", "num"),
                   ("has_values", "bool"), ("helpers", "num")],
    "terraform_module": [("tf_files", "num"), ("declaring", "num"),
                         ("has_variables", "bool"), ("has_outputs", "bool")],
    "ansible_role": [("task_files", "num"), ("has_defaults", "bool"),
                     ("has_handlers", "bool"), ("has_templates", "bool")],
    "manifest_set": [("manifests", "num")],
    "dockerfile": [("stages", "num"), ("multi_stage", "bool"),
                   ("instructions", "num"), ("has_user", "bool"),
                   ("has_healthcheck", "bool")],
    "workflow": [("jobs", "num"), ("steps", "num"), ("uses_actions", "num"),
                 ("has_permissions", "bool")],
    "compose": [("services", "num"), ("built_services", "num"),
                ("has_healthcheck", "bool"), ("has_volumes", "bool")],
}


def star_bucket(stars) -> str:
    stars = stars or 0
    for threshold, label in ((10_000, "10k+"), (1_000, "1k-10k"), (100, "100-1k"),
                             (10, "10-100"), (1, "1-10")):
        if stars >= threshold:
            return label
    return "0"


class ClassProfile:
    def __init__(self, unit_type: str, sample_size: int, seed: int) -> None:
        self.unit_type = unit_type
        self.count = 0
        self.licenses: Counter = Counter()
        self.stars: Counter = Counter()
        self.kinds: Counter = Counter()
        self.numeric: dict[str, list[int]] = {}
        self.booleans: Counter = Counter()
        self.files_total = 0
        self.bytes_total = 0
        self.reservoir: list[dict] = []
        self.sample_size = sample_size
        self.random = random.Random(seed)

    def add(self, record: dict) -> None:
        self.count += 1
        for name in record.get("license_types") or ["none"]:
            self.licenses[name] += 1
        self.stars[star_bucket(record.get("stars"))] += 1

        files = record.get("files") or []
        self.files_total += len(files)
        self.bytes_total += sum(f.get("size_bytes") or 0 for f in files)

        quality = record.get("quality") or {}
        for field, kind in QUALITY_FIELDS.get(self.unit_type, []):
            value = quality.get(field)
            if kind == "num" and isinstance(value, int):
                self.numeric.setdefault(field, []).append(value)
            elif kind == "bool":
                self.booleans[f"{field}={bool(value)}"] += 1
        for kind in quality.get("kinds") or []:
            self.kinds[kind] += 1

        # Reservoir sampling: every record gets an equal chance of being kept.
        if len(self.reservoir) < self.sample_size:
            self.reservoir.append(record)
        else:
            index = self.random.randrange(self.count)
            if index < self.sample_size:
                self.reservoir[index] = record

    def report(self) -> dict:
        def stats(values: list[int]) -> dict:
            if not values:
                return {}
            ordered = sorted(values)
            return {
                "min": ordered[0],
                "p50": ordered[len(ordered) // 2],
                "p90": ordered[int(len(ordered) * 0.9)],
                "max": ordered[-1],
                "mean": round(sum(ordered) / len(ordered), 2),
            }

        permissive = self.licenses.get("permissive", 0)
        return {
            "unit_type": self.unit_type,
            "units": self.count,
            "files": self.files_total,
            "bytes": self.bytes_total,
            "mean_files_per_unit": round(self.files_total / self.count, 2) if self.count else 0,
            "mean_bytes_per_unit": round(self.bytes_total / self.count) if self.count else 0,
            "permissive_pct": round(100 * permissive / self.count, 2) if self.count else 0,
            "licenses": dict(self.licenses.most_common()),
            "stars": dict(self.stars.most_common()),
            "quality_numeric": {f: stats(v) for f, v in sorted(self.numeric.items())},
            "quality_boolean": dict(sorted(self.booleans.items())),
            "top_kinds": dict(self.kinds.most_common(15)),
        }


def write_sample(root: str, record: dict, index: int) -> str:
    """Materialise one unit on disk as the directory of files it came from."""
    label = SAFE.sub("_", f"{index:02d}-{record.get('repo_path', 'unknown')}")
    base = os.path.join(root, record["unit_type"], label)
    prefix = record.get("unit_prefix") or ""
    for entry in record.get("files") or []:
        path = entry.get("path") or "file"
        relative = path[len(prefix):].lstrip("/") if prefix and path.startswith(prefix) else path
        if not relative:
            # Single-file units (Dockerfile, workflow, Compose) use the file's own
            # path as the unit prefix, which leaves nothing relative. Without this
            # the unit would be written as a file where its directory belongs.
            relative = os.path.basename(path) or "file"
        target = os.path.join(base, *[SAFE.sub("_", part) for part in relative.split("/") if part])
        os.makedirs(os.path.dirname(target) or base, exist_ok=True)
        with open(target, "w", encoding="utf-8", errors="replace") as handle:
            handle.write(entry.get("content") or "")
    with open(os.path.join(base, "_provenance.json"), "w") as handle:
        json.dump({
            "repo_path": record.get("repo_path"),
            "commit_id": record.get("commit_id"),
            "stars": record.get("stars"),
            "unit_prefix": prefix,
            "license_types": record.get("license_types"),
            "quality": record.get("quality"),
        }, handle, indent=2)
    return base


def profile_file(path: str, sample_size: int, seed: int) -> ClassProfile:
    unit_type = os.path.basename(path).split(".")[0]
    profile = ClassProfile(unit_type, sample_size, seed)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                profile.add(json.loads(line))
            except json.JSONDecodeError:
                continue
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="directory of *.jsonl.gz unit files")
    parser.add_argument("--samples", default="quality_samples")
    parser.add_argument("--sample-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--out", default="profile.json")
    args = parser.parse_args()

    paths = sorted(
        p for p in glob.glob(os.path.join(args.directory, "*.jsonl.gz"))
        if ".repaired." not in p
    )
    reports = {}
    for path in paths:
        profile = profile_file(path, args.sample_size, args.seed)
        reports[profile.unit_type] = profile.report()
        for index, record in enumerate(profile.reservoir, start=1):
            write_sample(args.samples, record, index)
        print(f"  profiled {profile.unit_type}: {profile.count:,} units",
              file=sys.stderr, flush=True)

    with open(args.out, "w") as handle:
        json.dump(reports, handle, indent=2)

    print()
    print(f"{'class':<18} {'units':>11} {'files/unit':>11} {'KB/unit':>9} {'permissive':>11}")
    print("-" * 66)
    total_units = total_files = total_bytes = 0
    for unit_type, report in reports.items():
        total_units += report["units"]
        total_files += report["files"]
        total_bytes += report["bytes"]
        print(f"{unit_type:<18} {report['units']:>11,} "
              f"{report['mean_files_per_unit']:>11.2f} "
              f"{report['mean_bytes_per_unit'] / 1024:>9.1f} "
              f"{report['permissive_pct']:>10.2f}%")
    print("-" * 66)
    print(f"{'TOTAL':<18} {total_units:>11,} {total_files:>11,} files "
          f"{total_bytes / 1e9:>6.2f} GB of source")
    print(f"\nwrote {args.out} and samples under {args.samples}/")


if __name__ == "__main__":
    main()
