"""Metadata-only sampling scan of The Stack v3 to size an IaC slice.

The corpus is 4.71 TB compressed across 8196 parquet shards, but 96.9% of every
shard is the `content` column. Reading only the metadata leaves (`file_path`,
`language`, `size_bytes`, `license_type`) costs ~1% of the bytes, so a shard's
worth of structure can be inspected for ~6 MB of transfer.

This scans an evenly spaced sample of shards and extrapolates to the full
corpus. Nothing is written to the dataset and no contents are downloaded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from .taxonomy import (
    ADJACENT_TOOLS,
    Classification,
    Precision,
    classify,
    detect_units,
)

REPO = "datasets/HuggingFaceCode/stack-v3-train"

# Pinned snapshot. Upstream applies opt-out removals in place: on 2026-07-28 it
# ran "Clear data before opt-out update" and re-uploaded the whole dataset under
# a new file UUID, so `main` stops resolving the shards a sweep was reading. The
# UUID and shard count below describe THIS revision and only this one. Old
# revisions stay readable by SHA, which is what makes a harvest reproducible.
REVISION = "de81e3ca7151"
SHARD_UUID = "4beed122-1346-42f6-82eb-5757f2b6305f"
TOTAL_SHARDS = 8196

COLUMNS = [
    "repo_path",
    "num_files",
    "github_metadata",
    "files.list.element.file_path",
    "files.list.element.language",
    "files.list.element.size_bytes",
    "files.list.element.license_type",
    "files.list.element.is_vendor",
]

# Recall-only prefilter: anything an infrastructure rule could match must pass,
# but false positives are harmless because the ordered rules run afterwards.
PREFILTER = re.compile(
    r"("
    r"\.(tf|tfvars|hcl|ya?ml|nix|rego|jsonnet|libsonnet|cue|bzl|pp|sls"
    r"|sh|bash|zsh|conf|cfg|service|timer|socket|mount|target|path|tpl|rb|json"
    r"|dockerfile|containerfile)$"
    r"|(^|/)(Dockerfile|Containerfile|Makefile|GNUmakefile|Jenkinsfile|Vagrantfile"
    r"|BUILD|WORKSPACE|MODULE|[Jj]ustfile|Earthfile|ansible|inventory|hosts)[^/]*$"
    r")",
    re.IGNORECASE,
)


def shard_path(index: int, revision: str = REVISION) -> str:
    """Path to one shard, pinned to a revision so a sweep cannot shift under it."""
    repo = f"{REPO}@{revision}" if revision else REPO
    return f"{repo}/data/part-{index:05d}-{SHARD_UUID}-c000.snappy.parquet"


def sample_indices(count: int) -> list[int]:
    """Evenly spaced shard indices, immune to any repo ordering within shards."""
    if count >= TOTAL_SHARDS:
        return list(range(TOTAL_SHARDS))
    step = TOTAL_SHARDS / count
    return sorted({int(i * step) for i in range(count)})


class CountingReader:
    """File wrapper that records how many bytes were actually pulled."""

    def __init__(self, handle) -> None:
        self._handle = handle
        self.bytes_read = 0

    def read(self, *args):
        chunk = self._handle.read(*args)
        self.bytes_read += len(chunk)
        return chunk

    def seek(self, *args):
        return self._handle.seek(*args)

    def tell(self):
        return self._handle.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def close(self):
        return self._handle.close()


@dataclass
class Totals:
    """Aggregated counters for one or many shards."""

    shards: int = 0
    bytes_fetched: int = 0
    row_groups_read: int = 0
    row_groups_total: int = 0
    repos: int = 0
    files: int = 0
    file_bytes: int = 0
    languages: Counter = field(default_factory=Counter)
    tool_files: Counter = field(default_factory=Counter)
    tool_bytes: Counter = field(default_factory=Counter)
    tool_repos: Counter = field(default_factory=Counter)
    tool_precision: Counter = field(default_factory=Counter)
    units: Counter = field(default_factory=Counter)
    unit_repos: Counter = field(default_factory=Counter)
    license_files: Counter = field(default_factory=Counter)
    core_repo_stars: Counter = field(default_factory=Counter)
    vendor_files: int = 0

    def merge(self, other: Totals) -> None:
        self.shards += other.shards
        self.bytes_fetched += other.bytes_fetched
        self.row_groups_read += other.row_groups_read
        self.row_groups_total += other.row_groups_total
        self.repos += other.repos
        self.files += other.files
        self.file_bytes += other.file_bytes
        self.vendor_files += other.vendor_files
        for name in (
            "languages",
            "tool_files",
            "tool_bytes",
            "tool_repos",
            "tool_precision",
            "units",
            "unit_repos",
            "license_files",
            "core_repo_stars",
        ):
            getattr(self, name).update(getattr(other, name))


def star_bucket(stars: int) -> str:
    if stars >= 10_000:
        return "10k+"
    if stars >= 1_000:
        return "1k-10k"
    if stars >= 100:
        return "100-1k"
    if stars >= 10:
        return "10-100"
    if stars >= 1:
        return "1-10"
    return "0"


def scan_shard(index: int, row_group_limit: int | None = None) -> Totals:
    """Read one shard's metadata columns and aggregate IaC statistics."""
    totals = Totals(shards=1)
    cache: dict[str, Classification | None] = {}
    fs = HfFileSystem()

    with fs.open(shard_path(index), "rb") as raw:
        reader = CountingReader(raw)
        parquet_file = pq.ParquetFile(reader)
        available = parquet_file.metadata.num_row_groups
        groups = range(available if row_group_limit is None else min(row_group_limit, available))
        totals.row_groups_total = available
        totals.row_groups_read = len(groups)

        for group in groups:
            table = parquet_file.read_row_group(group, columns=COLUMNS)
            files = table.column("files").combine_chunks()
            stars = table.column("github_metadata").combine_chunks().field("stars")

            flat = pc.list_flatten(files)
            parent = pc.list_parent_indices(files).to_pylist()
            paths = flat.field("file_path").to_pylist()
            languages = flat.field("language").to_pylist()
            sizes = flat.field("size_bytes").to_pylist()
            licenses = flat.field("license_type").to_pylist()
            vendors = flat.field("is_vendor").to_pylist()
            star_values = stars.to_pylist()

            totals.repos += table.num_rows
            totals.files += len(paths)
            totals.languages.update(lang or "unknown" for lang in languages)

            repo_paths: dict[int, list[str]] = defaultdict(list)
            repo_tools: dict[int, set[str]] = defaultdict(set)

            for position, path in enumerate(paths):
                size = sizes[position] or 0
                totals.file_bytes += size
                if vendors[position]:
                    totals.vendor_files += 1
                repo_paths[parent[position]].append(path)

                if not PREFILTER.search(path):
                    continue
                language = languages[position]
                key = f"{path}\x00{language}"
                if key in cache:
                    hit = cache[key]
                else:
                    hit = classify(path, language)
                    cache[key] = hit
                if hit is None:
                    continue

                totals.tool_files[hit.tool] += 1
                totals.tool_bytes[hit.tool] += size
                totals.tool_precision[f"{hit.tool}/{hit.precision.value}"] += 1
                totals.license_files[licenses[position] or "unknown"] += 1
                repo_tools[parent[position]].add(hit.tool)

            for repo_index, tools in repo_tools.items():
                for tool in tools:
                    totals.tool_repos[tool] += 1
                if tools - ADJACENT_TOOLS:
                    totals.core_repo_stars[star_bucket(star_values[repo_index] or 0)] += 1

            for repo_index, path_list in repo_paths.items():
                units = detect_units(path_list)
                for unit_type, found in units.items():
                    if found:
                        totals.units[unit_type] += len(found)
                        totals.unit_repos[unit_type] += 1

        totals.bytes_fetched = reader.bytes_read

    return totals


def scan(shard_count: int, workers: int, row_group_limit: int | None) -> Totals:
    indices = sample_indices(shard_count)
    combined = Totals()
    started = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scan_shard, index, row_group_limit): index for index in indices
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                combined.merge(future.result())
            except Exception as error:  # noqa: BLE001 - report and keep going
                print(f"  shard {index}: FAILED {error}", file=sys.stderr)
                continue
            done += 1
            elapsed = time.time() - started
            print(
                f"  [{done}/{len(indices)}] shard {index:05d}"
                f"  {combined.repos:,} repos  {combined.files:,} files"
                f"  {combined.bytes_fetched / 1e6:.0f} MB  {elapsed:.0f}s",
                file=sys.stderr,
            )

    return combined


def report(totals: Totals) -> dict:
    # Extrapolate over both dimensions of the sample: which shards were read,
    # and how much of each shard was read when row groups were limited.
    scale = (TOTAL_SHARDS / totals.shards) if totals.shards else 0
    if totals.row_groups_read:
        scale *= totals.row_groups_total / totals.row_groups_read
    core_files = sum(
        count for tool, count in totals.tool_files.items() if tool not in ADJACENT_TOOLS
    )
    core_bytes = sum(
        size for tool, size in totals.tool_bytes.items() if tool not in ADJACENT_TOOLS
    )

    return {
        "sample": {
            "shards_scanned": totals.shards,
            "shards_total": TOTAL_SHARDS,
            "coverage_pct": round(100 * totals.shards / TOTAL_SHARDS, 3),
            "bytes_fetched_mb": round(totals.bytes_fetched / 1e6, 1),
            "repos": totals.repos,
            "files": totals.files,
            "file_bytes_gb": round(totals.file_bytes / 1e9, 2),
        },
        "extrapolated": {
            "repos": int(totals.repos * scale),
            "files": int(totals.files * scale),
            "corpus_tb": round(totals.file_bytes * scale / 1e12, 2),
            "iac_core_files": int(core_files * scale),
            "iac_core_gb": round(core_bytes * scale / 1e9, 1),
            "iac_core_share_pct": round(
                100 * core_bytes / totals.file_bytes if totals.file_bytes else 0, 3
            ),
            "full_metadata_index_gb": round(
                totals.bytes_fetched * scale / 1e9, 1
            ),
        },
        "tools": {
            tool: {
                "files": totals.tool_files[tool],
                "files_extrapolated": int(totals.tool_files[tool] * scale),
                "gb_extrapolated": round(totals.tool_bytes[tool] * scale / 1e9, 2),
                "repos_pct": round(100 * totals.tool_repos[tool] / totals.repos, 2)
                if totals.repos
                else 0,
                "adjacent": tool in ADJACENT_TOOLS,
            }
            for tool, _ in totals.tool_files.most_common()
        },
        "precision": dict(totals.tool_precision.most_common()),
        "units": {
            unit: {
                "count": totals.units[unit],
                "count_extrapolated": int(totals.units[unit] * scale),
                "repos": totals.unit_repos[unit],
                "repos_extrapolated": int(totals.unit_repos[unit] * scale),
            }
            for unit, _ in totals.units.most_common()
        },
        "licenses": dict(totals.license_files.most_common()),
        "stars_of_iac_repos": dict(totals.core_repo_stars.most_common()),
        "top_languages": dict(totals.languages.most_common(30)),
    }


def print_report(data: dict) -> None:
    sample = data["sample"]
    extra = data["extrapolated"]

    print()
    print("=" * 78)
    print("STACK V3 IaC SLICE - METADATA SCAN")
    print("=" * 78)
    print(
        f"sampled {sample['shards_scanned']}/{sample['shards_total']} shards "
        f"({sample['coverage_pct']}%) for {sample['bytes_fetched_mb']} MB of transfer"
    )
    print(
        f"{sample['repos']:,} repos, {sample['files']:,} files, "
        f"{sample['file_bytes_gb']} GB of source in the sample"
    )
    print()
    print(f"corpus extrapolation      : {extra['repos']:,} repos, "
          f"{extra['files']:,} files, {extra['corpus_tb']} TB")
    print(f"full metadata index cost  : {extra['full_metadata_index_gb']} GB")
    print(f"IaC core slice            : {extra['iac_core_files']:,} files, "
          f"{extra['iac_core_gb']} GB ({extra['iac_core_share_pct']}% of corpus)")
    print()

    print(f"{'tool':<20} {'files (sample)':>14} {'extrapolated':>14} {'GB':>8} {'% repos':>8}")
    print("-" * 78)
    for tool, stats in data["tools"].items():
        marker = "~" if stats["adjacent"] else " "
        print(
            f"{marker}{tool:<19} {stats['files']:>14,} "
            f"{stats['files_extrapolated']:>14,} {stats['gb_extrapolated']:>8.2f} "
            f"{stats['repos_pct']:>7.2f}%"
        )
    print("-" * 78)
    print("~ = infrastructure-adjacent, counted but not part of the core slice")
    print()

    print(f"{'unit':<26} {'in sample':>12} {'extrapolated':>14} {'repos':>12}")
    print("-" * 78)
    for unit, stats in data["units"].items():
        print(
            f"{unit:<26} {stats['count']:>12,} {stats['count_extrapolated']:>14,} "
            f"{stats['repos_extrapolated']:>12,}"
        )
    print()

    print("license_type of matched files :", data["licenses"])
    print("stars of IaC-bearing repos    :", data["stars_of_iac_repos"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=16, help="shards to sample")
    parser.add_argument("--workers", type=int, default=8, help="parallel readers")
    parser.add_argument(
        "--row-groups", type=int, default=None, help="limit row groups per shard"
    )
    parser.add_argument("--out", default="scan_report.json")
    args = parser.parse_args()

    totals = scan(args.shards, args.workers, args.row_groups)
    data = report(totals)
    print_report(data)
    with open(args.out, "w") as handle:
        json.dump(data, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
