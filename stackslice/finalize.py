"""Turn the raw harvest into a publishable dataset.

Three things stand between the extracted units and something that can be released:

1. **Opt-out compliance.** Upstream applies opt-out removals in place and
   re-uploads. A harvest taken before that carries repositories whose owners have
   since asked to be removed, which neither ODC-By nor the dataset's own terms
   permit. Every unit is re-filtered against a target revision by `repo_path`,
   which needs only that one column: 0.1% of each shard.
2. **Deduplication.** The corpus repeats file rows inside a repository (10.4% of
   repositories, 14.5% of all file rows, byte-identical). Left alone, a model
   would see the same file up to eleven times.
3. **Derived flags.** Whether a Helm chart can render standalone, whether a
   Dockerfile pins a digest, and so on: cheap to compute once, impossible for a
   consumer to filter on without re-parsing everything.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from .scan import CountingReader, REPO

INCLUDE_CALL = re.compile(r"\{\{-?\s*(include|template)\s+\"")
DEFINE_CALL = re.compile(r"\{\{-?\s*define\s+\"")
DIGEST_PIN = re.compile(r"^\s*FROM\s+\S+@sha256:", re.MULTILINE | re.IGNORECASE)
LATEST_TAG = re.compile(r"^\s*FROM\s+\S+:latest\b", re.MULTILINE | re.IGNORECASE)
UNPINNED_ACTION = re.compile(r"uses:\s*\S+@(?:v?\d+|main|master)\s*$", re.MULTILINE)


def surviving_repos(revision: str, uuid: str, shards: int, workers: int = 12) -> set[str]:
    """Read `repo_path` from every shard of a revision. Costs ~0.1% of the data."""
    fs = HfFileSystem()

    def read(index: int) -> list[str]:
        path = f"{REPO}@{revision}/data/part-{index:05d}-{uuid}-c000.snappy.parquet"
        with fs.open(path, "rb") as raw:
            parquet_file = pq.ParquetFile(CountingReader(raw))
            names: list[str] = []
            for group in range(parquet_file.metadata.num_row_groups):
                table = parquet_file.read_row_group(group, columns=["repo_path"])
                names += table.column("repo_path").to_pylist()
            return names

    repos: set[str] = set()
    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read, i): i for i in range(shards)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                repos.update(future.result())
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"  shard {index:05d} FAILED: {error}", file=sys.stderr, flush=True)
                continue
            done += 1
            if done % 250 == 0 or done == shards:
                print(f"  [{done}/{shards}] {len(repos):,} repos  "
                      f"{time.time() - started:.0f}s", file=sys.stderr, flush=True)
    return repos


def dedupe_files(files: list[dict]) -> tuple[list[dict], int]:
    """Drop repeated (path, content) rows, preserving order."""
    seen: set[tuple[str, int]] = set()
    kept: list[dict] = []
    for entry in files:
        key = (entry.get("path") or "", hash(entry.get("content") or ""))
        if key in seen:
            continue
        seen.add(key)
        kept.append(entry)
    return kept, len(files) - len(kept)


def derived_flags(record: dict) -> dict:
    """Per-class properties a consumer would otherwise have to re-parse for."""
    unit_type = record.get("unit_type")
    files = record.get("files") or []
    blob = "\n".join(f.get("content") or "" for f in files)
    flags: dict[str, object] = {
        "file_count": len(files),
        "total_bytes": sum(len(f.get("content") or "") for f in files),
        "all_permissive": bool(record.get("license_types"))
        and set(record["license_types"]) == {"permissive"},
    }

    if unit_type == "helm_chart":
        references = bool(INCLUDE_CALL.search(blob))
        defines = bool(DEFINE_CALL.search(blob))
        # A chart calling a helper it does not carry cannot be rendered alone,
        # which is 27.1% of them and the single most important flag here.
        flags["self_contained"] = defines or not references
        flags["references_helpers"] = references
        flags["defines_helpers"] = defines
    elif unit_type == "dockerfile":
        flags["pins_digest"] = bool(DIGEST_PIN.search(blob))
        flags["uses_latest_tag"] = bool(LATEST_TAG.search(blob))
    elif unit_type == "workflow":
        flags["has_unpinned_action"] = bool(UNPINNED_ACTION.search(blob))
    return flags


def finalize_file(source: str, destination: str, keep: set[str] | None) -> dict:
    stats: Counter = Counter()
    with gzip.open(source, "rt", encoding="utf-8") as reader, \
            gzip.open(destination, "wt", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["unparseable"] += 1
                continue
            stats["read"] += 1

            if keep is not None and record.get("repo_path") not in keep:
                stats["dropped_opted_out"] += 1
                continue

            files, removed = dedupe_files(record.get("files") or [])
            if removed:
                stats["units_with_duplicates"] += 1
                stats["duplicate_files_removed"] += removed
            record["files"] = files
            record["flags"] = derived_flags(record)
            if record["flags"].get("self_contained") is False:
                stats["charts_not_self_contained"] += 1
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
    return {"source": os.path.basename(source), **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="directory of harvested *.jsonl.gz files")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--revision", default=None,
                        help="target revision to re-filter against (skip to only dedupe)")
    parser.add_argument("--uuid", default=None, help="shard file UUID of that revision")
    parser.add_argument("--shards", type=int, default=8196)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--repo-index", default="surviving_repos.txt",
                        help="cache of surviving repo_path values")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keep: set[str] | None = None

    if args.revision:
        if os.path.exists(args.repo_index):
            with open(args.repo_index) as handle:
                keep = {line.strip() for line in handle if line.strip()}
            print(f"loaded {len(keep):,} surviving repos from cache", file=sys.stderr)
        else:
            if not args.uuid:
                parser.error("--uuid is required when building the repo index")
            print(f"reading repo_path from {args.shards} shards of {args.revision}",
                  file=sys.stderr)
            keep = surviving_repos(args.revision, args.uuid, args.shards, args.workers)
            with open(args.repo_index, "w") as handle:
                handle.write("\n".join(sorted(keep)))
            print(f"wrote {len(keep):,} surviving repos to {args.repo_index}",
                  file=sys.stderr)

    results = []
    for source in sorted(glob.glob(os.path.join(args.directory, "*.jsonl.gz"))):
        destination = os.path.join(args.out, os.path.basename(source))
        result = finalize_file(source, destination, keep)
        results.append(result)
        print(f"  {result['source']:<28} {result.get('written', 0):>10,} kept  "
              f"opted_out={result.get('dropped_opted_out', 0):,}  "
              f"dup_files={result.get('duplicate_files_removed', 0):,}",
              file=sys.stderr, flush=True)

    with open(os.path.join(args.out, "finalize_stats.json"), "w") as handle:
        json.dump(results, handle, indent=2)

    total_written = sum(r.get("written", 0) for r in results)
    total_dropped = sum(r.get("dropped_opted_out", 0) for r in results)
    total_dupes = sum(r.get("duplicate_files_removed", 0) for r in results)
    print()
    print(f"units kept:                {total_written:,}")
    print(f"units dropped (opted out): {total_dropped:,}")
    print(f"duplicate files removed:   {total_dupes:,}")


if __name__ == "__main__":
    main()
