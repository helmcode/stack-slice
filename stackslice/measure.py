"""Score the final labels from `resolve.py` against an independent YAML parser.

`validate.py` measured path heuristics against regex content detectors. This
measures the finished pipeline (repository context plus content plus paths)
against `verify.py`, which loads each document with PyYAML and inspects its real
structure. Agreement between two independent mechanisms is evidence; agreement
between a regex and itself is not.

It also characterises the large bucket of YAML that is not infrastructure, by
collecting the actual top-level keys of those documents rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from .detect import GO_TEMPLATE
from .resolve import resolve_repo
from .scan import TOTAL_SHARDS, CountingReader, sample_indices, shard_path
from .validate import CONTENT_COLUMNS, YAML_PATH
from .verify import (
    top_level_keys,
    verify_ansible,
    verify_chart_metadata,
    verify_compose,
    verify_github_workflow,
    verify_k8s_manifest,
)

# Final label -> the arbiter that can confirm it by parsing.
ARBITERS = {
    "kubernetes": verify_k8s_manifest,
    "github_actions": verify_github_workflow,
    "ansible": verify_ansible,
    "compose": verify_compose,
}


class Scores:
    def __init__(self) -> None:
        self.shards = 0
        self.row_groups_read = 0
        self.row_groups_total = 0
        self.bytes_fetched = 0
        self.repos = 0
        self.files = 0
        self.yaml_files = 0
        self.labels: Counter = Counter()
        self.label_bytes: Counter = Counter()
        self.evidence: dict[str, Counter] = defaultdict(Counter)
        # final label -> arbiter verdict
        self.precision: dict[str, Counter] = defaultdict(Counter)
        # arbiter verdict -> final label (for recall)
        self.recall: dict[str, Counter] = defaultdict(Counter)
        self.helm_units: Counter = Counter()
        self.other_first_keys: Counter = Counter()
        self.other_signatures: Counter = Counter()
        self.other_basenames: Counter = Counter()
        self.other_unparseable = 0
        self.other_templated = 0

    def merge(self, other: Scores) -> None:
        for name in ("shards", "row_groups_read", "row_groups_total", "bytes_fetched",
                     "repos", "files", "yaml_files", "other_unparseable", "other_templated"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in ("labels", "label_bytes", "helm_units", "other_first_keys",
                     "other_signatures", "other_basenames"):
            getattr(self, name).update(getattr(other, name))
        for name in ("evidence", "precision", "recall"):
            mine, theirs = getattr(self, name), getattr(other, name)
            for key, counter in theirs.items():
                mine[key].update(counter)


def measure_shard(index: int, row_group_limit: int | None) -> Scores:
    scores = Scores()
    scores.shards = 1
    fs = HfFileSystem()

    with fs.open(shard_path(index), "rb") as raw:
        reader = CountingReader(raw)
        parquet_file = pq.ParquetFile(reader)
        available = parquet_file.metadata.num_row_groups
        groups = range(available if row_group_limit is None else min(row_group_limit, available))
        scores.row_groups_total = available
        scores.row_groups_read = len(groups)

        for group in groups:
            table = parquet_file.read_row_group(group, columns=CONTENT_COLUMNS)
            files = table.column("files").combine_chunks()
            flat = pc.list_flatten(files)
            contents = flat.field("content")
            paths = flat.field("file_path").to_pylist()
            languages = flat.field("language").to_pylist()
            sizes = flat.field("size_bytes").to_pylist()
            parent = pc.list_parent_indices(files).to_pylist()

            scores.repos += table.num_rows
            scores.files += len(paths)

            by_repo: dict[int, list[int]] = defaultdict(list)
            for position in range(len(paths)):
                by_repo[parent[position]].append(position)

            for positions in by_repo.values():
                repo_paths = [paths[p] for p in positions]
                repo_languages = [languages[p] for p in positions]
                # Content is only needed for YAML-ish files; everything else is
                # decided by path or context, so skip materialising it.
                repo_contents = [
                    contents[p].as_py() if YAML_PATH.search(paths[p]) else None
                    for p in positions
                ]
                resolved = resolve_repo(repo_paths, repo_languages, repo_contents)

                # Distinct units this repository contributes, counted once each.
                for unit_label, prefix in {
                    (r.tool, r.unit) for r in resolved if r and r.unit is not None
                }:
                    scores.helm_units[unit_label] += 1

                for offset, result in enumerate(resolved):
                    position = positions[offset]
                    path = paths[position]
                    content = repo_contents[offset]
                    is_yaml = bool(YAML_PATH.search(path))
                    if is_yaml:
                        scores.yaml_files += 1

                    label = result.tool if result else None
                    if result:
                        scores.labels[label] += 1
                        scores.label_bytes[label] += sizes[position] or 0
                        scores.evidence[label][result.evidence] += 1

                    if not is_yaml or content is None:
                        continue

                    # Precision: does the parser agree with our final label?
                    arbiter = ARBITERS.get(label) if label else None
                    if arbiter is not None:
                        scores.precision[label][
                            "confirmed" if arbiter(content) else "unconfirmed"
                        ] += 1
                    elif label == "helm":
                        if path.rsplit("/", 1)[-1].lower().startswith("chart."):
                            scores.precision["helm"][
                                "confirmed_chart_metadata"
                                if verify_chart_metadata(content)
                                else "unconfirmed"
                            ] += 1
                        elif GO_TEMPLATE.search(content):
                            scores.precision["helm"]["confirmed_templated"] += 1
                        else:
                            scores.precision["helm"]["chart_member_untemplated"] += 1

                    # Recall: what did we call the files the parser recognises?
                    for truth, verifier in ARBITERS.items():
                        if verifier(content):
                            scores.recall[truth][label or "unlabelled"] += 1
                            break

                    # Characterise everything we deliberately dropped.
                    if label is None:
                        keys = top_level_keys(content)
                        if not keys:
                            scores.other_unparseable += 1
                            if GO_TEMPLATE.search(content):
                                scores.other_templated += 1
                        else:
                            scores.other_first_keys[keys[0]] += 1
                            scores.other_signatures[",".join(sorted(keys[:4]))] += 1
                        scores.other_basenames[path.rsplit("/", 1)[-1].lower()] += 1

        scores.bytes_fetched = reader.bytes_read

    return scores


def report(scores: Scores) -> dict:
    scale = (TOTAL_SHARDS / scores.shards) if scores.shards else 0
    if scores.row_groups_read:
        scale *= scores.row_groups_total / scores.row_groups_read

    def rate(counter: Counter, positive: tuple[str, ...]) -> float | None:
        total = sum(counter.values())
        if not total:
            return None
        good = sum(counter[key] for key in positive if key in counter)
        return round(100 * good / total, 2)

    return {
        "sample": {
            "shards": scores.shards,
            "row_groups": f"{scores.row_groups_read}/{scores.row_groups_total}",
            "bytes_fetched_gb": round(scores.bytes_fetched / 1e9, 2),
            "repos": scores.repos,
            "files": scores.files,
            "yaml_files": scores.yaml_files,
        },
        "final_labels": {
            label: {
                "files": count,
                "files_extrapolated": int(count * scale),
                "gb_extrapolated": round(scores.label_bytes[label] * scale / 1e9, 2),
                "evidence": dict(scores.evidence[label].most_common()),
            }
            for label, count in scores.labels.most_common()
        },
        "precision_vs_parser": {
            label: {
                "counts": dict(counter),
                "confirmed_pct": rate(
                    counter,
                    ("confirmed", "confirmed_chart_metadata", "confirmed_templated"),
                ),
            }
            for label, counter in scores.precision.items()
        },
        "recall_vs_parser": {
            truth: {
                "counts": dict(counter.most_common()),
                "caught_pct": rate(counter, (truth,)),
                "missed_as_unlabelled_pct": rate(counter, ("unlabelled",)),
            }
            for truth, counter in scores.recall.items()
        },
        "units_detected": {
            label: {"count": count, "extrapolated": int(count * scale)}
            for label, count in scores.helm_units.most_common()
        },
        "non_infra_yaml": {
            "unparseable": scores.other_unparseable,
            "unparseable_templated": scores.other_templated,
            "top_first_keys": dict(scores.other_first_keys.most_common(30)),
            "top_key_signatures": dict(scores.other_signatures.most_common(25)),
            "top_basenames": dict(scores.other_basenames.most_common(30)),
        },
    }


def print_report(data: dict) -> None:
    sample = data["sample"]
    print()
    print("=" * 78)
    print("FINAL LABELS SCORED AGAINST AN INDEPENDENT YAML PARSER")
    print("=" * 78)
    print(f"{sample['shards']} shard(s), row groups {sample['row_groups']}, "
          f"{sample['bytes_fetched_gb']} GB fetched")
    print(f"{sample['repos']:,} repos, {sample['files']:,} files, "
          f"{sample['yaml_files']:,} YAML")
    print()

    print(f"{'label':<18} {'files':>9} {'extrapolated':>14} {'GB':>7}  evidence")
    print("-" * 78)
    for label, stats in data["final_labels"].items():
        evidence = " ".join(f"{k}={v}" for k, v in stats["evidence"].items())
        print(f"{label:<18} {stats['files']:>9,} {stats['files_extrapolated']:>14,} "
              f"{stats['gb_extrapolated']:>7.2f}  {evidence}")
    print()

    print("PRECISION of final labels (independent parser confirms?):")
    for label, stats in data["precision_vs_parser"].items():
        print(f"  {label:<18} {stats['confirmed_pct']}%  {stats['counts']}")
    print()

    print("RECALL against the parser (parser recognises it, we said):")
    for truth, stats in data["recall_vs_parser"].items():
        print(f"  {truth:<18} caught {stats['caught_pct']}%  "
              f"missed {stats['missed_as_unlabelled_pct']}%  {stats['counts']}")
    print()

    other = data["non_infra_yaml"]
    print(f"NON-INFRASTRUCTURE YAML  (unparseable: {other['unparseable']}, "
          f"of which templated: {other['unparseable_templated']})")
    print("  most common first key:")
    for key, count in list(other["top_first_keys"].items())[:20]:
        print(f"    {key:<28} {count:>7,}")
    print("  most common filenames:")
    for name, count in list(other["top_basenames"].items())[:20]:
        print(f"    {name:<28} {count:>7,}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--row-groups", type=int, default=None)
    parser.add_argument("--out", default="measure_report.json")
    args = parser.parse_args()

    combined = Scores()
    started = time.time()
    for number, index in enumerate(sample_indices(args.shards), start=1):
        combined.merge(measure_shard(index, args.row_groups))
        print(f"  [{number}] shard {index:05d} "
              f"{combined.bytes_fetched / 1e9:.2f} GB  {time.time() - started:.0f}s",
              file=sys.stderr)

    data = report(combined)
    print_report(data)
    with open(args.out, "w") as handle:
        json.dump(data, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
