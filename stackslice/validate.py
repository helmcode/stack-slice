"""Second pass: read contents to measure the accuracy of the path taxonomy.

Unlike the metadata scan, this pass cannot be cheap. Parquet stores one column
chunk per row group, so asking for `content` at all pulls the content of every
repository in that row group (~145 MB per row group, ~575 MB per shard). The
answer it buys is worth it on a small sample: how precise the heuristic classes
are, and how much real infrastructure sits in directories the path rules never
guessed.

It also extracts a handful of complete Helm charts and Terraform modules to
disk so the raw material for a benchmark can be inspected by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from .detect import YamlKind, classify_yaml, is_dockerfile, is_terraform, looks_generated
from .scan import TOTAL_SHARDS, CountingReader, sample_indices, shard_path
from .taxonomy import detect_units

CONTENT_COLUMNS = [
    "repo_path",
    "github_metadata",
    "files.list.element.file_path",
    "files.list.element.language",
    "files.list.element.size_bytes",
    "files.list.element.license_type",
    "files.list.element.content",
]

YAML_PATH = re.compile(r"\.ya?ml$", re.IGNORECASE)
DOCKERFILE_PATH = re.compile(r"(^|/)(Dockerfile|Containerfile)[^/]*$|\.dockerfile$", re.IGNORECASE)
SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]")


class Results:
    """Cross-tabulations between path-derived and content-derived labels."""

    def __init__(self) -> None:
        self.shards = 0
        self.row_groups_read = 0
        self.row_groups_total = 0
        self.bytes_fetched = 0
        self.repos = 0
        self.files = 0
        self.yaml_files = 0
        self.yaml_bytes = 0
        # content label -> path class (or "unclassified")
        self.recall: dict[str, Counter] = defaultdict(Counter)
        # path class -> content label
        self.precision: dict[str, Counter] = defaultdict(Counter)
        self.content_counts: Counter = Counter()
        self.content_bytes: Counter = Counter()
        self.terraform_confirmed = Counter()
        self.dockerfile_confirmed = Counter()
        self.generated = Counter()
        self.manifest_licenses: Counter = Counter()

    def merge(self, other: Results) -> None:
        self.shards += other.shards
        self.row_groups_read += other.row_groups_read
        self.row_groups_total += other.row_groups_total
        self.bytes_fetched += other.bytes_fetched
        self.repos += other.repos
        self.files += other.files
        self.yaml_files += other.yaml_files
        self.yaml_bytes += other.yaml_bytes
        for label, counter in other.recall.items():
            self.recall[label].update(counter)
        for label, counter in other.precision.items():
            self.precision[label].update(counter)
        for name in (
            "content_counts",
            "content_bytes",
            "terraform_confirmed",
            "dockerfile_confirmed",
            "generated",
            "manifest_licenses",
        ):
            getattr(self, name).update(getattr(other, name))


def _write_sample(root: str, repo: str, prefix: str, files: list[tuple[str, str]]) -> None:
    """Dump one extracted unit to disk for manual inspection."""
    safe_repo = SAFE_SEGMENT.sub("_", repo)
    base = os.path.join(root, safe_repo)
    for path, content in files:
        relative = path[len(prefix):].lstrip("/") if prefix else path
        target = os.path.join(base, *[SAFE_SEGMENT.sub("_", p) for p in relative.split("/")])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8", errors="replace") as handle:
            handle.write(content)


def validate_shard(
    index: int,
    row_group_limit: int | None,
    sample_root: str | None,
    max_samples: int,
    min_stars: int,
) -> Results:
    results = Results()
    results.shards = 1
    fs = HfFileSystem()
    samples_taken = 0

    with fs.open(shard_path(index), "rb") as raw:
        reader = CountingReader(raw)
        parquet_file = pq.ParquetFile(reader)
        available = parquet_file.metadata.num_row_groups
        groups = range(available if row_group_limit is None else min(row_group_limit, available))
        results.row_groups_total = available
        results.row_groups_read = len(groups)

        from .taxonomy import classify  # local import keeps the module import graph flat

        for group in groups:
            table = parquet_file.read_row_group(group, columns=CONTENT_COLUMNS)
            files = table.column("files").combine_chunks()
            repo_paths = table.column("repo_path").to_pylist()
            stars = table.column("github_metadata").combine_chunks().field("stars").to_pylist()

            flat = pc.list_flatten(files)
            contents = flat.field("content")
            paths = flat.field("file_path").to_pylist()
            languages = flat.field("language").to_pylist()
            sizes = flat.field("size_bytes").to_pylist()
            licenses = flat.field("license_type").to_pylist()
            parent = pc.list_parent_indices(files).to_pylist()

            results.repos += table.num_rows
            results.files += len(paths)

            for position, path in enumerate(paths):
                size = sizes[position] or 0
                hit = classify(path, languages[position])
                path_class = hit.tool if hit else "unclassified"

                is_yaml = YAML_PATH.search(path) or languages[position] == "YAML"
                if is_yaml:
                    content = contents[position].as_py() or ""
                    results.yaml_files += 1
                    results.yaml_bytes += size
                    label = classify_yaml(content, path).value
                    results.content_counts[label] += 1
                    results.content_bytes[label] += size
                    results.recall[label][path_class] += 1
                    if hit:
                        results.precision[path_class][label] += 1
                    if label == YamlKind.K8S_MANIFEST.value:
                        results.manifest_licenses[licenses[position] or "unknown"] += 1
                    if looks_generated(content):
                        results.generated[label] += 1
                elif path.endswith(".tf"):
                    content = contents[position].as_py() or ""
                    results.terraform_confirmed[
                        "confirmed" if is_terraform(content) else "rejected"
                    ] += 1
                elif DOCKERFILE_PATH.search(path):
                    content = contents[position].as_py() or ""
                    results.dockerfile_confirmed[
                        "confirmed" if is_dockerfile(content) else "rejected"
                    ] += 1

            if sample_root and samples_taken < max_samples:
                by_repo: dict[int, list[int]] = defaultdict(list)
                for position in range(len(paths)):
                    by_repo[parent[position]].append(position)

                for repo_index, positions in by_repo.items():
                    if samples_taken >= max_samples:
                        break
                    if (stars[repo_index] or 0) < min_stars:
                        continue
                    units = detect_units([paths[p] for p in positions])
                    for prefix in units["helm_chart"]:
                        if samples_taken >= max_samples:
                            break
                        wanted = [
                            (paths[p], contents[p].as_py() or "")
                            for p in positions
                            if paths[p].startswith(f"{prefix}/" if prefix else "")
                        ]
                        if not wanted:
                            continue
                        _write_sample(
                            os.path.join(sample_root, "helm"),
                            f"{repo_paths[repo_index]}-{prefix or 'root'}",
                            prefix,
                            wanted,
                        )
                        samples_taken += 1

        results.bytes_fetched = reader.bytes_read

    return results


def report(results: Results) -> dict:
    scale = (TOTAL_SHARDS / results.shards) if results.shards else 0
    if results.row_groups_read:
        scale *= results.row_groups_total / results.row_groups_read

    manifests = results.content_counts.get(YamlKind.K8S_MANIFEST.value, 0)
    templates = results.content_counts.get(YamlKind.HELM_TEMPLATE.value, 0)

    def as_pct(counter: Counter) -> dict:
        total = sum(counter.values())
        return {
            key: {"n": value, "pct": round(100 * value / total, 2)}
            for key, value in counter.most_common()
        } if total else {}

    return {
        "sample": {
            "shards": results.shards,
            "row_groups": f"{results.row_groups_read}/{results.row_groups_total}",
            "bytes_fetched_gb": round(results.bytes_fetched / 1e9, 2),
            "repos": results.repos,
            "files": results.files,
            "yaml_files": results.yaml_files,
        },
        "yaml_composition": as_pct(results.content_counts),
        "extrapolated": {
            "k8s_manifests": int(manifests * scale),
            "k8s_manifest_gb": round(
                results.content_bytes.get(YamlKind.K8S_MANIFEST.value, 0) * scale / 1e9, 2
            ),
            "helm_templates": int(templates * scale),
            "helm_template_gb": round(
                results.content_bytes.get(YamlKind.HELM_TEMPLATE.value, 0) * scale / 1e9, 2
            ),
            "yaml_files_total": int(results.yaml_files * scale),
        },
        "recall_of_path_rules": {
            label: as_pct(counter) for label, counter in results.recall.items()
        },
        "precision_of_path_rules": {
            label: as_pct(counter) for label, counter in results.precision.items()
        },
        "terraform_confirmation": dict(results.terraform_confirmed),
        "dockerfile_confirmation": dict(results.dockerfile_confirmed),
        "generated_by_label": dict(results.generated.most_common()),
        "k8s_manifest_licenses": dict(results.manifest_licenses),
    }


def print_report(data: dict) -> None:
    sample, extra = data["sample"], data["extrapolated"]
    print()
    print("=" * 78)
    print("CONTENT VALIDATION OF THE PATH TAXONOMY")
    print("=" * 78)
    print(
        f"{sample['shards']} shard(s), row groups {sample['row_groups']}, "
        f"{sample['bytes_fetched_gb']} GB fetched"
    )
    print(
        f"{sample['repos']:,} repos, {sample['files']:,} files, "
        f"{sample['yaml_files']:,} YAML files inspected"
    )
    print()

    print("what YAML in the corpus actually is:")
    for label, stats in data["yaml_composition"].items():
        print(f"  {label:<20} {stats['n']:>9,} {stats['pct']:>7.2f}%")
    print()

    print("extrapolated to the full corpus:")
    print(f"  real k8s manifests : {extra['k8s_manifests']:,} files, {extra['k8s_manifest_gb']} GB")
    print(f"  helm templates     : {extra['helm_templates']:,} files, {extra['helm_template_gb']} GB")
    print(f"  all YAML           : {extra['yaml_files_total']:,} files")
    print()

    for label in ("k8s_manifest", "helm_template", "ansible", "github_actions", "compose"):
        rows = data["recall_of_path_rules"].get(label)
        if not rows:
            continue
        print(f"RECALL - content says {label}, path rules said:")
        for path_class, stats in list(rows.items())[:6]:
            print(f"  {path_class:<20} {stats['n']:>9,} {stats['pct']:>7.2f}%")
        print()

    for path_class in ("kubernetes", "helm", "ansible"):
        rows = data["precision_of_path_rules"].get(path_class)
        if not rows:
            continue
        print(f"PRECISION - path rule said {path_class}, content says:")
        for label, stats in list(rows.items())[:6]:
            print(f"  {label:<20} {stats['n']:>9,} {stats['pct']:>7.2f}%")
        print()

    print("terraform .tf confirmation :", data["terraform_confirmation"])
    print("Dockerfile confirmation    :", data["dockerfile_confirmation"])
    print("k8s manifest license_type  :", data["k8s_manifest_licenses"])
    print("files marked generated     :", data["generated_by_label"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--row-groups", type=int, default=None)
    parser.add_argument("--samples", default=None, help="directory to extract units into")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--min-stars", type=int, default=10)
    parser.add_argument("--out", default="validate_report.json")
    args = parser.parse_args()

    combined = Results()
    started = time.time()
    for number, index in enumerate(sample_indices(args.shards), start=1):
        combined.merge(
            validate_shard(
                index, args.row_groups, args.samples, args.max_samples, args.min_stars
            )
        )
        print(
            f"  [{number}] shard {index:05d} "
            f"{combined.bytes_fetched / 1e9:.2f} GB  {time.time() - started:.0f}s",
            file=sys.stderr,
        )

    data = report(combined)
    print_report(data)
    with open(args.out, "w") as handle:
        json.dump(data, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
