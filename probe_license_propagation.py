"""Test whether Stack v3 license labels are under-propagated.

Hypothesis: `license_type == permissive` marks files carrying an inline license
header (an Apache-2.0 convention) rather than files belonging to a permissively
licensed repository. If true, repositories that ship a LICENSE file will still
be labelled `no_license` throughout, and the repo-level license can be recovered
from the LICENSE file that the corpus already contains as a data row.
"""

import re
import sys
from collections import Counter

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from stackslice.scan import COLUMNS, CountingReader, shard_path

LICENSE_FILE = re.compile(
    r"^(LICEN[SC]E|COPYING|COPYRIGHT|UNLICENSE|NOTICE)"
    r"(\.(md|txt|rst))?$",
    re.IGNORECASE,
)


def main() -> None:
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fs = HfFileSystem()

    stats: Counter = Counter()
    label_of_license_file: Counter = Counter()
    labels_in_licensed_repos: Counter = Counter()

    with fs.open(shard_path(index), "rb") as raw:
        reader = CountingReader(raw)
        parquet_file = pq.ParquetFile(reader)
        for group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group, columns=COLUMNS)
            files = table.column("files").combine_chunks()
            flat = pc.list_flatten(files)
            paths = flat.field("file_path").to_pylist()
            types = flat.field("license_type").to_pylist()
            parent = pc.list_parent_indices(files).to_pylist()

            has_license_file: dict[int, bool] = {}
            per_repo_labels: dict[int, Counter] = {}

            for position, path in enumerate(paths):
                repo = parent[position]
                per_repo_labels.setdefault(repo, Counter())[types[position] or "null"] += 1
                name = path.rsplit("/", 1)[-1]
                if "/" not in path and LICENSE_FILE.match(name):
                    has_license_file[repo] = True
                    label_of_license_file[types[position] or "null"] += 1

            for repo, labels in per_repo_labels.items():
                licensed = has_license_file.get(repo, False)
                permissive = labels.get("permissive", 0)
                stats["repos"] += 1
                if licensed:
                    stats["repos_with_license_file"] += 1
                    labels_in_licensed_repos.update(labels)
                    if permissive == 0:
                        stats["licensed_but_all_files_unlabelled"] += 1
                    elif permissive == sum(labels.values()):
                        stats["licensed_and_fully_labelled"] += 1
                    else:
                        stats["licensed_and_partly_labelled"] += 1
                elif permissive:
                    stats["unlicensed_repo_but_permissive_files"] += 1
        print(f"fetched {reader.bytes_read / 1e6:.1f} MB", file=sys.stderr)

    repos = stats["repos"]
    with_file = stats["repos_with_license_file"]
    print()
    print(f"shard {index:05d}: {repos:,} repos")
    print(f"repos shipping a root LICENSE file : {with_file:,} ({100 * with_file / repos:.2f}%)")
    print()
    if with_file:
        print("of those repos:")
        for key in (
            "licensed_but_all_files_unlabelled",
            "licensed_and_partly_labelled",
            "licensed_and_fully_labelled",
        ):
            print(f"  {key:<38} {stats[key]:>8,} ({100 * stats[key] / with_file:>6.2f}%)")
    print()
    print(f"repos with NO license file yet some permissive files: "
          f"{stats['unlicensed_repo_but_permissive_files']:,}")
    print()
    print("license_type assigned to the LICENSE file itself:", dict(label_of_license_file))
    print("license_type spread inside repos that have a LICENSE file:",
          dict(labels_in_licensed_repos))


if __name__ == "__main__":
    main()
