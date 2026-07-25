"""Measure the license_type distribution across whole shards.

The dataset card states files are labelled `permissive`, `no_license` (nothing
detected, or only non-license legal text) or `non_permissive`, and that
`non_permissive` was dropped before release. That leaves a mix of permissive and
unlicensed code, so any derived artifact we publish has to know the real split
before promising a permissive-only slice.
"""

import sys
from collections import Counter

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from stackslice.scan import COLUMNS, CountingReader, shard_path

EXTRA_COLUMNS = COLUMNS + ["files.list.element.detected_licenses"]


def main() -> None:
    indices = [int(a) for a in sys.argv[1:]] or [0, 4098, 8195]
    fs = HfFileSystem()

    files_by_type: Counter = Counter()
    bytes_by_type: Counter = Counter()
    repo_purity: Counter = Counter()
    licenses: Counter = Counter()
    fetched = 0

    for index in indices:
        with fs.open(shard_path(index), "rb") as raw:
            reader = CountingReader(raw)
            parquet_file = pq.ParquetFile(reader)
            for group in range(parquet_file.metadata.num_row_groups):
                table = parquet_file.read_row_group(group, columns=EXTRA_COLUMNS)
                files = table.column("files").combine_chunks()
                flat = pc.list_flatten(files)
                types = flat.field("license_type").to_pylist()
                sizes = flat.field("size_bytes").to_pylist()
                detected = flat.field("detected_licenses").to_pylist()
                parent = pc.list_parent_indices(files).to_pylist()

                for position, license_type in enumerate(types):
                    key = license_type or "null"
                    files_by_type[key] += 1
                    bytes_by_type[key] += sizes[position] or 0
                    for name in detected[position] or []:
                        licenses[name] += 1

                # Is a repository uniformly licensed, or a mix?
                per_repo: dict[int, set[str]] = {}
                for position, license_type in enumerate(types):
                    per_repo.setdefault(parent[position], set()).add(license_type or "null")
                for kinds in per_repo.values():
                    if kinds == {"permissive"}:
                        repo_purity["all_permissive"] += 1
                    elif "permissive" in kinds:
                        repo_purity["mixed"] += 1
                    else:
                        repo_purity["none_permissive"] += 1
            fetched += reader.bytes_read
        print(f"shard {index:05d} done ({fetched / 1e6:.0f} MB fetched)", file=sys.stderr)

    total_files = sum(files_by_type.values())
    total_bytes = sum(bytes_by_type.values())
    total_repos = sum(repo_purity.values())

    print()
    print(f"shards: {indices}")
    print(f"files : {total_files:,}   bytes: {total_bytes / 1e9:.2f} GB   repos: {total_repos:,}")
    print()
    print(f"{'license_type':<18} {'files':>14} {'share':>8} {'GB':>10} {'byte share':>11}")
    print("-" * 66)
    for key, count in files_by_type.most_common():
        print(
            f"{key:<18} {count:>14,} {100 * count / total_files:>7.2f}% "
            f"{bytes_by_type[key] / 1e9:>9.2f} {100 * bytes_by_type[key] / total_bytes:>10.2f}%"
        )
    print()
    print("repository purity:")
    for key, count in repo_purity.most_common():
        print(f"  {key:<18} {count:>10,} {100 * count / total_repos:>6.2f}%")
    print()
    print("top detected_licenses:")
    for name, count in licenses.most_common(15):
        print(f"  {name:<40} {count:>12,}")


if __name__ == "__main__":
    main()
