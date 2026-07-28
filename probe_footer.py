"""Inspect a Stack v3 parquet shard footer to size a metadata-only scan.

Reads only the parquet footer over HTTP range requests (no column data), then
reports per-leaf-column compressed sizes so we know what fraction of the corpus
a metadata-only pass has to pull.
"""

import sys
from collections import defaultdict

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from stackslice.scan import shard_path  # pinned revision lives there


def main() -> None:
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fs = HfFileSystem()
    path = shard_path(index)

    total_bytes = fs.info(path)["size"]
    with fs.open(path, "rb") as handle:
        meta = pq.read_metadata(handle)

    print(f"shard          : part-{index:05d}")
    print(f"file size      : {total_bytes / 1e9:.2f} GB")
    print(f"rows (repos)   : {meta.num_rows:,}")
    print(f"row groups     : {meta.num_row_groups}")
    print(f"leaf columns   : {meta.num_columns}")
    print()

    # Compressed bytes per leaf column, summed across row groups.
    per_column: dict[str, int] = defaultdict(int)
    for rg_index in range(meta.num_row_groups):
        rg = meta.row_group(rg_index)
        for col_index in range(rg.num_columns):
            col = rg.column(col_index)
            per_column[col.path_in_schema] += col.total_compressed_size

    grand_total = sum(per_column.values())
    print(f"{'leaf column':<52} {'compressed':>12} {'share':>7}")
    print("-" * 73)
    for name, size in sorted(per_column.items(), key=lambda kv: -kv[1]):
        print(f"{name:<52} {size / 1e6:>9.1f} MB {100 * size / grand_total:>6.2f}%")
    print("-" * 73)
    print(f"{'TOTAL':<52} {grand_total / 1e6:>9.1f} MB")
    print()

    # Cost of the metadata-only projection we care about for slicing.
    wanted = [
        "repo_path",
        "github_metadata.stars",
        "num_files",
        "files.list.element.file_path",
        "files.list.element.language",
        "files.list.element.size_bytes",
        "files.list.element.license_type",
        "files.list.element.is_vendor",
    ]
    meta_bytes = sum(per_column[name] for name in wanted if name in per_column)
    missing = [name for name in wanted if name not in per_column]
    if missing:
        print(f"WARNING unmatched leaf paths: {missing}")
    print(f"metadata projection : {meta_bytes / 1e6:.1f} MB "
          f"({100 * meta_bytes / grand_total:.2f}% of shard)")
    print(f"extrapolated 1090 shards: "
          f"{1090 * meta_bytes / 1e9:.1f} GB metadata "
          f"vs {1090 * grand_total / 1e12:.2f} TB full")
    print(f"rows extrapolated       : {1090 * meta.num_rows / 1e6:.1f}M repos")


if __name__ == "__main__":
    main()
