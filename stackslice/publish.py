"""Convert finalized units into parquet, one Hugging Face config per class.

Each class becomes its own config, which is why the schemas may differ between
them: a Helm chart's quality fields have nothing to say about a Dockerfile. Within
a config the schema is declared explicitly rather than inferred, because inference
over millions of records silently changes types when a field happens to be absent
from the first batch.

Files stay nested as a list of structs, mirroring The Stack v3's own shape, so a
unit arrives as the directory of files it was extracted from.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

# Hugging Face reads a directory per config; ~450 MB per file keeps streaming
# responsive without producing thousands of tiny files.
TARGET_FILE_BYTES = 450 * 1024 * 1024
BATCH_ROWS = 2_000

FILE_STRUCT = pa.struct([
    pa.field("path", pa.string()),
    pa.field("content", pa.string()),
    pa.field("license_type", pa.string()),
    pa.field("detected_licenses", pa.list_(pa.string())),
    pa.field("size_bytes", pa.int32()),
])

BASE_FIELDS = [
    pa.field("unit_type", pa.string()),
    pa.field("repo_path", pa.string()),
    pa.field("commit_id", pa.string()),
    pa.field("stars", pa.int32()),
    pa.field("unit_prefix", pa.string()),
    pa.field("shard", pa.int32()),
    pa.field("license_types", pa.list_(pa.string())),
    pa.field("files", pa.list_(FILE_STRUCT)),
]

BOOL, INT, STRINGS = pa.bool_(), pa.int32(), pa.list_(pa.string())

# Per-config quality and flag fields, declared so the schema never depends on
# which records happen to appear first.
CONFIG_FIELDS: dict[str, dict[str, list[tuple[str, pa.DataType]]]] = {
    "helm_chart": {
        "quality": [("templates", INT), ("templated_templates", INT),
                    ("has_values", BOOL), ("helpers", INT), ("files", INT)],
        "flags": [("file_count", INT), ("total_bytes", INT), ("all_permissive", BOOL),
                  ("self_contained", BOOL), ("references_helpers", BOOL),
                  ("defines_helpers", BOOL)],
    },
    "terraform_module": {
        "quality": [("tf_files", INT), ("declaring", INT), ("has_variables", BOOL),
                    ("has_outputs", BOOL), ("files", INT)],
        "flags": [("file_count", INT), ("total_bytes", INT), ("all_permissive", BOOL)],
    },
    "manifest_set": {
        "quality": [("manifests", INT), ("kinds", STRINGS), ("files", INT)],
        "flags": [("file_count", INT), ("total_bytes", INT), ("all_permissive", BOOL)],
    },
    "ansible_role": {
        "quality": [("task_files", INT), ("has_defaults", BOOL), ("has_handlers", BOOL),
                    ("has_templates", BOOL), ("files", INT)],
        "flags": [("file_count", INT), ("total_bytes", INT), ("all_permissive", BOOL)],
    },
    "dockerfile": {
        "quality": [("stages", INT), ("multi_stage", BOOL), ("instructions", INT),
                    ("has_user", BOOL), ("has_healthcheck", BOOL), ("files", INT)],
        "flags": [("file_count", INT), ("total_bytes", INT), ("all_permissive", BOOL),
                  ("pins_digest", BOOL), ("uses_latest_tag", BOOL)],
    },
    "workflow": {
        "quality": [("jobs", INT), ("steps", INT), ("uses_actions", INT),
                    ("triggers", STRINGS), ("has_permissions", BOOL), ("files", INT)],
        "flags": [("file_count", INT), ("total_bytes", INT), ("all_permissive", BOOL),
                  ("has_unpinned_action", BOOL)],
    },
    "compose": {
        "quality": [("services", INT), ("built_services", INT),
                    ("image_only_services", INT), ("has_healthcheck", BOOL),
                    ("has_volumes", BOOL), ("has_networks", BOOL), ("files", INT)],
        "flags": [("file_count", INT), ("total_bytes", INT), ("all_permissive", BOOL)],
    },
}


def schema_for(unit_type: str) -> pa.Schema:
    """Explicit schema for one config."""
    spec = CONFIG_FIELDS[unit_type]
    return pa.schema([
        *BASE_FIELDS,
        pa.field("quality", pa.struct([pa.field(n, t) for n, t in spec["quality"]])),
        pa.field("flags", pa.struct([pa.field(n, t) for n, t in spec["flags"]])),
    ])


def _sub(record: dict, key: str, fields: list[tuple[str, pa.DataType]]) -> dict:
    """Project a nested dict onto declared fields, filling absences with None."""
    source = record.get(key) or {}
    return {name: source.get(name) for name, _ in fields}


def to_batch(records: list[dict], unit_type: str, schema: pa.Schema) -> pa.RecordBatch:
    spec = CONFIG_FIELDS[unit_type]
    rows = []
    for record in records:
        rows.append({
            "unit_type": record.get("unit_type"),
            "repo_path": record.get("repo_path"),
            "commit_id": record.get("commit_id"),
            "stars": record.get("stars") or 0,
            "unit_prefix": record.get("unit_prefix") or "",
            "shard": record.get("shard"),
            "license_types": record.get("license_types") or [],
            "files": [
                {
                    "path": f.get("path"),
                    "content": f.get("content") or "",
                    "license_type": f.get("license_type"),
                    "detected_licenses": f.get("detected_licenses") or [],
                    "size_bytes": f.get("size_bytes") or 0,
                }
                for f in record.get("files") or []
            ],
            "quality": _sub(record, "quality", spec["quality"]),
            "flags": _sub(record, "flags", spec["flags"]),
        })
    return pa.RecordBatch.from_pylist(rows, schema=schema)


def convert(source: str, out_root: str, compression: str = "zstd") -> dict:
    """Write one class as `data/<config>/train-*.parquet`, rolled by size."""
    unit_type = os.path.basename(source).split(".")[0]
    if unit_type not in CONFIG_FIELDS:
        raise SystemExit(f"unknown config: {unit_type}")
    schema = schema_for(unit_type)
    directory = os.path.join(out_root, "data", unit_type)
    os.makedirs(directory, exist_ok=True)

    stats: Counter = Counter()
    parts: list[str] = []
    writer = None
    current = None

    def open_part(index: int):
        path = os.path.join(directory, f"train-{index:05d}.parquet")
        parts.append(path)
        return pq.ParquetWriter(path, schema, compression=compression), path

    batch: list[dict] = []

    def flush():
        nonlocal writer, current, batch
        if not batch:
            return
        record_batch = to_batch(batch, unit_type, schema)
        if writer is None:
            writer, current = open_part(0)
        writer.write_batch(record_batch)
        stats["rows"] += len(batch)
        batch = []
        # Roll to a new file once the current one is big enough.
        if os.path.getsize(current) >= TARGET_FILE_BYTES:
            writer.close()
            writer, current = open_part(len(parts))

    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError:
                stats["unparseable"] += 1
                continue
            if len(batch) >= BATCH_ROWS:
                flush()
        flush()

    if writer is not None:
        writer.close()
    # A rolled-over writer can leave a final empty file behind.
    parts = [p for p in parts if os.path.exists(p) and os.path.getsize(p) > 0]
    empties = [p for p in glob.glob(os.path.join(directory, "*.parquet")) if p not in parts]
    for path in empties:
        os.remove(path)

    return {
        "config": unit_type,
        "rows": stats["rows"],
        "unparseable": stats["unparseable"],
        "files": len(parts),
        "bytes": sum(os.path.getsize(p) for p in parts),
    }


def configs_yaml(results: list[dict]) -> str:
    """The `configs:` block a dataset card needs for multi-config loading."""
    lines = ["configs:"]
    for result in results:
        lines.append(f"  - config_name: {result['config']}")
        lines.append("    data_files:")
        lines.append("      - split: train")
        lines.append(f"        path: data/{result['config']}/train-*.parquet")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="directory of finalized *.jsonl.gz files")
    parser.add_argument("--out", required=True, help="dataset root to write into")
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip"])
    args = parser.parse_args()

    results = []
    for source in sorted(glob.glob(os.path.join(args.directory, "*.jsonl.gz"))):
        result = convert(source, args.out, args.compression)
        results.append(result)
        print(f"  {result['config']:<18} {result['rows']:>10,} rows  "
              f"{result['files']:>3} files  {result['bytes'] / 1e9:>6.2f} GB",
              file=sys.stderr, flush=True)

    total_rows = sum(r["rows"] for r in results)
    total_bytes = sum(r["bytes"] for r in results)
    with open(os.path.join(args.out, "publish_stats.json"), "w") as handle:
        json.dump({"configs": results, "rows": total_rows, "bytes": total_bytes},
                  handle, indent=2)
    print()
    print(f"{total_rows:,} rows across {len(results)} configs, "
          f"{total_bytes / 1e9:.2f} GB of parquet")
    print()
    print(configs_yaml(results))


if __name__ == "__main__":
    main()
