"""Tests for the parquet conversion and config layout."""

import glob
import gzip
import json
import os

import pyarrow.parquet as pq
import pytest

from stackslice.publish import CONFIG_FIELDS, configs_yaml, convert, schema_for, to_batch


def chart_record(name, self_contained=True, templates=3):
    return {
        "unit_type": "helm_chart",
        "repo_path": name,
        "commit_id": f"sha-{name}",
        "stars": 7,
        "unit_prefix": "charts/api",
        "shard": 42,
        "license_types": ["no_license"],
        "quality": {"templates": templates, "templated_templates": templates,
                    "has_values": True, "helpers": 1, "files": 3},
        "flags": {"file_count": 3, "total_bytes": 100, "all_permissive": False,
                  "self_contained": self_contained, "references_helpers": True,
                  "defines_helpers": self_contained},
        "files": [
            {"path": "charts/api/Chart.yaml", "content": "name: api\n",
             "license_type": "no_license", "detected_licenses": [], "size_bytes": 10},
            {"path": "charts/api/templates/a.yaml", "content": "kind: Service\n",
             "license_type": "no_license", "detected_licenses": ["MIT"], "size_bytes": 14},
        ],
    }


def write(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


@pytest.mark.parametrize("unit_type", sorted(CONFIG_FIELDS))
def test_every_config_has_a_schema(unit_type):
    schema = schema_for(unit_type)
    names = schema.names
    assert names[:3] == ["unit_type", "repo_path", "commit_id"]
    assert "files" in names and "quality" in names and "flags" in names
    files_type = schema.field("files").type
    assert files_type.value_type.field("content").type.equals(
        schema.field("repo_path").type
    ), "file content must be a string like other text columns"


def test_batch_matches_the_declared_schema():
    schema = schema_for("helm_chart")
    batch = to_batch([chart_record("a/b")], "helm_chart", schema)
    assert batch.schema.equals(schema)
    assert batch.num_rows == 1


def test_absent_quality_fields_become_null_not_a_new_type():
    """Inference would change types when a field is missing from a batch."""
    record = chart_record("a/b")
    del record["quality"]["helpers"]
    del record["flags"]["self_contained"]
    schema = schema_for("helm_chart")
    batch = to_batch([record], "helm_chart", schema)
    row = batch.to_pylist()[0]
    assert row["quality"]["helpers"] is None
    assert row["flags"]["self_contained"] is None
    assert batch.schema.equals(schema)


def test_convert_writes_a_readable_config(tmp_path):
    source = tmp_path / "helm_chart.jsonl.gz"
    write(source, [chart_record(f"repo/{i}") for i in range(50)])
    result = convert(str(source), str(tmp_path / "ds"))

    assert result["config"] == "helm_chart"
    assert result["rows"] == 50
    assert result["files"] == 1

    parts = glob.glob(str(tmp_path / "ds" / "data" / "helm_chart" / "*.parquet"))
    assert len(parts) == 1
    table = pq.read_table(parts[0])
    assert table.num_rows == 50
    assert table.column("repo_path").to_pylist()[0] == "repo/0"
    files = table.column("files").to_pylist()[0]
    assert files[0]["path"] == "charts/api/Chart.yaml"
    assert files[1]["detected_licenses"] == ["MIT"]


def test_nested_flags_survive_the_round_trip(tmp_path):
    source = tmp_path / "helm_chart.jsonl.gz"
    write(source, [chart_record("a/yes", True), chart_record("a/no", False)])
    convert(str(source), str(tmp_path / "ds"))
    table = pq.read_table(str(tmp_path / "ds" / "data" / "helm_chart"))
    flags = {row["repo_path"]: row["flags"]["self_contained"]
             for row in table.to_pylist()}
    assert flags == {"a/yes": True, "a/no": False}


def test_unknown_config_is_refused(tmp_path):
    source = tmp_path / "mystery.jsonl.gz"
    write(source, [chart_record("a/b")])
    with pytest.raises(SystemExit):
        convert(str(source), str(tmp_path / "ds"))


def test_no_empty_parquet_files_are_left_behind(tmp_path):
    source = tmp_path / "helm_chart.jsonl.gz"
    write(source, [chart_record(f"r/{i}") for i in range(5)])
    convert(str(source), str(tmp_path / "ds"))
    for path in glob.glob(str(tmp_path / "ds" / "data" / "helm_chart" / "*.parquet")):
        assert os.path.getsize(path) > 0


def test_configs_yaml_lists_every_config():
    results = [{"config": "helm_chart"}, {"config": "dockerfile"}]
    text = configs_yaml(results)
    assert text.splitlines()[0] == "configs:"
    assert "  - config_name: helm_chart" in text
    assert "        path: data/dockerfile/train-*.parquet" in text


def test_unparseable_lines_are_counted_not_fatal(tmp_path):
    source = tmp_path / "helm_chart.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(chart_record("good/one")) + "\n")
        handle.write("{broken\n")
        handle.write(json.dumps(chart_record("good/two")) + "\n")
    result = convert(str(source), str(tmp_path / "ds"))
    assert result["rows"] == 2
    assert result["unparseable"] == 1
