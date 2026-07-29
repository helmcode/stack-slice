"""Tests for opt-out filtering, deduplication and derived flags."""

import gzip
import json

from stackslice.finalize import dedupe_files, derived_flags, finalize_file


def write_units(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def unit(repo, unit_type="dockerfile", files=None, license_types=("no_license",)):
    return {
        "unit_type": unit_type,
        "repo_path": repo,
        "commit_id": "sha",
        "stars": 1,
        "unit_prefix": "Dockerfile",
        "license_types": list(license_types),
        "quality": {},
        "files": files if files is not None else [
            {"path": "Dockerfile", "content": "FROM alpine\n", "size_bytes": 12}
        ],
    }


def test_dedupe_removes_repeated_path_and_content():
    """The corpus repeats identical file rows inside one repository."""
    files = [
        {"path": "templates/svc.yaml", "content": "kind: Service\n"},
        {"path": "templates/svc.yaml", "content": "kind: Service\n"},
        {"path": "templates/svc.yaml", "content": "kind: Service\n"},
        {"path": "Chart.yaml", "content": "name: api\n"},
    ]
    kept, removed = dedupe_files(files)
    assert removed == 2
    assert [f["path"] for f in kept] == ["templates/svc.yaml", "Chart.yaml"]


def test_dedupe_keeps_same_path_with_different_content():
    """Only byte-identical repetition is duplication."""
    files = [
        {"path": "a.yaml", "content": "one\n"},
        {"path": "a.yaml", "content": "two\n"},
    ]
    kept, removed = dedupe_files(files)
    assert removed == 0
    assert len(kept) == 2


def test_dedupe_preserves_order():
    files = [{"path": f"{i}.tf", "content": str(i)} for i in range(5)]
    kept, _ = dedupe_files(files + files)
    assert [f["path"] for f in kept] == [f"{i}.tf" for i in range(5)]


def test_chart_referencing_a_missing_helper_is_not_self_contained():
    record = unit("acme/api", "helm_chart", files=[
        {"path": "Chart.yaml", "content": "name: api\nversion: 1.0.0\n"},
        {"path": "templates/deploy.yaml",
         "content": 'metadata:\n  name: {{ include "api.fullname" . }}\n'},
    ])
    flags = derived_flags(record)
    assert flags["references_helpers"] is True
    assert flags["defines_helpers"] is False
    assert flags["self_contained"] is False


def test_chart_carrying_its_helper_is_self_contained():
    record = unit("acme/api", "helm_chart", files=[
        {"path": "Chart.yaml", "content": "name: api\nversion: 1.0.0\n"},
        {"path": "templates/_helpers.tpl",
         "content": '{{- define "api.fullname" -}}api{{- end -}}\n'},
        {"path": "templates/deploy.yaml",
         "content": 'metadata:\n  name: {{ include "api.fullname" . }}\n'},
    ])
    assert derived_flags(record)["self_contained"] is True


def test_chart_using_no_helpers_is_self_contained():
    record = unit("acme/api", "helm_chart", files=[
        {"path": "Chart.yaml", "content": "name: api\nversion: 1.0.0\n"},
        {"path": "templates/deploy.yaml", "content": "kind: Deployment\n"},
    ])
    flags = derived_flags(record)
    assert flags["references_helpers"] is False
    assert flags["self_contained"] is True


def test_dockerfile_pinning_flags():
    pinned = unit("a/b", "dockerfile", files=[
        {"path": "Dockerfile", "content": "FROM alpine@sha256:" + "a" * 64 + "\n"}
    ])
    assert derived_flags(pinned)["pins_digest"] is True
    assert derived_flags(pinned)["uses_latest_tag"] is False

    loose = unit("a/b", "dockerfile", files=[
        {"path": "Dockerfile", "content": "FROM alpine:latest\n"}
    ])
    assert derived_flags(loose)["pins_digest"] is False
    assert derived_flags(loose)["uses_latest_tag"] is True


def test_workflow_unpinned_action_flag():
    loose = unit("a/b", "workflow", files=[
        {"path": ".github/workflows/ci.yml",
         "content": "jobs:\n  b:\n    steps:\n      - uses: actions/checkout@v4\n"}
    ])
    assert derived_flags(loose)["has_unpinned_action"] is True

    pinned = unit("a/b", "workflow", files=[
        {"path": ".github/workflows/ci.yml",
         "content": "jobs:\n  b:\n    steps:\n      - uses: actions/checkout@" + "b" * 40 + "\n"}
    ])
    assert derived_flags(pinned)["has_unpinned_action"] is False


def test_all_permissive_flag():
    assert derived_flags(unit("a/b", license_types=("permissive",)))["all_permissive"] is True
    assert derived_flags(unit("a/b", license_types=("permissive", "no_license")))["all_permissive"] is False
    assert derived_flags(unit("a/b", license_types=()))["all_permissive"] is False


def test_opted_out_repositories_are_dropped(tmp_path):
    source = tmp_path / "dockerfile.jsonl.gz"
    write_units(source, [unit("keep/one"), unit("gone/two"), unit("keep/three")])
    destination = tmp_path / "out.jsonl.gz"

    result = finalize_file(str(source), str(destination), keep={"keep/one", "keep/three"})
    assert result["read"] == 3
    assert result["written"] == 2
    assert result["dropped_opted_out"] == 1

    with gzip.open(destination, "rt") as handle:
        repos = [json.loads(line)["repo_path"] for line in handle]
    assert repos == ["keep/one", "keep/three"]


def test_no_filter_keeps_everything_but_still_dedupes(tmp_path):
    """A duplicated template is removed while the chart itself survives."""
    source = tmp_path / "helm_chart.jsonl.gz"
    files = [
        {"path": "Chart.yaml", "content": "name: api\nversion: 1.0.0\n"},
        {"path": "values.yaml", "content": "replicas: 1\n"},
        {"path": "templates/a.yaml", "content": "kind: Service\n{{ .Values.x }}\n"},
        {"path": "templates/a.yaml", "content": "kind: Service\n{{ .Values.x }}\n"},
        {"path": "templates/b.yaml", "content": "kind: Ingress\n{{ .Values.y }}\n"},
    ]
    record = unit("a/b", "helm_chart", files=files)
    record["unit_prefix"] = ""
    write_units(source, [record])
    destination = tmp_path / "out.jsonl.gz"

    result = finalize_file(str(source), str(destination), keep=None)
    assert result["written"] == 1
    assert result["duplicate_files_removed"] == 1
    assert "dropped_opted_out" not in result

    with gzip.open(destination, "rt") as handle:
        stored = json.loads(handle.readline())
    assert len(stored["files"]) == 4
    assert stored["flags"]["file_count"] == 4
    assert stored["quality"]["templates"] == 2, "the duplicate must not be counted"


def test_flags_are_attached_to_every_written_record(tmp_path):
    source = tmp_path / "dockerfile.jsonl.gz"
    write_units(source, [unit("a/b"), unit("c/d")])
    destination = tmp_path / "out.jsonl.gz"
    finalize_file(str(source), str(destination), keep=None)
    with gzip.open(destination, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            assert "flags" in record
            assert record["flags"]["total_bytes"] > 0


def test_harvest_repos_collects_distinct_repo_paths(tmp_path):
    """The index only needs the repositories a harvest actually cites."""
    from stackslice.finalize import harvest_repos

    write_units(tmp_path / "dockerfile.jsonl.gz", [unit("a/one"), unit("a/one"), unit("b/two")])
    write_units(tmp_path / "workflow.jsonl.gz", [unit("b/two", "workflow"), unit("c/three", "workflow")])
    assert harvest_repos(str(tmp_path)) == {"a/one", "b/two", "c/three"}


def test_harvest_repos_survives_bad_lines(tmp_path):
    from stackslice.finalize import harvest_repos

    path = tmp_path / "dockerfile.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(unit("good/one")) + "\n")
        handle.write("{not json\n")
        handle.write(json.dumps({"unit_type": "dockerfile"}) + "\n")
        handle.write(json.dumps(unit("good/two")) + "\n")
    assert harvest_repos(str(tmp_path)) == {"good/one", "good/two"}


def test_regate_corrects_counters_inflated_by_duplicates():
    """Three identical templates are one template, not three."""
    from stackslice.finalize import regate

    record = {
        "unit_type": "helm_chart",
        "unit_prefix": "charts/api",
        "quality": {"templates": 3, "files": 5},
        "files": [
            {"path": "charts/api/Chart.yaml", "content": "name: api\nversion: 1.0.0\n"},
            {"path": "charts/api/values.yaml", "content": "replicas: 1\n"},
            {"path": "charts/api/templates/a.yaml", "content": "kind: Service\n{{ .Values.x }}\n"},
            {"path": "charts/api/templates/b.yaml", "content": "kind: Ingress\n{{ .Values.y }}\n"},
        ],
    }
    passed, reason, quality = regate(record)
    assert passed, reason
    assert quality["templates"] == 2, "must count the files actually present"


def test_regate_drops_a_unit_that_only_passed_thanks_to_duplicates(tmp_path):
    """A chart whose templates were all the same file is not a two-template chart."""
    duplicated = [
        {"path": "c/Chart.yaml", "content": "name: api\nversion: 1.0.0\n"},
        {"path": "c/values.yaml", "content": "x: 1\n"},
        {"path": "c/templates/only.yaml", "content": "kind: Service\n{{ .Values.x }}\n"},
        {"path": "c/templates/only.yaml", "content": "kind: Service\n{{ .Values.x }}\n"},
    ]
    source = tmp_path / "helm_chart.jsonl.gz"
    record = unit("a/b", "helm_chart", files=duplicated)
    record["unit_prefix"] = "c"
    write_units(source, [record])

    result = finalize_file(str(source), str(tmp_path / "out.jsonl.gz"), keep=None)
    assert result.get("written", 0) == 0
    assert result["regated_out"] == 1
    assert result["regated/too_few_templates"] == 1


def test_single_file_units_survive_regating(tmp_path):
    """Their unit_prefix is the file path itself, which must still resolve."""
    from stackslice.finalize import regate, unit_relative

    assert unit_relative("build/Dockerfile", "build/Dockerfile") == "Dockerfile"
    record = {
        "unit_type": "dockerfile",
        "unit_prefix": "build/Dockerfile",
        "quality": {},
        "files": [{"path": "build/Dockerfile", "content": "FROM alpine\nRUN true\n"}],
    }
    passed, reason, quality = regate(record)
    assert passed, reason
    assert quality["stages"] == 1
