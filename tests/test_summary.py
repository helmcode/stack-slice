"""Tests for unit profiling and unbiased sampling."""

import json
import os

from stackslice.summary import ClassProfile, star_bucket, write_sample


def chart(name, templates=3, values=True, stars=0, license_types=("no_license",)):
    return {
        "unit_type": "helm_chart",
        "repo_path": name,
        "commit_id": f"sha-{name}",
        "stars": stars,
        "unit_prefix": "charts/api",
        "license_types": list(license_types),
        "quality": {
            "templates": templates,
            "templated_templates": templates,
            "has_values": values,
            "helpers": 1,
        },
        "files": [
            {"path": "charts/api/Chart.yaml", "content": "name: api\n", "size_bytes": 10},
            {"path": "charts/api/templates/a.yaml", "content": "kind: Service\n", "size_bytes": 14},
        ],
    }


def test_counts_and_averages():
    profile = ClassProfile("helm_chart", sample_size=2, seed=1)
    for i in range(5):
        profile.add(chart(f"repo{i}"))
    report = profile.report()
    assert report["units"] == 5
    assert report["files"] == 10
    assert report["mean_files_per_unit"] == 2.0
    assert report["mean_bytes_per_unit"] == 24


def test_permissive_share():
    profile = ClassProfile("helm_chart", sample_size=1, seed=1)
    profile.add(chart("a", license_types=("permissive",)))
    for i in range(3):
        profile.add(chart(f"b{i}", license_types=("no_license",)))
    assert profile.report()["permissive_pct"] == 25.0


def test_numeric_quality_percentiles():
    profile = ClassProfile("helm_chart", sample_size=1, seed=1)
    for templates in (2, 3, 4, 10, 40):
        profile.add(chart(f"r{templates}", templates=templates))
    stats = profile.report()["quality_numeric"]["templates"]
    assert stats["min"] == 2
    assert stats["max"] == 40
    assert stats["p50"] == 4


def test_boolean_quality_counts():
    profile = ClassProfile("helm_chart", sample_size=1, seed=1)
    profile.add(chart("a", values=True))
    profile.add(chart("b", values=False))
    booleans = profile.report()["quality_boolean"]
    assert booleans["has_values=True"] == 1
    assert booleans["has_values=False"] == 1


def test_star_buckets():
    assert star_bucket(None) == "0"
    assert star_bucket(0) == "0"
    assert star_bucket(5) == "1-10"
    assert star_bucket(50) == "10-100"
    assert star_bucket(500) == "100-1k"
    assert star_bucket(5_000) == "1k-10k"
    assert star_bucket(50_000) == "10k+"


def test_reservoir_keeps_a_bounded_sample():
    profile = ClassProfile("helm_chart", sample_size=3, seed=7)
    for i in range(1_000):
        profile.add(chart(f"repo{i}"))
    assert len(profile.reservoir) == 3
    assert profile.count == 1_000


def test_reservoir_is_not_just_the_first_records():
    """Head sampling would describe whichever shard was written first."""
    profile = ClassProfile("helm_chart", sample_size=5, seed=99)
    for i in range(5_000):
        profile.add(chart(f"repo{i}"))
    picked = [int(r["repo_path"].removeprefix("repo")) for r in profile.reservoir]
    assert max(picked) > 5, f"sample looks like the head of the file: {picked}"


def test_reservoir_is_deterministic_for_a_seed():
    def run():
        profile = ClassProfile("helm_chart", sample_size=4, seed=42)
        for i in range(500):
            profile.add(chart(f"repo{i}"))
        return [r["repo_path"] for r in profile.reservoir]

    assert run() == run()


def test_write_sample_materialises_the_unit(tmp_path):
    base = write_sample(str(tmp_path), chart("acme_api", stars=42), 1)
    assert os.path.isfile(os.path.join(base, "Chart.yaml"))
    assert os.path.isfile(os.path.join(base, "templates", "a.yaml"))

    with open(os.path.join(base, "Chart.yaml")) as handle:
        assert handle.read() == "name: api\n"

    with open(os.path.join(base, "_provenance.json")) as handle:
        provenance = json.load(handle)
    assert provenance["repo_path"] == "acme_api"
    assert provenance["stars"] == 42
    assert provenance["commit_id"] == "sha-acme_api"


def test_write_sample_strips_the_unit_prefix(tmp_path):
    """A chart must land as Chart.yaml, not charts/api/Chart.yaml."""
    base = write_sample(str(tmp_path), chart("x"), 2)
    assert not os.path.exists(os.path.join(base, "charts"))


def test_manifest_kinds_are_tallied():
    profile = ClassProfile("manifest_set", sample_size=1, seed=1)
    for kinds in (["Deployment", "Service"], ["Deployment", "Ingress"]):
        profile.add({
            "unit_type": "manifest_set",
            "repo_path": "r",
            "quality": {"manifests": len(kinds), "kinds": kinds},
            "files": [],
        })
    assert profile.report()["top_kinds"]["Deployment"] == 2
    assert profile.report()["top_kinds"]["Service"] == 1


def single_file_unit(unit_type, path, content):
    """Single-file classes use the file's own path as the unit prefix."""
    return {
        "unit_type": unit_type,
        "repo_path": "acme/app",
        "commit_id": "sha",
        "stars": 3,
        "unit_prefix": path,
        "license_types": ["no_license"],
        "quality": {"stages": 1},
        "files": [{"path": path, "content": content, "size_bytes": len(content)}],
    }


def test_single_file_units_land_inside_a_directory(tmp_path):
    """Regression: the unit was written as a file, so provenance could not be added."""
    for unit_type, path in (
        ("dockerfile", "build/Dockerfile"),
        ("workflow", ".github/workflows/ci.yml"),
        ("compose", "docker-compose.yml"),
    ):
        base = write_sample(str(tmp_path), single_file_unit(unit_type, path, "FROM alpine\n"), 1)
        assert os.path.isdir(base), f"{unit_type} must be a directory"
        assert os.path.isfile(os.path.join(base, os.path.basename(path)))
        assert os.path.isfile(os.path.join(base, "_provenance.json"))
