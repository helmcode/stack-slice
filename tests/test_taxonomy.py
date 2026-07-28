"""Tests for the IaC classifier and repository-level unit detector."""

import pytest

from stackslice.taxonomy import Precision, classify, detect_units


@pytest.mark.parametrize(
    ("path", "language", "tool", "precision"),
    [
        ("main.tf", "HCL", "terraform", Precision.EXACT),
        ("infra/prod/variables.tf", "HCL", "terraform", Precision.EXACT),
        ("envs/prod.tfvars", "HCL", "terraform", Precision.EXACT),
        ("terragrunt.hcl", "HCL", "terragrunt", Precision.EXACT),
        ("build.pkr.hcl", "HCL", "packer", Precision.EXACT),
        ("charts/api/Chart.yaml", "YAML", "helm", Precision.EXACT),
        ("charts/api/templates/_helpers.tpl", "Smarty", "helm", Precision.STRUCTURAL),
        ("overlays/prod/kustomization.yaml", "YAML", "kustomize", Precision.EXACT),
        ("Dockerfile", "Dockerfile", "dockerfile", Precision.EXACT),
        ("build/Dockerfile.ci", "Dockerfile", "dockerfile", Precision.EXACT),
        ("Containerfile", "Dockerfile", "dockerfile", Precision.EXACT),
        ("docker-compose.prod.yml", "YAML", "compose", Precision.EXACT),
        ("compose.yaml", "YAML", "compose", Precision.EXACT),
        (".github/workflows/release.yml", "YAML", "github_actions", Precision.EXACT),
        (".gitlab-ci.yml", "YAML", "gitlab_ci", Precision.EXACT),
        ("Jenkinsfile", "Groovy", "jenkins", Precision.EXACT),
        ("roles/nginx/tasks/main.yml", "YAML", "ansible", Precision.STRUCTURAL),
        ("group_vars/all.yml", "YAML", "ansible", Precision.STRUCTURAL),
        ("policy/authz.rego", "Open Policy Agent", "rego", Precision.EXACT),
        ("flake.nix", "Nix", "nix", Precision.EXACT),
        ("schema.cue", "CUE", "cue", Precision.EXACT),
        ("units/node-exporter.service", "INI", "systemd", Precision.EXACT),
        ("conf/nginx.conf", "Nginx", "nginx", Precision.EXACT),
        ("monitoring/prometheus.yml", "YAML", "prometheus", Precision.EXACT),
        ("BUILD.bazel", "Starlark", "bazel", Precision.EXACT),
        ("k8s/deployment.yaml", "YAML", "kubernetes", Precision.HEURISTIC),
    ],
)
def test_classify_known_paths(path, language, tool, precision):
    result = classify(path, language)
    assert result is not None, f"{path} was not classified"
    assert result.tool == tool
    assert result.precision == precision


def test_ci_paths_win_over_generic_manifest_hint():
    """A workflow lives under .github/workflows and must not read as Kubernetes."""
    result = classify(".github/workflows/deploy-to-k8s.yml", "YAML")
    assert result is not None
    assert result.tool == "github_actions"


def test_puppet_requires_matching_language():
    """`.pp` is Puppet only when enry agrees; it collides with other formats."""
    assert classify("manifests/site.pp", "Puppet").tool == "puppet"
    assert classify("manifests/site.pp", "Pascal") is None


@pytest.mark.parametrize(
    "path",
    [
        ".terraform/modules/vpc/main.tf",
        "node_modules/pkg/Dockerfile",
        "vendor/github.com/foo/main.tf",
        "charts/parent/charts/child/Chart.yaml",
        ".terragrunt-cache/abc/main.tf",
    ],
)
def test_vendored_and_cached_paths_are_dropped(path):
    assert classify(path) is None


def test_unrelated_source_is_not_classified():
    assert classify("src/components/Button.tsx", "TSX") is None
    assert classify("README.md", "Markdown") is None


def test_helm_chart_unit_needs_metadata_and_templates():
    units = detect_units(
        [
            "charts/api/Chart.yaml",
            "charts/api/values.yaml",
            "charts/api/templates/deployment.yaml",
            "charts/api/templates/service.yaml",
        ]
    )
    assert units["helm_chart"] == ["charts/api"]
    assert units["helm_chart_with_values"] == ["charts/api"]
    assert units["helm_umbrella"] == []


def test_chart_without_templates_is_an_umbrella():
    units = detect_units(["charts/stack/Chart.yaml", "charts/stack/values.yaml"])
    assert units["helm_chart"] == []
    assert units["helm_umbrella"] == ["charts/stack"]


def test_terraform_module_requires_more_than_one_file():
    single = detect_units(["infra/main.tf"])
    assert single["terraform_dir"] == ["infra"]
    assert single["terraform_module"] == []

    real = detect_units(["infra/main.tf", "infra/variables.tf", "infra/outputs.tf"])
    assert real["terraform_module"] == ["infra"]


def test_ansible_role_detection():
    units = detect_units(
        [
            "roles/postgres/tasks/main.yml",
            "roles/postgres/defaults/main.yml",
            "roles/redis/tasks/main.yml",
            "site.yml",
        ]
    )
    assert units["ansible_role"] == ["roles/postgres", "roles/redis"]


def test_units_ignore_vendored_subcharts():
    units = detect_units(
        [
            "charts/app/Chart.yaml",
            "charts/app/templates/deploy.yaml",
            "charts/app/charts/redis/Chart.yaml",
            "charts/app/charts/redis/templates/sts.yaml",
        ]
    )
    assert units["helm_chart"] == ["charts/app"]


def test_root_level_chart_is_found():
    units = detect_units(["Chart.yaml", "values.yaml", "templates/deployment.yaml"])
    assert units["helm_chart"] == [""]


def test_shard_paths_are_pinned_to_a_revision():
    """Upstream rewrites the dataset in place; an unpinned path breaks mid-sweep."""
    from stackslice.scan import REVISION, shard_path

    path = shard_path(0)
    assert f"@{REVISION}" in path
    assert path.endswith("part-00000-4beed122-1346-42f6-82eb-5757f2b6305f-c000.snappy.parquet")
    assert shard_path(8195).count("part-08195") == 1
    # An explicit empty revision still works, for reading whatever main holds.
    assert "@" not in shard_path(0, revision="")
