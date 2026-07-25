"""Tests for repository-context resolution and the YAML parser arbiter."""

import pytest

from stackslice.resolve import Resolved, resolve_repo
from stackslice.taxonomy import Precision
from stackslice.verify import (
    top_level_keys,
    verify_ansible,
    verify_chart_metadata,
    verify_compose,
    verify_github_workflow,
    verify_k8s_manifest,
)
from tests.test_detect import (
    ANSIBLE_PLAYBOOK,
    COMPOSE,
    HELM_TEMPLATE,
    HELM_VALUES,
    K8S_DEPLOYMENT,
    LOCALISATION_YAML,
    WORKFLOW,
)

CHART_YAML = "apiVersion: v2\nname: api\nversion: 1.2.3\n"
HELPERS_TPL = '{{- define "api.name" -}}api{{- end -}}\n'
ROLE_DEFAULTS = "nginx_port: 8080\nnginx_worker_processes: auto\n"


def labels(resolved):
    return [r.tool if r else None for r in resolved]


def test_chart_context_claims_values_and_templates():
    """values.yaml and _helpers.tpl carry no self-identifying content."""
    paths = [
        "charts/api/Chart.yaml",
        "charts/api/values.yaml",
        "charts/api/templates/deployment.yaml",
        "charts/api/templates/_helpers.tpl",
    ]
    contents = [CHART_YAML, HELM_VALUES, HELM_TEMPLATE, HELPERS_TPL]
    resolved = resolve_repo(paths, ["YAML"] * 4, contents)

    assert labels(resolved) == ["helm"] * 4
    assert all(r.evidence == "context" for r in resolved)
    assert all(r.unit == "charts/api" for r in resolved)


def test_chart_templates_are_no_longer_labelled_kubernetes():
    """The Phase 0.5 defect: templates/ under a chart read as Kubernetes."""
    paths = ["charts/api/Chart.yaml", "charts/api/templates/svc.yaml"]
    resolved = resolve_repo(paths, ["YAML", "YAML"], [CHART_YAML, HELM_TEMPLATE])
    assert labels(resolved) == ["helm", "helm"]


def test_plain_manifests_outside_a_chart_are_kubernetes():
    paths = ["manifests/deployment.yaml", "weird/place/svc.yaml"]
    resolved = resolve_repo(paths, ["YAML", "YAML"], [K8S_DEPLOYMENT, K8S_DEPLOYMENT])
    assert labels(resolved) == ["kubernetes", "kubernetes"]
    assert all(r.evidence in ("content", "path+content") for r in resolved)


def test_manifest_in_an_unguessable_directory_is_still_found():
    """60% of real manifests sit where no path rule would look."""
    resolved = resolve_repo(["src/main/resources/app.yaml"], ["YAML"], [K8S_DEPLOYMENT])
    assert labels(resolved) == ["kubernetes"]
    assert resolved[0].evidence == "content"


def test_deepest_chart_wins_for_subcharts():
    paths = [
        "chart/Chart.yaml",
        "chart/templates/a.yaml",
        "chart/charts/redis/Chart.yaml",
        "chart/charts/redis/templates/sts.yaml",
    ]
    resolved = resolve_repo(paths, ["YAML"] * 4, [CHART_YAML, HELM_TEMPLATE] * 2)
    assert labels(resolved) == ["helm"] * 4
    assert resolved[3].unit == "chart/charts/redis"


def test_role_context_claims_variable_files():
    """defaults/main.yml is Ansible only because of where it lives."""
    paths = [
        "roles/nginx/tasks/main.yml",
        "roles/nginx/defaults/main.yml",
        "roles/nginx/templates/nginx.conf.j2",
    ]
    contents = [ANSIBLE_PLAYBOOK, ROLE_DEFAULTS, "listen {{ nginx_port }};\n"]
    resolved = resolve_repo(paths, ["YAML", "YAML", "Jinja"], contents)
    assert labels(resolved) == ["ansible"] * 3
    assert all(r.unit == "roles/nginx" for r in resolved)


def test_unrelated_yaml_is_dropped_rather_than_guessed():
    paths = ["config/locales/en.yml", "k8s/notes.yaml"]
    resolved = resolve_repo(paths, ["YAML", "YAML"], [LOCALISATION_YAML, LOCALISATION_YAML])
    assert labels(resolved) == [None, None], "the kubernetes path guess must not survive"


def test_canonical_paths_survive_featureless_content():
    """An almost empty workflow is still a workflow."""
    resolved = resolve_repo(
        [".github/workflows/stale.yml"], ["YAML"], ["# configured elsewhere\n"]
    )
    assert labels(resolved) == ["github_actions"]
    assert resolved[0].evidence == "path"


def test_exact_path_rules_need_no_content():
    paths = ["main.tf", "Dockerfile", "flake.nix"]
    resolved = resolve_repo(paths, ["HCL", "Dockerfile", "Nix"], None)
    assert labels(resolved) == ["terraform", "dockerfile", "nix"]
    assert all(r.precision is Precision.EXACT for r in resolved)


def test_metadata_only_mode_downgrades_conventional_yaml_guesses():
    """Without content, a conventional guess is kept but marked heuristic."""
    resolved = resolve_repo(["k8s/deployment.yaml"], ["YAML"], None)
    assert labels(resolved) == ["kubernetes"]
    assert resolved[0].precision is Precision.HEURISTIC


def test_workflow_beats_a_kubernetes_directory_name():
    resolved = resolve_repo(
        [".github/workflows/deploy-k8s.yml"], ["YAML"], [WORKFLOW]
    )
    assert labels(resolved) == ["github_actions"]


# --- arbiter -----------------------------------------------------------------


def test_arbiter_accepts_real_artifacts():
    assert verify_k8s_manifest(K8S_DEPLOYMENT)
    assert verify_github_workflow(WORKFLOW)
    assert verify_ansible(ANSIBLE_PLAYBOOK)
    assert verify_compose(COMPOSE)
    assert verify_chart_metadata(CHART_YAML)


def test_arbiter_rejects_the_wrong_kind():
    assert not verify_k8s_manifest(COMPOSE)
    assert not verify_github_workflow(K8S_DEPLOYMENT)
    assert not verify_compose(K8S_DEPLOYMENT)
    assert not verify_ansible(K8S_DEPLOYMENT)
    assert not verify_chart_metadata(HELM_VALUES)


def test_arbiter_handles_yaml_boolean_on_key():
    """`on:` parses as the boolean True in YAML 1.1, which trips naive checks."""
    assert verify_github_workflow(WORKFLOW)
    assert top_level_keys(WORKFLOW)[:2] == ["name", "on"]


def test_arbiter_rejects_unparseable_templates():
    """Helm templates are not valid YAML, so the parser cannot bless them."""
    assert not verify_k8s_manifest(HELM_TEMPLATE)


def test_arbiter_survives_junk():
    for junk in ("", "\x00\x01", "a: b: c: d", "[unclosed", "@@@"):
        assert not verify_k8s_manifest(junk)
        assert not verify_github_workflow(junk)
        assert not verify_ansible(junk)


def test_top_level_keys_of_a_manifest():
    assert top_level_keys(K8S_DEPLOYMENT) == ["apiVersion", "kind", "metadata", "spec"]
