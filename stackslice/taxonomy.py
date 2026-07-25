"""Infrastructure-as-Code taxonomy for The Stack v3.

Language detection alone cannot separate infrastructure code: Helm templates,
Kubernetes manifests, Ansible playbooks, CI pipelines and Prometheus rules are
all reported as `YAML` by go-enry. Classification therefore combines the
detected language with path structure, and a second repository-level pass that
recognises whole units (a Helm chart, a Terraform module, an Ansible role).

Every rule carries a precision label so downstream reports never conflate a
high-confidence class (`*.tf` is Terraform) with a heuristic one (a YAML file
under `k8s/` is probably a manifest, but only content inspection can confirm).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

YAML_EXT = r"\.ya?ml$"


class Precision(str, Enum):
    """How much the classification can be trusted without reading contents."""

    EXACT = "exact"  # extension or filename is unambiguous
    STRUCTURAL = "structural"  # confirmed by repository-level structure
    HEURISTIC = "heuristic"  # path hint only, needs content validation


@dataclass(frozen=True)
class Rule:
    tool: str
    pattern: re.Pattern[str]
    precision: Precision
    languages: frozenset[str] | None = None


def _rule(
    tool: str,
    pattern: str,
    precision: Precision,
    languages: set[str] | None = None,
) -> Rule:
    return Rule(
        tool=tool,
        pattern=re.compile(pattern, re.IGNORECASE),
        precision=precision,
        languages=frozenset(languages) if languages else None,
    )


# Ordered by specificity: the first match wins, so narrow paths such as
# `.github/workflows/*.yml` must precede the generic YAML manifest hints.
RULES: tuple[Rule, ...] = (
    # --- CI / CD pipelines -------------------------------------------------
    _rule("github_actions", r"(^|/)\.github/workflows/[^/]+" + YAML_EXT, Precision.EXACT),
    _rule("github_actions", r"(^|/)\.github/actions/[^/]*action" + YAML_EXT, Precision.EXACT),
    _rule("gitlab_ci", r"(^|/)\.gitlab-ci([^/]*)?" + YAML_EXT, Precision.EXACT),
    _rule("gitlab_ci", r"(^|/)\.gitlab/ci/.+" + YAML_EXT, Precision.STRUCTURAL),
    _rule("jenkins", r"(^|/)Jenkinsfile[^/]*$", Precision.EXACT),
    _rule("circleci", r"(^|/)\.circleci/[^/]+" + YAML_EXT, Precision.EXACT),
    _rule("azure_pipelines", r"(^|/)azure-pipelines[^/]*" + YAML_EXT, Precision.EXACT),
    _rule("drone", r"(^|/)\.drone" + YAML_EXT, Precision.EXACT),
    _rule("buildkite", r"(^|/)\.buildkite/[^/]+" + YAML_EXT, Precision.EXACT),
    _rule("travis", r"(^|/)\.travis" + YAML_EXT, Precision.EXACT),
    _rule("codebuild", r"(^|/)buildspec[^/]*" + YAML_EXT, Precision.EXACT),
    _rule("tekton", r"(^|/)\.tekton/[^/]+" + YAML_EXT, Precision.STRUCTURAL),
    # --- Terraform / OpenTofu / Packer ------------------------------------
    _rule("packer", r"\.pkr\.(hcl|json)$", Precision.EXACT),
    _rule("terraform", r"\.tf$", Precision.EXACT),
    _rule("terraform", r"\.tfvars(\.json)?$", Precision.EXACT),
    _rule("terraform", r"\.tf\.json$", Precision.EXACT),
    _rule("terragrunt", r"(^|/)terragrunt\.hcl$", Precision.EXACT),
    # --- Kubernetes ecosystem ---------------------------------------------
    _rule("helm", r"(^|/)Chart" + YAML_EXT, Precision.EXACT),
    _rule("helm", r"(^|/)templates/[^/]+\.tpl$", Precision.STRUCTURAL),
    _rule("kustomize", r"(^|/)kustomization" + YAML_EXT, Precision.EXACT),
    _rule("skaffold", r"(^|/)skaffold" + YAML_EXT, Precision.EXACT),
    _rule("helmfile", r"(^|/)helmfile[^/]*" + YAML_EXT, Precision.EXACT),
    # --- Containers --------------------------------------------------------
    _rule("compose", r"(^|/)(docker-)?compose[^/]*" + YAML_EXT, Precision.EXACT),
    _rule("dockerfile", r"(^|/)(Dockerfile|Containerfile)[^/]*$", Precision.EXACT),
    _rule("dockerfile", r"\.(dockerfile|containerfile)$", Precision.EXACT),
    # --- Configuration management -----------------------------------------
    _rule("ansible", r"(^|/)roles/[^/]+/(tasks|handlers|defaults|vars|meta)/[^/]+" + YAML_EXT, Precision.STRUCTURAL),
    _rule("ansible", r"(^|/)(group_vars|host_vars)/[^/]+", Precision.STRUCTURAL),
    # `deploy*.yml` and `main.yml` are deliberately absent: they collide with
    # Kubernetes manifests (`deployment.yaml`) far more often than they identify
    # a playbook. Role-scoped `main.yml` is already covered structurally above.
    _rule("ansible", r"(^|/)(site|playbooks?|provision)(-[^/]*)?" + YAML_EXT, Precision.HEURISTIC),
    _rule("ansible", r"(^|/)ansible\.cfg$", Precision.EXACT),
    _rule("ansible", r"(^|/)(inventory|hosts)(\.(ini|ya?ml))?$", Precision.HEURISTIC),
    _rule("puppet", r"\.pp$", Precision.EXACT, {"Puppet"}),
    _rule("chef", r"(^|/)(recipes|attributes|libraries)/[^/]+\.rb$", Precision.STRUCTURAL),
    _rule("chef", r"(^|/)metadata\.rb$", Precision.EXACT),
    _rule("salt", r"\.sls$", Precision.EXACT),
    # --- Declarative / policy languages ------------------------------------
    _rule("nix", r"\.nix$", Precision.EXACT),
    _rule("rego", r"\.rego$", Precision.EXACT),
    _rule("jsonnet", r"\.(jsonnet|libsonnet)$", Precision.EXACT),
    _rule("cue", r"\.cue$", Precision.EXACT),
    _rule("bazel", r"(^|/)(BUILD|WORKSPACE|MODULE)(\.bazel)?$", Precision.EXACT),
    _rule("bazel", r"\.bzl$", Precision.EXACT),
    # --- Cloud-native app definitions -------------------------------------
    _rule("pulumi", r"(^|/)Pulumi([^/]*)?" + YAML_EXT, Precision.EXACT),
    _rule("cloudformation", r"(^|/)(cloudformation|cfn)/.+\.(ya?ml|json)$", Precision.HEURISTIC),
    _rule("serverless", r"(^|/)serverless" + YAML_EXT, Precision.EXACT),
    _rule("sam", r"(^|/)template" + YAML_EXT, Precision.HEURISTIC),
    _rule("cdk", r"(^|/)cdk" + YAML_EXT, Precision.HEURISTIC),
    _rule("vagrant", r"(^|/)Vagrantfile$", Precision.EXACT),
    _rule("cloud_init", r"(^|/)(cloud-init|user-data)[^/]*$", Precision.HEURISTIC),
    # --- Host and service configuration -----------------------------------
    _rule("systemd", r"\.(service|timer|socket|mount|target|path)$", Precision.EXACT),
    _rule("nginx", r"(^|/)nginx[^/]*\.conf$", Precision.EXACT),
    _rule("nginx", r"(^|/)(sites-available|sites-enabled|conf\.d)/[^/]+$", Precision.HEURISTIC),
    _rule("haproxy", r"(^|/)haproxy[^/]*\.cfg$", Precision.EXACT),
    _rule("envoy", r"(^|/)envoy[^/]*" + YAML_EXT, Precision.EXACT),
    _rule("traefik", r"(^|/)traefik[^/]*" + YAML_EXT, Precision.EXACT),
    # --- Observability -----------------------------------------------------
    _rule("prometheus", r"(^|/)(prometheus|alertmanager)[^/]*" + YAML_EXT, Precision.EXACT),
    _rule("prometheus", r"(^|/)[^/]*(rules|alerts)[^/]*" + YAML_EXT, Precision.HEURISTIC),
    _rule("grafana", r"(^|/)(dashboards|grafana)/[^/]+\.json$", Precision.HEURISTIC),
    _rule("otel", r"(^|/)otel[^/-]*(-config)?[^/]*" + YAML_EXT, Precision.EXACT),
    # --- Generic Kubernetes manifests (lowest specificity) -----------------
    _rule("kubernetes", r"(^|/)(k8s|kube|kubernetes|manifests?|deploy(ment)?s?|charts?)/.+" + YAML_EXT, Precision.HEURISTIC),
    _rule("kubernetes", r"(^|/)templates/[^/]+" + YAML_EXT, Precision.HEURISTIC),
)

# Files that are infrastructure-adjacent but so numerous and low-precision that
# they are counted separately rather than folded into the slice.
ADJACENT_RULES: tuple[Rule, ...] = (
    _rule("makefile", r"(^|/)(GNUmakefile|Makefile)[^/]*$", Precision.EXACT),
    _rule("shell", r"\.(sh|bash|zsh)$", Precision.EXACT),
    _rule("justfile", r"(^|/)[Jj]ustfile$", Precision.EXACT),
    _rule("earthly", r"(^|/)Earthfile$", Precision.EXACT),
)

# Vendored or generated paths that carry no signal even when they match a rule.
NOISE = re.compile(
    r"(^|/)("
    r"\.terraform/|\.terragrunt-cache/|"
    r"node_modules/|vendor/|third_party/|site-packages/|"
    # `build/` is NOT noise: Dockerfiles and CI glue legitimately live there.
    r"\.git/|dist/|target/|"
    r"charts/[^/]+/charts/"  # vendored Helm subchart tarball expansions
    r")",
    re.IGNORECASE,
)


# Cheap recall-only gate for the unit detector, which otherwise has to run
# several regexes against every path in every repository.
UNIT_PREFILTER = re.compile(
    r"(Chart\.ya?ml|values[^/]*\.ya?ml|kustomization\.ya?ml"
    r"|compose[^/]*\.ya?ml|\.tf$|templates/|roles/)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Classification:
    tool: str
    precision: Precision


def classify(file_path: str, language: str | None = None) -> Classification | None:
    """Classify one file by path and detected language, or return None."""
    if NOISE.search(file_path):
        return None
    for ruleset in (RULES, ADJACENT_RULES):
        for rule in ruleset:
            if rule.languages and (language or "") not in rule.languages:
                continue
            if rule.pattern.search(file_path):
                return Classification(rule.tool, rule.precision)
    return None


ADJACENT_TOOLS = frozenset(rule.tool for rule in ADJACENT_RULES)
CORE_TOOLS = frozenset(rule.tool for rule in RULES)


def _dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def detect_units(paths: list[str]) -> dict[str, list[str]]:
    """Find complete, self-contained infrastructure units inside one repo.

    Repository-level grouping is the reason The Stack v3 makes this possible at
    all: a Helm chart is only useful as a training or evaluation sample when
    `Chart.yaml`, `values.yaml` and `templates/` travel together.

    Returns a mapping of unit type to the directory prefixes that hold them.
    """
    clean = [
        p for p in paths if UNIT_PREFILTER.search(p) and not NOISE.search(p)
    ]

    chart_dirs: set[str] = set()
    template_dirs: set[str] = set()
    values_dirs: set[str] = set()
    terraform_dirs: dict[str, int] = {}
    ansible_role_dirs: set[str] = set()
    kustomize_dirs: set[str] = set()
    compose_dirs: set[str] = set()

    for path in clean:
        directory = _dirname(path)
        name = path.rsplit("/", 1)[-1]

        if re.fullmatch(r"Chart\.ya?ml", name, re.IGNORECASE):
            chart_dirs.add(directory)
        elif re.fullmatch(r"values[^/]*\.ya?ml", name, re.IGNORECASE):
            values_dirs.add(directory)
        elif re.fullmatch(r"kustomization\.ya?ml", name, re.IGNORECASE):
            kustomize_dirs.add(directory)
        elif re.fullmatch(r"(docker-)?compose[^/]*\.ya?ml", name, re.IGNORECASE):
            compose_dirs.add(directory)
        elif name.endswith(".tf"):
            terraform_dirs[directory] = terraform_dirs.get(directory, 0) + 1

        if "/templates/" in f"/{path}" or path.startswith("templates/"):
            head = re.sub(r"(^|/)templates/.*$", "", path)
            template_dirs.add(head)

        role_match = re.search(
            r"(^|/)roles/([^/]+)/(tasks|handlers|defaults)/[^/]+\.ya?ml$",
            path,
            re.IGNORECASE,
        )
        if role_match:
            ansible_role_dirs.add(path[: role_match.end(2)])

    # A Helm chart needs its metadata and its templates directory side by side.
    helm_charts = sorted(chart_dirs & template_dirs)
    # Charts missing templates/ are usually umbrella charts (dependencies only).
    umbrella_charts = sorted(chart_dirs - template_dirs)

    return {
        "helm_chart": helm_charts,
        "helm_umbrella": umbrella_charts,
        "helm_chart_with_values": sorted(set(helm_charts) & values_dirs),
        "terraform_module": sorted(d for d, n in terraform_dirs.items() if n >= 2),
        "terraform_dir": sorted(terraform_dirs),
        "ansible_role": sorted(ansible_role_dirs),
        "kustomize_dir": sorted(kustomize_dirs),
        "compose_project": sorted(compose_dirs),
    }
