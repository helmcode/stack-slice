"""Final labelling: repository context plus content, with paths as a prior only.

Phase 0.5 measured that path rules catch a third of real Kubernetes manifests
and are 57% precise, while extension-anchored rules (`.tf`, `Dockerfile`) are
99% correct. The conclusion drives this module:

- for YAML, the content label decides, and the low-precision path guesses are
  never used as a final answer
- repository context resolves what neither path nor content can: a templated
  manifest is a Helm template, but only a `Chart.yaml` next to it proves which
  chart it belongs to, and a variables file is Ansible only because it sits in a
  role
- path rules stay authoritative exactly where they are exact

The same function serves both passes: with contents it produces final labels,
without contents it produces the cheap metadata-only approximation, and every
result records which evidence decided it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .detect import YamlKind, classify_yaml
from .taxonomy import Precision, classify, detect_units

YAML_PATH = re.compile(r"\.(ya?ml|tpl)$", re.IGNORECASE)

# Files that belong to a Helm chart rooted at a given prefix.
CHART_MEMBER = re.compile(
    r"^(Chart\.ya?ml|Chart\.lock|values[^/]*\.ya?ml|values\.schema\.json"
    r"|\.helmignore|templates/.*|charts/.*|crds/.*)$",
    re.IGNORECASE,
)
# Files that belong to an Ansible role rooted at a given prefix.
ROLE_MEMBER = re.compile(
    r"^(tasks|handlers|defaults|vars|meta|templates|files|library|molecule)/",
    re.IGNORECASE,
)

# Content labels that map straight onto a tool name.
CONTENT_TO_TOOL = {
    YamlKind.K8S_MANIFEST: "kubernetes",
    YamlKind.HELM_TEMPLATE: "helm",
    YamlKind.ANSIBLE: "ansible",
    YamlKind.GITHUB_ACTIONS: "github_actions",
    YamlKind.GITLAB_CI: "gitlab_ci",
    YamlKind.COMPOSE: "compose",
    YamlKind.PROMETHEUS_RULES: "prometheus",
    YamlKind.CLOUDFORMATION: "cloudformation",
}

# Path classes trustworthy enough to stand on their own for YAML, because the
# path is canonical rather than conventional.
CANONICAL_YAML_PATHS = frozenset(
    {"github_actions", "gitlab_ci", "kustomize", "compose", "helm", "circleci",
     "azure_pipelines", "drone", "buildkite", "travis", "tekton", "pulumi",
     "codebuild",
     "serverless", "skaffold", "helmfile"}
)


@dataclass(frozen=True)
class Resolved:
    tool: str
    precision: Precision
    evidence: str  # "content", "context", "path", "path+content"
    unit: str | None = None  # prefix of the unit this file belongs to


def _relative(path: str, prefix: str) -> str | None:
    if not prefix:
        return path
    if path.startswith(f"{prefix}/"):
        return path[len(prefix) + 1:]
    return None


def resolve_repo(
    paths: list[str],
    languages: list[str | None],
    contents: list[str | None] | None = None,
) -> list[Resolved | None]:
    """Label every file in one repository, using all available evidence."""
    units = detect_units(paths)
    chart_prefixes = sorted(
        set(units["helm_chart"]) | set(units["helm_umbrella"]),
        key=len,
        reverse=True,  # deepest chart wins for nested subcharts
    )
    role_prefixes = sorted(set(units["ansible_role"]), key=len, reverse=True)

    resolved: list[Resolved | None] = []

    for index, path in enumerate(paths):
        language = languages[index] if index < len(languages) else None
        content = contents[index] if contents and index < len(contents) else None
        prior = classify(path, language)

        # 1. Repository context: chart and role membership beat everything,
        #    because they explain files that carry no self-identifying content
        #    (values.yaml, defaults/main.yml, _helpers.tpl).
        chart = next(
            (
                prefix
                for prefix in chart_prefixes
                if (relative := _relative(path, prefix)) is not None
                and CHART_MEMBER.match(relative)
            ),
            None,
        )
        if chart is not None:
            resolved.append(
                Resolved("helm", Precision.STRUCTURAL, "context", chart)
            )
            continue

        role = next(
            (
                prefix
                for prefix in role_prefixes
                if (relative := _relative(path, prefix)) is not None
                and ROLE_MEMBER.match(relative)
            ),
            None,
        )
        if role is not None:
            resolved.append(
                Resolved("ansible", Precision.STRUCTURAL, "context", role)
            )
            continue

        # 2. Content decides for YAML.
        if content is not None and YAML_PATH.search(path):
            label = classify_yaml(content, path)
            tool = CONTENT_TO_TOOL.get(label)
            if tool is not None:
                evidence = "path+content" if prior and prior.tool == tool else "content"
                resolved.append(Resolved(tool, Precision.EXACT, evidence))
                continue
            # Content says it is not infrastructure. Only a canonical path may
            # override that, for files whose content is inherently featureless
            # (an empty workflow, a values file outside a chart).
            if prior and prior.tool in CANONICAL_YAML_PATHS:
                resolved.append(Resolved(prior.tool, Precision.STRUCTURAL, "path"))
                continue
            resolved.append(None)
            continue

        # 3. No content available, or not a YAML file: trust the path only where
        #    it is exact, or where it is canonical rather than conventional.
        if prior is None:
            resolved.append(None)
            continue
        if prior.precision is Precision.EXACT or prior.tool in CANONICAL_YAML_PATHS:
            resolved.append(Resolved(prior.tool, prior.precision, "path"))
            continue
        if not YAML_PATH.search(path):
            # A structural non-YAML rule (chef recipes, gitlab ci includes).
            resolved.append(Resolved(prior.tool, prior.precision, "path"))
            continue
        # A conventional YAML guess with no content to confirm it: keep it, but
        # marked heuristic so reports never present it as established.
        resolved.append(Resolved(prior.tool, Precision.HEURISTIC, "path"))

    return resolved
