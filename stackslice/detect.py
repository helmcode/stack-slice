"""Content-based detectors for YAML that path rules cannot disambiguate.

The `kubernetes` class in `taxonomy.py` is a path heuristic, and path hints
cannot tell a Deployment from a GitHub Actions workflow from an OpenAPI spec:
they are all YAML in a directory with a plausible name. These detectors read the
first few kilobytes of a file and decide what it actually is, which gives both
the precision of the path rules (are matched files really manifests?) and their
recall (how much real infrastructure sits in directories we never guessed?).

Every detector is a pure function so it can be tested without network access.
"""

from __future__ import annotations

import re
from enum import Enum

# Only the head of a file is needed; manifests declare themselves up front.
HEAD_BYTES = 4096


class YamlKind(str, Enum):
    HELM_TEMPLATE = "helm_template"
    K8S_MANIFEST = "k8s_manifest"
    COMPOSE = "compose"
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    PROMETHEUS_RULES = "prometheus_rules"
    ANSIBLE = "ansible"
    CLOUDFORMATION = "cloudformation"
    OPENAPI = "openapi"
    HELM_VALUES = "helm_values"
    OTHER = "other"


K8S_API_VERSION = re.compile(r"^\s*apiVersion:\s*\S", re.MULTILINE)
K8S_KIND = re.compile(r"^\s*kind:\s*\S", re.MULTILINE)
# Go template actions, the unambiguous mark of a chart template.
GO_TEMPLATE = re.compile(r"\{\{[-\s]*[\w.$(\"]")
COMPOSE_SERVICES = re.compile(r"^services:\s*$", re.MULTILINE)
COMPOSE_HINT = re.compile(r"^(version|services|networks|volumes):", re.MULTILINE)
GHA_ON = re.compile(r"^(on|\"on\"|'on'):", re.MULTILINE)
GHA_JOBS = re.compile(r"^jobs:", re.MULTILINE)
GHA_STEPS = re.compile(r"^\s+(uses|runs-on):", re.MULTILINE)
GITLAB_CI = re.compile(r"^(stages|before_script|image|include):", re.MULTILINE)
GITLAB_JOB = re.compile(r"^\s+(script|extends):", re.MULTILINE)
PROM_GROUPS = re.compile(r"^groups:", re.MULTILINE)
PROM_RULE = re.compile(r"^\s*-\s*(alert|record):", re.MULTILINE)
ANSIBLE_PLAY = re.compile(r"^-\s+(hosts|name):", re.MULTILINE)
ANSIBLE_KEYS = re.compile(
    r"^\s*(tasks|roles|handlers|vars_files|gather_facts|become|pre_tasks|post_tasks):"
    # Fully qualified collection modules, whether or not they open a list item.
    r"|^\s*-?\s*ansible\.(builtin|posix|windows|netcommon)\."
    # Task-level keywords that only appear in Ansible.
    r"|^\s+(with_items|loop|when|register|notify|delegate_to|become):"
    r"|^\s*-\s*(include_tasks|import_tasks|set_fact):",
    re.MULTILINE,
)
# Ansible roles also keep a `templates/` directory, full of Jinja that looks
# exactly like Go templating. Without this guard every role template would be
# misfiled as a Helm chart template.
JINJA_PATH = re.compile(r"\.(j2|jinja2?)$|(^|/)roles/", re.IGNORECASE)
CFN = re.compile(r"AWSTemplateFormatVersion|^Resources:\s*$", re.MULTILINE)
CFN_TYPE = re.compile(r"^\s+Type:\s*AWS::", re.MULTILINE)
OPENAPI = re.compile(r"^(openapi|swagger):\s*[\"']?\d", re.MULTILINE)
# Helm values files have no apiVersion and no templating, just a config tree.
VALUES_HINT = re.compile(
    r"^(image|replicaCount|resources|ingress|serviceAccount|nameOverride):",
    re.MULTILINE,
)


def classify_yaml(content: str, file_path: str = "") -> YamlKind:
    """Decide what a YAML file actually is from its head.

    Order matters. Helm templates are checked first because they contain
    `apiVersion` and `kind` too and would otherwise be counted as manifests,
    which would overstate the amount of directly usable Kubernetes YAML.
    """
    head = content[:HEAD_BYTES]

    templated = bool(GO_TEMPLATE.search(head)) and not JINJA_PATH.search(file_path)
    has_kind = bool(K8S_API_VERSION.search(head)) and bool(K8S_KIND.search(head))
    in_templates_dir = "/templates/" in f"/{file_path}"

    if templated and (has_kind or in_templates_dir):
        return YamlKind.HELM_TEMPLATE
    if has_kind:
        return YamlKind.K8S_MANIFEST
    if COMPOSE_SERVICES.search(head) or (
        COMPOSE_HINT.search(head) and re.search(r"^\s+(image|build):", head, re.MULTILINE)
    ):
        return YamlKind.COMPOSE
    if GHA_ON.search(head) and (GHA_JOBS.search(head) or GHA_STEPS.search(head)):
        return YamlKind.GITHUB_ACTIONS
    if PROM_GROUPS.search(head) and PROM_RULE.search(head):
        return YamlKind.PROMETHEUS_RULES
    if CFN.search(head) and CFN_TYPE.search(head):
        return YamlKind.CLOUDFORMATION
    if ANSIBLE_PLAY.search(head) and ANSIBLE_KEYS.search(head):
        return YamlKind.ANSIBLE
    if ANSIBLE_KEYS.search(head) and re.search(r"^\s*-\s+name:", head, re.MULTILINE):
        return YamlKind.ANSIBLE
    if GITLAB_CI.search(head) and GITLAB_JOB.search(head):
        return YamlKind.GITLAB_CI
    if OPENAPI.search(head):
        return YamlKind.OPENAPI
    if VALUES_HINT.search(head):
        return YamlKind.HELM_VALUES
    return YamlKind.OTHER


TERRAFORM_BLOCK = re.compile(
    r"^\s*(resource|module|provider|variable|output|data|terraform|locals)\s",
    re.MULTILINE,
)


def is_terraform(content: str) -> bool:
    """Confirm an `.tf` file declares real Terraform blocks."""
    return bool(TERRAFORM_BLOCK.search(content[:HEAD_BYTES]))


DOCKERFILE_INSTRUCTION = re.compile(
    r"^\s*(FROM|ARG|RUN|COPY|ADD|CMD|ENTRYPOINT|ENV|WORKDIR|EXPOSE)\s",
    re.MULTILINE | re.IGNORECASE,
)


def is_dockerfile(content: str) -> bool:
    """Confirm a Dockerfile-named file contains Dockerfile instructions."""
    head = content[:HEAD_BYTES]
    return bool(re.search(r"^\s*FROM\s+\S", head, re.MULTILINE | re.IGNORECASE)) and bool(
        DOCKERFILE_INSTRUCTION.search(head)
    )


GENERATED = re.compile(
    r"(DO NOT EDIT|do not edit this file|autogenerated|auto-generated"
    r"|generated by|@generated|Code generated by)",
    re.IGNORECASE,
)


def looks_generated(content: str) -> bool:
    """Flag machine-written files that survived the corpus quality filter."""
    return bool(GENERATED.search(content[:1024]))
