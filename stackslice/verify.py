"""Independent arbiter: parse YAML properly instead of trusting our regexes.

`detect.py` classifies with regexes because they are fast enough to run over
billions of files. Measuring those regexes against themselves proves nothing, so
these functions load the document with a real YAML parser and check its actual
structure. They are slower and only ever run on samples, to score the detectors.

A parse failure is informative rather than fatal: Helm templates are not valid
YAML at all, so "fails to parse and contains Go template actions" is itself the
signature of a chart template.
"""

from __future__ import annotations

import yaml

# Parsing arbitrary YAML from the internet is unbounded work; cap the input.
MAX_PARSE_BYTES = 256 * 1024


class ParseOutcome(str):
    pass


def load_documents(content: str) -> list | None:
    """Parse a multi-document YAML file, or return None if it is not YAML."""
    if len(content) > MAX_PARSE_BYTES:
        return None
    try:
        return [doc for doc in yaml.safe_load_all(content)]
    except (yaml.YAMLError, UnicodeDecodeError, ValueError, RecursionError):
        return None


def verify_k8s_manifest(content: str) -> bool:
    """A Kubernetes manifest parses and declares apiVersion plus kind."""
    documents = load_documents(content)
    if documents is None:
        return False
    for document in documents:
        if not isinstance(document, dict):
            continue
        api_version = document.get("apiVersion")
        kind = document.get("kind")
        if isinstance(api_version, str) and isinstance(kind, str) and api_version and kind:
            return True
    return False


def verify_github_workflow(content: str) -> bool:
    """A workflow has triggers and jobs whose values look like jobs.

    YAML 1.1 parses the bare key `on` as the boolean True, which is exactly the
    trap that makes hand-rolled checks wrong, so both forms are accepted.
    """
    documents = load_documents(content)
    if not documents:
        return False
    document = documents[0]
    if not isinstance(document, dict):
        return False
    has_trigger = "on" in document or True in document
    jobs = document.get("jobs")
    if not has_trigger or not isinstance(jobs, dict) or not jobs:
        return False
    for job in jobs.values():
        if isinstance(job, dict) and (
            "steps" in job or "runs-on" in job or "uses" in job
        ):
            return True
    return False


ANSIBLE_TASK_KEYS = {
    "hosts",
    "tasks",
    "roles",
    "handlers",
    "become",
    "gather_facts",
    "vars_files",
    "pre_tasks",
    "post_tasks",
    "include_tasks",
    "import_tasks",
    "import_playbook",
    "block",
    "when",
    "loop",
    "with_items",
    "register",
    "notify",
    "delegate_to",
    "set_fact",
}


def verify_ansible(content: str) -> bool:
    """A playbook or task file is a list of mappings carrying Ansible keys."""
    documents = load_documents(content)
    if not documents:
        return False
    for document in documents:
        if not isinstance(document, list) or not document:
            continue
        for entry in document:
            if not isinstance(entry, dict):
                continue
            if ANSIBLE_TASK_KEYS & set(entry.keys()):
                return True
            # A task can be just `- name:` plus a module call, where the module
            # is any fully qualified or short name we cannot enumerate.
            if "name" in entry and len(entry) >= 2:
                return True
    return False


def verify_compose(content: str) -> bool:
    """A Compose file has a services mapping with at least one service."""
    documents = load_documents(content)
    if not documents:
        return False
    document = documents[0]
    if not isinstance(document, dict):
        return False
    services = document.get("services")
    if isinstance(services, dict) and services:
        return any(isinstance(value, dict) for value in services.values())
    return False


def verify_chart_metadata(content: str) -> bool:
    """A Chart.yaml parses and declares the fields Helm requires."""
    documents = load_documents(content)
    if not documents:
        return False
    document = documents[0]
    if not isinstance(document, dict):
        return False
    return isinstance(document.get("name"), str) and bool(document.get("version"))


def is_go_templated(content: str) -> bool:
    """Does this file fail to parse as YAML because it is a Go template?"""
    from .detect import GO_TEMPLATE

    if not GO_TEMPLATE.search(content):
        return False
    return load_documents(content) is None


def top_level_keys(content: str, limit: int = 12) -> list[str]:
    """Return the top-level mapping keys of a YAML document, if it is a mapping.

    Used to describe the large `other` bucket empirically rather than guessing
    what unclassified YAML contains.
    """
    documents = load_documents(content)
    if not documents:
        return []
    document = documents[0]
    if not isinstance(document, dict):
        return []
    keys = []
    for key in list(document.keys())[:limit]:
        keys.append("on" if key is True else str(key))
    return keys
