"""Render the dataset card from the stats the pipeline actually produced.

A card whose numbers were copied by hand drifts from the artifact it describes,
and every one of these figures is load-bearing: the opt-out count is a compliance
claim, the self-contained share tells consumers which charts render, the duplicate
count explains why the totals are lower than a naive extraction. So the card is a
template filled from `finalize_stats.json` and `publish_stats.json`, and a missing
placeholder is an error rather than a silently empty section.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from .publish import configs_yaml

PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")

# Human-facing order: scarcity and value first, bulk last.
CONFIG_ORDER = [
    "helm_chart",
    "terraform_module",
    "manifest_set",
    "ansible_role",
    "dockerfile",
    "workflow",
    "compose",
]

DESCRIPTIONS = {
    "helm_chart": "Complete charts: `Chart.yaml` plus templates, and `values.yaml` where present",
    "terraform_module": "Directories with two or more `.tf` files declaring real blocks",
    "manifest_set": "Directories of two or more Kubernetes manifests that parse",
    "ansible_role": "Roles with a verifiable task list, plus defaults, handlers and templates",
    "dockerfile": "Single files containing real Dockerfile instructions",
    "workflow": "GitHub Actions workflows with triggers and jobs",
    "compose": "Docker Compose files with a services mapping",
}


def ordered(configs: list[dict]) -> list[dict]:
    by_name = {c["config"]: c for c in configs}
    return [by_name[name] for name in CONFIG_ORDER if name in by_name] + [
        c for c in configs if c["config"] not in CONFIG_ORDER
    ]


def configs_table(configs: list[dict]) -> str:
    lines = [
        "| Config | Units | Parquet | What a unit is |",
        "|---|---|---|---|",
    ]
    for config in configs:
        lines.append(
            f"| `{config['config']}` | {config['rows']:,} | "
            f"{config['bytes'] / 1e9:.2f} GB | "
            f"{DESCRIPTIONS.get(config['config'], '')} |"
        )
    total_rows = sum(c["rows"] for c in configs)
    total_bytes = sum(c["bytes"] for c in configs)
    lines.append(f"| **total** | **{total_rows:,}** | **{total_bytes / 1e9:.2f} GB** | |")
    return "\n".join(lines)


def build_values(
    finalize_stats: list[dict],
    publish_stats: dict,
    source_revision: str,
    filter_revision: str,
) -> dict[str, str]:
    configs = ordered(publish_stats["configs"])
    by_source = {s["source"].split(".")[0]: s for s in finalize_stats}

    charts = by_source.get("helm_chart", {})
    chart_rows = next(
        (c["rows"] for c in configs if c["config"] == "helm_chart"), 0
    )
    not_self_contained = charts.get("charts_not_self_contained", 0)
    if chart_rows:
        broken_pct = 100 * not_self_contained / chart_rows
    else:
        broken_pct = 0.0

    return {
        "CONFIGS_YAML": configs_yaml(configs),
        "CONFIGS_TABLE": configs_table(configs),
        "TOTAL_UNITS": f"{publish_stats['rows']:,}",
        "SOURCE_REVISION": source_revision,
        "FILTER_REVISION": filter_revision,
        "OPTED_OUT": f"{sum(s.get('dropped_opted_out', 0) for s in finalize_stats):,}",
        "DUPLICATE_FILES": f"{sum(s.get('duplicate_files_removed', 0) for s in finalize_stats):,}",
        "DROPPED_AFTER_DEDUP": f"{sum(s.get('dropped_after_dedup', 0) for s in finalize_stats):,}",
        "CHART_SELF_CONTAINED_PCT": f"{100 - broken_pct:.1f}%",
        "CHART_NOT_SELF_CONTAINED_PCT": f"{broken_pct:.1f}%",
    }


def render(template: str, values: dict[str, str]) -> str:
    missing = {
        name for name in PLACEHOLDER.findall(template) if name not in values
    }
    if missing:
        raise KeyError(f"template placeholders with no value: {sorted(missing)}")
    unused = set(values) - set(PLACEHOLDER.findall(template))
    if unused:
        print(f"warning: unused values {sorted(unused)}", file=sys.stderr)
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="dataset_card/README.template.md")
    parser.add_argument("--finalize-stats", required=True)
    parser.add_argument("--publish-stats", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--filter-revision", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.template) as handle:
        template = handle.read()
    with open(args.finalize_stats) as handle:
        finalize_stats = json.load(handle)
    with open(args.publish_stats) as handle:
        publish_stats = json.load(handle)

    values = build_values(
        finalize_stats, publish_stats, args.source_revision, args.filter_revision
    )
    with open(args.out, "w") as handle:
        handle.write(render(template, values))
    print(f"wrote {args.out}")
    for key in sorted(values):
        if "\n" not in values[key]:
            print(f"  {key} = {values[key]}")


if __name__ == "__main__":
    main()
