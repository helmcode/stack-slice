"""Tests for rendering the dataset card from pipeline stats."""

import pytest

from stackslice.card import build_values, configs_table, ordered, render

FINALIZE = [
    {"source": "helm_chart.jsonl.gz", "read": 100, "written": 96,
     "dropped_opted_out": 4, "duplicate_files_removed": 30,
     "dropped_after_dedup": 7, "charts_not_self_contained": 24},
    {"source": "dockerfile.jsonl.gz", "read": 500, "written": 495,
     "dropped_opted_out": 5, "duplicate_files_removed": 70},
]
PUBLISH = {
    "rows": 591,
    "bytes": 3_000_000_000,
    "configs": [
        {"config": "dockerfile", "rows": 495, "files": 1, "bytes": 2_000_000_000},
        {"config": "helm_chart", "rows": 96, "files": 1, "bytes": 1_000_000_000},
    ],
}


def values():
    return build_values(FINALIZE, PUBLISH, "de81e3ca7151", "d7bc7991ea32")


def test_scarce_configs_are_listed_first():
    names = [c["config"] for c in ordered(PUBLISH["configs"])]
    assert names == ["helm_chart", "dockerfile"]


def test_totals_come_from_the_stats_not_prose():
    result = values()
    assert result["TOTAL_UNITS"] == "591"
    assert result["OPTED_OUT"] == "9"
    assert result["DUPLICATE_FILES"] == "100"
    assert result["DROPPED_AFTER_DEDUP"] == "7"


def test_self_contained_share_is_derived_from_chart_rows():
    """24 of 96 charts are not self-contained, so 75.0% are."""
    result = values()
    assert result["CHART_NOT_SELF_CONTAINED_PCT"] == "25.0%"
    assert result["CHART_SELF_CONTAINED_PCT"] == "75.0%"


def test_revisions_are_passed_through():
    result = values()
    assert result["SOURCE_REVISION"] == "de81e3ca7151"
    assert result["FILTER_REVISION"] == "d7bc7991ea32"


def test_configs_table_has_a_row_per_config_and_a_total():
    table = configs_table(ordered(PUBLISH["configs"]))
    assert "| `helm_chart` | 96 |" in table
    assert "| `dockerfile` | 495 |" in table
    assert "| **total** | **591** |" in table


def test_render_substitutes_every_placeholder():
    template = "units: {{TOTAL_UNITS}}, dropped: {{OPTED_OUT}}"
    output = render(template, values())
    assert output == "units: 591, dropped: 9"
    assert "{{" not in output


def test_render_refuses_an_unknown_placeholder():
    """A silently empty section in a published card is worse than a crash."""
    with pytest.raises(KeyError) as error:
        render("{{TOTAL_UNITS}} and {{NOT_A_REAL_KEY}}", values())
    assert "NOT_A_REAL_KEY" in str(error.value)


def test_the_real_template_renders_with_no_leftovers():
    with open("dataset_card/README.template.md") as handle:
        template = handle.read()
    output = render(template, values())
    assert "{{" not in output
    assert "license: odc-by" in output
    assert "config_name: helm_chart" in output
    # The licensing finding must survive into the published card.
    assert "header-based, not repository-based" in output


def test_missing_optional_counters_do_not_crash():
    finalize = [{"source": "compose.jsonl.gz", "read": 10, "written": 10}]
    publish = {"rows": 10, "bytes": 1, "configs": [
        {"config": "compose", "rows": 10, "files": 1, "bytes": 1}]}
    result = build_values(finalize, publish, "a", "b")
    assert result["OPTED_OUT"] == "0"
    assert result["DUPLICATE_FILES"] == "0"
    assert result["CHART_NOT_SELF_CONTAINED_PCT"] == "0.0%"
