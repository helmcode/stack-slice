"""Stream shards and extract complete, quality-gated infrastructure units.

Nothing is stored on disk except the output: pyarrow reads each shard over HTTP
range requests, one row group at a time, so a sweep of the corpus needs bandwidth
and CPU but almost no storage.

Units are emitted as gzipped JSONL, one record per unit, carrying every file of
that unit plus the provenance The Stack v3's licence requires (`repo_path`,
`commit_id`, per-file `license_type` and `detected_licenses`).

Quality gates are content-based rather than popularity-based. Star-gating charts
would mean reading roughly a third of the corpus to collect a couple of thousand,
whereas gating on structure keeps far more of what is actually usable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from .detect import DOCKERFILE_INSTRUCTION, is_dockerfile, is_terraform, looks_generated
from .resolve import resolve_repo
from .scan import CountingReader, sample_indices, shard_path
from .taxonomy import detect_units
from .verify import (
    load_documents,
    verify_ansible,
    verify_chart_metadata,
    verify_compose,
    verify_github_workflow,
    verify_k8s_manifest,
)

EXTRACT_COLUMNS = [
    "repo_path",
    "commit_id",
    "github_metadata",
    "files.list.element.file_path",
    "files.list.element.language",
    "files.list.element.size_bytes",
    "files.list.element.license_type",
    "files.list.element.detected_licenses",
    "files.list.element.content",
]

# Files whose content the resolver needs in order to label them at all.
YAML_CONTENT = re.compile(r"\.(ya?ml|tpl)$", re.IGNORECASE)

# Bounds that keep a unit reviewable by a human and cheap to render in a test.
MAX_FILE_BYTES = 256 * 1024
MAX_UNIT_BYTES = 2 * 1024 * 1024
MAX_UNIT_FILES = 60


@dataclass
class Rejections:
    counts: Counter = field(default_factory=Counter)

    def record(self, unit_type: str, reason: str) -> None:
        self.counts[f"{unit_type}/{reason}"] += 1


def _within_bounds(files: list[dict]) -> str | None:
    if not files:
        return "empty"
    if len(files) > MAX_UNIT_FILES:
        return "too_many_files"
    total = sum(len(f["content"].encode("utf-8", "replace")) for f in files)
    if total > MAX_UNIT_BYTES:
        return "unit_too_large"
    if any(len(f["content"]) > MAX_FILE_BYTES for f in files):
        return "file_too_large"
    return None


def gate_helm_chart(files: list[dict]) -> tuple[bool, str, dict]:
    """A chart must declare itself, ship values, and have real templates."""
    reason = _within_bounds(files)
    if reason:
        return False, reason, {}

    by_name = {f["relative"]: f for f in files}
    chart = next(
        (f for name, f in by_name.items() if name.lower() in ("chart.yaml", "chart.yml")),
        None,
    )
    if chart is None:
        return False, "no_chart_metadata", {}
    if not verify_chart_metadata(chart["content"]):
        return False, "unparseable_chart_metadata", {}

    templates = [
        f for name, f in by_name.items()
        if name.lower().startswith("templates/") and name.lower().endswith((".yaml", ".yml"))
    ]
    if len(templates) < 2:
        return False, "too_few_templates", {}

    # A chart without values.yaml is still valid Helm, so this is a richness
    # flag rather than a gate. Charts are the scarcest unit; do not throw away
    # valid ones for missing niceties.
    has_values = any(name.lower().startswith("values") for name in by_name)

    templated = sum(1 for f in templates if "{{" in f["content"])
    if templated == 0:
        return False, "no_templating", {}

    return True, "", {
        "templates": len(templates),
        "templated_templates": templated,
        "has_values": has_values,
        "helpers": sum(1 for n in by_name if n.lower().endswith(".tpl")),
        "files": len(files),
    }


def gate_terraform_module(files: list[dict]) -> tuple[bool, str, dict]:
    """A module needs several .tf files and at least one real declaration."""
    reason = _within_bounds(files)
    if reason:
        return False, reason, {}

    tf_files = [f for f in files if f["relative"].endswith(".tf")]
    if len(tf_files) < 2:
        return False, "too_few_tf_files", {}
    declaring = [f for f in tf_files if is_terraform(f["content"])]
    if not declaring:
        return False, "no_declarations", {}
    if all(looks_generated(f["content"]) for f in tf_files):
        return False, "generated", {}

    names = {f["relative"].lower() for f in tf_files}
    return True, "", {
        "tf_files": len(tf_files),
        "declaring": len(declaring),
        "has_variables": any("variable" in n for n in names),
        "has_outputs": any("output" in n for n in names),
        "files": len(files),
    }


def gate_ansible_role(files: list[dict]) -> tuple[bool, str, dict]:
    """A role needs a parseable task list, not just a directory shaped like one."""
    reason = _within_bounds(files)
    if reason:
        return False, reason, {}

    tasks = [
        f for f in files
        if f["relative"].lower().startswith("tasks/")
        and f["relative"].lower().endswith((".yml", ".yaml"))
    ]
    if not tasks:
        return False, "no_tasks", {}
    if not any(verify_ansible(f["content"]) for f in tasks):
        return False, "unverified_tasks", {}

    directories = {f["relative"].split("/")[0].lower() for f in files}
    return True, "", {
        "task_files": len(tasks),
        "has_defaults": "defaults" in directories,
        "has_handlers": "handlers" in directories,
        "has_templates": "templates" in directories,
        "files": len(files),
    }


def gate_manifest_set(files: list[dict]) -> tuple[bool, str, dict]:
    """A deployable set is two or more manifests that actually parse."""
    reason = _within_bounds(files)
    if reason:
        return False, reason, {}
    verified = [f for f in files if verify_k8s_manifest(f["content"])]
    if len(verified) < 2:
        return False, "too_few_manifests", {}
    kinds = set()
    for f in verified:
        for line in f["content"].splitlines():
            if line.startswith("kind:"):
                kinds.add(line.split(":", 1)[1].strip())
    return True, "", {
        "manifests": len(verified),
        "kinds": sorted(kinds)[:12],
        "files": len(files),
    }


def gate_dockerfile(files: list[dict]) -> tuple[bool, str, dict]:
    """A single-file unit: it must actually contain Dockerfile instructions."""
    reason = _within_bounds(files)
    if reason:
        return False, reason, {}
    content = files[0]["content"]
    if not is_dockerfile(content):
        return False, "not_a_dockerfile", {}
    if looks_generated(content):
        return False, "generated", {}

    lines = [line.strip() for line in content.splitlines()]
    stages = sum(1 for line in lines if line.upper().startswith("FROM "))
    return True, "", {
        "stages": stages,
        "multi_stage": stages > 1,
        "instructions": sum(
            1 for line in lines if DOCKERFILE_INSTRUCTION.match(line)
        ),
        "has_user": any(line.upper().startswith("USER ") for line in lines),
        "has_healthcheck": any(line.upper().startswith("HEALTHCHECK") for line in lines),
        "files": 1,
    }


def gate_workflow(files: list[dict]) -> tuple[bool, str, dict]:
    """A GitHub Actions workflow the parser accepts as having triggers and jobs."""
    reason = _within_bounds(files)
    if reason:
        return False, reason, {}
    content = files[0]["content"]
    if not verify_github_workflow(content):
        return False, "unverified_workflow", {}

    documents = load_documents(content) or [{}]
    document = documents[0] if isinstance(documents[0], dict) else {}
    jobs = document.get("jobs") or {}
    triggers = document.get("on", document.get(True))
    if isinstance(triggers, dict):
        trigger_names = sorted(str(k) for k in triggers)
    elif isinstance(triggers, list):
        trigger_names = sorted(str(t) for t in triggers)
    else:
        trigger_names = [str(triggers)] if triggers is not None else []

    steps = 0
    uses = 0
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                steps += 1
                if "uses" in step:
                    uses += 1
    return True, "", {
        "jobs": len(jobs),
        "steps": steps,
        "uses_actions": uses,
        "triggers": trigger_names[:8],
        "has_permissions": "permissions" in document,
        "files": 1,
    }


def gate_compose(files: list[dict]) -> tuple[bool, str, dict]:
    """A Compose file whose services mapping the parser accepts."""
    reason = _within_bounds(files)
    if reason:
        return False, reason, {}
    content = files[0]["content"]
    if not verify_compose(content):
        return False, "unverified_compose", {}

    documents = load_documents(content) or [{}]
    document = documents[0] if isinstance(documents[0], dict) else {}
    services = document.get("services") or {}
    built = sum(1 for s in services.values() if isinstance(s, dict) and "build" in s)
    return True, "", {
        "services": len(services),
        "built_services": built,
        "image_only_services": len(services) - built,
        "has_volumes": "volumes" in document,
        "has_networks": "networks" in document,
        "has_healthcheck": any(
            isinstance(s, dict) and "healthcheck" in s for s in services.values()
        ),
        "files": 1,
    }


GATES = {
    "helm_chart": gate_helm_chart,
    "terraform_module": gate_terraform_module,
    "ansible_role": gate_ansible_role,
    "manifest_set": gate_manifest_set,
    "dockerfile": gate_dockerfile,
    "workflow": gate_workflow,
    "compose": gate_compose,
}

# Classes whose unit is a single file: no composition to detect, so the resolved
# label plus a content gate is the whole story.
SINGLE_FILE_UNITS = {
    "dockerfile": "dockerfile",
    "github_actions": "workflow",
    "compose": "compose",
}


def _unit_candidates(paths: list[str], resolved) -> dict[str, dict[str, list[int]]]:
    """Group file positions by unit type and unit prefix."""
    units = detect_units(paths)
    candidates: dict[str, dict[str, list[int]]] = {
        name: defaultdict(list) for name in GATES
    }

    prefixes = {
        "helm_chart": units["helm_chart"],
        "terraform_module": units["terraform_module"],
        "ansible_role": units["ansible_role"],
    }
    for unit_type, unit_prefixes in prefixes.items():
        for prefix in unit_prefixes:
            for index, path in enumerate(paths):
                if prefix == "":
                    if "/" not in path or path.startswith(("templates/", "tasks/")):
                        candidates[unit_type][prefix].append(index)
                elif path.startswith(f"{prefix}/"):
                    candidates[unit_type][prefix].append(index)

    # Single-file units come straight from the resolved label.
    for index, result in enumerate(resolved):
        if result and result.tool in SINGLE_FILE_UNITS:
            candidates[SINGLE_FILE_UNITS[result.tool]][paths[index]] = [index]

    # Manifest sets are directories whose files we labelled kubernetes, and which
    # are not part of a chart (charts are captured as helm_chart instead).
    by_directory: dict[str, list[int]] = defaultdict(list)
    for index, result in enumerate(resolved):
        if result and result.tool == "kubernetes":
            directory = paths[index].rsplit("/", 1)[0] if "/" in paths[index] else ""
            by_directory[directory].append(index)
    for directory, indices in by_directory.items():
        if len(indices) >= 2:
            candidates["manifest_set"][directory] = indices

    return candidates


def _reset_http_session() -> None:
    """Drop huggingface_hub's cached HTTP session for this process.

    Long sweeps hit `Cannot send a request, as the client has been closed`: the
    session is cached per process and a pooled worker can inherit a closed one.
    Clearing the cache makes the next attempt build a fresh client.
    """
    try:
        from huggingface_hub.utils import get_session

        cache_clear = getattr(get_session, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
    except Exception:  # noqa: BLE001 - best effort, never fail a sweep on this
        pass


def with_retries(operation, attempts: int = 3, delay: float = 2.0, sleep=time.sleep):
    """Run `operation`, retrying transient failures with a fresh HTTP session."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:  # noqa: BLE001 - any transport error is retryable
            last_error = error
            if attempt == attempts:
                break
            _reset_http_session()
            sleep(delay * attempt)
    raise last_error  # type: ignore[misc]


def collect_shard(index: int, row_group_limit: int | None = None) -> dict:
    """Extract one shard, retrying the whole read on transport failures."""
    return with_retries(lambda: _collect_shard_once(index, row_group_limit))


def _collect_shard_once(index: int, row_group_limit: int | None = None) -> dict:
    """Extract every passing unit from one shard.

    Returns serialisable results rather than writing, so the sweep can run in a
    process pool: a single HTTP connection tops out around 8 MB/s against the
    Hugging Face CDN, and the YAML parsing is CPU-bound under the GIL, so neither
    threads nor a single process can saturate the host.
    """
    stats: Counter = Counter()
    rejections = Rejections()
    records: dict[str, list[str]] = {name: [] for name in GATES}
    fs = HfFileSystem()
    with fs.open(shard_path(index), "rb") as raw:
        reader = CountingReader(raw)
        parquet_file = pq.ParquetFile(reader)
        available = parquet_file.metadata.num_row_groups
        groups = range(available if row_group_limit is None else min(row_group_limit, available))

        for group in groups:
            table = parquet_file.read_row_group(group, columns=EXTRACT_COLUMNS)
            files_column = table.column("files").combine_chunks()
            repo_paths = table.column("repo_path").to_pylist()
            commits = table.column("commit_id").to_pylist()
            metadata = table.column("github_metadata").combine_chunks()
            stars = metadata.field("stars").to_pylist()
            forks = metadata.field("is_fork").to_pylist()

            flat = pc.list_flatten(files_column)
            contents = flat.field("content")
            paths = flat.field("file_path").to_pylist()
            languages = flat.field("language").to_pylist()
            licenses = flat.field("license_type").to_pylist()
            detected = flat.field("detected_licenses")
            parent = pc.list_parent_indices(files_column).to_pylist()

            by_repo: dict[int, list[int]] = defaultdict(list)
            for position in range(len(paths)):
                by_repo[parent[position]].append(position)

            stats["repos"] += table.num_rows

            for repo_index, positions in by_repo.items():
                if forks[repo_index]:
                    stats["skipped_forks"] += 1
                    continue

                repo_paths_local = [paths[p] for p in positions]
                repo_languages = [languages[p] for p in positions]
                # Content is required here, not optional: resolving with paths
                # alone finds only a third of real Kubernetes manifests and misses
                # the 17% of Compose files that are not named compose.yaml.
                repo_contents = [
                    contents[p].as_py() if YAML_CONTENT.search(paths[p]) else None
                    for p in positions
                ]
                resolved = resolve_repo(repo_paths_local, repo_languages, repo_contents)
                candidates = _unit_candidates(repo_paths_local, resolved)

                for unit_type, groups_by_prefix in candidates.items():
                    for prefix, local_indices in groups_by_prefix.items():
                        stats[f"candidate/{unit_type}"] += 1
                        unit_files = []
                        for local in local_indices:
                            position = positions[local]
                            path = repo_paths_local[local]
                            relative = path[len(prefix) + 1:] if prefix else path
                            unit_files.append({
                                "path": path,
                                "relative": relative,
                                "content": contents[position].as_py() or "",
                                "license_type": licenses[position],
                                "detected_licenses": detected[position].as_py() or [],
                                "size_bytes": len(contents[position].as_py() or ""),
                            })

                        passed, reason, quality = GATES[unit_type](unit_files)
                        if not passed:
                            rejections.record(unit_type, reason)
                            continue

                        record = {
                            "unit_type": unit_type,
                            "repo_path": repo_paths[repo_index],
                            "commit_id": commits[repo_index],
                            "stars": stars[repo_index],
                            "unit_prefix": prefix,
                            "shard": index,
                            "quality": quality,
                            "license_types": sorted(
                                {f["license_type"] for f in unit_files if f["license_type"]}
                            ),
                            "files": [
                                {k: v for k, v in f.items() if k != "relative"}
                                for f in unit_files
                            ],
                        }
                        records[unit_type].append(
                            json.dumps(record, ensure_ascii=False)
                        )
                        stats[f"extracted/{unit_type}"] += 1

        return {
            "index": index,
            "bytes": reader.bytes_read,
            "records": records,
            "stats": dict(stats),
            "rejections": dict(rejections.counts),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=20, help="how many shards to sweep")
    parser.add_argument("--offset", type=int, default=0, help="skip this many sampled shards")
    parser.add_argument("--workers", type=int, default=8, help="parallel shard readers")
    parser.add_argument("--row-groups", type=int, default=None)
    parser.add_argument("--out", default="units", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "state.json")
    done: set[int] = set()
    if os.path.exists(state_path):
        with open(state_path) as handle:
            done = set(json.load(handle).get("completed", []))
        print(f"resuming, {len(done)} shards already done", file=sys.stderr)

    indices = [
        i for i in sample_indices(args.shards + args.offset)[args.offset:]
        if i not in done
    ]
    writers = {
        name: gzip.open(os.path.join(args.out, f"{name}.jsonl.gz"), "at", encoding="utf-8")
        for name in GATES
    }
    stats: Counter = Counter()
    rejections = Rejections()
    total_bytes = 0
    started = time.time()
    completed = 0

    try:
        # Do NOT set max_tasks_per_child here. It deadlocks: after every worker
        # retires (workers * limit tasks, observed at exactly 300 with 12 workers
        # and a limit of 25) the pool stops spawning replacements and the parent
        # blocks forever in futex_wait_queue with no children left. Stale HTTP
        # sessions are handled by with_retries instead.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(collect_shard, index, args.row_groups): index
                for index in indices
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001 - keep sweeping
                    print(f"  shard {index:05d} FAILED: {error}", file=sys.stderr, flush=True)
                    continue

                for unit_type, lines in result["records"].items():
                    for line in lines:
                        writers[unit_type].write(line + "\n")
                    writers[unit_type].flush()
                stats.update(result["stats"])
                rejections.counts.update(result["rejections"])
                total_bytes += result["bytes"]
                done.add(index)
                completed += 1
                with open(state_path, "w") as handle:
                    json.dump({"completed": sorted(done)}, handle)

                elapsed = time.time() - started
                extracted = sum(v for k, v in stats.items() if k.startswith("extracted/"))
                print(
                    f"  [{completed}/{len(indices)}] shard {index:05d}  "
                    f"{total_bytes / 1e9:.1f} GB  {elapsed / 60:.1f} min  "
                    f"{total_bytes / 1e6 / max(elapsed, 1):.1f} MB/s  "
                    f"{extracted:,} units",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    elapsed = time.time() - started
    print()
    print("=" * 70)
    print(f"swept {completed} shards this run ({len(done)} total), "
          f"{total_bytes / 1e9:.1f} GB, {elapsed / 60:.1f} min, "
          f"{total_bytes / 1e6 / max(elapsed, 1):.1f} MB/s")
    print(f"repos seen: {stats['repos']:,}  forks skipped: {stats['skipped_forks']:,}")
    print()
    for unit_type in GATES:
        candidates = stats[f"candidate/{unit_type}"]
        extracted = stats[f"extracted/{unit_type}"]
        rate = 100 * extracted / candidates if candidates else 0
        print(f"{unit_type:<18} {extracted:>8,} kept of {candidates:>8,} candidates "
              f"({rate:.1f}%)")
    print()
    print("top rejection reasons:")
    for reason, count in rejections.counts.most_common(15):
        print(f"  {reason:<44} {count:>8,}")

    with open(os.path.join(args.out, "stats.json"), "w") as handle:
        json.dump(
            {"stats": dict(stats), "rejections": dict(rejections.counts),
             "bytes": total_bytes, "shards": sorted(done),
             "seconds": round(elapsed, 1)},
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
