---
license: odc-by
pretty_name: The Stack v3 DevOps Corpus
size_categories:
  - 10M<n<100M
task_categories:
  - text-generation
language:
  - code
tags:
  - infrastructure-as-code
  - devops
  - kubernetes
  - helm
  - terraform
  - ansible
  - docker
  - github-actions
  - sre
{{CONFIGS_YAML}}
---

# The Stack v3 DevOps Corpus

{{TOTAL_UNITS}} complete infrastructure units extracted from
[The Stack v3](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train),
grouped into seven classes and gated on content rather than popularity.

A unit is not a file, it is the thing an engineer would actually run: a Helm chart
arrives with its `Chart.yaml`, `values.yaml` and every template; a Terraform module
with all of its `.tf` files; an Ansible role with its tasks, defaults and handlers.
That is only possible because The Stack v3 groups rows by repository, which v2 did
not.

## Why this exists

Language detection cannot find infrastructure code. Helm templates, Kubernetes
manifests, Ansible playbooks, CI pipelines and Prometheus rules are all just
`YAML` to `go-enry`, and **59.6% of the YAML in the corpus is not infrastructure
at all** (Spring config, i18n plurals, Drupal exports, dbt models, Conda
environments). Path heuristics do not fix it either: a directory-name rule finds
only **33% of real Kubernetes manifests** and is **57% precise**.

So classification here is content-first. Every YAML unit was parsed and inspected,
and the resulting labels were scored against an independent YAML parser rather
than against more regexes:

| Class | Precision | Recall |
|---|---|---|
| kubernetes | 97.8% | 97.2% |
| github_actions | 98.9% | 100.0% |
| compose | 97.2% | 99.3% |
| helm | 93.9% (structural) | not measurable, templates are not valid YAML |
| ansible | 86.4% | 90.3% |
| terraform | 98.9% (extension-anchored) | |
| dockerfile | 99.7% (extension-anchored) | |

Roughly half of all Helm and Ansible labels come from repository context alone:
a `values.yaml` or a `defaults/main.yml` is a bare tree of variables, and no
amount of content inspection can tell you what it belongs to.

## Configs

{{CONFIGS_TABLE}}

### What is inside each one

- **`helm_chart`** the scarcest and richest class. `Chart.yaml`, `values.yaml`
  where present, every template and helper. Median 4 templates, up to 56.
- **`terraform_module`** a directory of two or more `.tf` files that declare real
  resources, modules, variables or outputs. Median 3 files. 70.7% declare
  variables, 47.8% outputs.
- **`manifest_set`** a directory of two or more Kubernetes manifests that parse.
  Median 3. Most common kinds: Deployment, Service, Kustomization, ConfigMap,
  Ingress, PersistentVolumeClaim, Secret.
- **`ansible_role`** `tasks/`, and whichever of `defaults/`, `handlers/`, `vars/`,
  `meta/`, `templates/`, `files/` the role ships. 24.6% carry defaults.
- **`dockerfile`** one file with real instructions. Median 8 instructions,
  20.3% multi-stage.
- **`workflow`** one GitHub Actions workflow with triggers and jobs. Median 1 job
  and 5 steps.
- **`compose`** one Compose file with a services mapping. Median 2 services.

### Using it

```python
from datasets import load_dataset

charts = load_dataset("Helmcode/stack-v3-devops", "helm_chart", split="train")
```

Charts that render standalone, which is what an executable benchmark needs:

```python
renderable = charts.filter(lambda row: row["flags"]["self_contained"])
```

Stream the large configs instead of downloading them:

```python
dockerfiles = load_dataset(
    "Helmcode/stack-v3-devops", "dockerfile", split="train", streaming=True
)
hardened = (row for row in dockerfiles if row["flags"]["pins_digest"])
```

Reconstruct a unit as files on disk, which is how you feed it to `helm lint`,
`terraform validate` or `hadolint`:

```python
import pathlib

def materialise(row, root):
    prefix = row["unit_prefix"]
    for entry in row["files"]:
        relative = entry["path"][len(prefix):].lstrip("/") if prefix else entry["path"]
        target = pathlib.Path(root, relative or pathlib.Path(entry["path"]).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry["content"])

materialise(charts[0], "/tmp/chart")
```

Restrict to units whose every file carries a permissive license header, but read
the licensing section first, because that is not the same as permissively
licensed code:

```python
permissive = charts.filter(lambda row: row["flags"]["all_permissive"])
```

### What it is good for

- **Evaluation.** Complete, self-contained units are what an executable benchmark
  needs: render the chart, validate the module, lint the Dockerfile, and score on
  whether real tools accept the output.
- **Fine-tuning on infrastructure tasks**, where the unit boundary matters more
  than the file: a model that writes one template without `values.yaml` has not
  written a chart.
- **Measuring practice.** The flags make questions like "what share of public
  Dockerfiles run as root" answerable in one pass instead of a research project.

It is **not** a pretraining corpus. 5.4 GB is small, and the classes are
deliberately unbalanced towards what exists rather than what would balance nicely.

## Schema

Every config shares a base schema and adds its own `quality` and `flags` structs.

| Field | Type | Notes |
|---|---|---|
| `unit_type` | string | one of the seven config names |
| `repo_path` | string | `owner/name`, for attribution |
| `commit_id` | string | the exact commit the files came from |
| `stars` | int32 | GitHub stars at crawl time |
| `unit_prefix` | string | directory the unit was rooted at, `""` for repo root |
| `shard` | int32 | source shard, for reproducibility |
| `license_types` | list\<string\> | distinct `license_type` values across the unit's files |
| `files` | list\<struct\> | `path`, `content`, `license_type`, `detected_licenses`, `size_bytes` |
| `quality` | struct | per class: template counts, stage counts, service counts, manifest kinds |
| `flags` | struct | derived booleans, below |

Flags worth knowing about:

- `self_contained` (helm_chart): the chart does not call a helper it lacks.
  **{{CHART_SELF_CONTAINED_PCT}}** of charts qualify; the rest cannot be rendered
  by `helm template` on their own.
- `pins_digest` / `uses_latest_tag` (dockerfile): supply-chain hygiene.
- `has_unpinned_action` (workflow): actions referenced by tag or branch instead of
  a commit SHA.
- `all_permissive`: every file in the unit is labelled `permissive`. Read the
  licensing section before relying on this.

## What this corpus says about real-world infrastructure

Measured across every unit, not a sample:

- **89.0% of Dockerfiles set no `USER`**, so the container runs as root
- **98.6% of Dockerfiles declare no `HEALTHCHECK`**
- **89.5% of workflows declare no `permissions`**, inheriting the default token scope
- **91.1% of Compose files define no healthcheck**
- 20.3% of Dockerfiles are multi-stage
- Top Kubernetes kinds: Deployment, Service, Kustomization, ConfigMap, Ingress

That is the baseline any model trained on public infrastructure code will imitate,
which is the point of publishing it as a measurable corpus rather than a curated
showcase.

## Provenance and how it was built

Built with [helmcode/stack-slice](https://github.com/helmcode/stack-slice)
(Apache-2.0). The corpus was surveyed and extracted **without downloading the
4.71 TB dataset**: `content` is 96.9% of every shard, so a metadata-only pass
costs 1% of the bytes, and extraction streams shards over HTTP range requests
without ever storing one.

- Source revision: **`{{SOURCE_REVISION}}`** of `HuggingFaceCode/stack-v3-train`
- Shards swept: **8,196 of 8,196**, covering 157.9M repositories
- Forks skipped, so units come from the repository that authored them
- Re-filtered for opt-out on **{{FILTER_DATE}}**, against the revision then at
  HEAD (`{{FILTER_REVISION}}`), and verified equivalent to the current HEAD
  `{{VERIFIED_HEAD}}` (see Licensing)

Gates are content-based, never popularity-based: a chart must have parseable
metadata, two or more templates and actual templating; a Terraform module must
declare real blocks and not be machine-generated; an Ansible role must have a task
list a parser accepts; a manifest set must have two or more manifests that load.

## Licensing, and a finding you should not skip

This dataset is released under **ODC-By 1.0**, inherited from The Stack v3.
**The code inside remains under its original licenses**, and `repo_path` plus
`commit_id` are included on every unit precisely so attribution is possible.

**The `license_type` labels are header-based, not repository-based.** In the source
corpus only 3.41% of files are labelled `permissive` and 98.2% of repositories
contain none at all. Apache-2.0 is detected 26,624 times against MIT's 442, which
inverts their real popularity on GitHub: the Apache convention puts a license
header in every source file, while MIT projects ship a single root `LICENSE`. So
`license_type == permissive` means **"this file carries an inline license header"**,
not "this file comes from a permissively licensed project".

Two consequences:

1. Filtering to `permissive` does not give you a representative permissive
   corpus, it gives you an Apache-2.0-skewed slice.
2. The remaining `no_license` majority is code with **no license grant at all**,
   not code that is permissively licensed. Treat it accordingly.

The repository-level license cannot be recovered from within The Stack v3 either:
plain-text `LICENSE` files were dropped by its quality filter, so only 8 of 20,923
repositories in a sample shard ship one. A provably permissive subset needs
external enrichment keyed on `repo_path`.

**Opt-out.** Upstream applies opt-out removals in place and re-uploads. This
dataset was re-filtered by `repo_path` on {{FILTER_DATE}}, dropping
{{OPTED_OUT}} units whose repositories had been removed.

Be aware that **upstream rewrites its own history**: the revision we filtered
against (`{{FILTER_REVISION}}`) was squashed away within a day, so source SHAs are
not durable references. We therefore record the filter date alongside the SHA, and
re-verify against the current HEAD (`{{VERIFIED_HEAD}}`) rather than assuming an
old SHA still resolves. If you find your code
here, use the
[Am I in The Stack?](https://huggingface.co/spaces/bigcode/in-the-stack) opt-out
process; we re-filter on each upstream patch release.

## Known limitations

- **The source corpus repeats file rows inside a repository**: 10.4% of
  repositories and 14.5% of all file rows, byte-identical by `content_id`. This
  dataset deduplicates by (path, content), removing {{DUPLICATE_FILES}} repeated
  files, and then **drops the {{DROPPED_AFTER_DEDUP}} units that only met their
  gate because of that repetition** (a "set of two manifests" whose two manifests
  were the same file is not a set of two). Counts here are therefore lower than a
  naive extraction would report, and correctly so. Quality counters such as
  `templates`, `tf_files` and `manifests` are recomputed after deduplication, so
  they describe the files actually present.
- **{{CHART_NOT_SELF_CONTAINED_PCT}} of Helm charts cannot render standalone**
  because they call helpers they do not carry. Filter on `self_contained`.
- **Ansible precision is a floor, not a measurement.** "A list of mappings with
  Ansible-ish keys" also matches ordinary YAML lists, and role variable files are
  indistinguishable from any other mapping by content alone.
- `manifest_set` groups manifests by directory, which is a convention, not a
  deployment boundary.
- Stars are as of the crawl and 58-76% of units come from repositories with none.
  Popularity was deliberately not used as a gate; see the card's reasoning above.

## Updates and versioning

Upstream applies opt-out removals in place, re-uploads the whole dataset and
squashes its history, which means the source moves and old revision SHAs stop
resolving. This dataset therefore records both the revision it was
extracted from and the revision it was last compliance-filtered against, and both
appear above. When upstream ships a patch release we re-filter and push a new
version; the extraction itself is not repeated unless the tooling changes.

If you need byte-for-byte reproducibility, pin the dataset revision you loaded.

## Reproducing this dataset

Everything here was produced by [helmcode/stack-slice](https://github.com/helmcode/stack-slice):

```bash
# Survey the corpus for 179 MB of transfer, no download
python -m stackslice.scan --shards 24

# Score the classifier against an independent YAML parser
python -m stackslice.measure --shards 3

# Sweep and extract units (streams shards, stores nothing but output)
python -m stackslice.extract --shards 8196 --workers 12 --out units

# Re-filter for opt-out, deduplicate, add flags
python -m stackslice.finalize units --out units_final \
    --revision <target-revision> --uuid <shard-uuid>

# Convert to parquet, one config per class
python -m stackslice.publish units_final --out dataset
```

The full measurement record, including the findings quoted in this card, is in
[FINDINGS.md](https://github.com/helmcode/stack-slice/blob/main/FINDINGS.md).

## Citation

```bibtex
@misc{stack_v3_devops,
  title  = {The Stack v3 DevOps Corpus},
  author = {Helmcode},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Helmcode/stack-v3-devops},
  note   = {Extracted from The Stack v3 with helmcode/stack-slice}
}
```

Please also cite the source corpus, [The Stack v3](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train).
