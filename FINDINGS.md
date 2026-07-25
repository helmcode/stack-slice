# Phase 0: sizing an IaC slice of The Stack v3

Measured on 2026-07-25 against `HuggingFaceCode/stack-v3-train` (revision `main`).

All numbers come from a metadata-only sample of 24 evenly spaced shards out of
8196 (0.293% coverage, 505,997 repositories, 8,585,754 files) using
`python -m stackslice.scan --shards 24`. Total transfer for the whole
measurement: **179 MB**. No file contents were downloaded.

## 1. The corpus is far cheaper to work with than its headline size

| Property | Measured | Dataset card |
|---|---|---|
| Parquet shards | 8196 | not stated |
| Compressed size on the wire | 4.71 TB | not stated |
| Decompressed source | 17.8 TB (extrapolated) | 15.9 TB |
| Repositories | 172,797,975 (extrapolated) | ~173M |
| Files | 2,932,034,991 (extrapolated) | not stated |

The extrapolated repository count lands within 0.2% of the published figure,
which validates the sampling method.

The decisive property is the column layout. `files.list.element.content`
accounts for **96.93%** of every shard, so the metadata leaves we need for
slicing (`file_path`, `language`, `size_bytes`, `license_type`, `is_vendor`,
`github_metadata`) cost **1.00% of the bytes**:

- one shard of metadata: ~6 MB instead of ~575 MB
- a full-corpus metadata index of all 2.9B files: **61 GB**, no GPU, no
  dataset download, plain HTTP range reads with anonymous access

Nested projection pushdown was verified empirically with a byte-counting
reader: requesting the metadata leaves of one row group pulled 2.15 MB and did
not touch the content column. The dataset is **not gated**; no token is needed.

## 2. The IaC slice is small, and that is the whole point

| | Files | Size |
|---|---|---|
| Core IaC (all tools below, excluding adjacent) | 37.2M | **58.2 GB** |
| Share of corpus | 1.27% of files | 0.326% of bytes |

Largest classes by volume (extrapolated to the full corpus):

| Tool | Files | GB | % of repos |
|---|---|---|---|
| nix | 10.4M | 25.40 | 0.07% |
| dockerfile | 5.1M | 3.02 | 1.90% |
| github_actions | 4.1M | 6.38 | 1.33% |
| terraform | 3.9M | 5.19 | 0.28% |
| kubernetes (path heuristic) | 3.3M | 4.69 | 0.18% |
| compose | 3.2M | 2.98 | 1.35% |
| ansible | 1.7M | 1.98 | 0.12% |
| bazel | 1.0M | 1.64 | 0.03% |
| helm | 755k | 0.89 | 0.07% |
| puppet, jenkins, nginx, prometheus, kustomize, chef, gitlab_ci, rego, salt, ... | tail | ~6 | |

Shell (35.7 GB) and Makefiles (36.2 GB) are counted separately as
infrastructure-adjacent: high volume, low precision as IaC.

Bytes per file are sane across every class (0.6 KB for Dockerfiles up to
13.9 KB for Rego bundles), so no class is inflated by a handful of giant
generated files.

58 GB is nothing as pretraining fuel. It is exactly the right size for a
curated fine-tuning corpus and an evaluation suite, which is what we should
build.

## 3. Repository grouping yields complete, executable units

This is what The Stack v3 enables and v2 did not: because a row is a whole
repository, self-contained units can be lifted out intact.

| Unit | Extrapolated count | Repositories |
|---|---|---|
| Compose project | 2,694,776 | 2,342,007 |
| Terraform module (>=2 `.tf` in a directory) | 850,676 | 350,720 |
| Ansible role (`roles/<name>/{tasks,defaults,handlers}`) | 457,610 | 103,474 |
| Kustomize directory | 202,851 | 71,032 |
| **Helm chart (`Chart.yaml` + `templates/`)** | **90,156** | 55,323 |
| Helm chart also carrying `values.yaml` | 63,860 | 45,078 |
| Helm umbrella chart (dependencies only) | 58,738 | 29,027 |

90k complete Helm charts and 850k Terraform modules is far more raw material
than an executable benchmark needs (a curated suite wants hundreds to low
thousands of samples).

## 4. Two findings that change what we can claim

### 4.1 The license labels are header-based, not repository-based

The dataset card states each file is labelled `permissive`, `no_license` ("no
licenses detected, or only non-license legal texts") or `non_permissive`, that
`non_permissive` files were excluded, and that licenses are detected with
ScanCode and "propagated through the repository file tree".

Measured across 3 full shards (896,971 files, 62,989 repos):

| license_type | Files | Bytes |
|---|---|---|
| `no_license` | 96.59% | 96.65% |
| `permissive` | 3.41% | 3.35% |

Repository purity: **98.21% of repositories contain zero `permissive` files**,
1.72% are entirely permissive, 0.07% are mixed.

Top detected licenses expose the mechanism: Apache-2.0 appears 26,624 times
while MIT appears 442. MIT overwhelmingly outnumbers Apache-2.0 on GitHub, but
the Apache convention puts a license header in every source file whereas MIT
projects ship only a root `LICENSE`. So `license_type == permissive` in
practice means **"this file carries an inline license header"**, not "this file
belongs to a permissively licensed project", and the promised tree propagation
is not visible in the released labels.

Consequence: filtering to `permissive` does not give you a representative
permissive corpus, it gives you an Apache-2.0-skewed 3.4% sample. Taking
everything means training mostly on code with no license grant at all.

Within the IaC slice specifically, `permissive` is **6.55%** of matched files
(11,805 of 180,102), roughly double the corpus-wide rate.

### 4.2 The repository license cannot be recovered from inside the dataset

The obvious fix would be to read each repository's own `LICENSE` file, which is
already in the corpus as a data row. It is not: only 8 of 20,923 repositories in
one shard ship a root license file, against the 30-40% typical of GitHub, and
the language histogram contains no plain-text language at all (Markdown
survives, plain text does not). The quality filter dropped them.

So a genuinely permissive-only subset requires enriching from an external
source keyed on `repo_path` and `commit_id`, which the dataset does provide for
exactly this purpose.

## 5. Quality distribution

Stars of repositories carrying core IaC files (23,515 in the sample):

| Stars | Share |
|---|---|
| 0 | 74.3% |
| 1-10 | 19.7% |
| 10-100 | 4.4% |
| 100-1k | 1.3% |
| 1k-10k | 0.28% |
| 10k+ | 0.05% |

Only **6.05%** sit in repositories with 10 or more stars. Star-gating is
therefore viable for a curated benchmark (thousands of charts survive) but
would gut a training corpus.

## 6. What this implies for what we build

1. **A metadata index is the cheapest high-value artifact.** 61 GB of transfer
   buys a queryable index of every file in the corpus. It makes the corpus
   usable for anyone who cannot host 4.71 TB, and it is a prerequisite for
   everything else we want to do.
2. **Do not promise a pretraining corpus.** 58 GB of IaC, of which a fraction
   is high quality, is fine-tuning and evaluation material.
3. **The executable benchmark is the strongest play.** 90k complete Helm charts
   and 850k Terraform modules, gated on stars and validated with `helm lint`,
   `helm template`, `kubeconform`, `terraform validate`, `tflint` and
   `hadolint`, is a benchmark nobody has published, and we have the tooling and
   the sandbox to run it.
4. **Content validation is still required for Kubernetes.** The `kubernetes`
   class is a path heuristic; only `apiVersion` plus `kind` in the content
   confirms a manifest. Expect the real count to move once contents are read.
5. **The license finding is publishable on its own** and should be stated
   plainly in any dataset card we release, alongside a `permissive` config for
   users who need a clean subset.

## 7. Phase 0.5: what the content says

Run with `python -m stackslice.validate --shards 3`. Unlike the metadata scan
this pass cannot be cheap: parquet stores one column chunk per row group, so
asking for `content` at all pulls every repository in that row group. Cost was
**1.45 GB and 2m24s** for 3 full shards (63,185 repos, 948,362 files, 13,137
YAML files inspected).

### 7.1 What YAML on GitHub actually is

| Content label | Share of YAML |
|---|---|
| other (config, locales, data, fixtures) | 59.58% |
| github_actions | 11.02% |
| compose | 10.92% |
| k8s_manifest | 9.10% |
| ansible | 5.05% |
| helm_values | 1.24% |
| helm_template | 1.21% |
| gitlab_ci | 0.91% |
| openapi | 0.56% |
| cloudformation | 0.30% |
| prometheus_rules | 0.09% |

So **40.4% of all YAML in the corpus is infrastructure**, spread across 35.9M
YAML files corpus-wide.

### 7.2 Corrected volumes

| | Files | Size |
|---|---|---|
| Real Kubernetes manifests (`apiVersion` + `kind`, no templating) | 3,267,472 | 2.81 GB |
| Helm templates (templated manifests) | 434,388 | 0.59 GB |

The path heuristic's file count for `kubernetes` (3.25M) happened to land close
to the true 3.27M, but that is a coincidence of two errors cancelling: its
composition was wrong in both directions.

### 7.3 The path rules are a prior, not a label

**Recall** (of files whose content proves what they are, how many did the path
rules catch?):

| Content label | Caught by the matching rule | Unclassified by any rule |
|---|---|---|
| github_actions | **97.10%** | 2.90% |
| compose | 77.89% | 17.43% |
| k8s_manifest | **33.11%** | 60.95% |
| ansible | 28.46% | 70.48% |

**Precision** (of files a path rule claimed, what are they really?):

| Path rule | Correct | Notable confusion |
|---|---|---|
| `kubernetes` | 56.81% | 17.22% are Helm templates, 23.53% unrelated |
| `ansible` | 35.80% | 64.20% labelled `other` |
| terraform (`.tf` content check) | **98.87%** (1308/1323) | |
| dockerfile (content check) | **99.73%** (1827/1832) | |

Three conclusions:

1. **For YAML, content must be the primary classifier.** Path rules miss two
   thirds of real Kubernetes manifests because manifests live anywhere, and they
   over-claim by 43%. Path rules remain useful as a cheap prior and as a router,
   never as the final label.
2. **Extension-anchored classes need no content pass.** Terraform and Dockerfile
   path rules are 99% correct, so the exact-precision classes can be trusted as
   labelled.
3. **`.github/workflows` is the one canonical path in the whole ecosystem**, at
   97% recall. Everything else is convention at best.

Two caveats on the numbers above, both in the direction of understating quality:

- The `ansible` precision figure is not trustworthy. Its rules deliberately
  include `group_vars/` and `host_vars/`, whose contents are plain variable trees
  with nothing that distinguishes them from any other YAML, so they land in
  `other` while genuinely being Ansible.
- The `helm` path class has no comparable YAML content label at all: it matches
  `Chart.yaml` and `_helpers.tpl`, which are chart metadata and template
  helpers, correctly not manifests.

### 7.4 A concrete taxonomy fix

17.22% of `kubernetes` path hits are Helm templates, caught by the
`(^|/)templates/[^/]+\.ya?ml` rule. The fix needs repository context, which
`detect_units` already computes: a `templates/` directory whose parent holds a
`Chart.yaml` is a chart, so its YAML belongs to `helm`, not `kubernetes`.

### 7.5 Licensing holds up

Of 1,196 real Kubernetes manifests, 68 are `permissive` (**5.69%**), consistent
with the 6.55% measured across the IaC slice by metadata alone. Machine-generated
files are negligible: 13 of 13,137 YAML files carry a generation marker.

### 7.6 Unit extraction works, and star-gating is expensive

Extracting complete units to disk produced coherent, self-contained charts
(`Chart.yaml` + `values.yaml` + templates, intact and consistent). One extracted
sample even ships a `policy/v1beta1` PodDisruptionBudget, a long-removed API
version, which is itself a good benchmark task.

The cost of star-gating is the planning number to keep: charts in repositories
with 10 or more stars occur at roughly 3.2e-5 per repository, so collecting
~2,000 of them means reading around 36% of the corpus, about **1.7 TB** of
content transfer.

The cheaper route is to gate on content quality rather than popularity: template
count, presence of `values.yaml`, parseable `Chart.yaml`, renderable templates.
Scanning ~5% of the corpus (~230 GB) yields roughly 4,500 charts to filter down
from, which is ample for a benchmark suite.

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pyarrow huggingface_hub fsspec pytest
.venv/bin/python -m pytest tests/ -q
.venv/bin/python probe_footer.py 0              # column sizes of one shard
.venv/bin/python probe_licenses.py 0 4098 8195  # license_type distribution
.venv/bin/python -m stackslice.scan --shards 24 # the metadata report
.venv/bin/python -m stackslice.validate --shards 3 # the content validation
```
