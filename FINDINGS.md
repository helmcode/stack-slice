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

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pyarrow huggingface_hub fsspec pytest
.venv/bin/python -m pytest tests/ -q
.venv/bin/python probe_footer.py 0              # column sizes of one shard
.venv/bin/python probe_licenses.py 0 4098 8195  # license_type distribution
.venv/bin/python -m stackslice.scan --shards 24 # the full report
```
