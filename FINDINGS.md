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

## 8. Phase 1: the four YAML classes are now usable

The Phase 0.5 verdict was that path rules cannot label YAML. `resolve.py`
replaces them with a three-source pipeline, and `measure.py` scores the result
against `verify.py`, which loads every document with PyYAML and inspects its real
structure. Scoring regexes against regexes proves nothing; scoring them against
an independent parser does.

The pipeline, in precedence order:

1. **Repository context wins.** A file inside a directory holding `Chart.yaml` is
   Helm, whatever its content says, which is the only way to claim `values.yaml`
   and `_helpers.tpl`. A file inside `roles/<name>/{tasks,defaults,vars}/` is
   Ansible for the same reason.
2. **Content decides for YAML.** The low-precision path guesses (`k8s/`,
   `manifests/`, `templates/`) are no longer emitted as labels at all.
3. **Paths stay authoritative where they are exact** (`.tf`, `Dockerfile`,
   `.nix`) or canonical rather than conventional (`.github/workflows/`).

### 8.1 Before and after, same 3 shards

| Class | Precision before | Precision after | Recall before | Recall after |
|---|---|---|---|---|
| kubernetes | 56.81% | **97.82%** | 33.11% | **97.17%** |
| github_actions | n/a | **98.90%** | 97.10% | **100.00%** |
| compose | n/a | **97.20%** | 77.89% | **99.26%** |
| ansible | 35.80% | **86.40%** | 28.46% | **90.26%** |
| helm | not measurable | **93.94%** | not measurable | see below |
| terraform | 98.87% | 98.87% (unchanged, path is exact) | | |
| dockerfile | 99.73% | 99.73% (unchanged, path is exact) | | |

So Kubernetes, Helm, GitHub Actions and Compose are all usable now, and Ansible
is usable with a caveat. Recall for Kubernetes went from catching one manifest in
three to catching 97 in 100, because content finds manifests wherever they live.

Where the residual disagreement goes is worth reading: of the manifests the
parser recognises, 97.17% are labelled `kubernetes`, 2.4% are labelled `helm`
(untemplated YAML sitting inside a chart, arguably correct) and only 0.42% are
dropped entirely.

### 8.2 Evidence mix, which is the interesting operational number

| Class | From content | From repo context | From path alone |
|---|---|---|---|
| kubernetes | 800 (+394 agreeing with path) | 0 | 0 |
| ansible | 473 (+26) | **460** | 38 |
| helm | 106 | **98** | 42 |
| github_actions | 42 (+1406) | 0 | 9 |
| compose | 115 (+1056) | 0 | 77 |

Ansible and Helm draw roughly half their labels from repository context alone.
Those files cannot be classified from their own bytes at any cost, which is the
concrete argument for The Stack v3's repository grouping over v2's flat files.

### 8.3 Two honest caveats

- **The Ansible arbiter is weak.** `apiVersion` plus `kind` is unambiguous, so
  the Kubernetes score is solid. "A list of mappings carrying Ansible-ish keys"
  also matches plenty of ordinary YAML lists, so Ansible's 86.40% is a soft
  floor and not a precise figure. Much of the unconfirmed remainder is role
  variable files (`defaults/main.yml`), which are plain variable trees that no
  parser can positively identify as Ansible.
- **Helm recall cannot be measured this way at all.** Chart templates are not
  valid YAML, so the parser can never bless them. Precision is measured
  structurally instead: 93.94%, counting a Go-templated chart member or a
  parseable `Chart.yaml` as confirmation.

### 8.4 Fixes this phase, all found by tests or by inspecting disagreements

- `templates/*.yaml` under a `Chart.yaml` now resolve to `helm`, not
  `kubernetes`. This was the 17.22% contamination measured in Phase 0.5.
- Compose precision rose from 84.91% to 97.20% by requiring `services:` to open
  a **mapping**. Travis CI declares a top-level `services:` as a *list*
  (`- docker`), and CodeBuild, Amplify and Read the Docs all pair `version:` with
  a phase called `build:`; the old rule swallowed all four.
- Jinja inside `roles/*/templates/` no longer reads as a Helm chart template.
  Ansible roles have `templates/` directories too.
- Bare Ansible task lists (a module call with no `hosts:` or `tasks:` header) are
  now detected.
- Added `buildspec.yml` (AWS CodeBuild) as its own class rather than leaving it
  to be mislabelled.

## 9. What the non-infrastructure YAML actually is

59.58% of YAML in the corpus is not infrastructure. Rather than guess, the
measurement pass collects the real top-level keys and filenames of every file it
drops. The families, from 13,088 YAML files across 3 shards:

| Family | Evidence | Count |
|---|---|---|
| **JVM / Spring app config** | `application.yml` (431), `application.yaml` (55), `bootstrap.yml` (71), key signatures `spring`, `server,spring`, `eureka,server,spring` | ~750 |
| **i18n and pluralisation** | signature `one,two` (336, plural forms), `en` (50), `en.yml` (40) | ~430 |
| **Drupal exported config** | signature `dependencies,langcode,status,uuid` (256), first key `uuid` (259) | ~260 |
| **dbt data models** | `sources.yml` (131), `schema.yml` (114), signatures `models,version` (157), `sources,version` (156) | ~300 |
| **Static site generators** | `_config.yml` (168, Jekyll) | ~170 |
| **Dart / Flutter packages** | `pubspec.yaml` (160), signature `description,name,publish_to,version` (100) | ~160 |
| **Conda environments and recipes** | `meta.yaml` (68), `environment.yml` (33), signature `channels,dependencies,name` (60) | ~100 |
| **ML experiment configs (Hydra)** | signatures `defaults,model,params,trainer` (82), `defaults,madx,model,params` (43), `MATCHER,MODEL,TEST,TRAIN` (45) | ~170 |
| **API specs and JSON Schema** | first key `openapi` (68), `swagger.yaml` (37), signature `$id,$schema,allOf,type` (51) | ~155 |
| **Repo hygiene** | `funding.yml` (63), `.pre-commit-config.yaml` (36) | ~100 |
| **Vendored copies of CI config** | `.travis.yml` (70) inside `node_modules/` and `vendor/`, correctly excluded as noise | ~70 |
| **Framework routing and DB config** | `database.yml` (44, Rails), `routing.yml` (34, Symfony), `config.yml`/`config.yaml` (152) | ~230 |
| **Unparseable** | 652 files, of which 72 contain template actions (fragments, broken YAML) | 652 |

The single largest key signature in the entire non-infrastructure bucket is
`one,two`: Rails-style pluralisation files. The second is Drupal's config export
format. Neither has anything to do with deployment, and both would have been
swept into a naive "all YAML is config" slice.

Practical consequence: the drop decisions are sound. Nothing in this list belongs
in an IaC dataset, and the two families that come closest (OpenAPI specs and
Conda environments) are already labelled separately rather than discarded blindly.

## 10. Phase 2: the full-corpus harvest

The whole corpus was swept on `nan-eu008`: **8,196 of 8,196 shards, 4.71 TB
pulled in 12h27m at 94 MB/s**, across 157.9M repositories, 114,761 forks skipped.
Nothing was stored but the output, because extraction streams shards over HTTP
range requests. `~/.cache/huggingface` grew to 1.2 MB after 4 TB of transfer, and
the host's free disk fell by 2 GB over the whole run.

| Unit | Extracted | Gate pass rate |
|---|---|---|
| **dockerfile** | **4,611,725** | 96.9% |
| **workflow** (GitHub Actions) | **3,384,907** | 98.4% |
| **compose** | **3,213,543** | 97.2% |
| **terraform_module** | **823,406** | 99.1% |
| **manifest_set** (Kubernetes) | **823,369** | 98.6% |
| **ansible_role** | **444,744** | 95.0% |
| **helm_chart** | **65,493** | 69.0% |

**13,367,187 units, 5.0 GB gzipped.** The 5%-sample projections held to within
10% on every class, which retroactively validates the evenly spaced sampling.

Rejections at scale show what the content gates buy, since none of these can be
filtered by path alone: 130,812 files named `Dockerfile` with no valid `FROM`;
85,376 non-Compose files matching Compose names (Travis, CodeBuild, Amplify and
Read the Docs); 21,797 single-template Helm charts, which is why Helm has the
lowest pass rate; 3,927 generated Dockerfiles; 478 unparseable `Chart.yaml`.

### 10.1 Three failures worth recording

**A deadlock I introduced.** `ProcessPoolExecutor(max_tasks_per_child=25)`, added
as belt-and-braces against stale HTTP sessions, hangs once every worker has
retired: at exactly `workers * limit` tasks (shard 300 with 12 workers) the pool
stops spawning replacements and the parent blocks in `futex_wait_queue` with no
children left. It failed silently, with no exception and no log line, so two
status reports called it healthy while it had been stopped for 68 minutes. Log
staleness, not log content, is what detects this. Retries in `with_retries` are
the correct fix for stale sessions; `max_tasks_per_child` is now documented as
forbidden.

**PyYAML raises outside YAMLError.** An explicit `!!bool` tag with a non-boolean
scalar (`flag: !!bool test`) raises a bare `KeyError` from
`construct_yaml_bool`; other malformed tags raise `TypeError` or
`AttributeError`. One such file aborted an entire shard. The arbiter now treats
any exception as "does not parse".

**Appending to gzip nearly cost the whole harvest.** `GzipFile.flush()` writes a
zlib sync point but does not terminate the member: no CRC, no ISIZE trailer.
Killing the deadlocked writer therefore left an unterminated member, and because
each run appended a new member to the same file, later runs landed behind that
wound. Every standard reader stops there, so `zcat` reported
`invalid compressed data--format violated` and counted 384,404 Dockerfiles out of
4.6M. The data was intact but unreachable.

`stackslice/repair.py` walks members, salvages whatever each yields before it
breaks (replaying byte at a time through the failing chunk, since zlib returns no
partial output from a call that raises), then resyncs on the next gzip magic.
Recovery was exact:

| Source | Units |
|---|---|
| 5% sample sweep | 658,325 |
| killed run, before the deadlock | 485,972 |
| full-corpus run | 12,221,264 |
| resume of the PyYAML shard | 1,626 |
| **sum** | **13,367,187** |
| **recovered by repair** | **13,367,187** |

Zero duplicates, zero unparseable lines, all 8,196 shards present. Not even the
record straddling the wound was lost. The root cause is now impossible: each run
writes `{unit_type}.{tag}.jsonl.gz` with mode `x`, so no run can reopen another
run's file.

## 11. Quality of the harvest, and three things to fix before publishing

Profiled by streaming all 13,367,187 units (`python -m stackslice.summary units_full`),
with reservoir sampling for representative examples rather than heads of files.

| Class | Units | Files/unit | KB/unit | `permissive` |
|---|---|---|---|---|
| helm_chart | 65,493 | 10.40 | 14.0 | 15.10% |
| terraform_module | 823,406 | 5.31 | 7.0 | 6.90% |
| manifest_set | 823,369 | 4.37 | 3.4 | 11.44% |
| ansible_role | 444,744 | 3.81 | 3.5 | 6.74% |
| workflow | 3,384,907 | 1.00 | 1.5 | 7.88% |
| compose | 3,213,543 | 1.00 | 0.9 | 3.59% |
| dockerfile | 4,611,725 | 1.00 | 0.6 | 5.14% |
| **total** | **13,367,187** | 21,550,716 files | **22.40 GB** | |

Substance checks (p50 / p90 / max):

- **helm_chart**: 4 / 9 / 56 templates, 80.3% ship `values.yaml`
- **terraform_module**: 3 / 8 / 60 `.tf` files, 67.5% declare variables, 45.6% outputs
- **manifest_set**: 3 / 8 / 60 manifests. Top kinds: Deployment 388,868, Service
  379,356, Kustomization 128,121, ConfigMap 117,936, Ingress 100,629
- **dockerfile**: 8 / 17 / 2857 instructions, 20.3% multi-stage, 11.0% set `USER`,
  1.4% declare `HEALTHCHECK`
- **workflow**: 1 / 2 / 96 jobs, 5 / 12 / 270 steps, 10.5% set `permissions`
- **compose**: 2 / 5 / 161 services, 32.6% mount volumes, 8.9% healthcheck

Those last three lines are why this is worth publishing as a benchmark corpus:
89% of Compose files have no healthcheck, 89% of Dockerfiles set no `USER`, and
89.5% of workflows declare no `permissions`. That is a measurable baseline for
what models trained on this data will imitate.

### 11.1 The corpus repeats file rows inside a repository

Sampling turned up a chart whose record listed `nginx-service.yaml` eleven times.
It is upstream, not us. In shard 0 of the pinned revision:

- **10.39% of repositories contain duplicate file paths**
- **14.5% of all file rows are duplicates**
- worst case: `LayerZero-Labs/LayerZero-v2` repeats
  `packages/layerzero-v2/evm/protocol/contracts/MessageLibManager.sol` ten times,
  with **one distinct `content_id`**, so it is byte-identical repetition

A git tree cannot hold ten files at one path, so this is an artifact of the
repository-grouping step. Consequence: unit file counts above are inflated by up
to ~14.5%, and any published artifact must deduplicate files by (path, content)
within each unit. No re-sweep is needed; it is a post-process.

### 11.2 Roughly a quarter of Helm charts cannot render standalone

Measured over all 65,493 charts:

| | Share |
|---|---|
| contains any `.tpl` file | 35.3% |
| references `include` or `template` | 63.1% |
| contains a `define` anywhere | 36.8% |
| **references a helper with no `define` present** | **27.1%** |
| ships `NOTES.txt` | 10.8% |

`.tpl` files do survive the corpus filters in general (0.0485% of files in shard
0), so this is mostly charts depending on a parent chart's helpers rather than a
wholesale filtering loss. Either way, `helm template` will fail on that 27.1%, so
renderability must be a published flag and the executable benchmark should draw
from the ~72.9% that are self-contained.

### 11.3 Upstream rewrites the dataset in place, and it happened mid-project

On **2026-07-28 between 08:52 and 09:28 UTC**, while this work was in progress,
upstream applied opt-out removals in a series of commits including
`Clear data before opt-out update`, deleted the data and began re-uploading it
under a new file UUID (`50e95205-…` in place of `4beed122-…`). Mid-afternoon the
new revision held 1,550 shards and 0.89 TB against the previous 8,196 and 4.71 TB,
so the re-upload was still running.

Two consequences:

1. **Paths must be pinned to a revision.** An unpinned path stops resolving the
   moment upstream republishes; `shard_path` now embeds
   `REVISION = de81e3ca7151` (2026-07-24 18:36:04), the revision this harvest read.
   Old revisions stay readable by SHA, so the harvest is reproducible and
   re-verifiable.
2. **The harvest predates those opt-out removals.** Publishing it as-is would
   redistribute code from developers who have since asked to be removed, which
   ODC-By and the dataset's own terms do not permit. Before publishing, every
   unit must be re-filtered against the new revision by `repo_path`, which is a
   metadata-only pass (1% of bytes) of exactly the kind Phase 0 established. That
   re-filter has to be repeated on each upstream patch release.

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pyarrow huggingface_hub fsspec pytest
.venv/bin/python -m pytest tests/ -q
.venv/bin/python probe_footer.py 0              # column sizes of one shard
.venv/bin/python probe_licenses.py 0 4098 8195  # license_type distribution
.venv/bin/python -m stackslice.scan --shards 24 # the metadata report
.venv/bin/python -m stackslice.validate --shards 3 # heuristics vs content
.venv/bin/python -m stackslice.measure  --shards 3 # final labels vs YAML parser
```
