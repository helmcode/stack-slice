# stack-slice

Carve targeted slices out of [The Stack v3](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train)
without downloading 4.71 TB.

The corpus ships as 8196 parquet shards where `files.list.element.content` is
96.9% of the bytes. Reading only the metadata leaves over HTTP range requests
costs 1% of the transfer, so the entire corpus can be surveyed, classified and
sized from a laptop. A full metadata index of all 2.9 billion files costs about
61 GB of transfer.

The first target slice is infrastructure-as-code, because language detection
alone cannot find it: Helm templates, Kubernetes manifests, Ansible playbooks,
CI pipelines and Prometheus rules are all just `YAML` to go-enry.

## Status

Phases 0 (sizing), 0.5 (content validation), 1 (final labelling) and 2 (the
full-corpus harvest) are complete. The whole of The Stack v3 was swept in 12h27m
at 94 MB/s, yielding **13,367,187 quality-gated units** in 5.0 GB gzipped:
4.6M Dockerfiles, 3.4M GitHub Actions workflows, 3.2M Compose files, 823k
Terraform modules, 823k Kubernetes manifest sets, 445k Ansible roles and 65k
complete Helm charts.
See [FINDINGS.md](FINDINGS.md) for the numbers, including two findings about the
dataset's license labels that affect what any derived artifact can claim.

Measured label quality against an independent YAML parser: Kubernetes 97.8%
precision / 97.2% recall, GitHub Actions 98.9% / 100%, Compose 97.2% / 99.3%,
Helm 93.9% precision, Ansible 86.4% / 90.3%, Terraform 98.9% and Dockerfile
99.7% from paths alone.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pyarrow huggingface_hub fsspec pytest
```

No Hugging Face token is required; the dataset is public and ungated.

## Usage

```bash
# Aggregate IaC statistics over an evenly spaced sample of shards
.venv/bin/python -m stackslice.scan --shards 24 --workers 8

# Per-column compressed sizes of a single shard
.venv/bin/python probe_footer.py 0

# Score final labels against the YAML parser, and profile dropped YAML
.venv/bin/python -m stackslice.measure --shards 3

# license_type distribution and repository purity
.venv/bin/python probe_licenses.py 0 4098 8195

# Whether license labels are propagated across a repository tree
.venv/bin/python probe_license_propagation.py 4098
```

## Layout

| Path | Purpose |
|---|---|
| `stackslice/taxonomy.py` | Path + language rules with precision labels, and repository-level unit detection |
| `stackslice/detect.py` | Fast content detectors for YAML that paths cannot disambiguate |
| `stackslice/resolve.py` | Final labelling: repository context, then content, then paths |
| `stackslice/verify.py` | Independent arbiter that parses YAML with PyYAML to score the detectors |
| `stackslice/scan.py` | Cheap sampling metadata scan (1% of bytes) |
| `stackslice/validate.py` | Content pass measuring the path heuristics |
| `stackslice/measure.py` | Scores final labels against the parser, and characterises dropped YAML |
| `tests/` | 84 tests over the classifier, detectors, resolver and arbiter |
| `stackslice/extract.py` | Streaming sweep that emits quality-gated units as JSONL |
| `stackslice/repair.py` | Recovers records from a gzip stream cut mid-member |
| `probe_*.py` | One-off measurements backing FINDINGS.md |

## Design notes

- **Precision labels.** Every classification is `exact` (unambiguous extension
  or filename), `structural` (confirmed by repository layout) or `heuristic`
  (path hint, needs content validation). Reports never mix them silently.
- **Units over files.** A Helm chart is only useful when `Chart.yaml`,
  `values.yaml` and `templates/` travel together. Repository-level grouping in
  v3 is what makes that extractable.
- **One output file per run.** Runs never share a gzip file. Appending is
  tempting and nearly cost a 12-hour harvest: `GzipFile.flush()` leaves the
  member unterminated, so a killed writer puts a wound in the file that stops
  every standard reader on everything written after it.
- **Sampling.** Shards are sampled evenly across the index range so no
  repository ordering inside the corpus can bias the result. Extrapolation
  corrects for both sampled shards and sampled row groups.

## Licensing

Code in this repository is Apache-2.0 (see LICENSE). The Stack v3 itself is
ODC-By 1.0 and its
contents remain under their original licenses; see FINDINGS.md section 4 before
assuming any subset is permissively licensed.
