# Installation

This repository is an overlay for a GEM checkout.

## Verified Upstream

The patch was tested against:

```text
GEM repository: https://github.com/lanxiang1017/GEM.git
GEM commit:     c8a580faae819c57c008e94fa080f5d3c6881769
```

For a fresh reproducible checkout, use:

```bash
bash scripts/prepare_gem_checkout.sh /path/to/GEM
```

This clones GEM if needed, checks out the verified commit, and applies the
TRACE-ECG patch.

## Apply GEM Patch Manually

If you already have a clean GEM checkout at the verified commit:

```bash
bash scripts/apply_gem_patch.sh /path/to/GEM
```

The patch command is idempotent:

1. if the tracked patch is already applied, it reports that state;
2. if the tracked patch can be applied cleanly, it applies it;
3. otherwise it stops and asks you to verify the upstream GEM commit.

The patch step:

1. applies tracked GEM source changes from `patches/gem_trace_ecg_tracked_changes.patch`;
2. copies new files from `patches/new_files/` into the GEM checkout.

## Environments

### Scoring-only environment

The public scoring utilities can run in a lightweight environment:

```bash
conda env create -f env/trace_ecg_eval_environment.yml
conda activate trace-ecg-eval
```

Equivalent pip requirements are in:

```text
env/trace_ecg_public_requirements.txt
```

### Training environment

Training uses the patched GEM training environment. After preparing GEM:

```bash
bash /path/to/GEM/setup.sh
conda activate gem
python -m pip install -r env/trace_ecg_public_requirements.txt
```

The public scripts assume a conda environment named `gem` unless `CONDA_ENV` is
set.

## Paths

For machine-specific paths, copy the example first:

```bash
cp configs/paths.example.env paths.env
source paths.env
```
