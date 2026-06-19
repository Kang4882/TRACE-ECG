# Installation

This repository is an overlay for a GEM checkout.

## Apply GEM Patch

```bash
bash scripts/apply_gem_patch.sh /path/to/GEM
```

The patch step:

1. applies tracked GEM source changes from `patches/gem_anchor_ecg_tracked_changes.patch`;
2. copies new files from `patches/new_files/` into the GEM checkout.

## Environment

Use the GEM training environment. The public scripts assume a conda environment
named `gem` unless `CONDA_ENV` is set.

```bash
source configs/paths.example.env
```

For machine-specific paths, copy the example first:

```bash
cp configs/paths.example.env paths.env
source paths.env
```

