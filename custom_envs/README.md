# Custom Environments

Local ManiSkill environments are defined here and registered when `custom_envs` is imported.

## Current Packages

- `custom_envs/mug_contact_probe/`: scripted mug-contact probe with optional multimodal dashboard.
- `custom_envs/mass_memory_bin_sort/`: mass-based object sorting benchmark.
- `custom_envs/sort_ycb_into_bins/`: earlier YCB sorting prototype.

## Run a Package

From repo root:

```bash
python -m custom_envs.mug_contact_probe.run_probe --skip-objectfolder
```

Use package modules (`python -m ...`) so imports and registration work consistently.
