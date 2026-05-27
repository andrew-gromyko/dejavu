# Custom Environments

Local ManiSkill environments are defined here and registered when `custom_envs` is imported.

## Current Packages

- `custom_envs/mug_contact_probe/`: scripted mug-contact probe with optional multimodal dashboard.

## Run a Package

From repo root:

```bash
python -m custom_envs.mug_contact_probe.run_probe --skip-objectfolder
```

Use package modules (`python -m ...`) so imports and registration work consistently.
