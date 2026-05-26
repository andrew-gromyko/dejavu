# ManiSkill Custom Env Sandbox

Small workspace for custom ManiSkill environments and probe scripts.

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install core dependencies:

```bash
pip install -r requirements.txt
```

3. Run the mug contact probe (video + dashboard, no ObjectFolder):

```bash
python -m custom_envs.mug_contact_probe.run_probe --skip-objectfolder
```

## Optional: ObjectFolder Tactile + Audio

Install optional dependencies:

```bash
pip install -r requirements-objectfolder.txt
```

Then run full 3-modality probe:

```bash
python -m custom_envs.mug_contact_probe.run_probe
```

## Project Layout

- `custom_envs/`: local environments and probe runners.
- `assets/`: checkpoints and object files used by optional multimodal features.
- `runs/`: generated outputs (images, logs, videos, wav files).
- `third_party/`: vendored upstream code (ObjectFolder, UniTouch, mplib).

## Notes

- On macOS, `afplay` is used for impact audio playback.
- If you only need contact extraction and rendering, use `--skip-objectfolder`.
