# ManiSkill Custom Env Sandbox

Small workspace for custom ManiSkill environments and probe scripts.

## Quick Start

1. Clone with submodules (or init submodules after clone):

```bash
git clone --recurse-submodules https://github.com/andrew-gromyko/dejavu.git
cd dejavu
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

3. Install core dependencies:

```bash
pip install -r requirements.txt
```

4. Run the mug contact probe (video + dashboard, no ObjectFolder):

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

## Assets Setup (Required For Full Multimodal Probe)

`assets/` is intentionally gitignored (large binary data). For full tactile+audio mode, place the following files locally:

- `assets/objectfolder2/025_mug_ObjectFile.pth`
- `assets/objectfolder2/ObjectFolder1-100.tar.gz` (or `assets/objectfolder/ObjectFolder1-100.tar.gz`)
- `assets/unitouch/last_new.ckpt`

After placing assets, verify setup:

```bash
python -m custom_envs.mug_contact_probe.verify_setup
```

If you only want RGB/contact probing, use `--skip-objectfolder` and no assets are needed.

## Project Layout

- `custom_envs/`: local environments and probe runners.
- `assets/`: checkpoints and object files used by optional multimodal features.
- `runs/`: generated outputs (images, logs, videos, wav files).
- `third_party/`: git submodules (ObjectFolder, UniTouch, mplib).

## Notes

- On macOS, `afplay` is used for impact audio playback.
- If you only need contact extraction and rendering, use `--skip-objectfolder`.
