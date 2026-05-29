# Mug Contact Probe

Deterministic Panda-to-mug contact probe with a live dashboard.

## Install

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional (tactile + audio via ObjectFolder):

```bash
pip install -r requirements-objectfolder.txt
```

## Run

Default (browser dashboard, video + tactile + audio):

```bash
python -m custom_envs.mug_contact_probe.run_probe
```

Core-only (no ObjectFolder models):

```bash
python -m custom_envs.mug_contact_probe.run_probe --skip-objectfolder
```

Native ManiSkill viewer only:

```bash
python -m custom_envs.mug_contact_probe.run_probe --render-mode human --no-dashboard
```

DejaVu memory benchmark viewer:

```bash
python -m custom_envs.mug_contact_probe.run_dejavu_viewer --task material --H 4 --D 2 --seed 22
```

Print the generated DejaVu episode without starting ManiSkill:

```bash
python -m custom_envs.mug_contact_probe.run_dejavu_viewer --task material --H 4 --D 2 --seed 22 --dry-run
```

## Outputs

- Prints first-contact world/local point and force.
- Saves `runs/mug_contact_probe/probe_rgb_first_contact.png`.
- Saves `runs/mug_contact_probe/probe_rgb_final.png`.
- During full multimodal mode, writes `realtime_impact.wav`.

## Code Layout

- `env.py`: environment definition.
- `dejavu_env.py`: reproducible DejaVu encounter/query generator.
- `dejavu_ms3_env.py`: ManiSkill3 visualization environment for DejaVu episodes.
- `run_dejavu_viewer.py`: scripted Panda viewer for DejaVu encounter actions.
- `run_probe.py`: scripted trajectory + orchestration.
- `dashboard.py`: dashboard composition and display backends.
- `objectfolder_worker.py`: background tactile/audio worker process.
- `objectfolder_query.py`: ObjectFolder query bridge.
