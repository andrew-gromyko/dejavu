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

## Outputs

- Prints first-contact world/local point and force.
- Saves `runs/mug_contact_probe/probe_rgb_first_contact.png`.
- Saves `runs/mug_contact_probe/probe_rgb_final.png`.
- During full multimodal mode, writes `realtime_impact.wav`.

## Code Layout

- `env.py`: environment definition.
- `run_probe.py`: scripted trajectory + orchestration.
- `dashboard.py`: dashboard composition and display backends.
- `objectfolder_worker.py`: background tactile/audio worker process.
- `objectfolder_query.py`: ObjectFolder query bridge.
