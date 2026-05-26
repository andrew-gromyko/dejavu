"""Background worker process for ObjectFolder tactile/audio queries."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from time import time_ns

import numpy as np
import torch
from scipy.io import wavfile


def objectfolder_worker(request_queue, response_queue, object_id):
    torch.set_num_threads(1)

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from custom_envs.mug_contact_probe.objectfolder_query import ObjectFolderModalityRenderer

    try:
        renderer = ObjectFolderModalityRenderer(object_id)
    except Exception as exc:
        print(f"Error loading models in background worker: {exc}", file=sys.stderr)
        return

    audio_procs = []
    while True:
        try:
            req = request_queue.get()
            if req is None:
                break

            req_type = req["type"]
            pad_id = req.get("pad_id")
            if req_type == "tactile":
                local_point = req["local_point"]
                press_depth = req["press_depth"]
                orientation_arr = np.array([np.radians(5.0), 0.0], dtype=np.float32)
                touch = renderer.make_touch_spec(local_point, orientation_arr, press_depth)
                tactile_rgb = renderer.render_tactile(touch)
                response_queue.put(("tactile", pad_id, tactile_rgb))
            elif req_type == "audio":
                local_point = req["local_point"]
                press_depth = req["press_depth"]
                orientation_arr = np.array([np.radians(5.0), 0.0], dtype=np.float32)
                touch = renderer.make_touch_spec(local_point, orientation_arr, press_depth)
                audio_waveform = renderer.render_audio(touch)

                pad_suffix = pad_id if pad_id is not None else "contact"
                audio_path = Path(
                    f"/Users/andrew/Documents/git/maniskill/runs/mug_contact_probe/realtime_impact_{pad_suffix}_{time_ns()}.wav"
                )
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(audio_path, 44100, audio_waveform.astype(np.float32))

                audio_procs = [proc for proc in audio_procs if proc.poll() is None]
                audio_procs.append(subprocess.Popen(["afplay", str(audio_path)]))

                response_queue.put(("audio", pad_id, audio_waveform))
        except Exception as exc:
            print(f"Error in background worker: {exc}", file=sys.stderr)

    for audio_proc in audio_procs:
        try:
            audio_proc.terminate()
        except Exception:
            pass
