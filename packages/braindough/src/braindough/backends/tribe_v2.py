"""Lazy TRIBE v2 backend integration."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from braindough.backends.base import BackendResult
from braindough.config import ExperimentSpec
from braindough.stimuli import Stimulus
from braindough.storage import BraindoughPaths


class TribeV2Backend:
    """Run Meta/Facebook TRIBE v2 when it is available locally."""

    name = "tribe-v2"

    def run(
        self,
        spec: ExperimentSpec,
        stimuli: list[Stimulus],
        paths: BraindoughPaths,
        run_dir: Path,
    ) -> BackendResult:
        del run_dir
        paths.init()
        os.environ.update(paths.env())
        repo_path = _resolve_tribe_repo(paths)
        if repo_path is not None and str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            import pandas as pd  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
            from tribev2.demo_utils import TribeModel  # type: ignore[import-not-found]
        except Exception as exc:
            return _skipped(
                "TRIBE v2 Python dependencies are not importable. "
                "Install or expose the TRIBE source/dependencies explicitly, then rerun. "
                f"Original error: {exc}"
            )

        checkpoint = spec.backend_config.get("checkpoint", "facebook/tribev2")
        checkpoint_path = Path(str(checkpoint)).expanduser()
        checkpoint_ref = checkpoint_path if checkpoint_path.exists() else checkpoint
        config_update = dict(spec.backend_config.get("config_update", {}))
        config_update.setdefault("data.num_workers", 0)
        config_update.setdefault("data.batch_size", 1)
        device_order = _device_order(torch, spec.backend_config.get("device"))
        cluster: Any = spec.backend_config.get("cluster")

        model = None
        load_errors: list[str] = []
        for device in device_order:
            try:
                device_config = {
                    **config_update,
                    "data.audio_feature.device": device,
                    "data.video_feature.image.device": device,
                    "data.image_feature.image.device": device,
                }
                model = TribeModel.from_pretrained(
                    checkpoint_ref,
                    cache_folder=paths.hf_cache / "tribe-v2-features",
                    cluster=cluster,
                    device=device,
                    config_update=device_config,
                )
                break
            except Exception as exc:
                load_errors.append(f"{device}: {exc}")
                model = None
        if model is None:
            return _skipped(
                "TRIBE v2 model could not be loaded on this machine. "
                + " | ".join(load_errors[-3:])
            )
        event_duration = float(
            spec.backend_config.get(
                "event_duration_seconds",
                getattr(model.data, "duration_trs", 100) * model.data.TR,
            )
        )

        responses: dict[str, np.ndarray] = {}
        events: list[dict[str, object]] = []
        prediction_errors: list[str] = []
        max_predictions = int(spec.backend_config.get("max_predictions", 0))
        attempted_predictions = 0
        for stimulus in stimuli:
            if max_predictions > 0 and (
                len(responses) >= max_predictions
                or attempted_predictions >= max_predictions
            ):
                events.append(
                    {
                        "event": "prediction_budget_reached",
                        "backend": self.name,
                        "max_predictions": max_predictions,
                        "attempted_predictions": attempted_predictions,
                    }
                )
                break
            if stimulus.modality not in {"video", "audio"}:
                continue
            event_type = "Video" if stimulus.modality == "video" else "Audio"
            event = {
                "type": event_type,
                "filepath": str(stimulus.path),
                "start": 0,
                "duration": event_duration,
                "timeline": stimulus.stimulus_id,
                "subject": "default",
            }
            try:
                attempted_predictions += 1
                dataframe = pd.DataFrame([event])
                prediction, _segments = model.predict(dataframe, verbose=False)
                responses[stimulus.stimulus_id] = np.asarray(
                    prediction, dtype=np.float32
                )
                events.append(
                    {
                        "event": "prediction",
                        "backend": self.name,
                        "stimulus_id": stimulus.stimulus_id,
                        "shape": list(responses[stimulus.stimulus_id].shape),
                    }
                )
            except Exception as exc:
                prediction_errors.append(
                    f"{stimulus.stimulus_id}: {exc}\n{traceback.format_exc(limit=2)}"
                )

        status = "completed" if responses else "skipped"
        blocker = (
            None
            if responses
            else "No TRIBE v2 predictions completed. "
            + " | ".join(prediction_errors[:3])
        )
        metrics: dict[str, Any] = {
            "backend": self.name,
            "n_stimuli": len(stimuli),
            "n_responses": len(responses),
            "prediction_errors": prediction_errors[:10],
            "max_predictions": max_predictions or None,
            "attempted_predictions": attempted_predictions,
            "device_attempts": device_order,
            "checkpoint": str(checkpoint_ref),
        }
        return BackendResult(
            status=status,
            responses=responses,
            events=events,
            metrics=metrics,
            blocker=blocker,
        )


def _resolve_tribe_repo(paths: BraindoughPaths) -> Path | None:
    configured = os.environ.get("BRAINDOUGH_TRIBE_REPO")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([paths.tribe_code_dir, Path("/tmp/braindough-tribev2-inspect")])
    for candidate in candidates:
        if (candidate / "tribev2" / "demo_utils.py").is_file():
            return candidate.resolve()
    return None


def _device_order(torch_module: Any, configured: object) -> list[str]:
    if isinstance(configured, str) and configured:
        return [configured, "cpu"] if configured != "cpu" else ["cpu"]
    devices: list[str] = []
    if (
        getattr(torch_module.backends, "mps", None) is not None
        and torch_module.backends.mps.is_available()
    ):
        devices.append("mps")
    devices.append("cpu")
    return devices


def _skipped(blocker: str) -> BackendResult:
    return BackendResult(
        status="skipped",
        responses={},
        events=[
            {"event": "backend_skipped", "backend": "tribe-v2", "blocker": blocker}
        ],
        metrics={"backend": "tribe-v2", "n_responses": 0},
        blocker=blocker,
    )
