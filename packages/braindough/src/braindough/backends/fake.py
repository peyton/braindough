"""Deterministic fake backend for CI and development."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from braindough.backends.base import BackendResult
from braindough.config import ExperimentSpec
from braindough.stimuli import Stimulus
from braindough.storage import BraindoughPaths


class FakeBackend:
    """A deterministic response generator with cortical-fingerprint structure."""

    name = "fake"

    def run(
        self,
        spec: ExperimentSpec,
        stimuli: list[Stimulus],
        paths: BraindoughPaths,
        run_dir: Path,
    ) -> BackendResult:
        del paths, run_dir
        vertices = int(spec.backend_config.get("vertices", 128))
        timesteps = int(spec.backend_config.get("timesteps", 6))
        responses: dict[str, np.ndarray] = {}
        events: list[dict[str, object]] = []

        for stimulus in stimuli:
            responses[stimulus.stimulus_id] = self._response(
                stimulus=stimulus,
                seed=spec.seed,
                timesteps=timesteps,
                vertices=vertices,
            )
            events.append(
                {
                    "event": "prediction",
                    "backend": self.name,
                    "stimulus_id": stimulus.stimulus_id,
                    "shape": list(responses[stimulus.stimulus_id].shape),
                }
            )

        metrics = {
            "backend": self.name,
            "n_stimuli": len(stimuli),
            "n_responses": len(responses),
            "response_vertices": vertices,
            "response_timesteps": timesteps,
        }
        return BackendResult(
            status="completed", responses=responses, events=events, metrics=metrics
        )

    @staticmethod
    def _response(
        stimulus: Stimulus, seed: int, timesteps: int, vertices: int
    ) -> np.ndarray:
        material = f"{seed}:{stimulus.stimulus_id}:{stimulus.sha256}".encode()
        digest = hashlib.sha256(material).digest()
        local_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(local_seed)
        response = rng.normal(0, 0.15, (timesteps, vertices)).astype(np.float32)

        suite_gain = {
            "image_activation": 0.7,
            "visual_controls": 0.25,
            "visual_perturbations": 0.55,
            "temporalization": 0.65,
            "audio_controls": 0.45,
            "latent_network_ica_explorer": 0.58,
            "virtual_lesion_lab": 0.52,
            "discrete_stimulus_optimizer": 0.62,
            "counterfactual_editing_workbench": 0.56,
        }.get(stimulus.suite, 0.35)
        modality_offset = 0.2 if stimulus.modality == "audio" else 0.0
        temporal = np.linspace(0.2, 1.0, timesteps, dtype=np.float32)[:, None]
        spatial = np.sin(np.linspace(0, np.pi * 4, vertices, dtype=np.float32))[None, :]
        response += (suite_gain + modality_offset) * temporal * spatial
        if "blank" in stimulus.kind or stimulus.kind == "silence":
            response *= 0.15
        return response
