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
            response = self._response(
                stimulus=stimulus,
                seed=spec.seed,
                timesteps=timesteps,
                vertices=vertices,
            )
            parent_id = str(stimulus.metadata.get("parent_id", ""))
            if parent_id in responses and stimulus.suite in {
                "virtual_lesion_lab",
                "counterfactual_editing_workbench",
                "focused_ultrasound_bridge",
            }:
                response = self._paired_response(
                    stimulus=stimulus,
                    parent=responses[parent_id],
                    fallback=response,
                )
            responses[stimulus.stimulus_id] = response
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
            "focused_ultrasound_bridge": 0.5,
        }.get(stimulus.suite, 0.35)
        modality_offset = 0.2 if stimulus.modality == "audio" else 0.0
        temporal = np.linspace(0.2, 1.0, timesteps, dtype=np.float32)[:, None]
        spatial = np.sin(np.linspace(0, np.pi * 4, vertices, dtype=np.float32))[None, :]
        response += (suite_gain + modality_offset) * temporal * spatial
        if "blank" in stimulus.kind or stimulus.kind == "silence":
            response *= 0.15
        if stimulus.suite == "discrete_stimulus_optimizer":
            index = int(stimulus.metadata.get("candidate_index", 0))
            shape = str(stimulus.metadata.get("params", {}).get("shape", ""))
            shape_gain = {
                "circle": 0.02,
                "square": 0.04,
                "triangle": 0.08,
                "stripe": 0.12,
            }
            response += (shape_gain.get(shape, 0.0) + 0.005 * index) * spatial
        return response

    @staticmethod
    def _paired_response(
        stimulus: Stimulus, parent: np.ndarray, fallback: np.ndarray
    ) -> np.ndarray:
        kind = str(
            stimulus.metadata.get("lesion_base_type")
            or stimulus.metadata.get("edit_base_type")
            or stimulus.metadata.get("condition")
            or stimulus.kind
        )
        strength = _metadata_float(stimulus.metadata.get("strength", 1.0), default=1.0)
        if stimulus.suite == "focused_ultrasound_bridge":
            strength = _metadata_float(
                stimulus.metadata.get("software_dose_index", 0.0), default=0.0
            )
            scale = {
                "baseline": 0.0,
                "sham": 0.012,
                "spatial_control": 0.045,
                "active": 0.08 + 0.12 * max(strength, 0.0),
            }.get(kind, 0.04)
            delta = fallback - np.mean(fallback, dtype=np.float32)
            return (parent + delta * np.float32(scale)).astype(np.float32)
        scale = {
            "sham_reencode": 0.01,
            "low_contrast": 0.04,
            "local_blur": 0.05,
            "blur_suppression": 0.06,
            "background_lighten": 0.07,
            "color_swap": 0.08,
            "mask_left": 0.11,
            "mask_right": 0.11,
            "central_occlusion": 0.14,
            "object_mask": 0.16,
            "random_patch_same_area": 0.18,
            "tile_scramble": 0.22,
            "blank_suppression": 0.26,
        }.get(kind, 0.1)
        delta = fallback - np.mean(fallback, dtype=np.float32)
        return (parent + delta * np.float32(scale * max(strength, 0.0))).astype(
            np.float32
        )


def _metadata_float(value: object, *, default: float) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return default
    return default
