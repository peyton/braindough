"""Deterministic stimulus generation for first-suite experiments."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from braindough.datasets.bold5000 import make_bold5000_stimuli
from braindough.storage import sha256_file, sha256_text
from braindough.suites import validate_suite_names

IMAGE_SIZE = 256
VIDEO_FPS = 4
VIDEO_SECONDS = 2
AUDIO_RATE = 16_000
AUDIO_SECONDS = 2

_FUS_TARGETS: dict[str, dict[str, Any]] = {
    "S1": {
        "label": "S1",
        "network": "somatosensory",
        "task_family": "somatosensory-discrimination",
        "schematic_xy": (0.42, 0.42),
        "control_xy": (0.72, 0.38),
    },
    "hMT_plus": {
        "label": "hMT+",
        "network": "visual-motion",
        "task_family": "visual-motion",
        "schematic_xy": (0.68, 0.52),
        "control_xy": (0.36, 0.66),
    },
    "M1": {
        "label": "M1",
        "network": "motor",
        "task_family": "motor-excitability",
        "schematic_xy": (0.50, 0.34),
        "control_xy": (0.78, 0.60),
    },
    "rIFG": {
        "label": "rIFG",
        "network": "response-inhibition",
        "task_family": "cognitive-control",
        "schematic_xy": (0.74, 0.48),
        "control_xy": (0.30, 0.46),
    },
}

_FUS_PROTOCOLS: dict[str, dict[str, Any]] = {
    "active_low_duty": {
        "condition": "active",
        "software_dose_index": 0.35,
        "virtual_duty_cycle_bins": 0.25,
        "virtual_burst_count": 2,
        "virtual_envelope": [0, 1, 0, 0, 1, 0, 0, 0],
        "sham_mode": "none",
        "nominal_center_frequency_mhz": "",
        "nominal_prf_hz": "",
        "nominal_duty_cycle": "",
        "nominal_sonication_seconds": "",
    },
    "active_mid_duty": {
        "condition": "active",
        "software_dose_index": 0.7,
        "virtual_duty_cycle_bins": 0.5,
        "virtual_burst_count": 4,
        "virtual_envelope": [0, 1, 0, 1, 0, 1, 0, 1],
        "sham_mode": "none",
        "nominal_center_frequency_mhz": "",
        "nominal_prf_hz": "",
        "nominal_duty_cycle": "",
        "nominal_sonication_seconds": "",
    },
    "sham_transmit_blocked": {
        "condition": "sham",
        "software_dose_index": 0.0,
        "virtual_duty_cycle_bins": 0.0,
        "virtual_burst_count": 0,
        "virtual_envelope": [0, 0, 0, 0, 0, 0, 0, 0],
        "sham_mode": "transmit-blocked coupling-pad proxy",
        "nominal_center_frequency_mhz": "",
        "nominal_prf_hz": "",
        "nominal_duty_cycle": "",
        "nominal_sonication_seconds": "",
    },
    "spatial_control": {
        "condition": "spatial_control",
        "software_dose_index": 0.35,
        "virtual_duty_cycle_bins": 0.25,
        "virtual_burst_count": 2,
        "virtual_envelope": [0, 1, 0, 0, 1, 0, 0, 0],
        "sham_mode": "off-target control proxy",
        "nominal_center_frequency_mhz": "",
        "nominal_prf_hz": "",
        "nominal_duty_cycle": "",
        "nominal_sonication_seconds": "",
    },
}


@dataclass(frozen=True)
class Stimulus:
    """One generated or user-supplied stimulus."""

    stimulus_id: str
    suite: str
    modality: str
    kind: str
    path: Path
    sha256: str
    license: str
    metadata: dict[str, Any]

    def to_manifest_input(self, run_dir: Path) -> dict[str, Any]:
        try:
            uri = str(self.path.relative_to(run_dir))
        except ValueError:
            uri = str(self.path)
        return {
            "id": self.stimulus_id,
            "kind": self.modality,
            "media_type": _media_type(self.path, self.modality),
            "uri": uri,
            "sha256": self.sha256,
            "license": self.license,
            "metadata": {
                "suite": self.suite,
                "variant": self.kind,
                **_relative_metadata(self.metadata, run_dir),
            },
        }

    def to_event(self, run_dir: Path) -> dict[str, Any]:
        return {
            "event": "stimulus",
            "stimulus_id": self.stimulus_id,
            "suite": self.suite,
            "modality": self.modality,
            "kind": self.kind,
            "path": str(self.path.relative_to(run_dir)),
            "sha256": self.sha256,
            "metadata": _relative_metadata(self.metadata, run_dir),
        }


def generate_stimuli(
    suites: tuple[str, ...],
    output_dir: Path,
    seed: int,
    config: dict[str, Any] | None = None,
) -> list[Stimulus]:
    """Generate deterministic local stimuli for requested suites."""

    config = config or {}
    validate_suite_names(suites)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base_images = _load_or_create_base_images(output_dir, rng, config)
    stimuli: list[Stimulus] = []

    builders = {
        "image_activation": _image_activation_stimuli,
        "visual_controls": _visual_control_stimuli,
        "visual_perturbations": _visual_perturbation_stimuli,
        "temporalization": _temporalization_stimuli,
        "audio_controls": _audio_control_stimuli,
        "latent_network_ica_explorer": _latent_network_stimuli,
        "virtual_lesion_lab": _virtual_lesion_stimuli,
        "discrete_stimulus_optimizer": _discrete_optimizer_stimuli,
        "counterfactual_editing_workbench": _counterfactual_stimuli,
        "focused_ultrasound_bridge": _focused_ultrasound_bridge_stimuli,
        "bold5000_roi_encoding": _bold5000_stimuli,
    }
    for suite in suites:
        stimuli.extend(builders[suite](base_images, output_dir, rng, config))

    return stimuli


def _bold5000_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del base_images, rng
    return make_bold5000_stimuli(output_dir, config)


def _stimulus(
    stimulus_id: str,
    suite: str,
    modality: str,
    kind: str,
    path: Path,
    duration: float,
    source_image: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Stimulus:
    payload: dict[str, Any] = {"duration_seconds": duration}
    if source_image is not None:
        payload["source_image"] = source_image
    if metadata:
        payload.update(metadata)
    return Stimulus(
        stimulus_id=stimulus_id,
        suite=suite,
        modality=modality,
        kind=kind,
        path=path,
        sha256=sha256_file(path),
        license="generated-unlicense",
        metadata=payload,
    )


def _relative_metadata(metadata: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        key: _relative_metadata_value(value, run_dir) for key, value in metadata.items()
    }


def _relative_metadata_value(value: Any, run_dir: Path) -> Any:
    if isinstance(value, Path):
        return _relative_path(value, run_dir)
    if isinstance(value, dict):
        return {
            str(key): _relative_metadata_value(child, run_dir)
            for key, child in value.items()
        }
    if isinstance(value, list | tuple):
        return [_relative_metadata_value(child, run_dir) for child in value]
    if isinstance(value, str):
        path = Path(value)
        return _relative_path(path, run_dir) if path.is_absolute() else value
    return value


def _relative_path(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return path.name


def _media_type(path: Path, modality: str) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".png":
        return "image/png"
    return f"application/x-braindough-{modality}"


def _image_activation_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del rng, config
    stimuli: list[Stimulus] = []
    for base_id, image_path in base_images.items():
        video_path = output_dir / "image_activation" / f"{base_id}-static.mp4"
        _write_static_video(image_path, video_path)
        stimuli.append(
            _stimulus(
                stimulus_id=f"image_activation:{base_id}:static",
                suite="image_activation",
                modality="video",
                kind="static_image_clip",
                path=video_path,
                duration=VIDEO_SECONDS,
                source_image=image_path,
                metadata={"base_id": base_id, "role": "baseline"},
            )
        )
    return stimuli


def _visual_control_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del base_images, config
    stimuli: list[Stimulus] = []
    for control_id, image in _control_images(rng).items():
        image_path = output_dir / "visual_controls" / f"{control_id}.png"
        _save_image(image, image_path)
        video_path = image_path.with_suffix(".mp4")
        _write_static_video(image_path, video_path)
        stimuli.append(
            _stimulus(
                stimulus_id=f"visual_controls:{control_id}",
                suite="visual_controls",
                modality="video",
                kind=control_id,
                path=video_path,
                duration=VIDEO_SECONDS,
                source_image=image_path,
                metadata={"control_id": control_id, "role": "control"},
            )
        )
    return stimuli


def _visual_perturbation_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del config
    stimuli: list[Stimulus] = []
    for base_id, image_path in base_images.items():
        image = Image.open(image_path).convert("RGB")
        parent_id = f"image_activation:{base_id}:static"
        for variant, perturbed in _perturbations(image, rng).items():
            variant_path = (
                output_dir / "visual_perturbations" / f"{base_id}-{variant}.png"
            )
            _save_image(perturbed, variant_path)
            video_path = variant_path.with_suffix(".mp4")
            _write_static_video(variant_path, video_path)
            stimuli.append(
                _stimulus(
                    stimulus_id=f"visual_perturbations:{base_id}:{variant}",
                    suite="visual_perturbations",
                    modality="video",
                    kind=variant,
                    path=video_path,
                    duration=VIDEO_SECONDS,
                    source_image=variant_path,
                    metadata={
                        "base_id": base_id,
                        "parent_id": parent_id,
                        "transform": variant,
                        "role": "perturbation",
                    },
                )
            )
    return stimuli


def _temporalization_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del config
    stimuli: list[Stimulus] = []
    for base_id, image_path in base_images.items():
        parent_id = f"image_activation:{base_id}:static"
        for variant in ["static", "pan_zoom", "montage"]:
            video_path = output_dir / "temporalization" / f"{base_id}-{variant}.mp4"
            _write_temporal_video(image_path, video_path, variant, rng)
            stimuli.append(
                _stimulus(
                    stimulus_id=f"temporalization:{base_id}:{variant}",
                    suite="temporalization",
                    modality="video",
                    kind=variant,
                    path=video_path,
                    duration=VIDEO_SECONDS,
                    source_image=image_path,
                    metadata={
                        "base_id": base_id,
                        "parent_id": parent_id,
                        "temporal_policy": variant,
                        "role": "temporalization",
                    },
                )
            )
    return stimuli


def _audio_control_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del base_images, config
    stimuli: list[Stimulus] = []
    for variant in ["silence", "tone_440hz", "chirp", "pink_noise"]:
        audio_path = output_dir / "audio_controls" / f"{variant}.wav"
        _write_audio(audio_path, variant, rng)
        stimuli.append(
            _stimulus(
                stimulus_id=f"audio_controls:{variant}",
                suite="audio_controls",
                modality="audio",
                kind=variant,
                path=audio_path,
                duration=AUDIO_SECONDS,
                metadata={"role": "audio_control"},
            )
        )
    return stimuli


def _latent_network_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del config
    stimuli: list[Stimulus] = []
    variants = ["static", "grayscale", "blur", "edge"]
    for base_id, image_path in base_images.items():
        source = Image.open(image_path).convert("RGB")
        variant_images = {"static": source, **_perturbations(source, rng)}
        for variant in variants:
            image = variant_images[variant]
            image_out = (
                output_dir / "latent_network_ica_explorer" / f"{base_id}-{variant}.png"
            )
            _save_image(image, image_out)
            video_path = image_out.with_suffix(".mp4")
            _write_static_video(image_out, video_path)
            stimuli.append(
                _stimulus(
                    stimulus_id=f"latent_network_ica_explorer:{base_id}:{variant}",
                    suite="latent_network_ica_explorer",
                    modality="video",
                    kind=variant,
                    path=video_path,
                    duration=VIDEO_SECONDS,
                    source_image=image_out,
                    metadata={
                        "base_id": base_id,
                        "latent_probe_family": "visual_factor_basis",
                        "role": "component_probe",
                    },
                )
            )
    return stimuli


def _virtual_lesion_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    stimuli: list[Stimulus] = []
    base_count = _config_int(config, "virtual_lesion_base_count", default=2, limit=8)
    lesion_types = _config_list(
        config,
        "virtual_lesion_types",
        default=[
            "mask_left",
            "mask_right",
            "central_occlusion",
            "low_contrast",
            "blur_suppression",
            "blank_suppression",
        ],
    )
    strengths = _config_float_list(config, "lesion_strengths", default=[1.0])
    for base_id, image_path in list(base_images.items())[:base_count]:
        source = Image.open(image_path).convert("RGB")
        baseline_id = f"virtual_lesion_lab:{base_id}:baseline"
        baseline_path = output_dir / "virtual_lesion_lab" / f"{base_id}-baseline.png"
        _save_image(source, baseline_path)
        baseline_video = baseline_path.with_suffix(".mp4")
        _write_static_video(baseline_path, baseline_video)
        stimuli.append(
            _stimulus(
                stimulus_id=baseline_id,
                suite="virtual_lesion_lab",
                modality="video",
                kind="baseline",
                path=baseline_video,
                duration=VIDEO_SECONDS,
                source_image=baseline_path,
                metadata={
                    "base_id": base_id,
                    "role": "baseline",
                    "intervention_family": "stimulus_factor_lesion",
                },
            )
        )
        for lesion_id, lesioned, lesion_metadata in _lesion_images(
            source,
            rng,
            lesion_types=lesion_types,
            strengths=strengths,
            masks_dir=output_dir / "virtual_lesion_lab" / "masks",
            base_id=base_id,
        ):
            lesion_path = (
                output_dir / "virtual_lesion_lab" / f"{base_id}-{lesion_id}.png"
            )
            _save_image(lesioned, lesion_path)
            video_path = lesion_path.with_suffix(".mp4")
            _write_static_video(lesion_path, video_path)
            stimuli.append(
                _stimulus(
                    stimulus_id=f"virtual_lesion_lab:{base_id}:{lesion_id}",
                    suite="virtual_lesion_lab",
                    modality="video",
                    kind=lesion_id,
                    path=video_path,
                    duration=VIDEO_SECONDS,
                    source_image=lesion_path,
                    metadata={
                        "base_id": base_id,
                        "parent_id": baseline_id,
                        "lesion_type": lesion_id,
                        "role": "stimulus_lesion",
                        "intervention_family": "stimulus_factor_lesion",
                        "source_image_sha256": sha256_file(baseline_path),
                        **lesion_metadata,
                    },
                )
            )
    return stimuli


def _discrete_optimizer_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del base_images
    count = int(config.get("optimizer_candidate_count", 12))
    count = max(1, min(count, 64))
    stimuli: list[Stimulus] = []
    shapes = ["circle", "square", "triangle", "stripe"]
    palettes = [
        ((236, 80, 74), (34, 38, 45)),
        ((62, 145, 220), (244, 241, 232)),
        ((236, 190, 76), (28, 82, 79)),
        ((145, 92, 182), (242, 242, 246)),
    ]
    for idx in range(count):
        shape = shapes[idx % len(shapes)]
        foreground, background = palettes[(idx // len(shapes)) % len(palettes)]
        angle = int(rng.integers(0, 360))
        params = {
            "shape": shape,
            "foreground_rgb": list(foreground),
            "background_rgb": list(background),
            "angle_degrees": angle,
        }
        image = _candidate_image(shape, foreground, background, angle)
        image_path = (
            output_dir / "discrete_stimulus_optimizer" / f"candidate-{idx:02d}.png"
        )
        _save_image(image, image_path)
        video_path = image_path.with_suffix(".mp4")
        _write_static_video(image_path, video_path)
        stimuli.append(
            _stimulus(
                stimulus_id=f"discrete_stimulus_optimizer:candidate_{idx:02d}",
                suite="discrete_stimulus_optimizer",
                modality="video",
                kind="candidate",
                path=video_path,
                duration=VIDEO_SECONDS,
                source_image=image_path,
                metadata={
                    "candidate_index": idx,
                    "optimizer_candidate": True,
                    "objective": "mean_abs_activation_minus_similarity_penalty",
                    "params": params,
                    "param_hash": _stable_hash(params),
                    "generation_policy": "shape_palette_angle_grid_v1",
                },
            )
        )
    return stimuli


def _counterfactual_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del rng
    stimuli: list[Stimulus] = []
    base_count = _config_int(config, "counterfactual_base_count", default=2, limit=8)
    edit_types = _config_list(
        config,
        "counterfactual_edit_types",
        default=[
            "color_swap",
            "local_blur",
            "background_lighten",
            "object_mask",
            "tile_scramble",
        ],
    )
    for base_id, image_path in list(base_images.items())[:base_count]:
        source = Image.open(image_path).convert("RGB")
        baseline_id = f"counterfactual_editing_workbench:{base_id}:baseline"
        baseline_path = (
            output_dir / "counterfactual_editing_workbench" / f"{base_id}-baseline.png"
        )
        _save_image(source, baseline_path)
        baseline_video = baseline_path.with_suffix(".mp4")
        _write_static_video(baseline_path, baseline_video)
        stimuli.append(
            _stimulus(
                stimulus_id=baseline_id,
                suite="counterfactual_editing_workbench",
                modality="video",
                kind="baseline",
                path=baseline_video,
                duration=VIDEO_SECONDS,
                source_image=baseline_path,
                metadata={
                    "base_id": base_id,
                    "pair_id": base_id,
                    "role": "counterfactual_source",
                },
            )
        )
        for edit_id, edited, edit_metadata in _counterfactual_edits(
            source, edit_types=edit_types
        ):
            edit_path = (
                output_dir
                / "counterfactual_editing_workbench"
                / f"{base_id}-{edit_id}.png"
            )
            _save_image(edited, edit_path)
            video_path = edit_path.with_suffix(".mp4")
            _write_static_video(edit_path, video_path)
            stimuli.append(
                _stimulus(
                    stimulus_id=f"counterfactual_editing_workbench:{base_id}:{edit_id}",
                    suite="counterfactual_editing_workbench",
                    modality="video",
                    kind=edit_id,
                    path=video_path,
                    duration=VIDEO_SECONDS,
                    source_image=edit_path,
                    metadata={
                        "base_id": base_id,
                        "parent_id": baseline_id,
                        "pair_id": base_id,
                        "edit_type": edit_id,
                        "role": "counterfactual_edit",
                        "edit_version": "rule_based_v1",
                        "source_image_sha256": sha256_file(baseline_path),
                        **edit_metadata,
                    },
                )
            )
    return stimuli


def _focused_ultrasound_bridge_stimuli(
    base_images: dict[str, Path],
    output_dir: Path,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[Stimulus]:
    del rng
    stimuli: list[Stimulus] = []
    base_count = _config_int(
        config, "focused_ultrasound_base_count", default=1, limit=4
    )
    target_ids = _known_config_list(
        config,
        "focused_ultrasound_targets",
        known=_FUS_TARGETS,
        default=["S1", "hMT_plus"],
    )
    protocol_ids = _known_config_list(
        config,
        "focused_ultrasound_protocols",
        known=_FUS_PROTOCOLS,
        default=[
            "active_low_duty",
            "active_mid_duty",
            "sham_transmit_blocked",
            "spatial_control",
        ],
    )
    suite_dir = output_dir / "focused_ultrasound_bridge"

    for base_id, image_path in list(base_images.items())[:base_count]:
        source = Image.open(image_path).convert("RGB")
        for target_id in target_ids:
            target = _FUS_TARGETS[target_id]
            pair_id = f"{base_id}:{target_id}"
            baseline_id = f"focused_ultrasound_bridge:{base_id}:{target_id}:baseline"
            baseline_path = suite_dir / f"{base_id}-{target_id}-baseline.png"
            baseline_protocol = _baseline_fus_protocol()
            _save_image(
                _focused_ultrasound_card(source, target, baseline_protocol),
                baseline_path,
            )
            baseline_video = baseline_path.with_suffix(".mp4")
            _write_static_video(baseline_path, baseline_video)
            stimuli.append(
                _stimulus(
                    stimulus_id=baseline_id,
                    suite="focused_ultrasound_bridge",
                    modality="video",
                    kind="baseline",
                    path=baseline_video,
                    duration=VIDEO_SECONDS,
                    source_image=baseline_path,
                    metadata=_fus_metadata(
                        base_id=base_id,
                        pair_id=pair_id,
                        target_id=target_id,
                        target=target,
                        protocol_id="baseline",
                        protocol=baseline_protocol,
                        parent_id="",
                        source_image_sha256="",
                    ),
                )
            )
            baseline_sha = sha256_file(baseline_path)
            for protocol_id in protocol_ids:
                protocol = _FUS_PROTOCOLS[protocol_id]
                card_path = suite_dir / f"{base_id}-{target_id}-{protocol_id}.png"
                _save_image(
                    _focused_ultrasound_card(source, target, protocol),
                    card_path,
                )
                video_path = card_path.with_suffix(".mp4")
                _write_static_video(card_path, video_path)
                stimuli.append(
                    _stimulus(
                        stimulus_id=(
                            "focused_ultrasound_bridge:"
                            f"{base_id}:{target_id}:{protocol_id}"
                        ),
                        suite="focused_ultrasound_bridge",
                        modality="video",
                        kind=protocol_id,
                        path=video_path,
                        duration=VIDEO_SECONDS,
                        source_image=card_path,
                        metadata=_fus_metadata(
                            base_id=base_id,
                            pair_id=pair_id,
                            target_id=target_id,
                            target=target,
                            protocol_id=protocol_id,
                            protocol=protocol,
                            parent_id=baseline_id,
                            source_image_sha256=baseline_sha,
                        ),
                    )
                )
    return stimuli


def _baseline_fus_protocol() -> dict[str, Any]:
    return {
        "condition": "baseline",
        "software_dose_index": 0.0,
        "virtual_duty_cycle_bins": 0.0,
        "virtual_burst_count": 0,
        "virtual_envelope": [0, 0, 0, 0, 0, 0, 0, 0],
        "sham_mode": "none",
        "nominal_center_frequency_mhz": "",
        "nominal_prf_hz": "",
        "nominal_duty_cycle": "",
        "nominal_sonication_seconds": "",
    }


def _fus_metadata(
    *,
    base_id: str,
    pair_id: str,
    target_id: str,
    target: dict[str, Any],
    protocol_id: str,
    protocol: dict[str, Any],
    parent_id: str,
    source_image_sha256: str,
) -> dict[str, Any]:
    condition = str(protocol["condition"])
    role = "fus_protocol_baseline" if condition == "baseline" else "fus_protocol_proxy"
    metadata: dict[str, Any] = {
        "base_id": base_id,
        "pair_id": pair_id,
        "parent_id": parent_id,
        "role": role,
        "intervention_family": "focused_ultrasound_protocol_proxy",
        "target_id": target_id,
        "target_label": target["label"],
        "target_network": target["network"],
        "task_family": target["task_family"],
        "target_coordinate_space": "schematic_2d_head",
        "target_coordinate": list(target["schematic_xy"]),
        "protocol_id": protocol_id,
        "condition": condition,
        "software_dose_index": protocol["software_dose_index"],
        "virtual_duty_cycle_bins": protocol["virtual_duty_cycle_bins"],
        "virtual_burst_count": protocol["virtual_burst_count"],
        "virtual_envelope": protocol["virtual_envelope"],
        "sham_mode": protocol["sham_mode"],
        "acoustic_modeling_status": "not_modeled",
        "safety_claim": "software_proxy_no_sonication_or_clinical_claim",
        "source_image_sha256": source_image_sha256,
        "itrusst_reporting_status": "synthetic_proxy_fields_only",
        "nominal_center_frequency_mhz": protocol["nominal_center_frequency_mhz"],
        "nominal_prf_hz": protocol["nominal_prf_hz"],
        "nominal_duty_cycle": protocol["nominal_duty_cycle"],
        "nominal_sonication_seconds": protocol["nominal_sonication_seconds"],
        "estimated_in_situ_pressure_mpa": "",
        "estimated_in_situ_ispta_mw_cm2": "",
        "mechanical_index": "",
        "thermal_index": "",
        "transducer_model": "not_applicable",
        "drive_system": "not_applicable",
        "primary_source_anchors": [
            "https://arxiv.org/abs/2402.10027",
            "https://www.nature.com/articles/s43586-024-00368-6",
            "https://www.nature.com/articles/nn.3620",
            "https://www.nature.com/articles/s41467-026-69853-8",
        ],
    }
    if condition == "spatial_control":
        metadata["target_coordinate"] = list(target["control_xy"])
        metadata["control_target_label"] = f"{target['label']}_off_target"
    return metadata


def _focused_ultrasound_card(
    source: Image.Image, target: dict[str, Any], protocol: dict[str, Any]
) -> Image.Image:
    source = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    canvas = Image.blend(
        source,
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (248, 249, 250)),
        alpha=0.38,
    )
    draw = ImageDraw.Draw(canvas)
    condition = str(protocol["condition"])
    dose = _metadata_float(protocol["software_dose_index"], default=0.0)
    color = {
        "active": (220, 70, 70),
        "sham": (80, 88, 100),
        "spatial_control": (220, 150, 55),
        "baseline": (55, 110, 170),
    }.get(condition, (80, 88, 100))
    coordinate = (
        target["control_xy"]
        if condition == "spatial_control"
        else target["schematic_xy"]
    )
    x = int(float(coordinate[0]) * IMAGE_SIZE)
    y = int(float(coordinate[1]) * IMAGE_SIZE)

    draw.rounded_rectangle((14, 14, 242, 242), radius=10, outline=color, width=3)
    draw.ellipse((46, 42, 210, 196), outline=(42, 48, 56), width=3)
    draw.line((128, 42, 128, 196), fill=(42, 48, 56), width=1)
    draw.line((72, 70, 184, 168), fill=(90, 96, 105), width=1)
    draw.ellipse((x - 14, y - 14, x + 14, y + 14), outline=color, width=4)
    draw.line((x - 22, y, x + 22, y), fill=color, width=2)
    draw.line((x, y - 22, x, y + 22), fill=color, width=2)

    pulse_left = 42
    pulse_top = 210
    pulse_count = 1 if condition == "baseline" else max(2, int(2 + 6 * dose))
    for idx in range(pulse_count):
        left = pulse_left + idx * 24
        height = 8 + int(18 * max(dose, 0.1))
        draw.rectangle(
            (left, pulse_top - height, left + 10, pulse_top),
            fill=color,
        )
    draw.line((36, pulse_top, 220, pulse_top), fill=(42, 48, 56), width=1)
    draw.text((24, 24), "FUS proxy", fill=(20, 24, 28))
    draw.text((24, 224), f"{target['label']} {condition}", fill=(20, 24, 28))
    return canvas


def _load_or_create_base_images(
    output_dir: Path, rng: np.random.Generator, config: dict[str, Any]
) -> dict[str, Path]:
    user_images = config.get("images", [])
    base_dir = output_dir / "base_images"
    base_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Path] = {}
    missing_images: list[str] = []

    for idx, raw_path in enumerate(user_images):
        source = Path(str(raw_path)).expanduser()
        if not source.is_file():
            missing_images.append(str(raw_path))
            continue
        image = Image.open(source).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        destination = base_dir / f"user_{idx:02d}_{source.stem}.png"
        _save_image(image, destination)
        images[f"user_{idx:02d}_{source.stem}"] = destination

    if missing_images:
        raise ValueError(f"Configured stimulus image(s) not found: {missing_images}")

    if images:
        return images

    for name, image in _default_base_images(rng).items():
        path = base_dir / f"{name}.png"
        _save_image(image, path)
        images[name] = path
    return images


def _default_base_images(rng: np.random.Generator) -> dict[str, Image.Image]:
    size = IMAGE_SIZE
    gradient = np.zeros((size, size, 3), dtype=np.uint8)
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(0, 255, size, dtype=np.uint8)
    gradient[:, :, 0] = x[None, :]
    gradient[:, :, 1] = y[:, None]
    gradient[:, :, 2] = 128

    shapes = Image.new("RGB", (size, size), (34, 38, 45))
    draw = ImageDraw.Draw(shapes)
    draw.rectangle((24, 30, 130, 150), fill=(220, 78, 65))
    draw.ellipse((100, 74, 226, 206), fill=(62, 145, 220))
    draw.polygon([(32, 220), (130, 112), (232, 232)], fill=(236, 190, 76))

    texture = rng.normal(128, 42, (size, size, 3)).clip(0, 255).astype(np.uint8)
    texture_image = Image.fromarray(texture)
    draw = ImageDraw.Draw(texture_image)
    draw.line((0, 220, 256, 30), fill=(20, 20, 20), width=10)
    draw.line((0, 238, 256, 48), fill=(245, 245, 245), width=3)

    return {
        "color_gradient": Image.fromarray(gradient),
        "geometric_scene": shapes,
        "textured_diagonal": texture_image,
    }


def _control_images(rng: np.random.Generator) -> dict[str, Image.Image]:
    size = IMAGE_SIZE
    checker = np.indices((size, size)).sum(axis=0) // 16 % 2
    checker_rgb = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    noise = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    blank = Image.new("RGB", (size, size), (0, 0, 0))
    gray = Image.new("RGB", (size, size), (127, 127, 127))
    simple_shape = Image.new("RGB", (size, size), (245, 245, 240))
    draw = ImageDraw.Draw(simple_shape)
    draw.ellipse((72, 72, 184, 184), fill=(40, 80, 180))
    return {
        "blank": blank,
        "gray": gray,
        "checker": Image.fromarray(checker_rgb),
        "noise": Image.fromarray(noise),
        "simple_shape": simple_shape,
    }


def _perturbations(
    image: Image.Image, rng: np.random.Generator
) -> dict[str, Image.Image]:
    arr = np.asarray(image)
    tiles = [
        arr[y : y + 32, x : x + 32].copy()
        for y in range(0, IMAGE_SIZE, 32)
        for x in range(0, IMAGE_SIZE, 32)
    ]
    rng.shuffle(tiles)
    scrambled = np.zeros_like(arr)
    idx = 0
    for y in range(0, IMAGE_SIZE, 32):
        for x in range(0, IMAGE_SIZE, 32):
            scrambled[y : y + 32, x : x + 32] = tiles[idx]
            idx += 1

    edge = image.convert("L").filter(ImageFilter.FIND_EDGES)
    crop = image.crop((32, 32, 224, 224)).resize((IMAGE_SIZE, IMAGE_SIZE))
    return {
        "grayscale": ImageOps.grayscale(image).convert("RGB"),
        "blur": image.filter(ImageFilter.GaussianBlur(radius=5)),
        "contrast": ImageOps.autocontrast(image, cutoff=8),
        "crop": crop,
        "scramble": Image.fromarray(scrambled),
        "edge": ImageOps.colorize(edge, black=(0, 0, 0), white=(240, 240, 240)),
    }


def _lesion_images(
    image: Image.Image,
    rng: np.random.Generator,
    *,
    lesion_types: list[str],
    strengths: list[float],
    masks_dir: Path,
    base_id: str,
) -> list[tuple[str, Image.Image, dict[str, Any]]]:
    fill = (127, 127, 127)
    variants: list[tuple[str, Image.Image, dict[str, Any]]] = []
    masks_dir.mkdir(parents=True, exist_ok=True)

    for lesion_type in lesion_types:
        if lesion_type == "sham_reencode":
            variants.append(
                (
                    lesion_type,
                    image.copy(),
                    {
                        "lesion_base_type": lesion_type,
                        "strength": 0.0,
                        "masked_fraction": 0.0,
                        "fill_rgb": list(fill),
                        "bbox": "",
                        "mask_sha256": "",
                        "mask_path": "",
                    },
                )
            )
            continue

        for strength in strengths:
            strength = min(max(float(strength), 0.0), 1.0)
            lesion_id = (
                lesion_type
                if len(strengths) == 1
                else f"{lesion_type}_s{round(strength * 100):03d}"
            )
            lesioned, mask, bbox = _apply_lesion(
                image=image,
                lesion_type=lesion_type,
                strength=strength,
                fill=fill,
                rng=rng,
            )
            mask_path = masks_dir / f"{base_id}-{lesion_id}-mask.png"
            _save_image(mask, mask_path)
            variants.append(
                (
                    lesion_id,
                    lesioned,
                    {
                        "lesion_base_type": lesion_type,
                        "strength": strength,
                        "masked_fraction": _mask_fraction(mask),
                        "fill_rgb": list(fill),
                        "bbox": list(bbox) if bbox else "",
                        "mask_sha256": sha256_file(mask_path),
                        "mask_path": mask_path,
                    },
                )
            )
    return variants


def _apply_lesion(
    image: Image.Image,
    lesion_type: str,
    strength: float,
    fill: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int] | None]:
    size = IMAGE_SIZE
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    bbox: tuple[int, int, int, int] | None = None

    if lesion_type == "mask_left":
        width = max(1, int((size // 2) * strength))
        bbox = (0, 0, width, size)
        draw.rectangle(bbox, fill=255)
    elif lesion_type == "mask_right":
        width = max(1, int((size // 2) * strength))
        bbox = (size - width, 0, size, size)
        draw.rectangle(bbox, fill=255)
    elif lesion_type == "central_occlusion":
        radius = max(1, int(52 * strength))
        bbox = (
            size // 2 - radius,
            size // 2 - radius,
            size // 2 + radius,
            size // 2 + radius,
        )
        draw.ellipse(bbox, fill=255)
    elif lesion_type == "random_patch_same_area":
        side = max(16, int(96 * max(strength, 0.2)))
        x = int(rng.integers(0, max(size - side, 1)))
        y = int(rng.integers(0, max(size - side, 1)))
        bbox = (x, y, x + side, y + side)
        draw.rectangle(bbox, fill=255)
    elif lesion_type in {"low_contrast", "blur_suppression", "blank_suppression"}:
        bbox = (0, 0, size, size)
        draw.rectangle(bbox, fill=255)

    if lesion_type == "low_contrast":
        contrast_factor = max(0.05, 1.0 - 0.75 * strength)
        return ImageEnhance.Contrast(image).enhance(contrast_factor), mask, bbox
    if lesion_type == "blur_suppression":
        return (
            image.filter(ImageFilter.GaussianBlur(radius=2 + 8 * strength)),
            mask,
            bbox,
        )
    if lesion_type == "blank_suppression":
        return (
            Image.blend(image, Image.new("RGB", (size, size), fill), strength),
            mask,
            bbox,
        )
    if lesion_type == "random_patch_same_area" and bbox:
        source = image.copy()
        sx = int(rng.integers(0, max(size - (bbox[2] - bbox[0]), 1)))
        sy = int(rng.integers(0, max(size - (bbox[3] - bbox[1]), 1)))
        patch = source.crop(
            (sx, sy, sx + (bbox[2] - bbox[0]), sy + (bbox[3] - bbox[1]))
        )
        target = image.copy()
        target.paste(patch, (bbox[0], bbox[1]))
        return target, mask, bbox
    if bbox:
        target = image.copy()
        overlay = Image.new("RGB", (size, size), fill)
        target.paste(Image.blend(target, overlay, strength), mask=mask)
        return target, mask, bbox
    raise ValueError(f"Unknown virtual lesion type: {lesion_type}")


def _mask_fraction(mask: Image.Image) -> float:
    arr = np.asarray(mask, dtype=np.float32)
    return float(np.count_nonzero(arr) / arr.size)


def _candidate_image(
    shape: str,
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
    angle: int,
) -> Image.Image:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), background)
    draw = ImageDraw.Draw(image)
    if shape == "circle":
        draw.ellipse((64, 64, 192, 192), fill=foreground)
    elif shape == "square":
        draw.rectangle((64, 64, 192, 192), fill=foreground)
    elif shape == "triangle":
        draw.polygon([(128, 42), (52, 210), (212, 210)], fill=foreground)
    else:
        for x in range(-IMAGE_SIZE, IMAGE_SIZE * 2, 24):
            draw.line((x, 0, x + IMAGE_SIZE, IMAGE_SIZE), fill=foreground, width=10)
    return image.rotate(angle, resample=Image.Resampling.BICUBIC)


def _counterfactual_edits(
    image: Image.Image, *, edit_types: list[str]
) -> list[tuple[str, Image.Image, dict[str, Any]]]:
    arr = np.asarray(image.convert("RGB"))
    swapped = Image.fromarray(arr[:, :, [2, 1, 0]])

    local_blur = image.copy()
    patch = local_blur.crop((64, 64, 192, 192)).filter(
        ImageFilter.GaussianBlur(radius=8)
    )
    local_blur.paste(patch, (64, 64))

    background_lighten = Image.blend(
        image,
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (245, 245, 245)),
        alpha=0.35,
    )

    object_mask = image.copy()
    ImageDraw.Draw(object_mask).rectangle((84, 84, 172, 172), fill=(127, 127, 127))

    tiles = [
        arr[y : y + 64, x : x + 64].copy()
        for y in range(0, IMAGE_SIZE, 64)
        for x in range(0, IMAGE_SIZE, 64)
    ]
    scrambled = np.zeros_like(arr)
    for idx, tile in enumerate(reversed(tiles)):
        y = (idx // 4) * 64
        x = (idx % 4) * 64
        scrambled[y : y + 64, x : x + 64] = tile

    edits = {
        "color_swap": swapped,
        "local_blur": local_blur,
        "background_lighten": background_lighten,
        "object_mask": object_mask,
        "tile_scramble": Image.fromarray(scrambled),
    }
    variants: list[tuple[str, Image.Image, dict[str, Any]]] = []
    for edit_type in edit_types:
        if edit_type not in edits:
            raise ValueError(f"Unknown counterfactual edit type: {edit_type}")
        edited = edits[edit_type]
        variants.append(
            (
                edit_type,
                edited,
                {
                    "edit_base_type": edit_type,
                    "semantic_class": _counterfactual_semantic_class(edit_type),
                    **_image_delta_metrics(image, edited),
                },
            )
        )
    return variants


def _counterfactual_semantic_class(edit_type: str) -> str:
    meaning_changing = {"object_mask", "tile_scramble"}
    return "meaning_changing" if edit_type in meaning_changing else "low_level"


def _image_delta_metrics(source: Image.Image, edited: Image.Image) -> dict[str, Any]:
    left = np.asarray(source.convert("RGB"), dtype=np.float32)
    right = np.asarray(edited.convert("RGB"), dtype=np.float32)
    diff = np.abs(right - left)
    changed = np.any(diff > 0, axis=2)
    if np.any(changed):
        ys, xs = np.nonzero(changed)
        bbox: list[int] | str = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
    else:
        bbox = ""
    return {
        "changed_pixel_fraction": float(np.count_nonzero(changed) / changed.size),
        "mean_rgb_l1": float(np.mean(diff)),
        "mean_rgb_l2": float(np.sqrt(np.mean(diff**2))),
        "edge_change_score": _edge_change_score(source, edited),
        "edit_bbox": bbox,
    }


def _edge_change_score(source: Image.Image, edited: Image.Image) -> float:
    left = np.asarray(
        source.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32
    )
    right = np.asarray(
        edited.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32
    )
    return float(np.mean(np.abs(right - left)))


def _config_int(config: dict[str, Any], key: str, *, default: int, limit: int) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, limit))


def _config_list(config: dict[str, Any], key: str, *, default: list[str]) -> list[str]:
    raw = config.get(key, default)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        return list(default)
    return list(raw) or list(default)


def _known_config_list(
    config: dict[str, Any],
    key: str,
    *,
    known: dict[str, Any],
    default: list[str],
) -> list[str]:
    requested = _config_list(config, key, default=default)
    filtered = [item for item in requested if item in known]
    return filtered or list(default)


def _config_float_list(
    config: dict[str, Any], key: str, *, default: list[float]
) -> list[float]:
    raw = config.get(key, default)
    if not isinstance(raw, list):
        return list(default)
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            continue
    return values or list(default)


def _metadata_float(value: object, *, default: float) -> float:
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _stable_hash(payload: dict[str, Any]) -> str:
    import json

    return sha256_text(json.dumps(payload, sort_keys=True))


def _save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_static_video(image_path: Path, video_path: Path) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    frames = [image for _ in range(VIDEO_FPS * VIDEO_SECONDS)]
    _write_video_frames(frames, video_path)


def _write_temporal_video(
    image_path: Path, video_path: Path, variant: str, rng: np.random.Generator
) -> None:
    base = Image.open(image_path).convert("RGB")
    frames: list[np.ndarray] = []
    count = VIDEO_FPS * VIDEO_SECONDS
    if variant == "static":
        frames = [np.asarray(base) for _ in range(count)]
    elif variant == "pan_zoom":
        for idx in range(count):
            margin = 24 - int(18 * idx / max(count - 1, 1))
            crop = base.crop((margin, margin, IMAGE_SIZE - margin, IMAGE_SIZE - margin))
            frames.append(np.asarray(crop.resize((IMAGE_SIZE, IMAGE_SIZE))))
    elif variant == "montage":
        variants = list(_perturbations(base, rng).values())[:count]
        while len(variants) < count:
            variants.extend(variants)
        frames = [
            np.asarray(frame.resize((IMAGE_SIZE, IMAGE_SIZE)))
            for frame in variants[:count]
        ]
    else:
        raise ValueError(f"Unknown temporal variant: {variant}")
    _write_video_frames(frames, video_path)


def _write_video_frames(frames: list[np.ndarray], video_path: Path) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frames_payload: Any = frames
    imageio.mimsave(
        video_path,
        frames_payload,
        fps=VIDEO_FPS,
        codec="libx264",
        macro_block_size=16,
        quality=6,
    )


def _write_audio(path: Path, variant: str, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = AUDIO_RATE * AUDIO_SECONDS
    t = np.arange(samples, dtype=np.float64) / AUDIO_RATE
    if variant == "silence":
        signal = np.zeros(samples, dtype=np.float64)
    elif variant == "tone_440hz":
        signal = 0.4 * np.sin(2 * math.pi * 440 * t)
    elif variant == "chirp":
        freq = 220 + 660 * t / AUDIO_SECONDS
        signal = 0.35 * np.sin(2 * math.pi * freq * t)
    elif variant == "pink_noise":
        white = rng.normal(0, 1, samples)
        spectrum = np.fft.rfft(white)
        scale = np.sqrt(np.maximum(np.arange(len(spectrum)), 1))
        pink = np.fft.irfft(spectrum / scale, n=samples)
        signal = 0.35 * pink / max(float(np.max(np.abs(pink))), 1e-9)
    else:
        raise ValueError(f"Unknown audio variant: {variant}")

    pcm = np.int16(np.clip(signal, -1, 1) * 32767)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(AUDIO_RATE)
        handle.writeframes(pcm.tobytes())
