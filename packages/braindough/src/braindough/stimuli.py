"""Deterministic stimulus generation for first-suite experiments."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from braindough.storage import sha256_file
from braindough.suites import validate_suite_names

IMAGE_SIZE = 256
VIDEO_FPS = 4
VIDEO_SECONDS = 2
AUDIO_RATE = 16_000
AUDIO_SECONDS = 2


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
    }
    for suite in suites:
        stimuli.extend(builders[suite](base_images, output_dir, rng, config))

    return stimuli


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
    del rng, config
    stimuli: list[Stimulus] = []
    for base_id, image_path in list(base_images.items())[:2]:
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
        for lesion_id, lesioned in _lesion_images(source).items():
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
                    "params": {
                        "shape": shape,
                        "foreground_rgb": list(foreground),
                        "background_rgb": list(background),
                        "angle_degrees": angle,
                    },
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
    del rng, config
    stimuli: list[Stimulus] = []
    for base_id, image_path in list(base_images.items())[:2]:
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
        for edit_id, edited in _counterfactual_edits(source).items():
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
                    },
                )
            )
    return stimuli


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


def _lesion_images(image: Image.Image) -> dict[str, Image.Image]:
    size = IMAGE_SIZE
    images: dict[str, Image.Image] = {}

    left = image.copy()
    ImageDraw.Draw(left).rectangle((0, 0, size // 2, size), fill=(127, 127, 127))
    images["mask_left"] = left

    right = image.copy()
    ImageDraw.Draw(right).rectangle((size // 2, 0, size, size), fill=(127, 127, 127))
    images["mask_right"] = right

    center = image.copy()
    ImageDraw.Draw(center).ellipse((76, 76, 180, 180), fill=(127, 127, 127))
    images["central_occlusion"] = center

    images["low_contrast"] = ImageOps.autocontrast(image, cutoff=35)
    images["blur_suppression"] = image.filter(ImageFilter.GaussianBlur(radius=9))
    images["blank_suppression"] = Image.new("RGB", (size, size), (127, 127, 127))
    return images


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


def _counterfactual_edits(image: Image.Image) -> dict[str, Image.Image]:
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

    return {
        "color_swap": swapped,
        "local_blur": local_blur,
        "background_lighten": background_lighten,
        "object_mask": object_mask,
        "tile_scramble": Image.fromarray(scrambled),
    }


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
