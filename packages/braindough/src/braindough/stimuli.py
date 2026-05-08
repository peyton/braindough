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
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base_images = _load_or_create_base_images(output_dir, rng, config)
    stimuli: list[Stimulus] = []

    if "image_activation" in suites:
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
                )
            )

    if "visual_controls" in suites:
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
                )
            )

    if "visual_perturbations" in suites:
        for base_id, image_path in base_images.items():
            image = Image.open(image_path).convert("RGB")
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
                    )
                )

    if "temporalization" in suites:
        for base_id, image_path in base_images.items():
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
                    )
                )

    if "audio_controls" in suites:
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
                )
            )

    return stimuli


def _stimulus(
    stimulus_id: str,
    suite: str,
    modality: str,
    kind: str,
    path: Path,
    duration: float,
    source_image: Path | None = None,
) -> Stimulus:
    metadata: dict[str, Any] = {"duration_seconds": duration}
    if source_image is not None:
        metadata["source_image"] = source_image
    return Stimulus(
        stimulus_id=stimulus_id,
        suite=suite,
        modality=modality,
        kind=kind,
        path=path,
        sha256=sha256_file(path),
        license="generated-unlicense",
        metadata=metadata,
    )


def _relative_metadata(metadata: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    relative: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, Path):
            relative[key] = _relative_path(value, run_dir)
        elif isinstance(value, str):
            path = Path(value)
            relative[key] = (
                _relative_path(path, run_dir) if path.is_absolute() else value
            )
        else:
            relative[key] = value
    return relative


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


def _load_or_create_base_images(
    output_dir: Path, rng: np.random.Generator, config: dict[str, Any]
) -> dict[str, Path]:
    user_images = config.get("images", [])
    base_dir = output_dir / "base_images"
    base_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Path] = {}

    for idx, raw_path in enumerate(user_images):
        source = Path(str(raw_path)).expanduser()
        if source.is_file():
            image = Image.open(source).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            destination = base_dir / f"user_{idx:02d}_{source.stem}.png"
            _save_image(image, destination)
            images[f"user_{idx:02d}_{source.stem}"] = destination

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
