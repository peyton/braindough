"""BOLD5000 dataset staging and lightweight metadata loading."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from braindough.storage import BraindoughPaths, sha256_file

BOLD5000_RELEASE = "release-1.0"
BOLD5000_VERSION = "figshare-v5"
BOLD5000_RELEASE_LABEL = "BOLD5000 Release 1.0"
RELEASE_2_URL = "https://figshare.com/articles/dataset/BOLD5000_Release_2_0/14456124"
RELEASE_2_REPO_URL = "https://github.com/BOLD5000-dataset/BOLD5000"
TERMS_URL = "https://bold5000-dataset.github.io/website/terms.html"
STIMULI_FILE_ID = "14042504"
ROI_FILE_ID = "12965447"
STIMULI_ZIP = "BOLD5000_Stimuli.zip"
ROI_ZIP = "BOLD5000_ROIs.zip"
FIGSHARE_ARTICLE = "https://figshare.com/articles/dataset/BOLD5000/6459449"
DATASET_WEBSITE = "https://bold5000-dataset.github.io/website/"
DOWNLOADS = {
    STIMULI_ZIP: {
        "file_id": STIMULI_FILE_ID,
        "url": f"https://ndownloader.figshare.com/files/{STIMULI_FILE_ID}",
        "md5": "a86dd0a6e86c7157dd9371c395dedd7e",
        "size": 4_912_929,
    },
    ROI_ZIP: {
        "file_id": ROI_FILE_ID,
        "url": f"https://ndownloader.figshare.com/files/{ROI_FILE_ID}",
        "md5": "f7f4a987e52c6f8d5f53da0b45ee8eba",
        "size": 4_212_682_377,
    },
}
DEFAULT_SUBJECTS = ("CSI1",)
DEFAULT_ROIS = ("LHEarlyVis", "RHEarlyVis", "LHLOC", "RHLOC", "LHPPA", "RHPPA")
DEFAULT_TR = "TR1"
SOURCE_LICENSE = (
    "BOLD5000 terms: fMRI and non-image dataset materials are CC0; stimulus "
    "images and original annotations are excluded and retain upstream terms."
)
SUITE = "bold5000_roi_encoding"


@dataclass(frozen=True)
class TrialRecord:
    """One BOLD5000 stimulus presentation row."""

    subject: str
    trial_index: int
    filename: str
    normalized_filename: str
    source_family: str
    label: str
    tokens: tuple[str, ...]
    repeated: bool


@dataclass(frozen=True)
class DatasetDoctor:
    """Current local BOLD5000 dataset status."""

    root: Path
    downloads: dict[str, dict[str, object]]
    extracted: dict[str, bool]

    @property
    def ready(self) -> bool:
        return all(item.get("ok") is True for item in self.downloads.values()) and all(
            self.extracted.values()
        )

    def blocker(self) -> str | None:
        if self.ready:
            return None
        missing = [
            name for name, item in self.downloads.items() if item.get("ok") is not True
        ]
        not_extracted = [name for name, ok in self.extracted.items() if not ok]
        parts: list[str] = []
        if missing:
            parts.append("missing or invalid archives: " + ", ".join(missing))
        if not_extracted:
            parts.append("missing extracted directories: " + ", ".join(not_extracted))
        return "; ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "ready": self.ready,
            "downloads": self.downloads,
            "extracted": self.extracted,
            "blocker": self.blocker(),
        }


class BOLD5000Dataset:
    """Resolve, stage, and read local BOLD5000 files under BRAINDOUGH_HOME."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else _default_root()

    @property
    def downloads_dir(self) -> Path:
        return self.root / "downloads"

    @property
    def extracted_dir(self) -> Path:
        return self.root / "extracted"

    @property
    def stimuli_root(self) -> Path:
        return self.extracted_dir / "BOLD5000_Stimuli"

    @property
    def rois_root(self) -> Path:
        return self.extracted_dir / "ROIs"

    @property
    def provenance_path(self) -> Path:
        return self.root / "provenance.json"

    def init(self) -> None:
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> DatasetDoctor:
        """Download and extract the direct Figshare BOLD5000 archives."""

        self.init()
        for filename, info in DOWNLOADS.items():
            destination = self.downloads_dir / filename
            expected_md5 = str(info["md5"])
            if destination.is_file() and _md5_file(destination) == expected_md5:
                continue
            if destination.is_file():
                destination.replace(
                    destination.with_suffix(destination.suffix + ".bad")
                )
            urllib.request.urlretrieve(str(info["url"]), destination)
            actual_md5 = _md5_file(destination)
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"{filename} md5 mismatch: expected {expected_md5}, got {actual_md5}"
                )
        self.extract()
        self._write_provenance()
        return self.doctor()

    def extract(self) -> None:
        """Extract archives idempotently."""

        self.init()
        targets = [
            (
                self.downloads_dir / STIMULI_ZIP,
                self.stimuli_root / ".braindough_extracted",
            ),
            (self.downloads_dir / ROI_ZIP, self.rois_root / ".braindough_extracted"),
        ]
        for archive, marker in targets:
            if marker.is_file():
                continue
            if not archive.is_file():
                continue
            with ZipFile(archive) as handle:
                handle.extractall(self.extracted_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok\n", encoding="utf-8")
        self._write_provenance()

    def doctor(self) -> DatasetDoctor:
        """Return local archive and extraction status."""

        fixture_mode = (self.root / ".braindough_fixture").is_file()
        downloads: dict[str, dict[str, object]] = {}
        for filename, info in DOWNLOADS.items():
            path = self.downloads_dir / filename
            exists = path.is_file()
            md5 = _md5_file(path) if exists else None
            downloads[filename] = {
                "path": str(path),
                "exists": exists,
                "size": path.stat().st_size if exists else None,
                "expected_size": info["size"],
                "file_id": info["file_id"],
                "md5": md5,
                "expected_md5": info["md5"],
                "ok": fixture_mode or (exists and md5 == info["md5"]),
                "url": info["url"],
            }
        return DatasetDoctor(
            root=self.root,
            downloads=downloads,
            extracted={
                "stimuli": (self.stimuli_root / ".braindough_extracted").is_file(),
                "rois": (self.rois_root / ".braindough_extracted").is_file(),
            },
        )

    def load_trials(
        self,
        subject: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        selection: str = "first",
        seed: int = 0,
    ) -> list[TrialRecord]:
        """Read stimulus-list rows for one BOLD5000 subject."""

        path = (
            self.rois_root
            / "stim_lists"
            / f"{_subject_stim_list_id(subject)}_stim_lists.txt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing BOLD5000 stimulus list: {path}")
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        indices = _select_trial_indices(
            total=len(lines),
            limit=limit,
            offset=offset,
            selection=selection,
            seed=seed,
            subject=subject,
        )
        image_labels = self._imagenet_labels()
        scene_labels = self._scene_labels()
        return [
            _trial_record(
                subject, int(index), lines[int(index)], image_labels, scene_labels
            )
            for index in indices
        ]

    def load_roi_matrix(
        self, subject: str, roi: str, *, tr: str = DEFAULT_TR, limit: int | None = None
    ) -> np.ndarray:
        """Load a subject ROI matrix as rows by trials and columns by voxels."""

        import h5py

        path = self.rois_root / subject / "h5" / f"{subject}_ROIs_{tr}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing BOLD5000 ROI file: {path}")
        with h5py.File(path, "r") as handle:
            if roi not in handle:
                available = ", ".join(sorted(handle.keys()))
                raise KeyError(
                    f"ROI {roi!r} not found in {path.name}; available: {available}"
                )
            dataset = cast(Any, handle[roi])
            array = np.asarray(dataset[:], dtype=np.float32)
        return array[:limit] if limit else array

    def make_stimuli(
        self,
        output_dir: Path,
        *,
        subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
        trial_limit: int = 64,
        offset: int = 0,
        selection: str = "first",
        trial_seed: int = 0,
    ) -> list[Any]:
        """Create inspectable label-card stimuli for real BOLD5000 trials."""

        from braindough.stimuli import Stimulus

        doctor = self.doctor()
        if not doctor.ready:
            return [
                _missing_dataset_stimulus(
                    output_dir, doctor.blocker() or "missing data"
                )
            ]
        stimuli: list[Stimulus] = []
        for subject in subjects:
            for trial in self.load_trials(
                subject,
                limit=trial_limit,
                offset=offset,
                selection=selection,
                seed=trial_seed,
            ):
                card_path = (
                    output_dir / SUITE / subject / f"trial-{trial.trial_index:05d}.png"
                )
                _write_label_card(card_path, trial)
                stimuli.append(
                    Stimulus(
                        stimulus_id=(
                            f"{SUITE}:{subject}:trial_{trial.trial_index:05d}"
                        ),
                        suite=SUITE,
                        modality="image",
                        kind="bold5000_label_card",
                        path=card_path,
                        sha256=sha256_file(card_path),
                        license=SOURCE_LICENSE,
                        metadata={
                            "subject": subject,
                            "trial_index": trial.trial_index,
                            "image_filename": trial.filename,
                            "normalized_image_filename": trial.normalized_filename,
                            "source_family": trial.source_family,
                            "label": trial.label,
                            "tokens": list(trial.tokens),
                            "repeated": trial.repeated,
                            "dataset": "BOLD5000",
                            "dataset_release": BOLD5000_RELEASE,
                            "dataset_release_label": BOLD5000_RELEASE_LABEL,
                            "dataset_version": BOLD5000_VERSION,
                            "dataset_article": FIGSHARE_ARTICLE,
                            "dataset_terms": TERMS_URL,
                            "dataset_website": DATASET_WEBSITE,
                            "release_2_recommended_url": RELEASE_2_URL,
                            "source_license_note": SOURCE_LICENSE,
                            "role": "real_dataset_trial",
                        },
                    )
                )
        return stimuli

    def _imagenet_labels(self) -> dict[str, str]:
        path = self.stimuli_root / "Image_Labels" / "imagenet_final_labels.txt"
        if not path.is_file():
            return {}
        labels: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            key, _, label = line.partition(" ")
            labels[key] = label.strip()
        return labels

    def _scene_labels(self) -> dict[str, str]:
        path = self.stimuli_root / "Image_Labels" / "scene_final_labels.txt"
        if not path.is_file():
            return {}
        labels: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            label = line.strip()
            if label:
                labels[label.lower()] = label
        return labels

    def _write_provenance(self) -> None:
        self.init()
        payload = {
            "dataset": "BOLD5000",
            "dataset_release": BOLD5000_RELEASE,
            "dataset_release_label": BOLD5000_RELEASE_LABEL,
            "version": BOLD5000_VERSION,
            "license": SOURCE_LICENSE,
            "article": FIGSHARE_ARTICLE,
            "terms": TERMS_URL,
            "website": DATASET_WEBSITE,
            "release_2_recommended_url": RELEASE_2_URL,
            "release_2_code_url": RELEASE_2_REPO_URL,
            "release_caveat": (
                "This adapter stages Release 1.0 processed ROI vectors and "
                "stimulus name/label metadata. The dataset authors recommend "
                "Release 2.0 for new functional analyses."
            ),
            "downloads": DOWNLOADS,
            "doctor": self.doctor().to_dict(),
        }
        self.provenance_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def make_bold5000_stimuli(
    output_dir: Path, config: dict[str, Any] | None = None
) -> list[Any]:
    """Build BOLD5000 stimulus records from experiment config."""

    config = config or {}
    dataset = BOLD5000Dataset(config.get("dataset_root"))
    subjects = tuple(str(item) for item in config.get("subjects", DEFAULT_SUBJECTS))
    trial_limit = int(config.get("trial_limit", 64))
    offset = int(config.get("trial_offset", 0))
    selection = str(config.get("trial_selection", "first"))
    trial_seed = int(config.get("trial_seed", 0))
    return dataset.make_stimuli(
        output_dir,
        subjects=subjects,
        trial_limit=trial_limit,
        offset=offset,
        selection=selection,
        trial_seed=trial_seed,
    )


def create_fixture_dataset(root: Path) -> BOLD5000Dataset:
    """Create a tiny BOLD5000-like fixture for tests."""

    dataset = BOLD5000Dataset(root)
    dataset.init()
    stimuli_root = dataset.stimuli_root
    rois_root = dataset.rois_root
    (stimuli_root / "Image_Labels").mkdir(parents=True, exist_ok=True)
    (rois_root / "stim_lists").mkdir(parents=True, exist_ok=True)
    (rois_root / "CSI1" / "h5").mkdir(parents=True, exist_ok=True)
    (stimuli_root / "Image_Labels" / "imagenet_final_labels.txt").write_text(
        "n00000001 object, thing\nn00000002 instrument\n", encoding="utf-8"
    )
    (stimuli_root / "Image_Labels" / "scene_final_labels.txt").write_text(
        "kitchen\nforest\n", encoding="utf-8"
    )
    filenames = [
        "n00000001_1.JPEG",
        "n00000002_2.JPEG",
        "COCO_train2014_000000000003.jpg",
        "kitchen1.jpg",
        "forest2.jpg",
        "rep_n00000001_1.JPEG",
        "COCO_train2014_000000000004.jpg",
        "kitchen2.jpg",
        "n00000002_3.JPEG",
        "forest3.jpg",
        "COCO_train2014_000000000005.jpg",
        "rep_kitchen1.jpg",
    ]
    (rois_root / "stim_lists" / "CSI01_stim_lists.txt").write_text(
        "\n".join(filenames) + "\n", encoding="utf-8"
    )
    import h5py

    x = np.arange(len(filenames), dtype=np.float32)
    source_signal = np.array(
        [
            0.2 if name.startswith("COCO") else 0.8 if name.startswith("n") else 1.3
            for name in filenames
        ],
        dtype=np.float32,
    )
    repeated = np.array([1.0 if name.startswith("rep_") else 0.0 for name in filenames])
    early = np.column_stack(
        [
            source_signal,
            source_signal * 0.5 + repeated,
            np.sin(x / 2.0),
            np.cos(x / 3.0),
        ]
    ).astype(np.float32)
    ppa = np.column_stack(
        [
            (source_signal > 1.0).astype(np.float32),
            source_signal * 0.25,
            repeated,
        ]
    ).astype(np.float32)
    with h5py.File(rois_root / "CSI1" / "h5" / "CSI1_ROIs_TR1.h5", "w") as handle:
        handle.create_dataset("LHEarlyVis", data=early)
        handle.create_dataset("LHPPA", data=ppa)
    (stimuli_root / ".braindough_extracted").write_text("ok\n", encoding="utf-8")
    (rois_root / ".braindough_extracted").write_text("ok\n", encoding="utf-8")
    (dataset.root / ".braindough_fixture").write_text("ok\n", encoding="utf-8")
    dataset._write_provenance()
    return dataset


def copy_or_download_dataset(
    *, root: str | Path | None = None, download: bool = False
) -> dict[str, object]:
    """CLI helper for staging or inspecting the dataset."""

    dataset = BOLD5000Dataset(root)
    doctor = dataset.download() if download else dataset.doctor()
    return doctor.to_dict()


def _default_root() -> Path:
    return BraindoughPaths.discover().home / "datasets" / "bold5000" / BOLD5000_VERSION


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_stim_list_id(subject: str) -> str:
    match = re.fullmatch(r"CSI0?([1-4])", subject)
    if not match:
        return subject
    return f"CSI0{match.group(1)}"


def _select_trial_indices(
    *,
    total: int,
    limit: int | None,
    offset: int,
    selection: str,
    seed: int,
    subject: str,
) -> np.ndarray:
    if offset < 0:
        raise ValueError("BOLD5000 trial offset must be non-negative")
    available = np.arange(min(offset, total), total, dtype=np.int64)
    count = len(available) if limit is None else min(max(0, limit), len(available))
    normalized = selection.strip().lower()
    if normalized in {"first", "sequential"}:
        return available[:count]
    if normalized == "random":
        rng = np.random.default_rng(seed + _stable_int(subject))
        return np.sort(rng.choice(available, size=count, replace=False))
    raise ValueError(
        f"Unknown BOLD5000 trial_selection {selection!r}; expected 'first' or 'random'"
    )


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _trial_record(
    subject: str,
    trial_index: int,
    filename: str,
    image_labels: dict[str, str],
    scene_labels: dict[str, str],
) -> TrialRecord:
    repeated = filename.startswith("rep_")
    normalized = filename.removeprefix("rep_")
    source = _source_family(normalized)
    label = _label_for_filename(normalized, image_labels, scene_labels)
    tokens = tuple(_tokens_for_filename(normalized, source, label, repeated))
    return TrialRecord(
        subject=subject,
        trial_index=trial_index,
        filename=filename,
        normalized_filename=normalized,
        source_family=source,
        label=label,
        tokens=tokens,
        repeated=repeated,
    )


def _source_family(filename: str) -> str:
    if filename.startswith("COCO_"):
        return "coco"
    if re.match(r"n\d+_", filename):
        return "imagenet"
    return "scene"


def _label_for_filename(
    filename: str, image_labels: dict[str, str], scene_labels: dict[str, str]
) -> str:
    if match := re.match(r"(n\d+)_", filename):
        return image_labels.get(match.group(1), match.group(1))
    if filename.startswith("COCO_"):
        return "coco natural image"
    stem = Path(filename).stem
    key = re.sub(r"\d+$", "", stem).lower()
    return scene_labels.get(key, key)


def _tokens_for_filename(
    filename: str, source: str, label: str, repeated: bool
) -> list[str]:
    stem = Path(filename).stem.lower()
    rough = re.split(r"[^a-z0-9]+", stem + " " + label.lower())
    tokens = [source, *[token for token in rough if token]]
    if repeated:
        tokens.append("repeated")
    return sorted(set(tokens))


def _write_label_card(path: Path, trial: TrialRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), _source_color(trial.source_family))
    draw = ImageDraw.Draw(image)
    lines = [
        "BOLD5000",
        trial.subject,
        f"trial {trial.trial_index}",
        trial.source_family,
        _wrap(trial.label, 22),
        _wrap(trial.normalized_filename, 26),
    ]
    y = 18
    for line in "\n".join(lines).splitlines():
        draw.text((16, y), line, fill=(20, 24, 32))
        y += 24
    image.save(path)


def _source_color(source: str) -> tuple[int, int, int]:
    return {
        "coco": (220, 235, 255),
        "imagenet": (226, 244, 219),
        "scene": (248, 226, 210),
    }.get(source, (235, 235, 235))


def _wrap(value: str, width: int) -> str:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _missing_dataset_stimulus(output_dir: Path, blocker: str) -> Any:
    from braindough.stimuli import Stimulus

    path = output_dir / SUITE / "dataset-missing.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), (245, 238, 220))
    draw = ImageDraw.Draw(image)
    draw.text((16, 24), "BOLD5000 data missing", fill=(20, 24, 32))
    draw.text((16, 64), _wrap(blocker, 28), fill=(20, 24, 32))
    image.save(path)
    return Stimulus(
        stimulus_id=f"{SUITE}:dataset_missing",
        suite=SUITE,
        modality="image",
        kind="dataset_missing",
        path=path,
        sha256=sha256_file(path),
        license=SOURCE_LICENSE,
        metadata={
            "dataset": "BOLD5000",
            "dataset_release": BOLD5000_RELEASE,
            "dataset_release_label": BOLD5000_RELEASE_LABEL,
            "dataset_version": BOLD5000_VERSION,
            "dataset_terms": TERMS_URL,
            "source_license_note": SOURCE_LICENSE,
            "role": "dataset_missing",
            "blocker": blocker,
        },
    )


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    """Write rows to CSV and return a manifest output row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {
        "id": f"table:{path.stem}",
        "path": str(path.relative_to(path.parents[2])),
        "sha256": sha256_file(path),
        "media_type": "text/csv",
        "rows": len(rows),
    }


def write_json_rows(path: Path, payload: dict[str, object]) -> dict[str, object]:
    """Write JSON and return a manifest output row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "id": f"table:{path.stem}",
        "path": str(path.relative_to(path.parents[2])),
        "sha256": sha256_file(path),
        "media_type": "application/json",
        "rows": 1,
    }
