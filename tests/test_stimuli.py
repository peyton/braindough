from pathlib import Path

from braindough.stimuli import generate_stimuli


def test_generate_first_suite_stimuli(tmp_path: Path) -> None:
    stimuli = generate_stimuli(
        suites=(
            "image_activation",
            "visual_controls",
            "visual_perturbations",
            "temporalization",
            "audio_controls",
        ),
        output_dir=tmp_path,
        seed=123,
    )

    assert len(stimuli) >= 20
    assert {stimulus.modality for stimulus in stimuli} == {"video", "audio"}
    assert all(stimulus.path.is_file() for stimulus in stimuli)
    assert all(len(stimulus.sha256) == 64 for stimulus in stimuli)
