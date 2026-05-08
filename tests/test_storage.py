from pathlib import Path

from braindough.storage import BraindoughPaths


def test_worktree_id_avoids_collisions(tmp_path: Path) -> None:
    first = BraindoughPaths.discover(home=tmp_path / "home", workspace=tmp_path / "a")
    second = BraindoughPaths.discover(home=tmp_path / "home", workspace=tmp_path / "b")

    assert first.worktree_id != second.worktree_id
    assert first.worktree_root != second.worktree_root


def test_storage_init_creates_expected_layout(tmp_path: Path) -> None:
    paths = BraindoughPaths.discover(home=tmp_path / "home", workspace=tmp_path)

    created = paths.init()

    assert paths.tribe_model_dir in created
    assert paths.scratch.is_dir()
    assert paths.runs_root.is_dir()
    assert paths.env()["BRAINDOUGH_WORKTREE_ID"] == paths.worktree_id
