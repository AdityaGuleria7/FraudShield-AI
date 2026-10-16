import runpy
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_ieee_challengers.py"
)


def load_check():
    namespace = runpy.run_path(
        str(SCRIPT),
        run_name="fraudshield_working_tree_test",
    )
    return namespace["require_clean_working_tree"]


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "FraudShield Test")
    git(repo, "config", "user.email", "fraudshield-test@example.invalid")
    (repo / "tracked.txt").write_text("baseline\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-m", "test baseline", "-q")
    return repo


def test_clean_working_tree_is_accepted(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)

    load_check()()


def test_untracked_file_is_rejected(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    (repo / "untracked.txt").write_text("untracked\n")
    monkeypatch.chdir(repo)

    check = load_check()
    try:
        check()
    except RuntimeError as exc:
        assert "Working tree is not clean" in str(exc)
    else:
        raise AssertionError("Untracked file should be rejected.")


def test_modified_tracked_file_is_rejected(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    (repo / "tracked.txt").write_text("modified\n")
    monkeypatch.chdir(repo)

    check = load_check()
    try:
        check()
    except RuntimeError as exc:
        assert "Working tree is not clean" in str(exc)
    else:
        raise AssertionError("Modified tracked file should be rejected.")
