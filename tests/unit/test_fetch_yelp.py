from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from user_signal_mining_agents.data import fetch_yelp


def test_format_bytes_scales_units() -> None:
    assert fetch_yelp._format_bytes(512) == "512.0 B"
    assert fetch_yelp._format_bytes(1024) == "1.0 KB"


def test_dataset_is_extracted_checks_expected_files(tmp_path: Path) -> None:
    assert fetch_yelp.dataset_is_extracted(tmp_path) is False
    for name in fetch_yelp.EXPECTED_YELP_FILES:
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert fetch_yelp.dataset_is_extracted(tmp_path) is True


def test_extract_zip_raises_when_no_tar_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("notes.txt", "hello")

    with pytest.raises(FileNotFoundError, match="No tar archive"):
        fetch_yelp.extract_zip(zip_path, tmp_path / "dataset")


def test_extract_zip_uses_existing_tar_when_present(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    tar_path = dataset_dir / "yelp_dataset.tar"
    tar_path.write_bytes(b"already-there")

    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("some/path/yelp_dataset.tar", b"new")

    out = fetch_yelp.extract_zip(zip_path, dataset_dir, overwrite=False)
    assert out == tar_path
    assert tar_path.read_bytes() == b"already-there"


def test_extract_tar_invokes_extractall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tar_path = tmp_path / "archive.tar"
    tar_path.write_bytes(b"not-used")
    dataset_dir = tmp_path / "dataset"

    calls: list[tuple[Path, str]] = []

    class _FakeArchive:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extractall(self, path: Path, filter: str | None = None) -> None:
            calls.append((path, filter or ""))

    monkeypatch.setattr(fetch_yelp, "dataset_is_extracted", lambda _p: False)
    monkeypatch.setattr(tarfile, "open", lambda _p: _FakeArchive())

    out = fetch_yelp.extract_tar(tar_path, dataset_dir, overwrite=False)
    assert out == dataset_dir
    assert calls == [(dataset_dir, "data")]


def test_ensure_yelp_dataset_skip_download_requires_existing_tar(tmp_settings) -> None:
    with pytest.raises(FileNotFoundError, match="Yelp tar archive not found"):
        fetch_yelp.ensure_yelp_dataset(tmp_settings, skip_download=True)


def test_ensure_yelp_dataset_download_path_calls_helpers(monkeypatch: pytest.MonkeyPatch, tmp_settings) -> None:
    seen: dict[str, object] = {}

    def _download(url: str, destination: Path, *, overwrite: bool = False) -> Path:
        seen["download"] = (url, destination, overwrite)
        return destination

    def _extract_zip(zip_path: Path, dataset_dir: Path, *, overwrite: bool = False) -> Path:
        seen["extract_zip"] = (zip_path, dataset_dir, overwrite)
        return dataset_dir / "yelp_dataset.tar"

    def _extract_tar(tar_path: Path, dataset_dir: Path, *, overwrite: bool = False) -> Path:
        seen["extract_tar"] = (tar_path, dataset_dir, overwrite)
        return dataset_dir

    monkeypatch.setattr(fetch_yelp, "download_file", _download)
    monkeypatch.setattr(fetch_yelp, "extract_zip", _extract_zip)
    monkeypatch.setattr(fetch_yelp, "extract_tar", _extract_tar)

    result = fetch_yelp.ensure_yelp_dataset(
        tmp_settings,
        skip_download=False,
        force_download=True,
        force_extract=True,
    )

    assert result == tmp_settings.yelp_dataset_dir
    assert "download" in seen and "extract_zip" in seen and "extract_tar" in seen


def test_ensure_yelp_dataset_skip_download_uses_existing_tar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_settings,
) -> None:
    tmp_settings.yelp_tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_settings.yelp_tar_path.write_bytes(b"x")

    called: list[tuple[Path, Path, bool]] = []

    def _extract_tar(tar_path: Path, dataset_dir: Path, *, overwrite: bool = False) -> Path:
        called.append((tar_path, dataset_dir, overwrite))
        return dataset_dir

    monkeypatch.setattr(fetch_yelp, "extract_tar", _extract_tar)

    result = fetch_yelp.ensure_yelp_dataset(
        tmp_settings,
        skip_download=True,
        force_extract=False,
    )

    assert result == tmp_settings.yelp_dataset_dir
    assert called == [(tmp_settings.yelp_tar_path, tmp_settings.yelp_dataset_dir, False)]
