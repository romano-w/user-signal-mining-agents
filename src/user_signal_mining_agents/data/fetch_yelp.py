from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from ..config import Settings


EXPECTED_YELP_FILES = (
    "yelp_academic_dataset_business.json",
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_tip.json",
    "yelp_academic_dataset_user.json",
    "yelp_academic_dataset_checkin.json",
)


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def dataset_is_extracted(dataset_dir: Path) -> bool:
    return all((dataset_dir / name).exists() for name in EXPECTED_YELP_FILES)


def download_file(url: str, destination: Path, *, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        print(f"Using existing download: {destination}")
        return destination

    print(f"Downloading Yelp dataset archive from {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        total_size = response.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size and total_size.isdigit() else None
        downloaded = 0
        chunk_size = 1024 * 1024 * 8
        next_progress_bytes = 0

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)

            if downloaded >= next_progress_bytes:
                if total_bytes:
                    percent = downloaded / total_bytes * 100
                    print(
                        f"Downloaded {_format_bytes(downloaded)} "
                        f"of {_format_bytes(total_bytes)} ({percent:.1f}%)"
                    )
                else:
                    print(f"Downloaded {_format_bytes(downloaded)}")
                next_progress_bytes = downloaded + chunk_size * 32

    print(f"Saved download to {destination}")
    return destination


def extract_zip(zip_path: Path, dataset_dir: Path, *, overwrite: bool = False) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        tar_members = [member for member in archive.namelist() if member.endswith(".tar")]
        if not tar_members:
            raise FileNotFoundError(f"No tar archive found inside {zip_path}")

        tar_member = tar_members[0]
        tar_path = dataset_dir / Path(tar_member).name
        if tar_path.exists() and not overwrite:
            print(f"Using existing tar archive: {tar_path}")
            return tar_path

        print(f"Extracting {tar_member} from {zip_path.name}")
        with archive.open(tar_member) as source, tar_path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024 * 8)

    print(f"Saved tar archive to {tar_path}")
    return tar_path


def extract_tar(tar_path: Path, dataset_dir: Path, *, overwrite: bool = False) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if dataset_is_extracted(dataset_dir) and not overwrite:
        print(f"Using existing extracted dataset in {dataset_dir}")
        return dataset_dir

    print(f"Extracting JSON files from {tar_path.name}")
    with tarfile.open(tar_path) as archive:
        archive.extractall(dataset_dir, filter="data")

    print(f"Extracted Yelp dataset into {dataset_dir}")
    return dataset_dir


def ensure_yelp_dataset(
    settings: Settings,
    *,
    skip_download: bool = False,
    force_download: bool = False,
    force_extract: bool = False,
) -> Path:
    dataset_dir = settings.yelp_dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        zip_path = download_file(
            settings.yelp_download_url,
            settings.yelp_download_zip_path,
            overwrite=force_download,
        )
        tar_path = extract_zip(zip_path, dataset_dir, overwrite=force_download)
    else:
        tar_path = settings.yelp_tar_path
        if not tar_path.exists():
            raise FileNotFoundError(
                "Yelp tar archive not found. Either place it at "
                f"{tar_path} or run without --skip-download."
            )

    extract_tar(tar_path, dataset_dir, overwrite=force_extract)
    return dataset_dir
