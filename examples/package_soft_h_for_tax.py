"""Stage a deterministic soft_entropy source package in the Tax image context."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _source_files(package_dir: Path) -> list[Path]:
    return sorted(
        path for path in package_dir.rglob("*.py") if "__pycache__" not in path.parts
    )


def package_digest(package_dir: Path) -> tuple[str, dict[str, str]]:
    """Hash Python source paths and contents in deterministic order."""
    digest = hashlib.sha256()
    file_hashes = {}
    for path in _source_files(package_dir):
        relative_path = path.relative_to(package_dir).as_posix()
        contents = path.read_bytes()
        file_hashes[relative_path] = hashlib.sha256(contents).hexdigest()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    if not file_hashes:
        raise ValueError(f"No Python source files found under {package_dir}.")
    return digest.hexdigest(), file_hashes


def stage_package(soft_h_repo: Path, output_dir: Path) -> str:
    """Copy soft_entropy into a new Tax build-context directory."""
    package_dir = (soft_h_repo / "soft_entropy").resolve()
    if not (package_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"soft_entropy package not found under {soft_h_repo}.")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing package at {output_dir}."
        )

    source_digest, source_hashes = package_digest(package_dir)
    staged_package = output_dir / "soft_entropy"
    staged_package.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        package_dir,
        staged_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    staged_digest, staged_hashes = package_digest(staged_package)
    if staged_digest != source_digest or staged_hashes != source_hashes:
        raise RuntimeError(
            "Staged soft_entropy source does not match the source checkout."
        )

    manifest = {
        "schema_version": 1,
        "package": "soft_entropy",
        "sha256": source_digest,
        "files": source_hashes,
    }
    (output_dir / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return source_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soft-h-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Print the source digest without creating a staged package.",
    )
    args = parser.parse_args()

    soft_h_repo = args.soft_h_repo.resolve()
    if args.hash_only:
        digest, _ = package_digest(soft_h_repo / "soft_entropy")
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required unless --hash-only is used.")
        digest = stage_package(soft_h_repo, args.output_dir.resolve())
    print(digest)


if __name__ == "__main__":
    main()
