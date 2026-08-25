"""Auto-detect install/package roots without hardcoded paths."""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent

CLI_BAT_NAME = "run_facehi_terminal.bat"
CLI_LAUNCHER_NAME = "open_facehi_terminal.ps1"
CLI_README_NAME = "CLI使用说明.txt"
CLI_WELCOME_BAT_NAME = "cli_terminal_welcome.bat"


def _unique_bases(*paths: Path) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for raw in paths:
        try:
            base = raw.resolve()
        except OSError:
            continue
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(base)
    return ordered


def find_package_root(start: Path | None = None) -> Path:
    """Locate install root from current app location (works on any drive/path)."""
    seeds: list[Path] = []
    if start is not None:
        seeds.append(start)
    seeds.extend([APP_ROOT, APP_ROOT.parent, *APP_ROOT.parents[:6]])

    for base in _unique_bases(*seeds):
        if (base / CLI_BAT_NAME).is_file():
            return base
        if (base / "python" / "python.exe").is_file() and (base / "main" / "app_simple.py").is_file():
            return base
        if (base / "python" / "python.exe").is_file() and (base / "main" / "cli_process.py").is_file():
            return base
    return APP_ROOT.resolve()


def find_resource_file(filename: str) -> Path | None:
    package_root = find_package_root()
    for candidate in (package_root / filename, APP_ROOT / filename):
        if candidate.is_file():
            return candidate
    return None


def get_cli_bat_path() -> Path:
    found = find_resource_file(CLI_BAT_NAME)
    return found if found is not None else find_package_root() / CLI_BAT_NAME


def get_cli_launcher_path() -> Path:
    found = find_resource_file(CLI_LAUNCHER_NAME)
    return found if found is not None else find_package_root() / CLI_LAUNCHER_NAME


def get_cli_readme_path() -> Path:
    found = find_resource_file(CLI_README_NAME)
    return found if found is not None else find_package_root() / CLI_README_NAME


def build_cli_sample_command() -> str:
    package_root = find_package_root()
    sample_rel = "data\\1.jpg"
    data_candidates = [
        package_root / "data",
        APP_ROOT.parent / "data",
    ]
    for data_dir in data_candidates:
        if not data_dir.is_dir():
            continue
        for name in ("1.png", "1.jpg", "1.jpeg", "1.webp"):
            image_path = data_dir / name
            if image_path.is_file():
                try:
                    rel = os.path.relpath(image_path, package_root)
                    sample_rel = rel.replace("/", "\\")
                except ValueError:
                    sample_rel = f"data\\{name}"
                break
        break
    return f'{CLI_BAT_NAME} -i "{sample_rel}"'
