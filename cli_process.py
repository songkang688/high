"""Command-line entry for highlight removal (single image or folder).

Examples:
  python cli_process.py -i photo.jpg
  python cli_process.py -i ./photos/
  python cli_process.py -i photo.jpg -o result.jpg
  python cli_process.py -i ./photos/ -o ./output/
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from highlight_removal.runtime_bootstrap import bootstrap_before_numpy, DEFAULT_RUNTIME_MODE, normalize_runtime_mode

bootstrap_before_numpy()

import cv2
import numpy as np

from highlight_removal.pipeline import process_image
from highlight_removal.utils import load_yaml, apply_runtime_mode, write_image
from highlight_removal.face_detect import get_last_landmarker_error, preload_face_landmarker

DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"
USER_SESSION_PATH = ROOT / "runtime_outputs" / "user_session.yaml"
INTENSITY_PATH = ROOT / "configs" / "removal_intensity_presets.yaml"
SENSITIVITY_PATH = ROOT / "configs" / "detection_sensitivity_presets.yaml"
MODE_CHOICES = ("常用模式", "高保真模式", "最高质量模式")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXT_LABEL = "、".join(sorted(ext.lstrip(".") for ext in IMAGE_EXTS))


@dataclass
class InputCheckResult:
    images: list[Path] = field(default_factory=list)
    resolved: Path | None = None
    is_directory: bool = False
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.images) and not self.messages


def _merge_preset(base_dict: dict, preset_path: Path, preset_name: str) -> dict:
    merged = deepcopy(base_dict)
    if preset_name == "正常":
        return merged
    data = load_yaml(preset_path)
    merged.update((data.get("presets") or {}).get(preset_name) or {})
    return merged


def load_cli_config(mode_name: str = "常用模式") -> dict:
    cfg = load_yaml(DEFAULT_CONFIG_PATH)
    runtime_mode = DEFAULT_RUNTIME_MODE
    if USER_SESSION_PATH.is_file():
        try:
            session = load_yaml(USER_SESSION_PATH)
            runtime_mode = normalize_runtime_mode(session.get("runtime_mode"))
            for section, values in (session.get("config") or {}).items():
                if section in cfg and isinstance(values, dict):
                    cfg[section].update(values)
        except Exception:
            pass
    cfg["runtime"] = apply_runtime_mode(runtime_mode)
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["enable_visualization"] = False
    cfg["pipeline"]["enable_roi_removal"] = True
    cfg["pipeline"]["roi_padding_ratio"] = 0.12
    cfg["pipeline"]["refine_upsampled_masks"] = False

    if mode_name == "常用模式":
        cfg["pipeline"]["process_scale"] = "compromise"
        cfg["highlight_detection"] = _merge_preset(cfg.get("highlight_detection", {}), SENSITIVITY_PATH, "灵敏")
        cfg["highlight_removal"] = _merge_preset(cfg.get("highlight_removal", {}), INTENSITY_PATH, "强力")
    elif mode_name == "高保真模式":
        cfg["pipeline"]["process_scale"] = "compromise"
        cfg["highlight_detection"] = _merge_preset(cfg.get("highlight_detection", {}), SENSITIVITY_PATH, "正常")
        cfg["highlight_removal"] = _merge_preset(cfg.get("highlight_removal", {}), INTENSITY_PATH, "正常")
    elif mode_name == "最高质量模式":
        cfg["pipeline"]["process_scale"] = 1.0
        cfg["highlight_detection"] = _merge_preset(cfg.get("highlight_detection", {}), SENSITIVITY_PATH, "正常")
        cfg["highlight_removal"] = _merge_preset(cfg.get("highlight_removal", {}), INTENSITY_PATH, "正常")
    return cfg


def normalize_path_text(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    for _ in range(3):
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
            text = text[1:-1].strip()
        else:
            break

    lowered = text.lower()
    if lowered.startswith("file:///"):
        text = text[8:]
    elif lowered.startswith("file://"):
        text = text[7:]

    text = text.strip().strip('"').strip("'")
    text = text.replace("/", os.sep)
    return os.path.normpath(text)


def resolve_user_path(raw: str, *, must_exist: bool = True) -> Path:
    text = normalize_path_text(raw)
    if not text:
        return Path("")

    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(os.path.normpath(str(candidate)))

    if must_exist and candidate.exists():
        try:
            return candidate.resolve()
        except OSError:
            return candidate
    return candidate


def format_path(path: Path | None) -> str:
    if path is None:
        return "(空)"
    return str(path)


def list_images_in_directory(directory: Path) -> list[Path]:
    def _sort_key(item: Path):
        if item.stem.isdigit():
            return (0, int(item.stem), item.suffix.lower())
        return (1, item.name.lower())

    files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    files.sort(key=_sort_key)
    return files


def _sample_names(paths: list[Path], limit: int = 5) -> str:
    if not paths:
        return "(无)"
    names = [p.name for p in paths[:limit]]
    if len(paths) > limit:
        names.append(f"... 等共 {len(paths)} 个")
    return "、".join(names)


def check_input_path(raw: str) -> InputCheckResult:
    result = InputCheckResult()
    text = normalize_path_text(raw)
    if not text:
        result.messages.append("[错误] 未提供输入路径，请使用 -i 指定图片或文件夹。")
        result.messages.append("       示例：run_facehi_terminal.bat -i \"D:\\photos\\face.jpg\"")
        return result

    resolved = resolve_user_path(raw, must_exist=False)
    result.resolved = resolved

    if not resolved.exists():
        parent = resolved.parent
        if parent.exists() and parent.is_dir():
            result.messages.append("[错误] 输入文件不存在，请检查文件名或后缀是否写错。")
            result.messages.append(f"       您输入的是：{raw}")
            result.messages.append(f"       解析后的路径：{format_path(resolved)}")
            result.messages.append(f"       所在目录存在：{format_path(parent)}")
            available = list_images_in_directory(parent)
            if available:
                result.messages.append(f"       该目录下可用图片示例：{_sample_names(available)}")
                same_stem = [p for p in available if p.stem.lower() == resolved.stem.lower()]
                if same_stem:
                    try:
                        rel = os.path.relpath(same_stem[0], Path.cwd())
                        rel = rel.replace("/", "\\")
                    except ValueError:
                        rel = str(same_stem[0])
                    result.messages.append(f"       提示：可改用 {same_stem[0].name}，例如 -i \"{rel}\"")
            else:
                siblings = sorted([p.name for p in parent.iterdir() if p.is_file()][:8])
                if siblings:
                    result.messages.append(f"       目录内其他文件示例：{_sample_names([parent / n for n in siblings])}")
        else:
            result.messages.append("[错误] 路径不存在，请检查是否写错。")
            result.messages.append(f"       您输入的是：{raw}")
            result.messages.append(f"       解析后的路径：{format_path(resolved)}")
            result.messages.append("       提示：可使用绝对路径，例如 C:\\Users\\你的用户名\\photos")
            result.messages.append("       也可使用相对路径，例如 .\\data 或 ..\\photos\\face.jpg")
        return result

    if resolved.is_file():
        if resolved.suffix.lower() not in IMAGE_EXTS:
            result.messages.append("[错误] 这是一个文件，但图片格式不支持。")
            result.messages.append(f"       文件路径：{format_path(resolved)}")
            result.messages.append(f"       当前后缀：{resolved.suffix or '(无后缀)'}")
            result.messages.append(f"       支持格式：{EXT_LABEL}")
            return result
        result.images = [resolved]
        return result

    if resolved.is_dir():
        result.is_directory = True
        images = list_images_in_directory(resolved)
        if not images:
            all_files = [f for f in resolved.iterdir() if f.is_file()]
            unsupported_images = [
                f for f in all_files if f.suffix.lower() not in IMAGE_EXTS and f.suffix.lower() in {".gif", ".tif", ".tiff", ".heic", ".webp"}
            ]
            result.messages.append("[错误] 文件夹内未找到可处理的图片。")
            result.messages.append(f"       文件夹路径：{format_path(resolved)}")
            result.messages.append(f"       支持格式：{EXT_LABEL}")
            if not all_files:
                result.messages.append("       该文件夹为空，请确认是否选错了目录。")
            else:
                result.messages.append(f"       文件夹内共有 {len(all_files)} 个文件，但都不符合支持格式。")
                result.messages.append(f"       文件示例：{_sample_names(all_files)}")
                if unsupported_images:
                    result.messages.append("       提示：部分图片可能是特殊格式，请先转换为 jpg/png 再处理。")
            result.messages.append("       说明：目前只处理该文件夹第一层的图片，不会自动扫描子文件夹。")
            return result
        result.images = images
        return result

    result.messages.append("[错误] 路径类型无法识别。")
    result.messages.append(f"       路径：{format_path(resolved)}")
    return result


def check_output_path(raw: str | None, *, batch_mode: bool, input_resolved: Path | None) -> tuple[Path | None, list[str]]:
    if not raw:
        return None, []

    text = normalize_path_text(raw)
    if not text:
        return None, ["[错误] 输出路径为空，请重新填写 -o，或去掉 -o 使用默认输出。"]

    resolved = resolve_user_path(raw, must_exist=False)
    messages: list[str] = []

    if batch_mode:
        if resolved.suffix and not resolved.exists():
            messages.append("[错误] 输入是文件夹时，-o 也必须填写文件夹路径，不能是单个文件名。")
            messages.append(f"       您填写的 -o：{raw}")
            messages.append(f"       解析后的路径：{format_path(resolved)}")
            messages.append("       正确示例：run_facehi_terminal.bat -i \"D:\\photos\\\" -o \"D:\\photos\\done\\\"")
            return None, messages
        if resolved.exists() and resolved.is_file():
            messages.append("[错误] 输入是文件夹时，-o 不能是文件。")
            messages.append(f"       当前 -o 指向文件：{format_path(resolved)}")
            messages.append("       请改为输出文件夹，例如：-o \"D:\\photos\\done\\\"")
            return None, messages
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved.resolve(), []
        except OSError as exc:
            messages.append("[错误] 无法创建输出文件夹。")
            messages.append(f"       目标路径：{format_path(resolved)}")
            messages.append(f"       系统提示：{exc}")
            return None, messages

    if resolved.exists() and resolved.is_dir():
        return resolved.resolve(), []

    if not resolved.suffix:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved.resolve(), []
        except OSError as exc:
            messages.append("[错误] 无法创建输出文件夹。")
            messages.append(f"       目标路径：{format_path(resolved)}")
            messages.append(f"       系统提示：{exc}")
            return None, messages

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        messages.append("[错误] 无法创建输出文件的父目录。")
        messages.append(f"       目标文件：{format_path(resolved)}")
        messages.append(f"       系统提示：{exc}")
        return None, messages

    if input_resolved and resolved.resolve() == input_resolved.resolve():
        messages.append("[错误] 输出文件不能与输入文件相同。")
        messages.append(f"       路径：{format_path(resolved)}")
        return None, messages

    return resolved.resolve(), []


def default_output_name(source_path: Path) -> str:
    return f"{source_path.stem}_output{source_path.suffix}"


def read_image_bgr(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def resolve_output_path(
    source_path: Path,
    output_resolved: Path | None,
    *,
    batch_mode: bool,
) -> Path:
    source_path = source_path.resolve()
    if output_resolved is None:
        return source_path.parent / default_output_name(source_path)
    if batch_mode or output_resolved.is_dir():
        return output_resolved / default_output_name(source_path)
    return output_resolved


def preload_models(cfg: dict) -> bool:
    fd = deepcopy(cfg.get("face_detection", {}))
    fd["_runtime"] = cfg["runtime"]
    model_rel = fd.get("mediapipe_task_model_path", "models/face_landmarker.task")
    model_path = (ROOT / model_rel).resolve()
    if not model_path.is_file():
        print("[错误] 人脸模型文件缺失。", file=sys.stderr)
        print(f"       期望位置：{model_path}", file=sys.stderr)
        return False
    fd["mediapipe_task_model_path"] = str(model_path)
    if not preload_face_landmarker(fd):
        detail = get_last_landmarker_error() or "未知错误"
        print("[错误] 人脸模型加载失败。", file=sys.stderr)
        print(f"       详情：{detail}", file=sys.stderr)
        return False
    return True


def print_messages(messages: list[str]) -> None:
    for line in messages:
        print(line, file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove facial highlights from image(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  run_facehi_terminal.bat -i photo.jpg\n"
            "  run_facehi_terminal.bat -i D:\\photos\\\n"
            "  run_facehi_terminal.bat -i C:/Users/Admin/photos\n"
            "  run_facehi_terminal.bat -i .\\data\\1.jpg -o result.jpg\n"
        ),
    )
    parser.add_argument("-i", "--input", required=True, help="输入图片或文件夹（必填）")
    parser.add_argument("-o", "--output", default=None, help="输出文件或文件夹（可选）")
    parser.add_argument(
        "--mode",
        default="常用模式",
        choices=MODE_CHOICES,
        help="处理模式（可选，默认：常用模式）",
    )
    return parser


def process_one(
    image_path: Path,
    cfg: dict,
    output_resolved: Path | None,
    *,
    batch_mode: bool,
) -> tuple[bool, Path | None, str]:
    image_bgr = read_image_bgr(image_path)
    if image_bgr is None:
        return False, None, "图片读取失败，文件可能损坏或格式异常"

    result = process_image(image_bgr, cfg)
    if not result.success:
        detail = result.status or "处理失败"
        if result.warnings:
            detail = f"{detail}；{'；'.join(result.warnings)}"
        return False, None, detail

    output_path = resolve_output_path(image_path, output_resolved, batch_mode=batch_mode)
    try:
        write_image(output_path, result.result_bgr)
    except OSError as exc:
        return False, None, f"结果保存失败：{exc}"
    return True, output_path, ""


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    input_check = check_input_path(args.input)
    if not input_check.ok:
        print_messages(input_check.messages)
        return 2

    batch_mode = input_check.is_directory or len(input_check.images) > 1
    output_resolved, output_messages = check_output_path(
        args.output,
        batch_mode=batch_mode,
        input_resolved=input_check.resolved if input_check.resolved and input_check.resolved.is_file() else None,
    )
    if output_messages:
        print_messages(output_messages)
        return 2

    cfg = load_cli_config(args.mode)
    if not preload_models(cfg):
        return 3

    if batch_mode:
        print(
            f"[信息] 将处理 {len(input_check.images)} 张图片，模式：{args.mode}",
            file=sys.stderr,
        )
        if input_check.resolved:
            print(f"[信息] 输入文件夹：{format_path(input_check.resolved)}", file=sys.stderr)

    ok_count = 0
    fail_count = 0
    for image_path in input_check.images:
        ok, output_path, detail = process_one(
            image_path,
            cfg,
            output_resolved,
            batch_mode=batch_mode,
        )
        if ok and output_path is not None:
            ok_count += 1
            print(str(output_path))
        else:
            fail_count += 1
            print(f"[失败] {image_path.name}：{detail}", file=sys.stderr)

    if len(input_check.images) > 1:
        print(
            f"[完成] 成功 {ok_count} 张，失败 {fail_count} 张，共 {len(input_check.images)} 张",
            file=sys.stderr,
        )

    if fail_count == 0:
        return 0
    if ok_count == 0:
        return 5
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
