import sys
import os
import time
import subprocess
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from highlight_removal.runtime_bootstrap import bootstrap_before_numpy, DEFAULT_RUNTIME_MODE, normalize_runtime_mode

bootstrap_before_numpy()

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup, QFileDialog,
    QProgressBar, QMessageBox, QSizePolicy, QDialog, QDialogButtonBox,
    QCheckBox, QListWidget, QListWidgetItem, QSplitter, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize, pyqtSlot, QTimer, QEvent
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon

from highlight_removal.pipeline import process_image
from highlight_removal.utils import load_yaml, apply_runtime_mode
from highlight_removal.face_detect import get_last_landmarker_error, preload_face_landmarker

DEFAULT_CONFIG_PATH = ROOT / "configs/default.yaml"
USER_SESSION_PATH = ROOT / "runtime_outputs/user_session.yaml"
FAVORITES_PATH = ROOT / "runtime_outputs/favorites.yaml"
INTENSITY_PATH = ROOT / "configs/removal_intensity_presets.yaml"
SENSITIVITY_PATH = ROOT / "configs/detection_sensitivity_presets.yaml"
from install_paths import (
    build_cli_sample_command,
    find_package_root,
    get_cli_bat_path,
    get_cli_launcher_path,
    get_cli_readme_path,
)

DEFAULT_DATA_DIR_REL = "../data"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
APP_ICON_PATH = ROOT / "assets" / "app_icon.ico"


def get_install_root() -> Path:
    return find_package_root()


def _resolve_path(path_text: str | Path) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.expanduser().resolve()
    return p


def _resolve_dir_path(dir_text: str | Path | None = None) -> Path:
    raw = str(dir_text or "").strip() or DEFAULT_DATA_DIR_REL
    p = Path(raw)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    else:
        p = p.expanduser().resolve()
    if p.is_file():
        return p.parent
    return p


def _normalize_stored_path(path: str | Path) -> str:
    text = str(path or "").strip()
    if not text:
        return DEFAULT_DATA_DIR_REL
    p = _resolve_path(text)
    try:
        rel = os.path.relpath(p, ROOT)
        return rel.replace("\\", "/") if os.sep == "\\" else rel
    except ValueError:
        return text.replace("\\", "/")


def _display_path(path: str | Path) -> str:
    return _normalize_stored_path(path)


def _path_is_dir(path: str | Path) -> bool:
    return _resolve_dir_path(path).is_dir()


def _dialog_dir(path: str | Path) -> str:
    p = _resolve_dir_path(path)
    return str(p if p.is_dir() else p.parent)


def _read_image(path: str | Path):
    resolved = str(_resolve_path(path))
    img = cv2.imread(resolved, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(resolved, cv2.IMREAD_UNCHANGED)
    return img


def _ensure_bgr_color(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    img_bgr = _ensure_bgr_color(img_bgr)
    img_rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _default_image_dir_rel() -> str:
    if USER_SESSION_PATH.is_file():
        try:
            session = load_yaml(USER_SESSION_PATH)
            image_dir = str(session.get("image_dir") or "").strip()
            if image_dir:
                return _normalize_stored_path(image_dir)
        except Exception:
            pass
    return DEFAULT_DATA_DIR_REL


def _output_dir_rel_for_files(files: list[str]) -> str:
    if not files:
        return ""
    out_dir = _resolve_path(files[0]).parent / "output"
    return _normalize_stored_path(out_dir)




def _load_favorites() -> set[str]:
    if not FAVORITES_PATH.is_file():
        return set()
    try:
        data = load_yaml(FAVORITES_PATH)
        items = data.get("favorites") or []
        return {_normalize_stored_path(p) for p in items if str(p).strip()}
    except Exception:
        return set()


def _save_favorites(favorites: set[str]) -> None:
    FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(favorites, key=lambda p: (_resolve_path(p).name.lower(), p))
    with open(FAVORITES_PATH, "w", encoding="utf-8") as fh:
        fh.write("favorites:\n")
        for path in ordered:
            fh.write(f"  - {path}\n")

def _merge_preset(base_dict: dict, preset_path: Path, preset_name: str) -> dict:
    d = deepcopy(base_dict)
    if preset_name == "正常":
        return d
    data = load_yaml(preset_path)
    d.update((data.get("presets") or {}).get(preset_name) or {})
    return d


def load_base_config() -> dict:
    """default.yaml + user_session.config + CPU 狂暴 runtime（GPU 禁用）。"""
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
    return cfg


def get_config_for_mode(mode_name: str) -> dict:
    cfg = load_base_config()
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


def list_images(path: str | Path) -> list[str]:
    p = Path(path)
    if p.suffix.lower() in IMAGE_EXTS and _resolve_path(p).is_file():
        return [_normalize_stored_path(p)]
    root = _resolve_dir_path(path)
    if not root.is_dir():
        return []

    def _sort_key(item: Path):
        if item.stem.isdigit():
            return (0, int(item.stem), item.suffix.lower())
        return (1, item.name.lower())

    files = [f for f in root.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    files.sort(key=_sort_key)
    return [_normalize_stored_path(f) for f in files]


def default_output_name(source_path: str) -> str:
    return f"output_{Path(source_path).name}"


class BatchImageSelectDialog(QDialog):
    """选择参与批量去高光的图片。"""

    def __init__(self, image_files: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择批量处理的图片")
        self.setMinimumSize(420, 480)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("勾选需要批量去高光的图片："))

        tool_row = QHBoxLayout()
        self.btn_select_all = ModernButton("全选")
        self.btn_select_all.clicked.connect(self._select_all)
        tool_row.addWidget(self.btn_select_all)
        tool_row.addStretch()
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color: #666; font-size: 13px;")
        tool_row.addWidget(self.lbl_count)
        layout.addLayout(tool_row)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(SingleModeWidget.LIST_STYLE)
        for path in image_files:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
        self.list_widget.itemChanged.connect(lambda _: self._update_count())
        layout.addWidget(self.list_widget, stretch=1)
        self._update_count()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_all(self):
        all_checked = all(
            self.list_widget.item(i).checkState() == Qt.Checked
            for i in range(self.list_widget.count())
        )
        state = Qt.Unchecked if all_checked else Qt.Checked
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)
        self.list_widget.blockSignals(False)
        self.btn_select_all.setText("全不选" if state == Qt.Checked else "全选")
        self._update_count()

    def _update_count(self):
        n = len(self.selected_paths())
        total = self.list_widget.count()
        self.lbl_count.setText(f"已选 {n} / {total}")

    def selected_paths(self) -> list[str]:
        paths = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                paths.append(item.data(Qt.UserRole))
        return paths


class BatchModeDialog(QDialog):
    """批量处理前选择模式。"""

    MODES = ["常用模式", "高保真模式", "最高质量模式"]

    def __init__(self, image_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量去高光")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"即将处理 {image_count} 张图片，请选择处理模式："))

        self.mode_group = QButtonGroup(self)
        for i, name in enumerate(self.MODES):
            rb = QRadioButton(name)
            if i == 0:
                rb.setChecked(True)
            self.mode_group.addButton(rb, i)
            layout.addWidget(rb)

        layout.addSpacing(8)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("开始批量处理")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_mode(self) -> str:
        btn = self.mode_group.checkedButton()
        return btn.text() if btn else self.MODES[0]


class CliModeDialog(QDialog):
    """命令行模式说明与快捷入口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("命令行模式")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("可通过命令行批量或单张处理图片，适合脚本调用与自动化。"))

        layout.addWidget(QLabel("默认可直接复制试用的命令："))
        self.sample_cmd = build_cli_sample_command()
        self.input_cmd = QLineEdit(self.sample_cmd)
        self.input_cmd.setReadOnly(True)
        self.input_cmd.setStyleSheet(
            "QLineEdit { padding: 8px 10px; border: 1px solid #D0D0D0; border-radius: 6px; "
            "background: #F8F9FA; color: #222; font-family: Consolas, 'Courier New', monospace; }"
        )
        layout.addWidget(self.input_cmd)

        hint = QLabel(
            "说明：\n"
            "• 只需必填 -i，可传单张图片或文件夹\n"
            "• 不填 -o 时，结果保存在当前目录，文件名加 _output\n"
            "• 不填 --mode 时，默认使用「常用模式」"
        )
        hint.setStyleSheet("color: #555;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        self.btn_copy = ModernButton("复制命令")
        self.btn_copy.clicked.connect(self._copy_command)
        self.btn_open_readme = ModernButton("打开说明书")
        self.btn_open_readme.clicked.connect(self._open_readme)
        self.btn_open_terminal = ModernButton("打开命令行终端", primary=True)
        self.btn_open_terminal.clicked.connect(self._open_terminal)
        btn_row.addWidget(self.btn_copy)
        btn_row.addWidget(self.btn_open_readme)
        btn_row.addWidget(self.btn_open_terminal)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        install_root = get_install_root()
        path_hint = QLabel(f"安装目录：{install_root}")
        path_hint.setStyleSheet("color: #888; font-size: 12px;")
        path_hint.setWordWrap(True)
        layout.addWidget(path_hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn is not None:
            close_btn.setText("关闭")
        layout.addWidget(buttons)

    def _copy_command(self):
        QApplication.clipboard().setText(self.sample_cmd)
        QMessageBox.information(self, "已复制", "示例命令已复制到剪贴板，可在命令行终端中粘贴运行。")

    def _open_readme(self):
        readme = get_cli_readme_path()
        if not readme.is_file():
            QMessageBox.warning(
                self,
                "未找到说明书",
                f"未找到文件：\n{readme}\n\n请确认已使用安装包或便携包运行。",
            )
            return
        os.startfile(str(readme))

    def _open_terminal(self):
        bat_path = get_cli_bat_path()
        launcher_path = get_cli_launcher_path()
        install_root = get_install_root()
        if not bat_path.is_file():
            QMessageBox.warning(
                self,
                "未找到启动脚本",
                f"未找到：\n{bat_path}\n\n"
                f"程序目录：{ROOT}\n"
                f"识别到的安装目录：{install_root}\n\n"
                "请确认使用完整安装包运行；若刚更新过，请重新安装最新版本。",
            )
            return
        try:
            work_dir = bat_path.parent
            if launcher_path.is_file():
                subprocess.Popen(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(launcher_path),
                    ],
                    cwd=str(work_dir),
                )
            else:
                subprocess.Popen([str(bat_path)], cwd=str(work_dir))
        except OSError as exc:
            QMessageBox.warning(self, "打开失败", f"无法启动命令行终端：\n{exc}")


class BatchProcessWorker(QThread):
    finished = pyqtSignal(object, float, str, str)
    progress = pyqtSignal(int, int)
    item_done = pyqtSignal(str, object, float, int, int)

    def __init__(self, mode: str, files: list[str]):
        super().__init__()
        self.mode = mode
        self.files = files
        self.cfg = get_config_for_mode(self.mode)

    def run(self):
        if not self.files:
            self.finished.emit(None, 0, "没有找到图片", "")
            return

        total = len(self.files)
        first_file = _resolve_path(self.files[0])
        out_dir = first_file.parent / "output"
        out_dir.mkdir(exist_ok=True)

        for i, f in enumerate(self.files):
            try:
                img = _read_image(f)
                if img is None:
                    continue

                t0 = time.perf_counter()
                out = process_image(img, self.cfg)
                wall = time.perf_counter() - t0
                if not out.success:
                    continue

                core_time = float((out.stage_times or {}).get("除可视化用时", 0.0))
                if core_time <= 0:
                    core_time = wall

                p = _resolve_path(f)
                out_path = out_dir / f"output_{p.name}"
                cv2.imwrite(str(out_path), out.result_bgr)

                self.item_done.emit(f, out.result_bgr, core_time * 1000, i + 1, total)
            except Exception:
                pass
            self.progress.emit(i + 1, total)

        self.finished.emit(None, 0, "DONE", "")

class ProcessWorker(QThread):
    finished = pyqtSignal(object, float, str, str)

    def __init__(self, mode: str, image_path: str):
        super().__init__()
        self.mode = mode
        self.image_path = image_path
        self.cfg = get_config_for_mode(self.mode)

    def run(self):
        try:
            img = _read_image(self.image_path)
            if img is None:
                self.finished.emit(None, 0, "无法读取图片", self.image_path)
                return

            t0 = time.perf_counter()
            out = process_image(img, self.cfg)
            wall = time.perf_counter() - t0
            if not out.success:
                self.finished.emit(None, 0, out.status or "处理失败", self.image_path)
                return

            core_time = float((out.stage_times or {}).get("除可视化用时", 0.0))
            if core_time <= 0:
                core_time = wall
            self.finished.emit(out.result_bgr, core_time * 1000, "", self.image_path)
        except Exception as exc:
            self.finished.emit(None, 0, str(exc), self.image_path)


class ImagePanel(QWidget):
    IMAGE_HEIGHT = 420

    def __init__(self, title: str, placeholder: str, droppable: bool = False, with_download: bool = False):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("color: #333333; font-size: 15px; font-weight: bold;")
        title_row.addWidget(self.lbl_title)
        title_row.addStretch()
        self.btn_download = None
        if with_download:
            self.btn_download = QPushButton("下载")
            self.btn_download.setToolTip("下载当前结果")
            self.btn_download.setFixedHeight(28)
            self.btn_download.setMinimumWidth(52)
            self.btn_download.setEnabled(False)
            self.btn_download.setCursor(Qt.PointingHandCursor)
            self.btn_download.setStyleSheet(
                """
                QPushButton {
                    border: 1px solid #CCCCCC; border-radius: 6px;
                    background: #FFFFFF; color: #0056B3; font-size: 13px; font-weight: 600;
                    padding: 0 10px;
                }
                QPushButton:hover { background: #EBF5FF; border-color: #0056B3; }
                QPushButton:disabled { color: #AAAAAA; background: #F5F5F5; border-color: #E0E0E0; }
                """
            )
            title_row.addWidget(self.btn_download)
        title_host = QWidget()
        title_host.setLayout(title_row)
        title_host.setFixedHeight(28)

        self.image_host = QWidget()
        self.image_host.setFixedHeight(self.IMAGE_HEIGHT)
        self.image_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        host_layout = QVBoxLayout(self.image_host)
        host_layout.setContentsMargins(0, 0, 0, 0)

        if droppable:
            self.lbl_image = DropLabel(placeholder)
        else:
            self.lbl_image = ImageLabel(placeholder)
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        host_layout.addWidget(self.lbl_image)

        self.lbl_info = QLabel("文件: - | 分辨率: - | 用时: -")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setFixedHeight(32)
        self.lbl_info.setStyleSheet("color: #666666; font-size: 13px; font-weight: 500;")

        layout.addWidget(title_host)
        layout.addWidget(self.image_host)
        layout.addWidget(self.lbl_info)

    def setEnabled(self, enabled: bool):
        super().setEnabled(True)
        self.lbl_image.set_interactive(enabled)

    def set_info(self, filename: str, resolution: str, elapsed: str):
        self.lbl_info.setText(f"文件: {filename} | 分辨率: {resolution} | 用时: {elapsed}")

    def show_image(self, img_bgr: np.ndarray):
        self.lbl_image.setPixmap(_bgr_to_qpixmap(img_bgr))
        self.lbl_image.ensure_vivid_display()

    def show_placeholder(self, text: str):
        self.lbl_image.clear_image(text)


class ImageLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap = None
        self._placeholder = text
        self._interactive = True
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "QLabel { background-color: #F8F9FA; border: 1px solid #E0E0E0; "
            "border-radius: 8px; color: #666666; font-size: 15px; }"
        )

    def sizeHint(self):
        return QSize(480, ImagePanel.IMAGE_HEIGHT)

    def minimumSizeHint(self):
        return QSize(200, ImagePanel.IMAGE_HEIGHT)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.setText("")
        self.ensure_vivid_display()
        self._refresh()

    def ensure_vivid_display(self):
        super().setEnabled(True)

    def setEnabled(self, enabled: bool):
        # Qt 禁用 QLabel 会把 pixmap 渲染成灰蒙蒙的「黑白」效果；处理中只锁交互，不锁显示。
        super().setEnabled(True)
        self._interactive = enabled

    def changeEvent(self, event):
        if event.type() == QEvent.EnabledChange and not self.isEnabled():
            super().setEnabled(True)
            self._refresh()
        super().changeEvent(event)

    def _refresh(self):
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        if self._pixmap and not self._pixmap.isNull():
            super().setPixmap(self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            super().setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def clear_image(self, placeholder: str):
        self._pixmap = None
        self._placeholder = placeholder
        self.setText(placeholder)
        super().setPixmap(QPixmap())

    def set_interactive(self, enabled: bool):
        self._interactive = enabled
        self.ensure_vivid_display()
        self._refresh()


class DropLabel(ImageLabel):
    file_dropped = pyqtSignal(str)
    clicked = pyqtSignal()

    def __init__(self, text):
        super().__init__(text)
        self.setAcceptDrops(True)
        self.setStyleSheet(
            """
            QLabel {
                border: 1px dashed #CCCCCC; border-radius: 8px;
                background-color: #F8F9FA; color: #666666; font-size: 15px;
            }
            QLabel:hover { border: 1px dashed #0056B3; background-color: #EBF5FF; color: #0056B3; }
            """
        )

    def dragEnterEvent(self, event):
        if not self._interactive:
            event.ignore()
            return
        event.accept() if event.mimeData().hasUrls() else event.ignore()

    def dropEvent(self, event):
        if not self._interactive:
            event.ignore()
            return
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path) or path.lower().endswith(tuple(IMAGE_EXTS)):
                self.file_dropped.emit(path)

    def mousePressEvent(self, event):
        if not self._interactive:
            return
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class ModernButton(QPushButton):
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(36)
        if primary:
            self.setStyleSheet(
                "QPushButton { background-color: #0056B3; color: white; border: none; "
                "border-radius: 6px; font-size: 14px; font-weight: bold; padding: 0 16px; }"
                "QPushButton:hover { background-color: #004494; }"
                "QPushButton:disabled { background-color: #CCCCCC; color: #888888; }"
            )
        else:
            self.setStyleSheet(
                "QPushButton { background-color: #FFFFFF; color: #333333; border: 1px solid #CCCCCC; "
                "border-radius: 6px; font-size: 14px; padding: 0 16px; }"
                "QPushButton:hover { background-color: #F0F0F0; }"
                "QPushButton:disabled { background-color: #F5F5F5; color: #AAAAAA; }"
            )


class FileListRowWidget(QWidget):
    favorite_toggled = pyqtSignal(str)

    def __init__(self, path: str, favorited: bool, parent=None):
        super().__init__(parent)
        self.path = path
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 10, 2)
        layout.setSpacing(8)
        self.lbl_name = QLabel(Path(path).name)
        self.lbl_name.setStyleSheet("color: #333333; font-size: 13px;")
        self.btn_star = QPushButton("☆")
        self.btn_star.setFlat(True)
        self.btn_star.setFixedSize(34, 34)
        self.btn_star.setCursor(Qt.PointingHandCursor)
        self.btn_star.clicked.connect(lambda: self.favorite_toggled.emit(self.path))
        layout.addWidget(self.lbl_name, stretch=1)
        layout.addWidget(self.btn_star, alignment=Qt.AlignRight | Qt.AlignVCenter)
        self.set_favorited(favorited)

    def set_favorited(self, favorited: bool):
        if favorited:
            self.btn_star.setText("★")
            self.btn_star.setStyleSheet(
                "QPushButton { border: none; background: transparent; font-size: 20px; color: #FFB800; padding: 0; }"
                "QPushButton:hover { color: #E6A600; }"
            )
        else:
            self.btn_star.setText("☆")
            self.btn_star.setStyleSheet(
                "QPushButton { border: none; background: transparent; font-size: 20px; color: #C8C8C8; padding: 0; }"
                "QPushButton:hover { color: #FFB800; }"
            )


class SingleModeWidget(QWidget):
    LIST_STYLE = """
        QListWidget {
            border: 1px solid #E0E0E0; border-radius: 6px;
            background-color: #F8F9FA; font-size: 13px; color: #333333;
        }
        QListWidget::item { padding: 0; margin: 0; border-bottom: 1px solid #EEEEEE; }
        QListWidget::item:selected { background-color: #EBF5FF; color: #0056B3; }
    """

    def __init__(self, parent_app):
        super().__init__()
        self.app = parent_app
        self.current_input_path = _default_image_dir_rel()
        self.image_files: list[str] = []
        self.current_index = 0
        self.worker = None
        self.batch_worker = None
        self._processing = False
        self._batch_processing = False
        self._worker_generation = 0
        self._list_block = False
        self.list_filter = "all"
        self.favorite_paths = _load_favorites()
        self._pending_preview_path: str | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._run_list_preview)
        self.result_cache: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
        self.current_result_bgr = None
        self.current_result_source = None
        self.download_dir: str | None = None
        self.last_output_dir: str | None = None
        self._batch_mode_name: str = "常用模式"
        self._pending_reprocess = False
        self.auto_save = False
        self._init_ui()

    def _init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(12)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("工作目录:"))
        self.lbl_current_path = QLabel(self.current_input_path)
        self.lbl_current_path.setStyleSheet(
            "color: #555; background: #F0F0F0; padding: 6px 10px; border-radius: 6px; border: 1px solid #E0E0E0;"
        )
        self.lbl_current_path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.btn_select_file = ModernButton("选择图片")
        self.btn_select_file.clicked.connect(self._select_image)
        self.btn_select_dir = ModernButton("选择目录")
        self.btn_select_dir.clicked.connect(self._select_directory)
        path_row.addWidget(self.lbl_current_path)
        path_row.addWidget(self.btn_select_file)
        path_row.addWidget(self.btn_select_dir)
        main.addLayout(path_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("处理模式:"))
        self.mode_group = QButtonGroup(self)
        for i, name in enumerate(["常用模式", "高保真模式", "最高质量模式"]):
            rb = QRadioButton(name)
            if i == 0:
                rb.setChecked(True)
            rb.toggled.connect(self._on_mode_changed)
            self.mode_group.addButton(rb, i)
            mode_row.addWidget(rb)
        mode_row.addStretch()
        self.chk_auto_save = QCheckBox("自动保存")
        self.chk_auto_save.setStyleSheet("color: #333333; font-size: 14px;")
        self.chk_auto_save.toggled.connect(lambda c: setattr(self, "auto_save", c))
        mode_row.addWidget(self.chk_auto_save)
        self.btn_process = ModernButton("开始处理", primary=True)
        self.btn_process.clicked.connect(lambda: self._process_current(force=True))
        self.btn_batch = ModernButton("批量去高光")
        self.btn_batch.clicked.connect(self._on_batch_clicked)
        self.btn_open_output = ModernButton("打开输出目录")
        self.btn_open_output.clicked.connect(self._open_output_dir)
        self.btn_open_output.setEnabled(False)
        self.btn_cli_mode = ModernButton("命令行模式")
        self.btn_cli_mode.clicked.connect(self._open_cli_mode_dialog)
        mode_row.addWidget(self.btn_process)
        mode_row.addWidget(self.btn_batch)
        mode_row.addWidget(self.btn_open_output)
        mode_row.addWidget(self.btn_cli_mode)
        main.addLayout(mode_row)

        workspace = QSplitter(Qt.Horizontal)
        sidebar = QWidget()
        sidebar.setMinimumWidth(300)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)

        self.lbl_list_title = QLabel("文件列表 (0)")
        self.lbl_list_title.setStyleSheet("color: #333333; font-size: 14px; font-weight: bold;")
        filter_row = QHBoxLayout()
        self.btn_filter_all = ModernButton("全部")
        self.btn_filter_fav = ModernButton("只看收藏")
        self.btn_filter_all.setCheckable(True)
        self.btn_filter_fav.setCheckable(True)
        self.btn_filter_all.setChecked(True)
        self.btn_filter_all.clicked.connect(lambda: self._set_list_filter("all"))
        self.btn_filter_fav.clicked.connect(lambda: self._set_list_filter("favorites"))
        filter_row.addWidget(self.btn_filter_all)
        filter_row.addWidget(self.btn_filter_fav)
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(self.LIST_STYLE)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.itemSelectionChanged.connect(self._on_list_selection_changed)
        self.list_widget.itemClicked.connect(self._on_list_item_clicked)
        self.list_widget.itemDoubleClicked.connect(lambda _: self._process_current(force=True))
        sidebar_layout.addWidget(self.lbl_list_title)
        sidebar_layout.addLayout(filter_row)
        sidebar_layout.addWidget(self.list_widget, stretch=1)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: #666666; font-size: 12px;")
        sidebar_layout.addWidget(self.lbl_status)

        panels = QSplitter(Qt.Horizontal)
        self.panel_input = ImagePanel("原图预览", "点击或拖入图片/目录", droppable=True)
        self.panel_output = ImagePanel("去高光结果", "等待处理", with_download=True)
        self.panel_input.lbl_image.file_dropped.connect(self._handle_drop)
        self.panel_input.lbl_image.clicked.connect(self._select_image_area_clicked)
        self.panel_output.btn_download.clicked.connect(self._on_download_clicked)
        panels.addWidget(self.panel_input)
        panels.addWidget(self.panel_output)
        panels.setSizes([520, 520])

        workspace.addWidget(sidebar)
        workspace.addWidget(panels)
        workspace.setSizes([320, 1120])
        main.addWidget(workspace, stretch=1)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.hide()
        main.addWidget(self.progress)

    def _busy(self) -> bool:
        return self._processing or self._batch_processing

    def _current_mode(self) -> str:
        btn = self.mode_group.checkedButton()
        return btn.text() if btn else "常用模式"

    def _cache_key(self, path: str, mode: str | None = None) -> tuple[str, str]:
        mode_name = mode or self._current_mode()
        return (path, mode_name)

    def _get_cached_result(self, path: str, mode: str | None = None):
        return self.result_cache.get(self._cache_key(path, mode))

    def _paths_for_list(self) -> list[str]:
        if self.list_filter == "favorites":
            return [p for p in self.image_files if p in self.favorite_paths]
        return list(self.image_files)

    def _populate_file_list(self, select_path: str | None = None):
        self._list_block = True
        self.list_widget.clear()
        visible = self._paths_for_list()
        self.lbl_list_title.setText(f"文件列表 ({len(visible)})")
        select_row = 0
        for i, path in enumerate(visible):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, path)
            item.setSizeHint(QSize(280, 40))
            row = FileListRowWidget(path, path in self.favorite_paths)
            row.favorite_toggled.connect(self._toggle_favorite_path)
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, row)
            if select_path and path == select_path:
                select_row = i
        if visible:
            self.list_widget.setCurrentRow(select_row)
        self._list_block = False

    def _sync_list_selection(self):
        if not self.image_files or self.current_index < 0:
            return
        path = self.image_files[self.current_index]
        self._populate_file_list(select_path=path)

    def _set_list_filter(self, mode: str):
        self.list_filter = mode
        self.btn_filter_all.setChecked(mode == "all")
        self.btn_filter_fav.setChecked(mode == "favorites")
        current = self.image_files[self.current_index] if self.image_files else None
        self._populate_file_list(select_path=current)

    def _toggle_favorite_path(self, path: str):
        if path in self.favorite_paths:
            self.favorite_paths.remove(path)
        else:
            self.favorite_paths.add(path)
        _save_favorites(self.favorite_paths)
        keep = path
        if self.list_filter == "favorites" and path not in self.favorite_paths:
            visible = self._paths_for_list()
            keep = visible[0] if visible else (self.image_files[self.current_index] if self.image_files else None)
        self._populate_file_list(select_path=keep)

    def _schedule_list_preview(self, path: str):
        if self._list_block or self._busy() or path not in self.image_files:
            return
        self._pending_preview_path = path
        self._preview_timer.start(0)

    def _run_list_preview(self):
        path = self._pending_preview_path
        if not path or path not in self.image_files:
            return
        self._show_image_at_index(self.image_files.index(path), auto_process=True)

    def _on_list_selection_changed(self):
        if self._list_block or self._busy():
            return
        items = self.list_widget.selectedItems()
        if not items:
            return
        self._schedule_list_preview(items[0].data(Qt.UserRole))

    def _on_list_item_clicked(self, item):
        if item is None:
            return
        self._schedule_list_preview(item.data(Qt.UserRole))

    def _on_mode_changed(self, checked: bool):
        if not checked or not self.image_files or self._busy():
            return
        path = self.image_files[self.current_index]
        if self._processing:
            self._pending_reprocess = True
            self.panel_output.show_placeholder("处理中...")
            self.panel_output.set_info(default_output_name(path), "-", "处理中...")
            return
        cached = self._get_cached_result(path)
        if cached:
            result, elapsed = cached
            self._show_result(path, result, elapsed)
            return
        self.panel_output.show_placeholder("处理中...")
        self.panel_output.set_info(default_output_name(path), "-", "处理中...")
        self._process_current(force=True)

    def _abort_worker(self):
        self._worker_generation += 1
        self._processing = False
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.wait(100)
        self._batch_processing = False
        self.progress.hide()
        self._apply_busy_ui(False)

    def _load_input_source(self, path: str, auto_process_first: bool = False):
        self._abort_worker()
        stored_path = _normalize_stored_path(path)
        self.current_input_path = stored_path
        self.lbl_current_path.setText(stored_path)
        self.image_files = list_images(stored_path)
        self.current_index = 0
        self.result_cache.clear()
        self.last_output_dir = None
        self.btn_open_output.setEnabled(False)
        self._clear_current_result()
        self._populate_file_list()

        if not self.image_files:
            self.panel_input.show_placeholder("未找到可用图片")
            self.panel_output.show_placeholder("等待处理")
            self.panel_input.set_info("-", "-", "-")
            self.panel_output.set_info("-", "-", "-")
            self.lbl_status.setText("目录中没有图片")
            return

        self.lbl_status.setText(f"已加载 {len(self.image_files)} 张图片")
        self._show_image_at_index(0, auto_process=auto_process_first)

    def _clear_current_result(self):
        self.current_result_bgr = None
        self.current_result_source = None
        if self.panel_output.btn_download:
            self.panel_output.btn_download.setEnabled(False)

    def _show_image_at_index(self, index: int, auto_process: bool = False):
        if not self.image_files:
            return
        if self._processing:
            self._abort_worker()
        self.current_index = max(0, min(index, len(self.image_files) - 1))
        path = self.image_files[self.current_index]
        img = _read_image(path)
        if img is None:
            self.panel_input.show_placeholder("无法读取图片")
            self.panel_input.set_info(Path(path).name, "-", "-")
            return

        h, w = img.shape[:2]
        self.panel_input.show_image(img)
        self.panel_input.set_info(Path(path).name, f"{w}x{h}", "-")
        self._sync_list_selection()

        cached = self._get_cached_result(path)
        if cached:
            result, elapsed = cached
            self._show_result(path, result, elapsed)
            return

        self.panel_output.show_placeholder("处理中..." if auto_process else "点击列表项自动预览")
        self.panel_output.set_info(default_output_name(path), "-", "处理中..." if auto_process else "-")
        self._clear_current_result()

        if auto_process:
            self._process_current()

    def _show_result(self, source_path: str, result_bgr: np.ndarray, elapsed_ms: float):
        self.current_result_bgr = result_bgr
        self.current_result_source = source_path
        self.panel_output.show_image(result_bgr)
        h, w = result_bgr.shape[:2]
        self.panel_output.set_info(default_output_name(source_path), f"{w}x{h}", f"{elapsed_ms:.0f} ms")
        if self.panel_output.btn_download:
            self.panel_output.btn_download.setEnabled(True)

    def _show_batch_sync(self, source_path: str, result_bgr: np.ndarray, elapsed_ms: float):
        """批量处理时左右预览同步更新。"""
        img = _read_image(source_path)
        if img is not None:
            h, w = img.shape[:2]
            self.panel_input.show_image(img)
            self.panel_input.set_info(Path(source_path).name, f"{w}x{h}", "-")
        self._show_result(source_path, result_bgr, elapsed_ms)
        if source_path in self.image_files:
            self.current_index = self.image_files.index(source_path)
            self._sync_list_selection()

    def _select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择包含图片的目录", _dialog_dir(self.current_input_path)
        )
        if dir_path:
            self._load_input_source(dir_path, auto_process_first=True)

    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", _dialog_dir(self.current_input_path),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if path:
            self._load_input_source(path, auto_process_first=True)

    def _select_image_area_clicked(self):
        self._select_directory() if _path_is_dir(self.current_input_path) else self._select_image()

    def _handle_drop(self, path: str):
        self._load_input_source(path, auto_process_first=True)

    def _process_current(self, force: bool = False):
        if not self.image_files or self._processing or self._batch_processing:
            return
        path = self.image_files[self.current_index]
        if not force:
            cached = self._get_cached_result(path)
            if cached:
                result, elapsed = cached
                self._show_result(path, result, elapsed)
                return

        self._processing = True
        self._apply_busy_ui(True)
        self.panel_output.show_placeholder("处理中...")
        self.panel_output.set_info(default_output_name(path), "-", "处理中...")
        self.lbl_status.setText(f"处理中: {Path(path).name}")
        self.progress.show()
        self.progress.setRange(0, 0)

        worker = ProcessWorker(self._current_mode(), path)
        worker.generation = self._worker_generation
        self.worker = worker
        worker.finished.connect(self._on_process_finished)
        worker.start()

    @pyqtSlot(object, float, str, str)
    def _on_process_finished(self, result_bgr, elapsed_ms: float, error_msg: str, source_path: str):
        worker = self.sender()
        if not isinstance(worker, ProcessWorker) or worker.generation != self._worker_generation:
            return

        self.progress.hide()
        self._processing = False

        if source_path in self.image_files:
            self.current_index = self.image_files.index(source_path)

        if error_msg:
            self.panel_output.show_placeholder("处理失败")
            self.panel_output.set_info(default_output_name(source_path), "-", "失败")
            self._clear_current_result()
            self.lbl_status.setText(f"失败: {Path(source_path).name}")
            if self._pending_reprocess:
                self._pending_reprocess = False
                self._process_current(force=True)
            else:
                self._apply_busy_ui(False)
            return

        self.result_cache[self._cache_key(source_path)] = (result_bgr, elapsed_ms)
        self._show_result(source_path, result_bgr, elapsed_ms)
        self.lbl_status.setText(f"完成: {Path(source_path).name} ({elapsed_ms:.0f} ms)")

        if self.auto_save and result_bgr is not None:
            source = _resolve_path(source_path)
            save_path = source.parent / default_output_name(source_path)
            cv2.imwrite(str(save_path), result_bgr)

        if self._pending_reprocess:
            self._pending_reprocess = False
            self._process_current(force=True)
        else:
            self._apply_busy_ui(False)

    def _on_batch_clicked(self):
        if not self.image_files:
            QMessageBox.warning(self, "错误", "当前目录没有可用图片")
            return
        if self._busy():
            return

        pick_dlg = BatchImageSelectDialog(self.image_files, self)
        if pick_dlg.exec_() != QDialog.Accepted:
            return
        selected = pick_dlg.selected_paths()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一张图片")
            return

        mode_dlg = BatchModeDialog(len(selected), self)
        if mode_dlg.exec_() != QDialog.Accepted:
            return
        mode_name = mode_dlg.selected_mode()
        self._start_batch(selected, mode_name)

    def _start_batch(self, files: list[str], mode_name: str):
        self._batch_mode_name = mode_name
        self._batch_processing = True
        self._apply_busy_ui(True)
        self.progress.show()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m")
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.lbl_status.setText(f"批量处理中 ({mode_name})... 0/{len(files)}")

        self.batch_worker = BatchProcessWorker(mode_name, files)
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.item_done.connect(self._on_batch_item_done)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_worker.start()

    def _on_batch_progress(self, current: int, total: int):
        self.progress.setValue(current)
        self.lbl_status.setText(f"批量处理中... {current}/{total}")

    def _on_batch_item_done(self, path: str, result_bgr: np.ndarray, elapsed_ms: float, current: int, total: int):
        self.result_cache[self._cache_key(path, self._batch_mode_name)] = (result_bgr, elapsed_ms)
        self._show_batch_sync(path, result_bgr, elapsed_ms)
        self.lbl_status.setText(
            f"批量处理中 ({self._batch_mode_name})... {current}/{total} | {Path(path).name}"
        )

    def _on_batch_finished(self, res, elapsed, msg, source):
        self._batch_processing = False
        self.progress.hide()
        self.progress.setTextVisible(False)
        self.lbl_status.setText("批量处理完成")
        if self.batch_worker and self.batch_worker.files:
            self.last_output_dir = _output_dir_rel_for_files(self.batch_worker.files)
            self.btn_open_output.setEnabled(_resolve_path(self.last_output_dir).exists())

        out_dir = _display_path(self.last_output_dir or DEFAULT_DATA_DIR_REL)
        box = QMessageBox(self)
        box.setWindowTitle("批量处理完成")
        box.setText(f"所有图片已处理完毕。\n结果保存在:\n{out_dir}")
        btn_open = box.addButton("打开输出目录", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec_()
        if box.clickedButton() is btn_open:
            self._open_output_dir()
        self._apply_busy_ui(False)

    def _open_output_dir(self):
        if self.last_output_dir is None:
            if self.image_files:
                self.last_output_dir = _output_dir_rel_for_files(self.image_files)
            else:
                QMessageBox.information(self, "提示", "还没有输出目录。")
                return
        if not _resolve_path(self.last_output_dir).exists():
            QMessageBox.warning(self, "错误", f"目录不存在:\n{_display_path(self.last_output_dir)}")
            return
        os.startfile(str(_resolve_path(self.last_output_dir)))

    def _open_cli_mode_dialog(self):
        CliModeDialog(self).exec_()

    def _on_download_clicked(self):
        if self.current_result_bgr is None or not self.current_result_source:
            return
        source = _resolve_path(self.current_result_source)
        if self.download_dir is None:
            picked = QFileDialog.getExistingDirectory(self, "选择下载保存目录", str(source.parent))
            if not picked:
                return
            self.download_dir = _normalize_stored_path(picked)
        save_path = _resolve_path(self.download_dir) / default_output_name(self.current_result_source)
        if not cv2.imwrite(str(save_path), self.current_result_bgr):
            QMessageBox.warning(self, "错误", "保存失败，请检查目标路径是否可写")
            return
        QMessageBox.information(self, "下载完成", f"已保存到:\n{_display_path(save_path)}")

    def _refresh_input_display(self):
        if not self.image_files:
            return
        idx = self.current_index
        if idx < 0 or idx >= len(self.image_files):
            return
        img = _read_image(self.image_files[idx])
        if img is not None:
            self.panel_input.show_image(img)

    def _apply_busy_ui(self, busy: bool):
        for btn in self.mode_group.buttons():
            btn.setEnabled(not busy)
        self.btn_process.setEnabled(not busy)
        self.btn_batch.setEnabled(not busy)
        self.btn_select_file.setEnabled(not busy)
        self.btn_select_dir.setEnabled(not busy)
        self.btn_filter_all.setEnabled(not busy)
        self.btn_filter_fav.setEnabled(not busy)
        self.list_widget.setEnabled(not busy)
        self.panel_input.setEnabled(not busy)
        self.panel_input.lbl_image.set_interactive(not busy)
        if busy:
            self.btn_open_output.setEnabled(False)
            if self.panel_output.btn_download:
                self.panel_output.btn_download.setEnabled(False)
        else:
            if self.last_output_dir and _resolve_path(self.last_output_dir).exists():
                self.btn_open_output.setEnabled(True)
            if self.panel_output.btn_download:
                self.panel_output.btn_download.setEnabled(self.current_result_bgr is not None)
        self._refresh_input_display()


class AppLite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("面部去高光 Simple")
        self.setMinimumSize(1480, 780)
        self.resize(1520, 820)
        if APP_ICON_PATH.is_file():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.setStyleSheet("QMainWindow { background-color: #FFFFFF; }")
        self._init_ui()
        if not self._preload_model():
            model_path = ROOT / "models" / "face_landmarker.task"
            detail = get_last_landmarker_error() or f"期望模型路径:\n{model_path}"
            QMessageBox.critical(
                self,
                "启动失败",
                "人脸检测模型未就绪。\n\n"
                f"{detail}\n\n"
                "若使用安装包，请重新运行 FaceHighlight-Simple-Setup 安装；"
                "若杀毒软件拦截，请将安装目录加入白名单。",
            )
            sys.exit(1)
        self._load_defaults()

    def _preload_model(self) -> bool:
        cfg = load_base_config()
        fd = dict(cfg.get("face_detection", {}))
        model_rel = fd.get("mediapipe_task_model_path", "models/face_landmarker.task")
        model_path = (ROOT / model_rel).resolve()
        if not model_path.is_file():
            return False
        fd["mediapipe_task_model_path"] = str(model_path)
        fd["_runtime"] = cfg.get("runtime", {})
        return preload_face_landmarker(fd)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(24, 24, 24, 24)
        main.setSpacing(12)
        self.workspace = SingleModeWidget(self)
        main.addWidget(self.workspace, stretch=1)

    def _load_defaults(self):
        default_path = _default_image_dir_rel()
        self.workspace._load_input_source(default_path, auto_process_first=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if APP_ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    font = QFont()
    font.setFamilies(["PingFang SC", "Microsoft YaHei", "Segoe UI", "Arial", "sans-serif"])
    font.setPointSize(10)
    app.setFont(font)
    window = AppLite()
    window.show()
    sys.exit(app.exec_())
