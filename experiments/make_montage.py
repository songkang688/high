"""生成 before/after 对比拼图（人脸 ROI 裁剪，横向拼接）。

用法：.venv/bin/python experiments/make_montage.py 6 17 base drm06 ...
  位置参数里数字是样本名，其他是变体名；输出到 experiments/out/montage/。
  --full 输出整脸原始分辨率；默认缩到高 420px 方便入库。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, imread, load_face  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

MONT_DIR = OUT_DIR / "montage"


def face_crop_box(name: str, shape, pad: float = 0.18):
    face = load_face(name)
    x, y, w, h = face.bbox
    H, W = shape[:2]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    return x0, y0, x1, y1


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main(argv) -> int:
    names = [a for a in argv if a.isdigit()]
    variants = [a for a in argv if not a.isdigit() and a != "--full"]
    full = "--full" in argv
    target_h = None if full else 420
    MONT_DIR.mkdir(parents=True, exist_ok=True)
    for n in names:
        img = imread(n)
        x0, y0, x1, y1 = face_crop_box(n, img.shape)
        tiles = [label(img[y0:y1, x0:x1], "original")]
        for v in variants:
            p = OUT_DIR / "variants" / v / f"{n}.png"
            if not p.is_file():
                p = OUT_DIR / "audit" / f"{n}_result.png"
            res = cv2.imread(str(p))
            tiles.append(label(res[y0:y1, x0:x1], v))
        if target_h:
            tiles = [cv2.resize(t, (int(t.shape[1] * target_h / t.shape[0]), target_h)) for t in tiles]
        mont = np.hstack(tiles)
        out = MONT_DIR / f"{n}_{'_'.join(variants)}.jpg"
        cv2.imwrite(str(out), mont, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
