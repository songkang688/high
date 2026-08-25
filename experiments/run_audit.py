"""A 部分：对 data/1..17 用「常用模式」全量检测审计。

- 完整跑 cli_process.load_cli_config("常用模式") + process_image；
- 记录人脸检测 / 高光检测 / 修复质量全指标；
- 保存 landmarks 到 experiments/out/landmarks/ 供 B 部分实验复用；
- 保存 result / hard / soft mask 到 experiments/out/audit/。

用法：.venv/bin/python experiments/run_audit.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    NAMES,
    OUT_DIR,
    component_count,
    dump_json,
    eval_partition_masks,
    imread,
    lap_var,
    residual_shine,
    save_face,
    zone_report,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cli_process import load_cli_config, preload_models  # noqa: E402
from highlight_removal.pipeline import process_image  # noqa: E402
from highlight_removal.utils import mask_to_uint8  # noqa: E402

AUDIT_DIR = OUT_DIR / "audit"


def main() -> int:
    cfg = load_cli_config("常用模式")
    if not preload_models(cfg):
        return 3
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in NAMES:
        img = imread(name)
        t0 = time.perf_counter()
        out = process_image(img, cfg)
        elapsed = time.perf_counter() - t0

        row: dict = {"样本": name, "尺寸": f"{img.shape[1]}x{img.shape[0]}", "耗时s": round(elapsed, 2)}
        if not out.success or out.face is None:
            row.update({"检测成功": False, "状态": out.status, "警告": out.warnings})
            rows.append(row)
            print(f"== {name}: 检测失败 {out.status}")
            continue

        face = out.face
        save_face(name, face)
        hard = mask_to_uint8(out.highlight_mask)
        soft = mask_to_uint8(out.soft_mask)
        regions = out.regions
        face_mask = mask_to_uint8(regions.masks["face_mask"]) > 0
        treatable = mask_to_uint8(regions.masks.get("treatable_skin", regions.masks["skin"]))
        protect = mask_to_uint8(regions.masks["protect"]) > 0

        yaw_warn = [w for w in out.warnings if "侧脸" in w]
        row.update({
            "检测成功": True,
            "detector": face.detector,
            "置信度": round(float(face.confidence), 4),
            "关键点数": 0 if face.landmarks is None else int(len(face.landmarks)),
            "检出人脸数": int(out.detection_count),
            "多脸": bool(out.detection_count > 1),
            "侧脸警告": yaw_warn[0] if yaw_warn else "",
        })

        face_px = max(1, int(face_mask.sum()))
        row["高光"] = {
            "hard占人脸": round(float((hard > 0).sum() / face_px), 6),
            "soft占人脸": round(float((soft > 0).sum() / face_px), 6),
            "连通域数": component_count(hard),
            "hard∩protect像素": int(((hard > 0) & protect).sum()),
        }

        zones = eval_partition_masks(img, face, cfg.get("regions", {}))
        row["分区点亮"] = {k: zone_report(img, z, soft) for k, z in zones.items()}

        row["quality_metrics"] = dict(out.metrics)
        row["警告"] = list(out.warnings)
        row["阶段耗时"] = {k: round(float(v), 3) for k, v in out.stage_times.items()}

        # 修复效果：残留油光 / 纹理 / 色偏
        gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(out.result_bgr, cv2.COLOR_BGR2GRAY)
        rs_pre = residual_shine(img, hard, treatable)
        rs_post = residual_shine(out.result_bgr, hard, treatable)
        lv0 = lap_var(gray0, hard)
        lv1 = lap_var(gray1, hard)
        row["修复"] = {
            "残留dL_处理前": rs_pre["residual_dl"],
            "残留dL_处理后": rs_post["residual_dl"],
            "纹理LapVar_前": round(lv0, 2),
            "纹理LapVar_后": round(lv1, 2),
            "纹理保留比": round(lv1 / lv0, 4) if lv0 > 0 else 1.0,
        }

        cv2.imwrite(str(AUDIT_DIR / f"{name}_result.png"), out.result_bgr)
        cv2.imwrite(str(AUDIT_DIR / f"{name}_hard.png"), hard)
        cv2.imwrite(str(AUDIT_DIR / f"{name}_soft.png"), soft)
        np.savez_compressed(
            AUDIT_DIR / f"{name}_masks.npz",
            hard=hard, soft=soft,
            treatable=treatable.astype(np.uint8),
            protect=protect.astype(np.uint8),
            face_mask=face_mask.astype(np.uint8),
        )

        print(
            f"== {name}: conf={row['置信度']} hard占脸={row['高光']['hard占人脸']:.4f} "
            f"连通域={row['高光']['连通域数']} 残留dL {rs_pre['residual_dl']:+.2f}→{rs_post['residual_dl']:+.2f} "
            f"纹理比={row['修复']['纹理保留比']:.3f} 警告数={len(out.warnings)}"
        )
        rows.append(row)

    dump_json(OUT_DIR / "audit.json", rows)
    print(f"\n审计完成，结果写入 {OUT_DIR / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
