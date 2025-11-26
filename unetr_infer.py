#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_local_infer.py
# 以本機環境批次執行 CardiacSegV2/expers/infer.py 的推論
# 支援預設的 UNETR / UNET3D preset，比較好記、比較不會拿錯 checkpoint

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List

# ========= 這裡定義你常用的模型 preset =========
# key = preset 名稱，執行時用 --preset 選
MODEL_PRESETS = {
    # UNETR - 對應你訓練腳本裡的 exp_name
    "unetr_f32_d2-3-6-3_lr2e-4": {
        "model_name":   "unetr",
        "exp_name":     "AICUP_unetr_f32_d2-3-6-3_lr2e-4",
        "patch_size":   16,
        "feature_size": 32,
        "drop_rate":    0.1,
        "depths":       [2, 3, 6, 3],
        "kernel_size":  7,
        "exp_rate":     4,
        "norm_name":    "layer",
        "roi_z":        112,
        "lr":           2e-4,
        "weight_decay": 1e-5,
    },
    "unetr_f48_d3-3-9-3_lr1e-4": {
        "model_name":   "unetr",
        "exp_name":     "AICUP_unetr_f48_d3-3-9-3_lr1e-4",
        "patch_size":   16,
        "feature_size": 48,
        "drop_rate":    0.1,
        "depths":       [3, 3, 9, 3],
        "kernel_size":  7,
        "exp_rate":     4,
        "norm_name":    "layer",
        "roi_z":        112,
        "lr":           1e-4,
        "weight_decay": 1e-4,
    },
    "unetr_f64_d3-4-10-3_lr5e-5": {
        "model_name":   "unetr",
        "exp_name":     "AICUP_unetr_f64_d3-4-10-3_lr5e-5",
        "patch_size":   16,
        "feature_size": 64,
        "drop_rate":    0.1,
        "depths":       [3, 4, 10, 3],
        "kernel_size":  7,
        "exp_rate":     4,
        "norm_name":    "layer",
        "roi_z":        112,
        "lr":           5e-5,
        "weight_decay": 1e-4,
    },
    # 如果之後有 unet3d，這裡可以再加一個 preset
    # "unet3d_baseline": {...}
}


def list_all_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for r, _, fs in os.walk(root):
        for name in fs:
            files.append(Path(r) / name)
    files.sort()
    return files


def run(cmd_list):
    print("\n>>> 執行指令：")
    print(" ".join([str(c) for c in cmd_list]), "\n")
    subprocess.run(cmd_list, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run CardiacSegV2 inference locally.")

    # 根路徑（跟訓練腳本一致）
    parser.add_argument(
        "--workspace_dir",
        type=Path,
        default=Path("/home/fazonglin/ml2025/CardiacSegV2"),
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path("/home/fazonglin/ml2025"),
    )

    # 資料設定
    parser.add_argument("--data_name", default="chgh")

    # 選擇要用哪個 preset（上面 MODEL_PRESETS 的 key）
    parser.add_argument(
        "--preset",
        type=str,
        default="unetr_f48_d3-3-9-3_lr1e-4",
        choices=list(MODEL_PRESETS.keys()),
        help="選擇要用哪一個模型 preset（上面 MODEL_PRESETS 的 key）",
    )

    # 輸入/輸出（預設放在 project_root/myo_pred/<data_name>/...）
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=None,
        help="要推論的影像資料夾（會遞迴掃描所有檔案）",
    )
    parser.add_argument(
        "--infer_dir",
        type=Path,
        default=None,
        help="推論輸出資料夾（會再自動加上 preset 子資料夾）",
    )

    # 資料根目錄（infer.py 需求）
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="dataset/<data_name>",
    )

    # （可選）手動指定 checkpoint；若提供則不依照 preset 的 exp_name 找
    parser.add_argument("--checkpoint", type=Path, default=None)

    # 其他通用超參（影像 normalize / spacing / ROI XY 一般不變）
    parser.add_argument("--out_channels", type=int, default=4)
    parser.add_argument("--a_min", type=float, default=-42)
    parser.add_argument("--a_max", type=float, default=423)
    parser.add_argument("--space_x", type=float, default=0.7)
    parser.add_argument("--space_y", type=float, default=0.7)
    parser.add_argument("--space_z", type=float, default=1.0)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)

    # flags
    parser.add_argument(
        "--infer_post_process",
        action="store_true",
        default=True,
        help="是否啟用 infer.py 的後處理（建議開啟）",
    )

    args = parser.parse_args()

    # ====== 檢查 infer.py 是否存在 ======
    workspace_dir = args.workspace_dir
    expers_dir = workspace_dir / "expers"
    infer_py = expers_dir / "infer.py"
    if not infer_py.exists():
        print(f"找不到 {infer_py}，請確認 --workspace_dir 設定正確。")
        sys.exit(1)

    # ====== 套用 preset 的設定 ======
    preset_cfg = MODEL_PRESETS[args.preset]

    model_name = preset_cfg["model_name"]
    exp_name = preset_cfg["exp_name"]
    patch_size = preset_cfg["patch_size"]
    feature_size = preset_cfg["feature_size"]
    drop_rate = preset_cfg["drop_rate"]
    depths = preset_cfg["depths"]
    kernel_size = preset_cfg["kernel_size"]
    exp_rate = preset_cfg["exp_rate"]
    norm_name = preset_cfg["norm_name"]
    roi_z = preset_cfg["roi_z"]

    print(f"[INFO] 使用 preset: {args.preset}")
    print(f"[INFO] model_name={model_name}, exp_name={exp_name}")

    # ====== 路徑推導 ======
    if args.data_dir is None:
        args.data_dir = workspace_dir / "dataset" / args.data_name

    base_pred_dir = args.project_root / "myo_pred" / args.data_name
    if args.images_dir is None:
        args.images_dir = base_pred_dir / "image"
    if args.infer_dir is None:
        # 爲了不同模型不互相覆蓋，在 infer_dir 底下再加一層 preset 名稱
        args.infer_dir = base_pred_dir / "infer" / args.preset

    # 準備資料夾
    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.infer_dir.mkdir(parents=True, exist_ok=True)

    # 增加 import 專案路徑
    sys.path.append(str(workspace_dir))

    # ====== 決定 checkpoint 路徑 ======
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        if not checkpoint_path.is_file():
            print(f"[ERROR] 指定的 --checkpoint 不存在：{checkpoint_path}")
            sys.exit(1)
        model_dir = checkpoint_path.parent
        print(f"[INFO] 使用指定權重：{checkpoint_path}")
    else:
        # 依照訓練腳本的規則：models/<exp_name>/best_model.pth
        model_dir = args.project_root / "models" / exp_name
        checkpoint_path = model_dir / "best_model.pth"
        if not checkpoint_path.is_file():
            print(f"[ERROR] 找不到 checkpoint：{checkpoint_path}")
            sys.exit(1)
        print(f"[INFO] 使用預設權重：{checkpoint_path}")

    # ====== 收集影像 ======
    pred_imgs = list_all_files(args.images_dir)
    if not pred_imgs:
        print(f"[WARN] 在 {args.images_dir} 找不到檔案可推論。")
        sys.exit(0)

    print(f"[INFO] 共 {len(pred_imgs)} 個影像要推論。輸出會存到：{args.infer_dir}")

    # ====== 逐檔推論 ======
    for idx, img_pth in enumerate(pred_imgs, 1):
        print(f"[{idx}/{len(pred_imgs)}] Inference on: {img_pth}")
        cmd = [
            sys.executable,
            str(infer_py),
            f"--model_name={model_name}",
            f"--data_name={args.data_name}",
            f"--data_dir={args.data_dir}",
            f"--model_dir={model_dir}",
            f"--infer_dir={args.infer_dir}",
            f"--checkpoint={checkpoint_path}",
            f"--img_pth={img_pth}",
            f"--out_channels={args.out_channels}",
            f"--patch_size={patch_size}",
            f"--feature_size={feature_size}",
            f"--drop_rate={drop_rate}",
            "--depths",
            *[str(d) for d in depths],
            "--kernel_size",
            str(kernel_size),
            "--exp_rate",
            str(exp_rate),
            "--norm_name",
            norm_name,
            f"--a_min={args.a_min}",
            f"--a_max={args.a_max}",
            f"--space_x={args.space_x}",
            f"--space_y={args.space_y}",
            f"--space_z={args.space_z}",
            f"--roi_x={args.roi_x}",
            f"--roi_y={args.roi_y}",
            f"--roi_z={roi_z}",
        ]
        if args.infer_post_process:
            cmd.append("--infer_post_process")

        try:
            run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 推論失敗：{img_pth}\n{e}")
            # 不中止，繼續下一張
            continue

    print("\n[INFO] 批次推論完成。輸出目錄：", args.infer_dir)


if __name__ == "__main__":
    main()
