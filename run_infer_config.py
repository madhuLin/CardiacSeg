#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_infer_from_config.py

使用 experiments_config.EXPERIMENTS 中的設定，
對多個已訓練好的模型做批次推論。

- 每個模型對同一批病人做推論
- 各模型輸出存在：<project_root>/myo_pred/<data_name>/infer/<exp_id>/
- 詳細 log 寫入檔案：<project_root>/logs/infer_logs/infer_<exp_id>.log
- 終端機只顯示簡短進度，適合丟著睡覺跑 😴
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# 讓 Python 找得到 experiments_config.py
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from experiments_config import EXPERIMENTS  # type: ignore


def run(cmd: List[str], log_file: Path):
    """
    執行一個命令，stdout/stderr 寫入 log 檔案。
    """
    print("[CMD]", " ".join(cmd))
    with log_file.open("a") as f:
        f.write("\n>>> CMD: " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        proc.wait()
        if proc.returncode != 0:
            print(f"[WARN] 指令失敗，更多細節請看：{log_file}")


def collect_images(root: Path) -> List[Path]:
    imgs: List[Path] = []
    for r, _, fs in os.walk(root):
        for name in fs:
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                imgs.append(Path(r) / name)
    imgs.sort()
    return imgs


def main():
    parser = argparse.ArgumentParser(description="Run inference for multiple models from EXPERIMENTS config.")

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
    parser.add_argument("--data_name", type=str, default="chgh")

    # 指定要跑哪幾個實驗 id（就是 EXPERIMENTS 的 key）
    parser.add_argument(
        "--exp_ids",
        nargs="*",
        default=None,
        help="要推論的實驗 id（EXPERIMENTS 的 key），預設：跑所有 enabled=True 且有模型的。",
    )

    # 影像來源：病人的 .nii / .nii.gz
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=None,
        help="要推論的影像路徑（遞迴掃描 .nii / .nii.gz）。預設：<project_root>/myo_pred/<data_name>/image",
    )

    # 推論輸出根目錄
    parser.add_argument(
        "--infer_root",
        type=Path,
        default=None,
        help="各模型推論結果的根目錄（裡面會再加 <exp_id> 子資料夾）。預設：<project_root>/myo_pred/<data_name>/infer",
    )

    parser.add_argument(
        "--infer_post_process",
        action="store_true",
        default=True,
        help="是否啟用 infer.py 的後處理（建議 True）",
    )

    # 一般不變的參數（跟訓練對齊）
    parser.add_argument("--out_channels", type=int, default=4)
    parser.add_argument("--a_min", type=float, default=-42)
    parser.add_argument("--a_max", type=float, default=423)
    parser.add_argument("--space_x", type=float, default=0.7)
    parser.add_argument("--space_y", type=float, default=0.7)
    parser.add_argument("--space_z", type=float, default=1.0)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)

    # 方便 debug：只跑前 N 個 case
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="只推論前 N 個影像（不設代表全部）。",
    )

    args = parser.parse_args()

    workspace_dir = args.workspace_dir
    expers_dir = workspace_dir / "expers"
    infer_py = expers_dir / "infer.py"
    if not infer_py.exists():
        print(f"[ERROR] 找不到 {infer_py}，請確認 --workspace_dir 是否正確。")
        sys.exit(1)

    data_dir = workspace_dir / "dataset" / args.data_name

    # 預設 images_dir / infer_root
    if args.images_dir is None:
        args.images_dir = args.project_root / "myo_pred" / args.data_name / "image"
    if args.infer_root is None:
        args.infer_root = args.project_root / "myo_pred" / args.data_name / "infer"

    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.infer_root.mkdir(parents=True, exist_ok=True)

    # 收集要推論的影像
    all_imgs = collect_images(args.images_dir)
    if not all_imgs:
        print(f"[WARN] 在 {args.images_dir} 找不到 .nii / .nii.gz，先確認路徑。")
        sys.exit(0)

    if args.max_cases is not None:
        all_imgs = all_imgs[: args.max_cases]

    print(f"[INFO] 共 {len(all_imgs)} 個影像會被推論。")

    # 要跑哪些 exp_id
    if args.exp_ids:
        exp_ids = args.exp_ids
    else:
        # 預設：EXPERIMENTS 裡 enabled=True 的都跑
        exp_ids = [k for k, v in EXPERIMENTS.items() if v.get("vote", True)]

    print("[INFO] 本次要推論的實驗：", exp_ids)

    logs_dir = args.project_root / "logs" / "infer_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 每個實驗 id 依序處理
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"[WARN] exp_id={exp_id} 不在 EXPERIMENTS 裡，跳過。")
            continue

        cfg = EXPERIMENTS[exp_id]
        if not cfg.get("enabled", True):
            print(f"[INFO] exp_id={exp_id} enabled=False，跳過。")
            continue

        model_name = cfg["model_name"]
        params: Dict[str, Any] = cfg["params"]
        print(f"\n========== Inference for {exp_id} ({model_name}) ==========")

        # 找 checkpoint：<project_root>/models/<exp_id>/best_model.pth
        model_dir = args.project_root / "models" / exp_id
        checkpoint_path = model_dir / "best_model.pth"
        if not checkpoint_path.is_file():
            print(f"[WARN] 找不到 checkpoint: {checkpoint_path}，略過此模型。")
            continue

        infer_dir = args.infer_root / exp_id
        infer_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"infer_{exp_id}.log"
        log_file.write_text(f"[INFO] Inference log for {exp_id}\n", encoding="utf-8")

        # 逐病人跑 infer.py
        for idx, img_pth in enumerate(all_imgs, 1):
            print(f"[{exp_id}] [{idx}/{len(all_imgs)}] {img_pth.name}")

            cmd = [
                sys.executable,
                str(infer_py),
                f"--model_name={model_name}",
                f"--data_name={args.data_name}",
                f"--data_dir={data_dir}",
                f"--model_dir={model_dir}",
                f"--infer_dir={infer_dir}",
                f"--checkpoint={checkpoint_path}",
                f"--img_pth={img_pth}",
                f"--out_channels={args.out_channels}",
                f"--patch_size={params.get('patch_size', 2)}",
                f"--feature_size={params.get('feature_size', 32)}",
                f"--drop_rate={params.get('drop_rate', 0.0)}",
                "--depths",
                *[str(d) for d in params.get("depths", [2, 2, 2, 2])],
                "--kernel_size",
                str(params.get("kernel_size", 3)),
                "--exp_rate",
                str(params.get("exp_rate", 4)),
                "--norm_name",
                params.get("norm_name", "instance"),
                f"--a_min={args.a_min}",
                f"--a_max={args.a_max}",
                f"--space_x={args.space_x}",
                f"--space_y={args.space_y}",
                f"--space_z={args.space_z}",
                f"--roi_x={args.roi_x}",
                f"--roi_y={args.roi_y}",
                f"--roi_z={params.get('roi_z', 112)}",
            ]
            if args.infer_post_process:
                cmd.append("--infer_post_process")

            run(cmd, log_file)

    print("\n[INFO] 所有模型推論結束。")
    print("[INFO] 推論結果在：", args.infer_root)
    print("[INFO] log 檔在：", logs_dir)


if __name__ == "__main__":
    main()


# python run_infer_config.py --data_name chgh --infer_post_process
# AICUP_swinunetr