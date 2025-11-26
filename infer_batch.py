#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_local_infer.py
# 以本機環境批次執行 CardiacSegV2/expers/infer.py 的推論
# 與 run_local_train_test.py 參數風格一致

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

def get_tune_model_dir(root_exp_dir: str, exp_name: str) -> str:
    import os, glob
    from ray import tune

    def _project_root():
        pr = os.environ.get("PROJECT_ROOT", "/home/fazonglin/ml2025")
        return pr

    experiment_path = os.path.join(root_exp_dir, exp_name)
    print(f"[INFO] 從 Ray Tune 還原：{experiment_path}")

    restored_tuner = tune.Tuner.restore(experiment_path, trainable="main")
    result_grid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric="tt_dice", mode="max")

    trial_dir = (
        getattr(best_result, "log_dir", None)
        or getattr(best_result, "path", None)
        or getattr(best_result, "artifact_path", None)
        or getattr(best_result, "local_path", None)
    )

    # 1) trial_dir 下的 models
    if trial_dir:
        print(f"[INFO] Trial dir: {trial_dir}")
        for cand in [
            os.path.join(trial_dir, "models"),
            os.path.join(trial_dir, "artifacts", "models"),
        ]:
            if os.path.isdir(cand):
                return cand

    # 2) 掃描 tune_results/<exp_name> 內所有 models/
    print("[WARN] trial_dir 沒有 models/，改掃描實驗資料夾。")
    candidates = [p for p in glob.glob(os.path.join(experiment_path, "**", "models"), recursive=True)
                  if os.path.isdir(p)]
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        print(f"[INFO] 使用掃描到的 models 目錄：{candidates[0]}")
        return candidates[0]

    # 3) 專案根目錄 models/
    pr = _project_root()
    root_models = os.path.join(pr, "models")
    if os.path.isfile(os.path.join(root_models, "best_model.pth")):
        print(f"[INFO] 回退到專案根目錄 models：{root_models}")
        return root_models

    # 4) runs/**/models/
    runs_candidates = [p for p in glob.glob(os.path.join(pr, "runs", "**", "models"), recursive=True)
                       if os.path.isdir(p)]
    if runs_candidates:
        runs_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        print(f"[INFO] 使用 runs 內較新的 models：{runs_candidates[0]}")
        return runs_candidates[0]

    raise RuntimeError(f"在 {experiment_path} 與專案根目錄都找不到任何 models/ 目錄。")

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

    # 與你的訓練腳本相同的根路徑預設
    parser.add_argument("--workspace_dir", type=Path, default=Path("/home/fazonglin/ml2025/CardiacSegV2"))
    parser.add_argument("--project_root", type=Path,  default=Path("/home/fazonglin/ml2025"))

    # 任務設定
    parser.add_argument("--model_name", default="unet3d")
    parser.add_argument("--data_name",  default="chgh")
    parser.add_argument("--exp_name",   default="AICUP_training")

    # 輸入/輸出（預設放在 project_root/myo_pred/<data_name>/...）
    parser.add_argument("--images_dir", type=Path, default=None,
                        help="要推論的影像資料夾（會遞迴掃描所有檔案）")
    parser.add_argument("--infer_dir",  type=Path, default=None,
                        help="推論輸出資料夾")

    # Ray Tune 實驗根目錄（沿用你的結構）
    parser.add_argument("--root_exp_dir", type=Path, default=None,
                        help="exps/exps/<model>/<data>/tune_results")
    # 資料根目錄（infer.py 需求）
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="dataset/<data_name>")

    # （可選）手動指定 checkpoint；若提供則不從 Ray Tune 還原
    parser.add_argument("--checkpoint", type=Path, default=None)

    # infer.py 模型/資料超參（對齊你的訓練預設，可覆寫）
    parser.add_argument("--out_channels", type=int, default=4)
    parser.add_argument("--patch_size",  type=int, default=2)         # 你原推論腳本用 2
    parser.add_argument("--feature_size",type=int, default=48)
    parser.add_argument("--drop_rate",   type=float, default=0.1)
    parser.add_argument("--depths",      type=int, nargs="+", default=[3, 3, 9, 3])
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--exp_rate",    type=int, default=4)
    parser.add_argument("--norm_name",   type=str, default="layer")

    parser.add_argument("--a_min",  type=float, default=-42)
    parser.add_argument("--a_max",  type=float, default=423)
    parser.add_argument("--space_x",type=float, default=0.7)
    parser.add_argument("--space_y",type=float, default=0.7)
    parser.add_argument("--space_z",type=float, default=1.0)
    parser.add_argument("--roi_x",  type=int,   default=128)
    parser.add_argument("--roi_y",  type=int,   default=128)
    parser.add_argument("--roi_z",  type=int,   default=128)

    # flags
    parser.add_argument("--infer_post_process", action="store_true", default=False)

    args = parser.parse_args()

    # ====== 路徑推導 ======
    workspace_dir = args.workspace_dir
    expers_dir    = workspace_dir / "expers"
    infer_py      = expers_dir / "infer.py"
    if not infer_py.exists():
        print(f"找不到 {infer_py}，請確認 --workspace_dir 設定正確。")
        sys.exit(1)

    if args.root_exp_dir is None:
        args.root_exp_dir = workspace_dir / "exps" / "exps" / args.model_name / args.data_name / "tune_results"

    if args.data_dir is None:
        args.data_dir = workspace_dir / "dataset" / args.data_name

    # 輸入/輸出資料夾預設：project_root/myo_pred/<data_name>/image|infer
    base_pred_dir = args.project_root / "myo_pred" / args.data_name
    if args.images_dir is None: args.images_dir = base_pred_dir / "image"
    if args.infer_dir  is None: args.infer_dir  = base_pred_dir / "infer"

    # 準備資料夾
    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.infer_dir.mkdir(parents=True, exist_ok=True)

    # 增加 import 專案路徑
    sys.path.append(str(workspace_dir))

    # ====== 找 checkpoint ======
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        if not checkpoint_path.is_file():
            print(f"[ERROR] 指定的 --checkpoint 不存在：{checkpoint_path}")
            sys.exit(1)
        print(f"[INFO] 使用指定權重：{checkpoint_path}")
    else:
        # 從 Tune 結果取最佳 trial
        models_dir = get_tune_model_dir(args.root_exp_dir, args.exp_name)
        checkpoint_path = models_dir / "best_model.pth"
        if not checkpoint_path.is_file():
            print(f"[ERROR] 找不到 best_model.pth：{checkpoint_path}")
            sys.exit(1)
        print(f"[INFO] 使用 Tune 最佳權重：{checkpoint_path}")

    # ====== 收集影像 ======
    pred_imgs = list_all_files(args.images_dir)
    if not pred_imgs:
        print(f"[WARN] 在 {args.images_dir} 找不到檔案可推論。")
        sys.exit(0)

    # ====== 逐檔推論 ======
    for idx, img_pth in enumerate(pred_imgs, 1):
        print(f"[{idx}/{len(pred_imgs)}] Inference on: {img_pth}")
        cmd = [
            sys.executable, str(infer_py),
            f"--model_name={args.model_name}",
            f"--data_name={args.data_name}",
            f"--data_dir={args.data_dir}",
            f"--model_dir={checkpoint_path.parent}",
            f"--infer_dir={args.infer_dir}",
            f"--checkpoint={checkpoint_path}",
            f"--img_pth={img_pth}",
            f"--out_channels={args.out_channels}",
            f"--patch_size={args.patch_size}",
            f"--feature_size={args.feature_size}",
            f"--drop_rate={args.drop_rate}",
            "--depths", *[str(d) for d in args.depths],
            "--kernel_size", str(args.kernel_size),
            "--exp_rate", str(args.exp_rate),
            "--norm_name", args.norm_name,
            f"--a_min={args.a_min}",
            f"--a_max={args.a_max}",
            f"--space_x={args.space_x}",
            f"--space_y={args.space_y}",
            f"--space_z={args.space_z}",
            f"--roi_x={args.roi_x}",
            f"--roi_y={args.roi_y}",
            f"--roi_z={args.roi_z}",
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
