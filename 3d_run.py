#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_local_train_test.py
# 以本機環境執行 CardiacSegV2/expers/tune.py 的訓練與測試
# 將原本 Colab 版本（含 !python、/content/...）改為可執行的 .py 腳本

import os
import sys
import argparse
import subprocess
from pathlib import Path

jsonFile = "AICUP_training_50_1.json"
# jsonFile = "AICUP_training_3.json"


def build_common_args(args, for_test: bool = False):
    """
    產生共用 CLI 參數清單；flag 參數僅在為 True 時加入
    """
    l = [
        f"--exp_name={args.exp_name}",
        f"--data_name={args.data_name}",
        f"--data_dir={str(args.data_dir)}",
        f"--root_exp_dir={str(args.root_exp_dir)}",
        f"--model_name={args.model_name}",
        f"--model_dir={str(args.model_dir)}",
        f"--log_dir={str(args.log_dir)}",
        f"--eval_dir={str(args.eval_dir)}",
        f"--data_dicts_json={str(args.data_dicts_json)}",
        "--out_channels", str(args.out_channels),
        "--patch_size", str(args.patch_size),
        "--feature_size", str(args.feature_size),
        "--drop_rate", str(args.drop_rate),
        "--depths", *[str(x) for x in args.depths],
        "--kernel_size", str(args.kernel_size),
        "--exp_rate", str(args.exp_rate),
        "--norm_name", args.norm_name,
        "--a_min", str(args.a_min),
        "--a_max", str(args.a_max),
        "--space_x", str(args.space_x),
        "--space_y", str(args.space_y),
        "--space_z", str(args.space_z),
        "--roi_x", str(args.roi_x),
        "--roi_y", str(args.roi_y_test if for_test else args.roi_y),
        "--roi_z", str(args.roi_z_test if for_test else args.roi_z),
        "--optim", args.optim,
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        f"--checkpoint={str(args.final_checkpoint)}",
    ]
    # flags（沒有值的參數）
    if args.pin_memory:
        l.append("--pin_memory")
    if args.use_init_weights:
        l.append("--use_init_weights")
    if args.infer_post_process:
        l.append("--infer_post_process")
    if for_test:
        if args.resume_tuner:
            l.append("--resume_tuner")
        if args.save_eval_csv:
            l.append("--save_eval_csv")
        if args.test_mode:
            l.append("--test_mode")
    return l


def run(cmd_list):
    print("\n>>> 執行指令：")
    print(" ".join([str(c) for c in cmd_list]), "\n")
    subprocess.run(cmd_list, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run CardiacSegV2 train & test locally.")

    # 根路徑（依你貼的資料夾結構）
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

    # 任務/資料設定
    parser.add_argument("--data_name", default="chgh")
    parser.add_argument("--exp_name", default="AICUP_training")
    parser.add_argument("--data_dict_file_name", default=jsonFile)
    # 路徑（可自動由上面推導）
    parser.add_argument("--root_exp_dir", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--data_dicts_json", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--eval_dir", type=Path, default=None)

    # 訓練相關
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--max_early_stop_count", type=int, default=10)
    parser.add_argument("--max_epoch", type=int, default=200)

    # 模型超參（UNETR 會用到的部分）
    parser.add_argument("--out_channels", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--feature_size", type=int, default=64)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 10, 3])
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--exp_rate", type=int, default=4)
    parser.add_argument("--norm_name", type=str, default="layer")
    parser.add_argument("--a_min", type=float, default=-42)
    parser.add_argument("--a_max", type=float, default=423)
    parser.add_argument("--space_x", type=float, default=0.7)
    parser.add_argument("--space_y", type=float, default=0.7)
    parser.add_argument("--space_z", type=float, default=1.0)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)
    parser.add_argument("--roi_z", type=int, default=120)
    # 測試時 ROI（你的原碼 test 用 z=128）
    parser.add_argument("--roi_y_test", type=int, default=128)
    parser.add_argument("--roi_z_test", type=int, default=128)

    # 優化器
    parser.add_argument("--optim", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # flags
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--use_init_weights", action="store_true", default=True)
    parser.add_argument("--infer_post_process", action="store_true", default=True)

    # 測試 flags
    parser.add_argument("--resume_tuner", action="store_true", default=True)
    parser.add_argument("--save_eval_csv", action="store_true", default=True)
    parser.add_argument("--test_mode", action="store_true", default=True)

    # unet3d / swinunetr / attention_unet / cotr / unetr
    parser.add_argument("--model_name", default="unetr")

    args = parser.parse_args()

    # ====== 檢查 tune.py 是否存在 ======
    tune_py = args.workspace_dir / "expers" / "tune.py"
    if not tune_py.exists():
        print(f"找不到 {tune_py}，請確認 --workspace_dir 設定正確。")
        sys.exit(1)

    # ====== UNETR 多組超參設定 ======
    candidate_configs = [
        {
            "tag": "f32_d2-3-6-3_lr2e-4",
            "feature_size": 32,
            "depths": [2, 3, 6, 3],
            "drop_rate": 0.1,
            "lr": 2e-4,
            "weight_decay": 1e-5,
        },
        {
            "tag": "f48_d3-3-9-3_lr1e-4",
            "feature_size": 48,
            "depths": [3, 3, 9, 3],
            "drop_rate": 0.1,
            "lr": 1e-4,
            "weight_decay": 1e-4,
        },
        {
            "tag": "f64_d3-4-10-3_lr5e-5",
            "feature_size": 64,
            "depths": [3, 4, 10, 3],
            "drop_rate": 0.1,
            "lr": 5e-5,
            "weight_decay": 1e-4,
        },
    ]

    # ====== 逐組 config 執行訓練 ======
    for i, cfg in enumerate(candidate_configs):
        print(f"\n===== Run UNETR config {i}: {cfg['tag']} =====")

        # 固定使用 UNETR
        args.model_name = "unetr"

        # 套用當前 config 的超參
        args.feature_size = cfg["feature_size"]
        args.depths = cfg["depths"]
        args.drop_rate = cfg["drop_rate"]
        args.lr = cfg["lr"]
        args.weight_decay = cfg["weight_decay"]

        # 每組實驗用不同 exp_name，避免覆蓋
        args.exp_name = f"AICUP_unetr_{cfg['tag']}"

        # root_exp_dir：沿用原本規則（Ray Tune 會在底下再分資料夾）
        if args.root_exp_dir is None:
            args.root_exp_dir = (
                args.workspace_dir
                / "exps"
                / "exps"
                / args.model_name
                / args.data_name
                / "tune_results"
            )

        # 資料集與 data_dicts 路徑（同一份 json split）
        if args.data_dir is None:
            args.data_dir = args.workspace_dir / "dataset" / args.data_name
        if args.data_dicts_json is None:
            args.data_dicts_json = (
                args.workspace_dir
                / "exps"
                / "data_dicts"
                / args.data_name
                / args.data_dict_file_name
            )

        # 每組實驗分開存 model / log / eval
        model_root = args.project_root / "models"
        log_root = args.project_root / "logs"
        eval_root = args.project_root / "evals"

        args.model_dir = model_root / args.exp_name
        args.log_dir = log_root / args.exp_name
        args.eval_dir = eval_root / args.exp_name

        # 檢查與建立資料夾
        for p in [args.model_dir, args.log_dir, args.eval_dir, args.root_exp_dir]:
            os.makedirs(p, exist_ok=True)

        # 權重檔路徑（供參考；實際存檔由 tune.py 控制）
        args.best_checkpoint = args.model_dir / "best_model.pth"
        args.final_checkpoint = args.model_dir / "final_model.pth"

        # ====== 執行訓練 ======
        train_cmd = [
            sys.executable,
            str(tune_py),
            "--tune_mode=train",
            f"--start_epoch={args.start_epoch}",
            f"--val_every={args.val_every}",
            f"--max_early_stop_count={args.max_early_stop_count}",
            f"--max_epoch={args.max_epoch}",
        ] + build_common_args(args, for_test=False)

        run(train_cmd)

        # 如果你之後確定某一組是最好的，再單獨開啟 test 模式就好
        # test_cmd = [sys.executable, str(tune_py), "--tune_mode=test"] + build_common_args(args, for_test=True)
        # run(test_cmd)


# export PYTHONPATH=$PYTHONPATH:/home/fazonglin/ml2025/CardiacSegV2
if __name__ == "__main__":
    main()
