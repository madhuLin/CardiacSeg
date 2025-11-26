#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_local_train_preset.py
# 用「preset」方式訓練不同模型：unet3d / attention_unet / unetr / swinunetr / cotr
# 不再手動塞一大串 CLI 參數

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ========= 這裡定義你常用的「訓練 preset」 =========
# 每個 preset 同時決定：
# - model_name
# - exp_name（會用來決定 logs / models / evals 目錄）
# - 模型超參（patch_size, feature_size, depths, ...）
# - 訓練超參（lr, weight_decay, max_epoch, val_every, ...）

TRAIN_PRESETS = {
    # ---- 1) UNET3D：穩定 baseline ----
    "unet3d_f48": {
        "model_name": "unet3d",
        "exp_name":   "AICUP_unet3d_f48",
        "patch_size": 2,
        "feature_size": 48,
        "drop_rate":  0.1,
        "depths":     [3, 3, 9, 3],
        "kernel_size": 7,
        "exp_rate":    4,
        "norm_name":   "layer",
        "roi_z":       120,

        "lr":            1e-4,
        "weight_decay":  1e-5,
        "max_epoch":     250,
        "val_every":     5,
        "max_early_stop_count": 20,
    },

    # ---- 2) Attention UNet ----
    "attunet_f48": {
        "model_name": "attention_unet",
        "exp_name":   "AICUP_attunet_f48",
        "patch_size": 2,
        "feature_size": 48,
        "drop_rate":  0.1,
        "depths":     [3, 3, 9, 3],
        "kernel_size": 7,
        "exp_rate":    4,
        "norm_name":   "layer",
        "roi_z":       120,

        "lr":            1e-4,
        "weight_decay":  1e-5,
        "max_epoch":     250,
        "val_every":     5,
        "max_early_stop_count": 20,
    },

    # ---- 3) UNETR（你已經在用的設定，留一個乾淨版）----
    "unetr_f64_d3-4-10": {
        "model_name": "unetr",
        "exp_name":   "AICUP_unetr_f64_d3-4-10",
        "patch_size": 16,
        "feature_size": 64,
        "drop_rate":  0.1,
        "depths":     [3, 4, 10, 3],
        "kernel_size": 7,
        "exp_rate":    4,
        "norm_name":   "layer",
        "roi_z":       112,

        "lr":            1e-4,
        "weight_decay":  1e-4,
        "max_epoch":     250,
        "val_every":     2,
        "max_early_stop_count": 10,
    },

    # ---- 4) SwinUNETR：比較重，先給小一點的設定 ----
    "swinunetr_small": {
        "model_name": "swinunetr",
        "exp_name":   "AICUP_swinunetr_small",
        "patch_size": 2,          # swinunetr 通常用小 patch
        "feature_size": 48,
        "drop_rate":  0.1,
        "depths":     [2, 2, 2, 2],  # 比較小的 encoder 深度
        "kernel_size": 3,
        "exp_rate":    4,
        "norm_name":   "layer",
        "roi_z":       96,        # 稍微小一點，避免 OOM

        "lr":            1e-4,
        "weight_decay":  1e-4,
        "max_epoch":     250,
        "val_every":     2,
        "max_early_stop_count": 15,
    },

    # ---- 5) CoTr：如果你之後想玩可以打開 ----
    "cotr_f32": {
        "model_name": "cotr",
        "exp_name":   "AICUP_cotr_f32",
        "patch_size": 2,
        "feature_size": 32,
        "drop_rate":  0.1,
        "depths":     [2, 3, 6, 3],
        "kernel_size": 7,
        "exp_rate":    4,
        "norm_name":   "layer",
        "roi_z":       112,

        "lr":            2e-4,
        "weight_decay":  1e-5,
        "max_epoch":     150,
        "val_every":     5,
        "max_early_stop_count": 20,
    },
}


def build_common_args(args, preset_cfg, for_test: bool = False):
    """
    把 args + preset_cfg 組成傳給 tune.py 的 CLI 參數
    """
    l = [
        f"--exp_name={preset_cfg['exp_name']}",
        f"--data_name={args.data_name}",
        f"--data_dir={str(args.data_dir)}",
        f"--root_exp_dir={str(args.root_exp_dir)}",
        f"--model_name={preset_cfg['model_name']}",
        f"--model_dir={str(args.model_dir)}",
        f"--log_dir={str(args.log_dir)}",
        f"--eval_dir={str(args.eval_dir)}",
        f"--data_dicts_json={str(args.data_dicts_json)}",
        "--out_channels", str(args.out_channels),
        "--patch_size",  str(preset_cfg["patch_size"]),
        "--feature_size",str(preset_cfg["feature_size"]),
        "--drop_rate",   str(preset_cfg["drop_rate"]),
        "--depths",      *[str(x) for x in preset_cfg["depths"]],
        "--kernel_size", str(preset_cfg["kernel_size"]),
        "--exp_rate",    str(preset_cfg["exp_rate"]),
        "--norm_name",   preset_cfg["norm_name"],
        "--a_min",       str(args.a_min),
        "--a_max",       str(args.a_max),
        "--space_x",     str(args.space_x),
        "--space_y",     str(args.space_y),
        "--space_z",     str(args.space_z),
        "--roi_x",       str(args.roi_x),
        "--roi_y",       str(args.roi_y),
        "--roi_z",       str(preset_cfg["roi_z"]),
        "--optim",       args.optim,
        "--lr",          str(preset_cfg["lr"]),
        "--weight_decay",str(preset_cfg["weight_decay"]),
        f"--checkpoint={str(args.final_checkpoint)}",
    ]

    # flags
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
    parser = argparse.ArgumentParser(description="Run CardiacSegV2 train with presets.")

    # 路徑
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

    # 資料與 split
    parser.add_argument("--data_name", default="chgh")
    parser.add_argument("--data_dict_file_name", default="AICUP_training_smal.json")

    # 選擇 preset
    parser.add_argument(
        "--preset",
        type=str,
        default="unet3d_f48",
        choices=list(TRAIN_PRESETS.keys()),
        help="選擇要訓練的模型 preset",
    )

    # optional override: 你想改 max_epoch 或 lr 也可以額外給
    parser.add_argument("--max_epoch_override", type=int, default=None)
    parser.add_argument("--lr_override", type=float, default=None)

    # 路徑（可以不給，自動推）
    parser.add_argument("--root_exp_dir", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--data_dicts_json", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--eval_dir", type=Path, default=None)

    # 通用訓練相關
    parser.add_argument("--start_epoch", type=int, default=0)

    # flag / 其他超參
    parser.add_argument("--out_channels", type=int, default=4)
    parser.add_argument("--a_min", type=float, default=-42)
    parser.add_argument("--a_max", type=float, default=423)
    parser.add_argument("--space_x", type=float, default=0.7)
    parser.add_argument("--space_y", type=float, default=0.7)
    parser.add_argument("--space_z", type=float, default=1.0)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)

    parser.add_argument("--optim", type=str, default="AdamW")

    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--use_init_weights", action="store_true", default=True)
    parser.add_argument("--infer_post_process", action="store_true", default=True)

    # 測試 flags（先保留，之後若要跑 test 可以打開）
    parser.add_argument("--resume_tuner", action="store_true", default=True)
    parser.add_argument("--save_eval_csv", action="store_true", default=True)
    parser.add_argument("--test_mode", action="store_true", default=True)

    args = parser.parse_args()

    preset_cfg = TRAIN_PRESETS[args.preset]

    # 若有 override，就改掉 preset 裡的值
    if args.max_epoch_override is not None:
        preset_cfg["max_epoch"] = args.max_epoch_override
    if args.lr_override is not None:
        preset_cfg["lr"] = args.lr_override

    # ====== 路徑推導 ======
    tune_py = args.workspace_dir / "expers" / "tune.py"
    if not tune_py.exists():
        print(f"找不到 {tune_py}，請確認 --workspace_dir 設定正確。")
        sys.exit(1)

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

    if args.root_exp_dir is None:
        args.root_exp_dir = (
            args.workspace_dir
            / "exps"
            / "exps"
            / preset_cfg["model_name"]
            / args.data_name
            / "tune_results"
        )

    # model/log/eval 路徑：每個 preset 一個資料夾
    model_root = args.project_root / "models"
    log_root = args.project_root / "logs"
    eval_root = args.project_root / "evals"

    exp_name = preset_cfg["exp_name"]
    if args.model_dir is None:
        args.model_dir = model_root / exp_name
    if args.log_dir is None:
        args.log_dir = log_root / exp_name
    if args.eval_dir is None:
        args.eval_dir = eval_root / exp_name

    for p in [args.model_dir, args.log_dir, args.eval_dir, args.root_exp_dir]:
        os.makedirs(p, exist_ok=True)

    # checkpoint 路徑
    args.best_checkpoint = args.model_dir / "best_model.pth"
    args.final_checkpoint = args.model_dir / "final_model.pth"

    # ====== 組訓練指令 ======
    train_cmd = [
        sys.executable,
        str(tune_py),
        "--tune_mode=train",
        f"--start_epoch={args.start_epoch}",
        f"--val_every={preset_cfg['val_every']}",
        f"--max_early_stop_count={preset_cfg['max_early_stop_count']}",
        f"--max_epoch={preset_cfg['max_epoch']}",
    ] + build_common_args(args, preset_cfg, for_test=False)

    run(train_cmd)

    # ====== 如果之後要自動接 test，可以在這裡打開 ======
    # test_cmd = [sys.executable, str(tune_py), "--tune_mode=test"] + build_common_args(args, preset_cfg, for_test=True)
    # run(test_cmd)


if __name__ == "__main__":
    main()
