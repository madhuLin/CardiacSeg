#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_models.py
讀取 experiments_config.EXPERIMENTS，一次把多個模型的訓練排隊跑完。
每個實驗輸出會寫到獨立 log 檔，不塞爆命令列。
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# --- 確保可以 import 同資料夾的 experiments_config.py ---
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiments_config import EXPERIMENTS  # noqa: E402


# ========== 把公共 CLI 參數包成一個小工具 ==========
def build_common_args(args):
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
        "--roi_y", str(args.roi_y),
        "--roi_z", str(args.roi_z),
        "--optim", args.optim,
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        f"--checkpoint={str(args.final_checkpoint)}",
    ]
    if args.pin_memory:
        l.append("--pin_memory")
    if args.use_init_weights:
        l.append("--use_init_weights")
    if args.infer_post_process:
        l.append("--infer_post_process")
    return l


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple CardiacSegV2 experiments (all models) from config."
    )

    # 根路徑（依你目前結構）
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

    # 資料 / split
    parser.add_argument("--data_name", default="chgh")
    parser.add_argument(
        "--data_dict_file_name",
        default="AICUP_training_50_1.json",  # 或你小資料 AICUP_training_3.json
    )
    
    parser.add_argument(
        "--kfold",
        type=int,
        default=1,
        help="使用 K-fold training。k>1 時，會自動載入 AICUP_training_50_foldX.json，並在 exp_name 後加 _fX。",
    )


    # 訓練相關共用預設（可被每個 experiment override）
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--max_early_stop_count", type=int, default=10)
    parser.add_argument("--max_epoch", type=int, default=200)

    # 通用超參
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
    parser.add_argument("--roi_z", type=int, default=112)

    # 優化器
    parser.add_argument("--optim", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # flags
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--use_init_weights", action="store_true", default=True)
    parser.add_argument("--infer_post_process", action="store_true", default=True)

    # 選擇要跑哪些實驗：
    parser.add_argument(
        "--exp_ids",
        nargs="+",
        default=None,
        help="要跑哪些實驗 id，對應 EXPERIMENTS 的 key。不指定就跑所有 enabled=True 的。",
    )

    args = parser.parse_args()

    # ===== 檢查 tune.py =====
    tune_py = args.workspace_dir / "expers" / "tune.py"
    if not tune_py.exists():
        print(f"[ERROR] 找不到 {tune_py}，請確認 --workspace_dir 設定正確。")
        sys.exit(1)

    # ===== 路徑推導 =====
    args.data_dir = args.workspace_dir / "dataset" / args.data_name
    args.data_dicts_json = (
        args.workspace_dir
        / "exps"
        / "data_dicts"
        / args.data_name
        / args.data_dict_file_name
    )

    model_root = args.project_root / "models"
    log_root = args.project_root / "logs"
    eval_root = args.project_root / "evals"
    cli_log_root = args.project_root / "logs_cli"  # 存 command line log
    cli_log_root.mkdir(parents=True, exist_ok=True)

    # Ray Tune 的實驗根目錄：把不同模型都丟在 mixed_models 下面
    args.root_exp_dir = (
        args.workspace_dir
        / "exps"
        / "exps"
        / "mixed_models"
        / args.data_name
        / "tune_results"
    )
    args.root_exp_dir.mkdir(parents=True, exist_ok=True)

    # 要跑哪些實驗
    if args.exp_ids is None:
        exp_ids = [k for k, v in EXPERIMENTS.items() if v.get("enabled", True)]
    else:
        exp_ids = args.exp_ids

    print("[INFO] 將執行的實驗：", exp_ids)

    # base env：降低 Ray 輸出
    base_env = os.environ.copy()
    base_env["AIR_VERBOSITY"] = "1"  # 0/1 都可以，比較安靜
    # 確保 CardiacSegV2 在 PYTHONPATH 裡
    ws_str = str(args.workspace_dir)
    base_env["PYTHONPATH"] = (
        ws_str
        + ":" + base_env.get("PYTHONPATH", "")
        if base_env.get("PYTHONPATH")
        else ws_str
    )

    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"[WARN] exp_id={exp_id} 不在 EXPERIMENTS 裡，跳過。")
            continue

        cfg = EXPERIMENTS[exp_id]
        model_name = cfg["model_name"]
        params = cfg.get("params", {})

        # k-fold 外層
        k = max(1, args.kfold)
        for fold_idx in range(1, k + 1):
            # 如果是 k-fold，就換不同 data_dict JSON & exp_name
            if k > 1:
                data_dict_name = f"AICUP_training_50_fold{fold_idx}.json"
                exp_name = f"{exp_id}_f{fold_idx}"
            else:
                data_dict_name = args.data_dict_file_name
                exp_name = exp_id

            args.exp_name = exp_name
            args.model_name = model_name

            # data_dicts_json 路徑
            args.data_dicts_json = (
                args.workspace_dir
                / "exps"
                / "data_dicts"
                / args.data_name
                / data_dict_name
            )

            # 用 config 覆蓋共用設定
            for k_param, v in params.items():
                if hasattr(args, k_param):
                    setattr(args, k_param, v)

            # 每個實驗自己的 model/log/eval 資料夾
            args.model_dir = model_root / exp_name
            args.log_dir = log_root / exp_name
            args.eval_dir = eval_root / exp_name
            for p in [args.model_dir, args.log_dir, args.eval_dir]:
                p.mkdir(parents=True, exist_ok=True)

            args.best_checkpoint = args.model_dir / "best_model.pth"
            args.final_checkpoint = args.model_dir / "final_model.pth"

            train_cmd = [
                sys.executable,
                str(tune_py),
                "--tune_mode=train",
                f"--start_epoch={args.start_epoch}",
                f"--val_every={args.val_every}",
                f"--max_early_stop_count={args.max_early_stop_count}",
                f"--max_epoch={params.get('max_epoch', args.max_epoch)}",
            ] + build_common_args(args)

            log_file = cli_log_root / f"{exp_name}.log"

            print(f"\n===== [RUN] {exp_name} | model={model_name} =====")
            print(f"[INFO] 使用 data_dicts_json: {args.data_dicts_json}")
            print(f"[INFO] log 檔案：{log_file}")

            with open(log_file, "w") as lf:
                try:
                    subprocess.run(
                        train_cmd,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        env=base_env,
                        check=True,
                    )
                    print(f"[OK] {exp_name} 完成（詳細請看 log）")
                except subprocess.CalledProcessError:
                    print(f"[ERROR] {exp_name} 訓練失敗，請看 {log_file}")
                    continue

    print("\n[INFO] 所有實驗都已嘗試執行完畢。")


if __name__ == "__main__":
    main()

"""
python run_all_models.py \
  --data_name chgh \
  --kfold 5
  
  python train_k.py \
--data_name chgh \
  --exp_ids AICUP_uxnet_small \
  --kfold 5
"""