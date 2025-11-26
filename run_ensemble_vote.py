#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ensemble_vote.py

對多個模型的 segmentation 結果做 voxel-wise 投票集成。

支援：
- 一般 majority vote
- 模型加權 weighted vote（依照每個 exp_id 的 Dice 等）
- 類別加權（例如 myocardium class 權重加大）

推論結果路徑假設：
    <project_root>/myo_pred/<data_name>/infer/<exp_id>/patientXXXX.nii.gz

輸出：
    <project_root>/myo_pred/<data_name>/infer/ensemble_<ensemble_name>/patientXXXX.nii.gz
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import nibabel as nib


def list_nii_files(root: Path) -> Dict[str, Path]:
    """
    遞迴掃描 root 下所有 .nii / .nii.gz 檔，
    回傳 dict: {basename_without_ext: full_path}
    例如 patient0001.nii.gz -> key: patient0001
    """
    result: Dict[str, Path] = {}
    for r, _, fs in os.walk(root):
        for name in fs:
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                p = Path(r) / name
                stem = p.name
                for ext in [".nii.gz", ".nii"]:
                    if stem.endswith(ext):
                        stem = stem[: -len(ext)]
                        break
                result[stem] = p
    return result


def majority_vote(volumes: np.ndarray) -> np.ndarray:
    """
    volumes: shape (M, H, W, D) ，M 個模型的 hard label(pred)。
    回傳：ensemble label，shape (H, W, D)

    做法：對每一個 class 計票，再 argmax。
    假設 label 是 0,1,2,3... 的整數。
    """
    max_label = int(volumes.max())
    num_classes = max_label + 1

    votes = np.zeros((num_classes,) + volumes.shape[1:], dtype=np.int16)

    for c in range(num_classes):
        votes[c] = (volumes == c).sum(axis=0)

    ensemble = np.argmax(votes, axis=0)
    return ensemble.astype(np.uint8)


def weighted_vote(volumes: np.ndarray,
                  model_weights: np.ndarray,
                  class_weights: np.ndarray | None = None) -> np.ndarray:
    """
    volumes: shape (M, H, W, D)
    model_weights: shape (M,)
    class_weights: shape (C,) or None

    先對模型做加權投票，再乘上類別權重。
    """
    M, H, W, D = volumes.shape
    max_label = int(volumes.max())
    num_classes = max_label + 1

    if class_weights is None:
        class_weights = np.ones((num_classes,), dtype=np.float32)
    else:
        # 長度對不上的話，補 1 或截斷
        cw = np.ones((num_classes,), dtype=np.float32)
        up_to = min(num_classes, len(class_weights))
        cw[:up_to] = class_weights[:up_to]
        class_weights = cw

    # votes shape: (C, H, W, D)
    votes = np.zeros((num_classes, H, W, D), dtype=np.float32)
    mw = model_weights.reshape(-1, 1, 1, 1)  # (M,1,1,1)

    for c in range(num_classes):
        mask = (volumes == c)  # (M,H,W,D)
        # 模型權重加總，再乘類別權重
        votes[c] = (mask * mw).sum(axis=0) * class_weights[c]

    ensemble = np.argmax(votes, axis=0)
    return ensemble.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Ensemble 多模型 segmentation（majority / weighted vote）")

    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path("/home/fazonglin/ml2025"),
        help="專案根目錄（預設：/home/fazonglin/ml2025）",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="chgh",
        help="資料集名稱（會用在 myo_pred/<data_name>）",
    )

    # 參與投票的 exp_id
    parser.add_argument(
        "--exp_ids",
        nargs="+",
        required=True,
        help="要參與 ensemble 的實驗 id（例如：AICUP_unet3d_base AICUP_attention_unet ...）",
    )

    parser.add_argument(
        "--ensemble_name",
        type=str,
        default=None,
        help="ensemble 輸出子資料夾名稱（預設：join_exp_id，例如 unet3d+attn+dyn）",
    )

    parser.add_argument(
        "--infer_root",
        type=Path,
        default=None,
        help="各模型推論結果根目錄（預設：<project_root>/myo_pred/<data_name>/infer）",
    )

    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="只 ensemble 前 N 個 case（debug 用；預設全部）",
    )

    # ⭐ 模型權重：可以多次指定，例如
    # --weight AICUP_unet3d_base=1.2 --weight AICUP_attention_unet=1.5
    parser.add_argument(
        "--weight",
        dest="weights",
        action="append",
        default=[],
        help="模型權重，格式：exp_id=weight，可重複指定多個（沒指定的模型預設 1.0）",
    )

    # ⭐ 類別權重：例如 (bg, LV, MYO, RV) -> 1 1 2 1
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs="+",
        default=None,
        help="類別權重列表 [w0 w1 w2 ...]，長度對不上會自動補 1 或截斷",
    )

    args = parser.parse_args()

    # 路徑處理
    if args.infer_root is None:
        args.infer_root = args.project_root / "myo_pred" / args.data_name / "infer"

    if args.ensemble_name is None:
        short = "+".join(args.exp_ids)
        if len(short) > 40:
            short = "+".join([e.split("_")[1] if "_" in e else e for e in args.exp_ids])
        args.ensemble_name = short

    out_dir = args.infer_root / f"ensemble_{args.ensemble_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] project_root:", args.project_root)
    print("[INFO] data_name   :", args.data_name)
    print("[INFO] infer_root  :", args.infer_root)
    print("[INFO] exp_ids     :", args.exp_ids)
    print("[INFO] ensemble out:", out_dir)

    # 解析模型權重字串 exp_id=weight
    weight_map: Dict[str, float] = {}
    for w in args.weights:
        try:
            name, val = w.split("=")
            name = name.strip()
            val = float(val.strip())
            weight_map[name] = val
        except Exception as e:
            print(f"[WARN] 無法解析 weight='{w}'，格式應為 exp_id=weight，將略過。錯誤：{e}")

    # 1. 讀取每個 exp_id 的檔案列表
    exp_files: Dict[str, Dict[str, Path]] = {}
    for exp_id in args.exp_ids:
        exp_dir = args.infer_root / exp_id
        if not exp_dir.exists():
            print(f"[WARN] 推論資料夾不存在，略過：{exp_dir}")
            continue
        files = list_nii_files(exp_dir)
        if not files:
            print(f"[WARN] 在 {exp_dir} 找不到任何 .nii 檔，略過此模型。")
            continue
        exp_files[exp_id] = files
        print(f"[INFO] {exp_id}: {len(files)} files")

    if len(exp_files) < 2:
        print("[ERROR] 參與投票的有效模型數量 < 2，無法做 ensemble。")
        sys.exit(1)

    # 2. 建立有效模型列表 + 對應 model_weights
    active_exp_ids = list(exp_files.keys())
    print("[INFO] 實際參與投票的模型:", active_exp_ids)

    model_weights = np.ones((len(active_exp_ids),), dtype=np.float32)
    for i, exp_id in enumerate(active_exp_ids):
        if exp_id in weight_map:
            model_weights[i] = weight_map[exp_id]

    use_weighted = (args.weights or args.class_weights is not None)
    if use_weighted:
        # 正規化模型權重
        model_weights = model_weights / (model_weights.sum() + 1e-8)
        print("[INFO] 使用 weighted vote")
        print("[INFO] 模型權重 (norm):")
        for eid, w in zip(active_exp_ids, model_weights):
            print(f"       {eid}: {w:.4f}")
    else:
        print("[INFO] 使用一般 majority vote（沒有指定 --weight / --class_weights）")

    class_weights = None
    if args.class_weights is not None:
        class_weights = np.array(args.class_weights, dtype=np.float32)
        print("[INFO] 類別權重(原始):", class_weights)

    # 3. 找出「所有模型都有」的 patient 交集
    all_case_sets = [set(files.keys()) for files in exp_files.values()]
    common_cases = set.intersection(*all_case_sets)
    common_cases = sorted(common_cases)

    if args.max_cases is not None:
        common_cases = common_cases[: args.max_cases]

    if not common_cases:
        print("[ERROR] 沒有共同的 patient 檔案可以 ensemble，請檢查各模型 infer 輸出。")
        sys.exit(1)

    print(f"[INFO] 共同的 case 數量：{len(common_cases)}")
    print("[INFO] 例如：", ", ".join(common_cases[:5]), ("..." if len(common_cases) > 5 else ""))

    # 4. 逐個 patient 做 ensemble
    for idx, case_id in enumerate(common_cases, 1):
        print(f"[{idx}/{len(common_cases)}] Ensemble case: {case_id}")

        vols = []
        ref_img = None
        ref_shape = None

        for exp_id in active_exp_ids:
            files = exp_files[exp_id]
            p = files[case_id]
            img = nib.load(str(p))
            data = img.get_fdata().astype(np.int16)

            if ref_img is None:
                ref_img = img
                ref_shape = data.shape
            else:
                if data.shape != ref_shape:
                    raise RuntimeError(
                        f"Shape mismatch for case {case_id} between models. "
                        f"{exp_id} has shape {data.shape}, expected {ref_shape}"
                    )
            vols.append(data)

        stack = np.stack(vols, axis=0)  # (M, H, W, D)

        if use_weighted:
            ensemble = weighted_vote(stack, model_weights, class_weights)
        else:
            ensemble = majority_vote(stack)

        out_path = out_dir / f"{case_id}.nii.gz"
        ensemble_img = nib.Nifti1Image(ensemble, affine=ref_img.affine, header=ref_img.header)
        nib.save(ensemble_img, str(out_path))

    print("\n[INFO] Ensemble 完成 ✅")
    print("[INFO] 輸出目錄：", out_dir)


if __name__ == "__main__":
    main()


"""
python run_ensemble_vote.py \
  --data_name chgh \
  --exp_ids \
    AICUP_attention_unet \
    AICUP_dynunet \
    AICUP_swinunetr \
    AICUP_uxnet_small \
  --ensemble_name myo_focus \
  --weight AICUP_attention_unet=0.83909 \
  --weight AICUP_dynunet=0.83786 \
  --weight AICUP_swinunetr=0.83494 \
  --weight AICUP_uxnet_small=0.82502 \
  --class_weights 1.0 1.0 2.0 1.0
  
  
python run_ensemble_vote.py \
  --data_name chgh \
  --exp_ids \
    AICUP_attention_unet \
    AICUP_dynunet \
    AICUP_swinunetr \
    AICUP_uxnet_small \
  --ensemble_name myo_focus \
  --class_weights 1.0 1.0 2.0 1.0
"""