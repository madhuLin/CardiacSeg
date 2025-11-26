#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ensemble_vote.py

對多個模型的 segmentation 結果做 voxel-wise 投票集成（majority vote）。

假設單一模型輸出路徑：
    <project_root>/myo_pred/<data_name>/infer/<exp_id>/patientXXXX.nii.gz

本腳本會：
    - 讀取多個 exp_id 的預測
    - 對每個 patient 做 majority vote
    - 輸出到：
        <project_root>/myo_pred/<data_name>/infer/ensemble_<ensemble_name>/patientXXXX.nii.gz
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict

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
                # 去掉所有副檔名：.nii / .gz
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
    # 取得所有 label range（假設從 0 開始）
    max_label = int(volumes.max())
    num_classes = max_label + 1

    # votes shape: (C, H, W, D)
    votes = np.zeros((num_classes,) + volumes.shape[1:], dtype=np.int16)

    for c in range(num_classes):
        votes[c] = (volumes == c).sum(axis=0)

    ensemble = np.argmax(votes, axis=0)
    return ensemble.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Ensemble 多模型 segmentation（majority vote）")

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

    # 要參與投票的 exp_id（就是 models 跟 infer 的子資料夾名稱）
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

    args = parser.parse_args()

    if args.infer_root is None:
        args.infer_root = args.project_root / "myo_pred" / args.data_name / "infer"

    if args.ensemble_name is None:
        # 預設把 exp_ids 拼起來當名字，避免太長就簡單壓縮一下
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

    # 2. 找出「所有模型都有」的 patient 交集
    all_case_sets = [set(files.keys()) for files in exp_files.values()]
    common_cases = set.intersection(*all_case_sets)
    common_cases = sorted(common_cases)

    if args.max_cases is not None:
        common_cases = common_cases[: args.max_cases]

    if not common_cases:
        print("[ERROR] 沒有共同的 patient 檔案可以 ensemble，請檢查各模型 infer 輸出。")
        sys.exit(1)

    print(f"[INFO] 共同的 case 數量：{len(common_cases)}")
    # 列出前幾個示意
    print("[INFO] 例如：", ", ".join(common_cases[:5]), ("..." if len(common_cases) > 5 else ""))

    # 3. 逐個 patient 做 majority vote
    for idx, case_id in enumerate(common_cases, 1):
        print(f"[{idx}/{len(common_cases)}] Ensemble case: {case_id}")

        vols = []
        ref_img = None

        for exp_id, files in exp_files.items():
            p = files[case_id]
            img = nib.load(str(p))
            data = img.get_fdata().astype(np.int16)  # 假設已經是 label mask
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

        # shape: (M, H, W, D)
        stack = np.stack(vols, axis=0)
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
  --ensemble_name smalAll

python run_ensemble_vote.py \
  --data_name chgh \
  --exp_ids \
    AICUP_unet3d_base \
    AICUP_attention_unet \
    AICUP_vnet \
    AICUP_unetr_pp \
    AICUP_dynunet \
    AICUP_unetr_light \
    AICUP_swinunetr \
    AICUP_uxnet_small \
    unetr_f48_d3-3-9-3_lr1e-4 \
  --ensemble_name all8
  
  
  --weight AICUP_unet3d_base=1.2 \
  --weight AICUP_unetr_light=1.5 \
  """