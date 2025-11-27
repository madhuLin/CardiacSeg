#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_kfold_splits.py

把一份「50 病人」的 data_dict JSON，切成 K 個 fold，
輸出成：
  AICUP_training_50_fold1.json
  AICUP_training_50_fold2.json
  ...
"""

import json
import random
from pathlib import Path
import argparse
from collections import defaultdict


def extract_patient_id(path: str) -> str:
    """
    從影像路徑推 patient id，例如:
      .../patient0033_image.nii.gz -> patient0033
      .../patient0033.nii.gz       -> patient0033
    你可以依實際檔名微調這裡。
    """
    name = Path(path).name
    stem = name
    for ext in [".nii.gz", ".nii"]:
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break

    # 常見命名：patientXXXX_*.nii.gz 或 patientXXXX.nii.gz
    # 這裡簡單找出含 "patient" 的前綴
    if "patient" in stem:
        idx = stem.index("patient")
        # 取到下一個 '_' 或整個尾巴
        sub = stem[idx:]
        if "_" in sub:
            sub = sub.split("_")[0]
        return sub

    # fallback：直接用 stem
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace_dir",
        type=Path,
        default=Path("/home/fazonglin/ml2025/CardiacSegV2"),
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="chgh",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="AICUP_training_50.json",
        help="原本的 training JSON 檔名（位於 exps/data_dicts/<data_name>/ 底下）",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=5,
        help="fold 數（預設 5-fold）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
    )
    args = parser.parse_args()

    data_dict_dir = (
        args.workspace_dir / "exps" / "data_dicts" / args.data_name
    )
    src_json = data_dict_dir / args.input_json

    print("[INFO] 讀取 JSON:", src_json)
    data = json.loads(src_json.read_text())

    # 假設格式是：
    # {
    #   "training": [ { "image": "...", "label": "..." }, ... ],
    #   "validation": [ ... ]  # 有可能沒有
    #   ...
    # }
    train_items = data.get("train", [])
    if not train_items:
        print("[ERROR] 此 JSON 中沒有 'train' 欄位，請檢查格式。")
        return

    # 依 patient 分組
    patient_to_items = defaultdict(list)
    for item in train_items:
        img_path = item.get("image") or item.get("img") or ""
        pid = extract_patient_id(img_path)
        patient_to_items[pid].append(item)

    patients = sorted(patient_to_items.keys())
    print(f"[INFO] 總病人數: {len(patients)}")
    random.seed(args.seed)
    random.shuffle(patients)

    k = args.kfold
    folds = [[] for _ in range(k)]
    for i, pid in enumerate(patients):
        folds[i % k].append(pid)

    for fold_idx in range(k):
        val_pids = set(folds[fold_idx])
        train_pids = set(patients) - val_pids

        fold_train = []
        fold_val = []

        for pid in train_pids:
            fold_train.extend(patient_to_items[pid])
        for pid in val_pids:
            fold_val.extend(patient_to_items[pid])

        out = {}
        out["train"] = fold_train
        out["val"] = fold_val

        # 如果原 JSON 還有其他欄位（像 test），維持原樣
        for key, val in data.items():
            if key in ["train", "val"]:
                continue
            out[key] = val

        out_name = f"AICUP_training_50_fold{fold_idx+1}.json"
        out_path = data_dict_dir / out_name
        out_path.write_text(json.dumps(out, indent=2))
        print(
            f"[INFO] 寫出 fold{fold_idx+1}: "
            f"train={len(fold_train)}, val={len(fold_val)} -> {out_path}"
        )

    print("[INFO] K-fold JSON 產生完成 ✅")


if __name__ == "__main__":
    main()
