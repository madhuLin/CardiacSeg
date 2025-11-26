cd /home/fazonglin/ml2025

python infer_batch.py \
  --checkpoint /home/fazonglin/ml2025/models/best_model.pth \
  --images_dir /home/fazonglin/ml2025/myo_pred/chgh/image \
  --infer_dir  /home/fazonglin/ml2025/myo_pred/chgh/infer \
  --infer_post_process
