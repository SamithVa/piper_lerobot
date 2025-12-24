# CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch \
#   --multi_gpu \
#   --num_processes=4 \
#   $(which lerobot-train) \
#   --dataset.repo_id=samithva/pick_and_place \
#   --policy.type=act \
#   --output_dir=outputs/train/act_with_language \
#   --job_name=act_finetune_with_language \
#   --policy.device=cuda \
#   --wandb.enable=true \
#   --policy.repo_id=samitva/act_with_language \
#   --batch_size=128 \
#   --steps=12_000 \
#   --num_workers=32 \
#   --policy.use_text_conditioning=true \
#   --policy.text_encoder_model=distilbert-base-uncased

CUDA_VISIBLE_DEVICES=2 lerobot-train \
  --dataset.repo_id=samithva/pick_and_place \
  --policy.type=act \
  --output_dir=outputs/train/act_with_language \
  --job_name=act_finetune_with_language \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=samitva/act_with_language \
  --batch_size=128 \
  --steps=12_000 \
  --num_workers=32 \
  --policy.use_text_conditioning=true \
  --policy.text_encoder_model=distilbert-base-uncased \
  --log_freq=50