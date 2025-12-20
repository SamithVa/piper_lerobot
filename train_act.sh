CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
--multi_gpu \
--num_processes=2 \
$(which lerobot-train)  \
--dataset.repo_id=jokeru/pick_and_place \
--policy.type=act \
--output_dir=outputs/train/act_pick_place_vit_dino \
--job_name=act_pick_place_vit_dino \
--policy.device=cuda \
--wandb.enable=true  \
--policy.repo_id=samithva/act_pick_place_vit_dino \
--batch_size=64  \
--steps=12000 \
--num_workers=96 \
--optimizer.lr=5e-5