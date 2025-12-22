CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
--multi_gpu \
--num_processes=4 \
$(which lerobot-train)  \
--dataset.repo_id=jokeru/pick_and_place \
--policy.type=act \
--output_dir=outputs/train/act_flow \
--job_name=act_pick_place_flow \
--policy.device=cuda \
--wandb.enable=true  \
--policy.repo_id=samithva/act_pick_place_flow \
--batch_size=64  \
--steps=12000 \
--num_workers=96 \
--optimizer.lr=5e-5 \
--log_freq=10