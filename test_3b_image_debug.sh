# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

export HOME=/home/projects/polyullm/congkai/data/verl
export WANDB_MODE=offline
export RAY_DEDUP_LOGS_ALLOW_REGEX='xieck13'
export NCCL_TIMEOUT=7200000  # 2小时
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
DATA_DIR="$HOME/data/0_1_2_visual_toolbok_v2"
# WANDB_API_KEY=XXX
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='deepeyes_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.max_prompt_length=1536 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="/lustre/projects/polyullm/models/Qwen/Qwen2.5-VL-3B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.profiler.ranks=[0,1] \
    actor_rollout_ref.actor.profiler.all_ranks=True \
    actor_rollout_ref.actor.profiler.discrete=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="deepeyes_multiturn" \
    trainer.experiment_name="Qwen2.5-VL-3B-Tool" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    data.train_files=test.parquet \
    data.val_files=test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/image_zoom_in_tool_config.yaml" \
    trainer.total_epochs=15 $@