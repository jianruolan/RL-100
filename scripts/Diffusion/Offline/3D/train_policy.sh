# Examples:
# bash scripts/train_policy.sh rl100 adroit_door_medium 0112 100
# bash scripts/train_policy.sh rl100 adroit_door_medium 0112 100
# bash scripts/train_policy.sh rl100 adroit_door_medium 0112 100



DEBUG=False

save_ckpt=True

alg_name=${1}
task_name=${2}
config_name='rl100_3d_epsilon' #${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}


gpu_id=$(bash scripts/find_gpu.sh)
# gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
fi

cd RL-100


export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1 
export MUJOCO_EGL_DEVICE_ID=${gpu_id}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUDA_LAUNCH_BLOCKING=1
act='relu'
encoder_type='dp3vib'
model='skipnet'
run_dir="data/outputs_vib_ablation/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}"
for lr in 1e-6 2e-6 1e-5
do
    for rollout_length in 3 10 5 15 20
    do
    for clip_std_max in 0.1 0.8 
    do
    python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            unio4.bppo_lr=${lr} \
                            unio4.rollout_length=${rollout_length} \
                            training.resume=True \
                            policy._target_=rl_100.policy.rl100_3d.RL1003D \
                            policy.ddim_noise_scheduler.num_train_timesteps=100 \
                            policy.cm_noise_scheduler.num_train_timesteps=50 \
                            use_action_embed=False \
                            horizon=3 \
                            n_action_steps=1 \
                            n_obs_steps=3 \
                            ft_all_actions=False \
                            num_inference_steps=10 \
                            policy.model=$model \
                            policy.encoder_type=$encoder_type \
                            policy.act=$act \
                            unio4.bppo_steps=5000 \
                            policy.encoder_output_dim=64 \
                            policy.diffusion_step_embed_dim=256 \
                            policy.down_dims="[256,512,1024]" \
                            task.env_runner.eval_episodes=30 \
                            task.env_runner.env_num=1 \
                            offline=True \
                            policy.scheduler_type='ddim' \
                            policy.use_agent_pos=True \
                            only_bc=True \
                            unio4.idql_eval=False \
                            policy.mlp_policy_depth=3 \
                            use_wandb=True \
                            critic.omega=0.7 \
                            critic.gamma=0.99 \
                            policy.use_vib=True \
                            policy.use_recon=True \
                            dynamics_type='mlp' \
                            training.num_epochs=600 \
                            training.num_critic_epochs=400 \
                            dynamics.dynamics_max_epochs=150 \
                            distill_phase='after_offlin' \
                            clip_std_max=${clip_std_max} \
                            kl_annealing=False \
                            policy.beta_kl=1e-5
        done
    done               
done
