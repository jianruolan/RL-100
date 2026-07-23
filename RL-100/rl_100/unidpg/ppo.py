import gym
import torch
import numpy as np
from copy import deepcopy

from rl_100.unidpg.buffer import OnlineReplayBuffer
from rl_100.unidpg.net import GaussPolicyMLP
from rl_100.unidpg.critic import ValueLearner, QLearner
from rl_100.unidpg.utils import orthogonal_initWeights, log_prob_func
from rl_100.unidpg.diffusion_policy.bc_diffusion_ddim import Diffusion_BC
import os

from termcolor import cprint

class ProximalPolicyOptimization:
    _device: torch.device
    _policy: GaussPolicyMLP
    _optimizer: torch.optim
    _policy_lr: float
    _old_policy: GaussPolicyMLP
    _scheduler: torch.optim
    _clip_ratio: float
    _entropy_weight: float
    _decay: float
    _omega: float
    _batch_size: int


    def __init__(
        self,
        policy: Diffusion_BC,
        device: torch.device,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int,
        is_iql:bool,
        ratio_strategy: str,
        fix_encoder: bool
    ) -> None:
        super().__init__()
        self._is_iql = is_iql
        self._device = device
        self._policy = deepcopy(policy).to(device)
        self._fix_encoder = fix_encoder
        self.policy_lr = policy_lr
        #orthogonal_initWeights(self._policy)
        if fix_encoder:
            self._optimizer = torch.optim.Adam(
                self._policy.model.parameters(),
                lr=policy_lr
                )
        else:
            models_params = (
            list(self._policy.model.parameters()) +
            list(self._policy.obs_encoder.parameters()) 
        )
            self._optimizer = torch.optim.Adam(
                models_params,
                lr=policy_lr
                )

        self._policy_lr = policy_lr
        self._old_policy = deepcopy(self._policy).to(device)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=2,
            gamma=0.98
            )
        
        self._clip_ratio = clip_ratio
        self._entropy_weight = entropy_weight
        self._decay = decay
        self._omega = omega
        self._batch_size = batch_size
        self._ratio_strategy = ratio_strategy


    def weighted_advantage(
        self,
        advantage: torch.Tensor
    ) -> torch.Tensor:
        if self._omega == 0.5:
            return advantage
        else:

            weight = torch.where(advantage > 0, self._omega, (1 - self._omega))
            weight.to(self._device)
            return weight * advantage


    def loss(
        self, 
        replay_buffer: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
        is_linear_decay, clip_ratio_now
    ) -> torch.Tensor:
        # -------------------------------------Advantage-------------------------------------
        s, a, _, _, _, _, _, advantage = replay_buffer.sample(self._batch_size)
        old_dist = self._old_policy(s)
        # -------------------------------------Advantage-------------------------------------
        new_dist = self._policy(s)
        
        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp()
        
        advantage = self.weighted_advantage(advantage)

        loss1 =  ratio * advantage 

        if is_clip_decay:
            if is_linear_decay:
                self._clip_ratio = clip_ratio_now
            else:
                self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio

        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage 

        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight
        
        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss


    def update(
        self, 
        replay_buffer: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
        is_lr_decay: bool,
        iql =  None,
        is_linear_decay =  None,
        bppo_lr_now =  None, 
        clip_ratio_now =  None
    ) -> float:
        policy_loss = self.loss(replay_buffer, Q, value, is_clip_decay, iql, is_linear_decay, clip_ratio_now)
        
        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.model.parameters(), 0.5)
        self._optimizer.step()
        
        if is_lr_decay:
            self._scheduler.step()
        if is_linear_decay:
            for p in self._optimizer.param_groups:
                p['lr'] = bppo_lr_now    
        return policy_loss.item()


    def select_action(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        action = self._policy.model.sample_action(s)
        return action


    def evaluate(
        self,
        env_name: str,
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
        eval_episodes: int=10
        ) -> float:
        env = gym.make(env_name)
        env.seed(seed)

        total_reward = 0
        for _ in range(eval_episodes):
            s, done = env.reset(), False
            while not done:
                s = ((np.array(s).reshape(1, -1) - mean) / std).to(self._device)
                a = self.select_action(s)
                s, r, done, _ = env.step(a)
                total_reward += r
        
        avg_reward = total_reward / eval_episodes
        return avg_reward
    
    def save(
        self, path: str
    ) -> None:
        torch.save(self._policy.model.state_dict(), os.path.join(path, 'model.pt'))
        torch.save(self._policy.obs_encoder.state_dict(), os.path.join(path, 'encoder.pt'))
        if getattr(self._policy, 'gripper_head', None) is not None:
            torch.save(
                self._policy.gripper_head.state_dict(),
                os.path.join(path, 'gripper_head.pt'),
            )
        print('Policy parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._policy.model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=self._device))
       
        # encoder_params = torch.load(os.path.join(path, 'encoder.pt'), map_location=self._device)
        # modified_encoder_params = {f"state_mlp.{k}": v for k, v in encoder_params.items()}
        # self._policy.obs_encoder.load_state_dict(modified_encoder_params)
        self._policy.obs_encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pt'), map_location=self._device))
        gripper_head_path = os.path.join(path, 'gripper_head.pt')
        if getattr(self._policy, 'gripper_head', None) is not None:
            if not os.path.exists(gripper_head_path):
                raise FileNotFoundError(
                    f'当前策略启用了夹爪头，但checkpoint缺少 {gripper_head_path}'
                )
            self._policy.gripper_head.load_state_dict(
                torch.load(gripper_head_path, map_location=self._device)
            )
        self._old_policy.model.load_state_dict(self._policy.model.state_dict())
        self._old_policy.obs_encoder.load_state_dict(self._policy.obs_encoder.state_dict())
        if getattr(self._policy, 'gripper_head', None) is not None:
            self._old_policy.gripper_head.load_state_dict(self._policy.gripper_head.state_dict())
        self._policy.to(self._device)
        self._old_policy.to(self._device)
        cprint('1. policy loaded from {}'.format(path), 'green')


        print('Policy parameters loaded from: {}'.format(path))
    def set_policy(
        self, policy, device: torch.device = None,
    ) -> None:
        self._policy = deepcopy(policy).to(device)
        #orthogonal_initWeights(self._policy)
        if self._fix_encoder:
            self._optimizer = torch.optim.Adam(
                self._policy.model.parameters(),
                lr=self._policy_lr
                )
        else:
            models_params = (
            list(self._policy.model.parameters()) +
            list(self._policy.obs_encoder.parameters()) 
            )
            self._optimizer = torch.optim.Adam(
                models_params,
                lr=self._policy_lr
                )

    def set_old_policy(
        self,
    ) -> None:
        self._old_policy = deepcopy(self._policy).to(self._device)
        # self._old_policy.model.load_state_dict(self._policy.model.state_dict())
