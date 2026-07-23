from typing import Dict
import os
import torch
import numpy as np
import copy
import zarr
from rl_100.common.pytorch_util import dict_apply
from rl_100.common.replay_buffer import ReplayBuffer
from rl_100.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from rl_100.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.unidpg.utils import RewardScaling

from termcolor import cprint
from tqdm import tqdm
def compute_return(reward, not_done, gamma: float == 0.99
    ) -> np.ndarray:
        size_ = len(reward)
        return_ = np.zeros((size_, 1))
        pre_return = 0
        for i in tqdm(reversed(range(size_)), desc='Computing the returns'):
            return_[i] = reward[i] + gamma * pre_return * not_done[i]
            pre_return = return_[i]
        return return_
class AdroitDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            scale_strategy=None,
            pre_image_norm=False,
            sequence_stride=1,
            controlled_dims=None,
            action_dims=None,
            action_key='action',
            use_gripper_head=False,
            n_obs_steps=1,
            gripper_horizon=12,
            gripper_future_offset=3,
            gripper_event_radius=2,
            gripper_event_weight=10.0,
            ):
        super().__init__()
        self.task_name = task_name
        self.controlled_dims = controlled_dims
        # agent_pos可以保留7维（含夹爪反馈），同时让arm-only diffusion只学习前6维动作。
        self.action_dims = controlled_dims if action_dims is None else action_dims
        self.action_key = action_key
        self.next_action_key = (
            'next_action' if action_key == 'action' else f'next_{action_key}'
        )
        self.use_gripper_head = use_gripper_head
        self.n_obs_steps = n_obs_steps
        self.gripper_horizon = gripper_horizon
        self.gripper_future_offset = gripper_future_offset
        self.gripper_event_radius = gripper_event_radius
        self.gripper_event_weight = gripper_event_weight

        if self.use_gripper_head:
            if self.gripper_horizon < 1 or self.gripper_future_offset < 0:
                raise ValueError('gripper_horizon必须大于0，gripper_future_offset不能小于0')
            if self.gripper_event_radius < 0 or self.gripper_event_weight < 1:
                raise ValueError('gripper_event_radius不能小于0，gripper_event_weight不能小于1')

        keys = [
            'state', self.action_key, 'point_cloud', 'img',
            'next_state', self.next_action_key, 'next_point_cloud', 'next_img',
            'reward', 'done', 'timeout', 'return'
        ]
        if self.use_gripper_head:
            keys.extend([
                'gripper_command_state',
                'gripper_open_event',
                'gripper_close_event',
            ])
        # 在复制整个数据集前先给出明确的缺字段错误，避免把旧 action 静默用于新任务。
        root = zarr.open(os.path.expanduser(zarr_path), mode='r')
        missing_keys = sorted(set(keys) - set(root['data'].keys()))
        if missing_keys:
            raise KeyError(
                f'数据集 {zarr_path} 缺少训练字段: {missing_keys}; '
                f'action_key={self.action_key!r}, use_gripper_head={self.use_gripper_head}'
            )
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys)
        # construct scaled reward and return
        # import pdb; pdb.set_trace()
        if scale_strategy == 'dynamic':
            print('scaling reward dynamically')
            reward_norm = RewardScaling(1, gamma=0.99)
            rewards = self.replay_buffer['reward'].flatten()
            for i, not_done in enumerate(1 - self.replay_buffer['done'].flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            self.replay_buffer.root['data']['reward'] = rewards.reshape(-1, 1)
            self.replay_buffer.root['data']['return'] = compute_return(self.replay_buffer['reward'], 1 - self.replay_buffer['done'], gamma=0.99)
            self.reward_norm = reward_norm
            cprint('reward and return scaled', 'green')
        # self.replay_buffer.root {'meta', 'data'}

        # for key, value in self.replay_buffer.items():
        #     cprint(f'Replay Buffer: {key}, shape {value.shape}, dtype {value.dtype}, range {value.min():.2f}~{value.max():.2f}', 'green')
        # cprint("--------------------------", 'green')

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler_horizon = horizon
        sampler_pad_after = pad_after
        if self.use_gripper_head:
            # 夹爪头需要看到观察窗口之后更远的标签；这些额外帧不会传给 diffusion。
            sampler_horizon = max(
                horizon,
                (n_obs_steps - 1) + gripper_future_offset + gripper_horizon
                + gripper_event_radius,
            )
            # 保持与原短 horizon 相同的有效观测起点数量，并为末尾标签生成 padding mask。
            sampler_pad_after = max(pad_after, sampler_horizon - n_obs_steps)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=sampler_horizon,
            pad_before=pad_before,
            pad_after=sampler_pad_after,
            episode_mask=train_mask,
            sequence_stride=sequence_stride)
        self.train_mask = train_mask
        self.horizon = horizon
        self.sampler_horizon = sampler_horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sampler_pad_after = sampler_pad_after
        self.sequence_stride = sequence_stride
    def reward_scaling(self, scaling_strategy = 'dynamic', gamma = 0.99):
        if scaling_strategy == 'dynamic':
            print('scaling reward dynamically')
            reward_norm = RewardScaling(1, gamma)
            rewards = self.replay_buffer['reward'].flatten()
            for i, not_done in enumerate(1 - self.replay_buffer['done'].flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            self.replay_buffer['reward'] = rewards.reshape(-1, 1)
            self.replay_buffer['return'] = compute_return(self.replay_buffer['reward'], 1 - self.replay_buffer['done'], gamma)
        

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.sampler_horizon,
            pad_before=self.pad_before,
            pad_after=self.sampler_pad_after,
            episode_mask=~self.train_mask,
            sequence_stride=self.sequence_stride,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        state_slice = slice(None, self.controlled_dims)
        action_slice = slice(None, self.action_dims)
        data = {
            'action': self.replay_buffer[self.action_key][..., action_slice],
            'agent_pos': self.replay_buffer['state'][..., state_slice],
            'point_cloud': self.replay_buffer['point_cloud'],

            'next_action': self.replay_buffer[self.next_action_key][..., action_slice],
            'next_agent_pos': self.replay_buffer['next_state'][..., state_slice],
            'next_point_cloud': self.replay_buffer['next_point_cloud'],

            # 'reward': self.replay_buffer['reward'],
            # 'not_done': 1. - self.replay_buffer['done'],
            # 'return': self.replay_buffer['return'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample, valid_mask=None):
        state_slice = slice(None, self.controlled_dims)
        action_slice = slice(None, self.action_dims)
        # 扩展采样窗口只服务于夹爪标签，主训练数据仍严格保持配置中的 horizon。
        main = slice(0, self.horizon)
        agent_pos = sample['state'][main, state_slice].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][main,].astype(np.float32) # (T, 1024, 6)
        image = sample['img'][main,].astype(np.float32) # (T, 3, 64, 64)
        
        next_agent_pos = sample['next_state'][main, state_slice].astype(np.float32) # (agent_posx2, block_posex3)
        next_point_cloud = sample['next_point_cloud'][main,].astype(np.float32) # (T, 1024, 6)
        next_image = sample['next_img'][main,].astype(np.float32) # (T, 3, 64, 64)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
                'image': image, # T, 84, 84, 3
            },
            'next_obs': {
                'point_cloud': next_point_cloud, # T, 1024, 6
                'agent_pos': next_agent_pos, # T, D_pos
                'image': next_image, # T, 84, 84, 3
            }, 
            'reward': sample['reward'][main].astype(np.float32), # T, D_action
            'not_done': 1. - sample['done'][main].astype(np.bool_), # T, D_action
            'return': sample['return'][main].astype(np.float32), # T, D_action
            'action': sample[self.action_key][main, action_slice].astype(np.float32), # T, D_action
            'next_action': sample[self.next_action_key][main, action_slice].astype(np.float32) # T, D_action
        }

        if self.use_gripper_head:
            target_start = (self.n_obs_steps - 1) + self.gripper_future_offset
            target_end = target_start + self.gripper_horizon
            target_slice = slice(target_start, target_end)
            target = sample['gripper_command_state'][target_slice, 0].astype(np.float32)

            open_events = sample['gripper_open_event'][:, 0].astype(bool)
            close_events = sample['gripper_close_event'][:, 0].astype(bool)

            def dilate(events):
                if self.gripper_event_radius == 0:
                    return events
                kernel = np.ones(2 * self.gripper_event_radius + 1, dtype=np.int32)
                return np.convolve(events.astype(np.int32), kernel, mode='same') > 0

            # 损失在事件邻域加权；指标只在真正收到命令的帧上统计，避免把事件前
            # 尚未切换的正确状态误算成open/close召回失败。
            event_neighborhood = (
                dilate(open_events) | dilate(close_events)
            )[target_slice]
            open_mask = open_events[target_slice]
            close_mask = close_events[target_slice]
            if valid_mask is None:
                target_valid = np.ones(self.gripper_horizon, dtype=np.float32)
            else:
                target_valid = valid_mask[target_slice].astype(np.float32)
            event_weight = np.where(
                event_neighborhood, self.gripper_event_weight, 1.0
            ).astype(np.float32)
            data.update({
                'gripper_target': target,
                'gripper_valid_mask': target_valid,
                'gripper_event_weight': event_weight,
                'gripper_open_event_mask': open_mask.astype(np.float32),
                'gripper_close_event_mask': close_mask.astype(np.float32),
            })

        return data
    def get_shape_info(self, n_action_steps, n_obs_steps):
        sample = self.sampler.sample_sequence(10)
        state_slice = slice(None, self.controlled_dims)
        action_slice = slice(None, self.action_dims)
        agent_pos = sample['state'][:, state_slice].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        image = sample['img'][:,].astype(np.float32) # (T, 3, 64, 64)

        shape_info = {
        'obs': {
            'point_cloud': (n_obs_steps,) + point_cloud.shape[1:],
            'agent_pos': (n_obs_steps,) + agent_pos.shape[1:],
            'image': (n_obs_steps,) + image.shape[1:],
        },
        'action': (n_action_steps, sample[self.action_key][:, action_slice].shape[-1]),
        }
        return shape_info
    def get_all_data(self,) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(range(self.replay_buffer[self.action_key].shape[0]))
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_length(self, ):
        return len(self.sampler.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        _, _, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        valid_mask = np.zeros(self.sampler_horizon, dtype=bool)
        valid_mask[sample_start_idx:sample_end_idx] = True
        data = self._sample_to_data(sample, valid_mask=valid_mask)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
