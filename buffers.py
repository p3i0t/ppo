import imp
from typing import Tuple
from typing import Union, Tuple

import torch
import numpy as np


class VecPPOBuffer:
    def __init__(self, num_env: int, obs_dim: Union[int, Tuple[int]], act_dim: int, size: int, gamma=0.99, lam=0.95, device='cuda') -> None:

        if isinstance(obs_dim, int):
            obs_buffer_shape = (size, num_env, obs_dim)
        elif isinstance(obs_dim, Tuple):
            obs_buffer_shape = (size, num_env, *obs_dim)

        self.obs_buf = torch.empty(obs_buffer_shape, dtype=torch.float32, device=device)
        self.act_buf = torch.empty((size, num_env, act_dim), dtype=torch.float32, device=device)
        self.adv_buf = torch.empty((size, num_env), dtype=torch.float32, device=device)
        self.reward_buf = torch.empty((size, num_env), dtype=torch.float32, device=device) # one-step reward
        self.return_buf = torch.empty((size, num_env), dtype=torch.float32, device=device) # reward-to-go
        self.value_buf = torch.empty((size, num_env), dtype=torch.float32, device=device)  # state value
        self.logp_buf = torch.empty((size, num_env), dtype=torch.float32, device=device)   # lop_prob of actions
        self.gamma, self.lam = gamma, lam
        self.cur_idx, self.start_idx, self.max_size = 0, 0, size

    def append(self, obs: torch.Tensor, act: torch.Tensor, reward: torch.Tensor,
               value: torch.Tensor, logp: torch.Tensor):
        assert self.cur_idx < self.max_size
        self.obs_buf[self.cur_idx] = obs
        self.act_buf[self.cur_idx] = act
        self.reward_buf[self.cur_idx] = reward
        self.value_buf[self.cur_idx] = value
        self.logp_buf[self.cur_idx] = logp
        self.cur_idx += 1

    def finish_and_compute_gae(self, last_value: torch.Tensor):
        path_slice = slice(self.start_idx, self.cur_idx)
        rewards = torch.cat([self.reward_buf[path_slice], last_value[None, :]], dim=0)
        values = torch.cat([self.value_buf[path_slice], last_value[None, :]], dim=0)
        path_len = values.shape[0] - 1

        deltas = rewards[:-1, :] + self.gamma * values[1:, :] - values[:-1]
        discount = self.gamma * self.lam

        pre_adv = 0
        pre_reward = last_value # state values of next state
        for i in range(reversed(path_len)):
            #往前推，推的距离越长，方差越大，越需要trade-off bias and variance.
            #如果推的距离不长，就简单的单步估计就可以了，
            # 距离越长，bias越小。
            # adv for last timestamp is delta itself.
            self.adv_buf[i] = deltas[i] * discount * pre_adv
            pre_adv = self.adv_buf[i]

            self.return_buf[i] = rewards[i] + self.gamma * pre_reward
            pre_reward = self.return_buf[i]


