import os
import pickle
from dataclasses import dataclass
from typing import Optional, Dict, Union

import numpy as np
import torch as t


@dataclass
class Episode:
    states: Union[t.Tensor, Dict[str, t.Tensor]]
    actions: t.Tensor
    rewards: t.Tensor
    size: int


@dataclass
class Batch:
    state: Union[t.Tensor, Dict[str, t.Tensor]]
    action: t.Tensor
    reward: Optional[t.Tensor]
    next_state: Optional[Union[t.Tensor, Dict[str, t.Tensor]]] = None
    info: Optional[Dict[str, t.Tensor]] = None


def batch_to_device(batch: Batch, device: t.device):
    return Batch(
        state=batch.state.to(device),
        action=batch.action.to(device) if batch.action is not None else None,
        reward=batch.reward.to(device) if batch.reward is not None else None,
        next_state=batch.next_state.to(device) if batch.next_state is not None else None,
        info={
            key: value.to(device) for key, value in batch.info.items()
        } if batch.info is not None else None
    )


class ReplayBuffer:

    def __init__(
            self,
            ep_num,
            ep_len,
            state_shapes: Dict[str, tuple],
            action_len,
            load_dir: Optional[str] = None,
            save_dir: Optional[str] = None,
            save_episode_suffix: str = '',
    ):
        self._ep_num = ep_num
        self._ep_len = ep_len
        self._state_shapes = state_shapes
        self._action_len = action_len
        self._save_dir = save_dir
        if self._save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self._save_episode_suffix = save_episode_suffix

        self._states = {}
        for mod_name, mod_shape in self._state_shapes.items():
            self._states[mod_name] = t.zeros(
                (self._ep_num, self._ep_len + 1) + tuple(mod_shape),
                dtype=t.float32,
            )
        self._actions = t.zeros((self._ep_num, self._ep_len, self._action_len), dtype=t.float32)
        self._rewards = t.zeros((self._ep_num, self._ep_len, 1), dtype=t.float32)
        self._ep_sizes = t.zeros((self._ep_num, 1), dtype=t.int32)
        self._ep_ind = 0
        self._ep_stored = 0
        self._ep_pushed = 0
        self._tr_stored = 0

    @property
    def transitions_pushed(self):
        return self._tr_stored

    @property
    def eps_pushed(self) -> int:
        return self._ep_pushed

    def push_episode(self, episode: Episode):
        for mod_name, mod_value in episode.states.items():
            self._states[mod_name][self._ep_ind, :episode.size + 1] = mod_value
        self._actions[self._ep_ind, :episode.size] = episode.actions
        self._rewards[self._ep_ind, :episode.size] = episode.rewards
        self._ep_sizes[self._ep_ind][0] = episode.size

        if self._save_dir:
            self._save_episode(episode, self._ep_ind)

        self._ep_ind += 1
        self._ep_ind %= self._ep_num
        self._ep_stored += 1
        self._ep_stored = min(self._ep_num, self._ep_stored)
        self._tr_stored += episode.size
        self._ep_pushed += 1

    def _save_episode(self, ep: Episode, ep_ind: int):
        with open(os.path.join(self._save_dir, f'ep_{self._save_episode_suffix}_{ep_ind}.pkl'), 'wb') as f:
            pickle.dump(ep, f)

    def sample_batch(self, batch_size) -> Batch:
        ep_inds = np.random.randint(0, self._ep_stored, size=batch_size)
        transition_inds = np.random.randint(0, self._ep_len, size=batch_size)
        transition_inds = np.mod(transition_inds, self._ep_sizes[ep_inds, 0])
        state = {mod_name: mod_value[ep_inds, transition_inds] for mod_name, mod_value in self._states.items()}
        next_state = {mod_name: mod_value[ep_inds, transition_inds + 1] for mod_name, mod_value in self._states.items()}
        return Batch(
            state=state,
            action=self._actions[ep_inds, transition_inds],
            reward=self._rewards[ep_inds, transition_inds],
            next_state=next_state,
        )

    def sample_seq_batch(self, batch_size, seq_len) -> Batch:
        ep_inds = np.random.randint(0, self._ep_stored, size=batch_size)
        transition_inds = np.random.randint(0, self._ep_sizes[ep_inds, 0] - seq_len, size=batch_size)
        transition_ranges = [range(ind, ind + seq_len) for ind in transition_inds]
        ep_inds_to_broadcast = t.as_tensor(ep_inds).unsqueeze(-1)
        info = {}
        state = {mod_name: mod_value[ep_inds_to_broadcast, transition_ranges] for mod_name, mod_value in self._states.items()}
        return Batch(
            state=state,
            action=self._actions[ep_inds_to_broadcast, transition_ranges],
            reward=self._rewards[ep_inds_to_broadcast, transition_ranges],
            info=info
        )

    def sample_episode(self):
        ep_ind = np.random.randint(0, self._ep_stored, size=1)[0]
        return self.get_episode(ep_ind)

    def get_episode(self, ep_ind):
        size = self._ep_sizes[ep_ind]
        return Episode(
            states={mod: val[ep_ind, :size] for mod, val in self._states.items()},
            actions=self._actions[ep_ind, :size],
            rewards=self._rewards[ep_ind, :size],
            size=size,
        )


def save_buffer_episodes_to_dir(buf: ReplayBuffer, dir_path:str):
    os.makedirs(dir_path, exist_ok=True)
    for ep_ind in range(buf.eps_pushed):
        ep = buf.get_episode(ep_ind)
        with open(os.path.join(dir_path, f'ep_{ep_ind}.pkl'), 'wb') as f:
            pickle.dump(ep, f)
