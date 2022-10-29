import glob
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Union

import numpy as np
import torch as t


@dataclass
class Episode:
    states: Union[t.Tensor, Dict[str, t.Tensor]]
    actions: t.Tensor
    rewards: t.Tensor
    size: int
    done: bool

    def copy(self):
        if isinstance(self.states, t.Tensor):
            states = self.states.clone()
        else:
            states = {mod: val.clone() for mod, val in self.states.items()}
        return Episode(
            states=states,
            actions=self.actions.clone(),
            rewards=self.rewards.clone(),
            size=self.size,
            done=self.done,
        )

    def __repr__(self):
        ep_reward = t.sum(self.rewards).item()
        state_shapes = {mod: val.shape for mod, val in self.states.items()}
        return f'episode size {self.size} ' \
            f'reward {ep_reward} ' \
            f'action min {self.actions.min()} ' \
            f'max {self.actions.max()} ' \
            f'done {self.done}'
            # f'shapes {state_shapes} ' \
            # f'actions {self.actions.shape} ' \
            # f'rewards {self.rewards.shape} ' \


@dataclass
class Batch:
    state: Union[t.Tensor, Dict[str, t.Tensor]]
    action: t.Tensor
    reward: Optional[t.Tensor]
    done: t.Tensor
    next_state: Optional[Union[t.Tensor, Dict[str, t.Tensor]]] = None
    info: Optional[Dict[str, t.Tensor]] = None


def batch_to_device(batch: Batch, device: t.device):
    return Batch(
        state=batch.state.to(device),
        action=batch.action.to(device) if batch.action is not None else None,
        reward=batch.reward.to(device) if batch.reward is not None else None,
        done=batch.done.to(device),
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
        self._ends_with_done = t.zeros((self._ep_num, 1), dtype=t.int32)
        self._ep_ind = 0
        self._ep_stored = 0
        self._ep_pushed = 0
        self._tr_pushed = 0

    @property
    def transitions_pushed(self):
        return self._tr_pushed

    @property
    def eps_pushed(self) -> int:
        return self._ep_pushed

    def append_episode(self, episode: Episode, add_last_obs: bool = False):
        start_ind = self._ep_sizes[self._ep_ind][0]
        end_ind = start_ind + episode.size
        obs_end_ind = end_ind + (1 if add_last_obs else 0)
        for mod_name, mod_value in episode.states.items():
            self._states[mod_name][self._ep_ind, start_ind:obs_end_ind] = mod_value if add_last_obs else mod_value[:-1]
        self._actions[self._ep_ind, start_ind:end_ind] = episode.actions
        self._rewards[self._ep_ind, start_ind:end_ind] = episode.rewards
        self._ep_sizes[self._ep_ind][0] += episode.size
        self._ends_with_done[self._ep_ind][0] = episode.done

    def end_episode(self):

        if self._save_dir:
            ep = self.get_episode(self._ep_ind)
            self._save_episode(ep.copy(), self._ep_ind)

        self._tr_pushed += self._ep_sizes[self._ep_ind][0]
        self._ep_pushed += 1
        self._ep_stored += 1
        self._ep_stored = min(self._ep_num, self._ep_stored)
        self._ep_ind += 1
        self._ep_ind %= self._ep_num

        self.clean_last_episode()

    def clean_last_episode(self):
        self._ep_sizes[self._ep_ind][0] = 0
        self._ends_with_done[self._ep_ind][0] = 0

    def push_episode(self, episode: Episode):
        self.append_episode(episode, add_last_obs=True)
        self.end_episode()

    def _save_episode(self, ep: Episode, ep_ind: int):
        fpath = f'ep_{self._save_episode_suffix}_{ep_ind}.pkl'
        with open(os.path.join(self._save_dir, fpath), 'wb') as f:
            pickle.dump(ep, f)

    def sample_batch(self, batch_size) -> Batch:
        if self._ep_stored == 0:
            raise RuntimeError(f'Stored episodes: {self._ep_stored} should be greater than zero')
        ep_inds = np.random.randint(0, self._ep_stored, size=batch_size)
        transition_inds = np.random.randint(0, self._ep_len, size=batch_size)
        ep_sizes = self._ep_sizes[ep_inds, 0]
        transition_inds = np.mod(transition_inds, ep_sizes)  # ep size is number of taken actions
        state = {mod_name: mod_value[ep_inds, transition_inds] for mod_name, mod_value in self._states.items()}
        next_state = {mod_name: mod_value[ep_inds, transition_inds + 1] for mod_name, mod_value in self._states.items()}
        return Batch(
            state=state,
            action=self._actions[ep_inds, transition_inds],
            reward=self._rewards[ep_inds, transition_inds],
            next_state=next_state,
            done=t.as_tensor(transition_inds == ep_sizes - 1, dtype=t.float32).unsqueeze(-1) * self._ends_with_done[ep_inds],
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
        size = self._ep_sizes[ep_ind][0]
        return Episode(
            states={mod: val[ep_ind, :size + 1] for mod, val in self._states.items()},
            actions=self._actions[ep_ind, :size],
            rewards=self._rewards[ep_ind, :size],
            size=size,
            done=self._ends_with_done[ep_ind][0],
        )

    def get_last_episode(self):
        return self.get_episode((self._ep_ind - 1) % self._ep_num)


def save_buffer_episodes_to_dir(buf: ReplayBuffer, dir_path: str):
    os.makedirs(dir_path, exist_ok=True)
    for ep_ind in range(buf.eps_pushed):
        ep = buf.get_episode(ep_ind)
        with open(os.path.join(dir_path, f'ep_{ep_ind}.pkl'), 'wb') as f:
            pickle.dump(ep, f)


def load_buffer_episodes_from_dir(buf: ReplayBuffer, dir_path: str):
    # for ind, path in enumerate(glob.glob(os.path.join(dir_path, '**', 'ep_*.pkl'))):
    for ind, path in enumerate(Path(dir_path).rglob("ep_*.pkl")):
        if ind % 100 == 0:
            print(f'loading {ind}')
        with open(path, 'rb') as f:
            try:
                buf.push_episode(pickle.load(f))
            except pickle.UnpicklingError:
                print(f'--- UnpicklingError: skipping {path}')
