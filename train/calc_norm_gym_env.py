import argparse
import os
import pickle
import random
from copy import deepcopy
from distutils.util import strtobool
from typing import Dict, Any

import gym
import torch as t
import numpy as np

from rlblocks.awac.awac_optimizer import AWACOptimizer
from rlblocks.data.replay_buffer import ReplayBuffer, Episode, batch_to_device, load_buffer_episodes_from_dir
from rlblocks.data_collection.episode import collect_episode, collect_episodes
from rlblocks.data_collection.exploration import NormalExploration
from rlblocks.model.actor import Actor
from rlblocks.model.gaussian_actor import GaussianActor
from rlblocks.model.min_ensamble import MinEnsamble
from rlblocks.model.mlp import MLP
from rlblocks.model.q_func import QFunc
from rlblocks.model.stochastic_actor import DeterministicActorWrapper
from rlblocks.q_learning.q_optimizer import QOptimizer
from rlblocks.tb.tb_logger import TensorboardLogger
from rlblocks.utils.env_utils import VecToDictObsWrapper, get_state_shapes


def parse_args():
    parser = argparse.ArgumentParser(description='Train policy for gym env')
    parser.add_argument(
        '--env-name', '--env',
        required=True,
        type=str,
        help='OpenAI Gym env name',
    )
    parser.add_argument(
        '--episodes-num', '--en',
        default=500,
        type=int,
        help='Number of episodes',
    )
    parser.add_argument(
        '--episode-len', '--el',
        default=1000,
        type=int,
        help='Episode length',
    )
    parser.add_argument(
        '--save-to', '-s',
        required=True,
        type=str,
        help='Save normalization to file',
    )

    return parser.parse_args()


def main():

    args = parse_args()
    eps_num = args.episodes_num

    # init env
    env = VecToDictObsWrapper(gym.make(args.env_name))
    print(f'env {env}')
    print(f'observation space {env.observation_space}')
    print(f'action space {env.action_space}')
    ep_len = args.episode_len

    def random_actor(inp: t.Tensor):
        return t.as_tensor([env.action_space.sample()])

    modalities = ['obs']

    def state_preproc(state: Dict[str, t.Tensor]) -> t.Tensor:
        cated_state = t.cat([
            t.as_tensor(state[mod], dtype=t.float32)
            for mod in modalities
        ], dim=-1)
        return cated_state

    eps = collect_episodes(
        eps_num=eps_num,
        env=env,
        actor=random_actor,
        ep_len=ep_len,
        state_preproc=state_preproc,
        action_min=env.action_space.low,
        action_max=env.action_space.high,
        debug_print_prefix='collect',
    )

    all_states = np.concatenate([ep.states["obs"] for ep in eps], axis=0)
    print(f'Collected states {all_states.shape}')
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0)
    print(f'State mean min {state_mean.min()} max {state_mean.max()} mean {state_mean.mean()}')
    print(f'State std  min {state_std.min()} max {state_std.max()} mean {state_std.mean()}')

    with open(args.save_to, 'wb') as f:
        pickle.dump({
            'state_mean': state_mean,
            'state_std': state_std,
        }, f)


if __name__ == '__main__':
    main()
