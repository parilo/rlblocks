import argparse
import os
import random
from copy import deepcopy
from distutils.util import strtobool
from typing import Dict, Any

import gym
import torch as t
import numpy as np

from rlblocks.awac.awac_optimizer import AWACOptimizer
from rlblocks.data.replay_buffer import ReplayBuffer, Episode, batch_to_device
from rlblocks.data_collection.episode import collect_episode
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
        help='OpenAI Gym env name'
    )
    parser.add_argument(
        '--replay-buffer-ep-num', '--rbep',
        default=100,
        type=int,
        help='Replay buffer capacity in episodes'
    )
    parser.add_argument(
        '--train-epochs', '--ep',
        default=1,
        type=int,
        help='Number of train epochs'
    )
    parser.add_argument(
        '--train-steps-per-epoch', '--st',
        default=1,
        type=int,
        help='Number of train steps per epoch'
    )
    parser.add_argument(
        '--episode-len', '--el',
        default=1000,
        type=int,
        help='Episode length'
    )
    parser.add_argument(
        '--episodes-per-epoch', '--eps',
        default=1,
        type=int,
        help='Number of train episodes per epoch'
    )
    parser.add_argument(
        '--batch-size', '--bs',
        default=256,
        type=int,
        help='Batch size'
    )
    parser.add_argument(
        '--exploration-prob', '--explp',
        default=0.8,
        type=float,
        help='Probability of using exploration during episode collection',
    )
    parser.add_argument(
        '--device', '-d',
        default='cpu',
        help='PyTorch device'
    )
    parser.add_argument(
        '--tb-log-dir', '--tb',
        default=None,
        type=str,
        help='Tensorboard log dir'
    )
    parser.add_argument(
        '--visualize', '--vis',
        default=True,
        type=strtobool,
        help='Visualize episodes'
    )
    parser.add_argument(
        '--visualize-every', '--visev',
        default=1,
        type=int,
        help='Sparse episode visualization'
    )
    parser.add_argument(
        '--replay-buffer-pre-fill', '--rbpf',
        default=0,
        type=int,
        help='Pre fill replay buffer with random actions. Number of episodes.'
    )

    return parser.parse_args()


def main():

    args = parse_args()
    ep_num = args.replay_buffer_ep_num

    # init env
    env = VecToDictObsWrapper(gym.make(args.env_name))
    print(f'env {env}')
    print(f'observation space {env.observation_space}')
    print(f'action space {env.action_space}')
    ep_len = args.episode_len
    state_len = sum([val.shape[0] for val in env.observation_space.values()])
    action_len = env.action_space.shape[0]

    state_shapes = get_state_shapes(env.observation_space)
    print(f'State shapes: {state_shapes}')

    # init replay buffer
    replay_buffer = ReplayBuffer(
        ep_num=ep_num,
        ep_len=ep_len,
        state_shapes=state_shapes,
        action_len=action_len
    )

    modalities = ['obs']

    def state_preproc(state: Dict[str, t.Tensor]) -> t.Tensor:
        cated_state = t.cat([
            t.as_tensor(state[mod], dtype=t.float32)
            for mod in modalities
        ], dim=-1)
        return cated_state

    # init models
    actor = GaussianActor(
        model=MLP(
            input_size=state_len,
            output_size=2 * action_len,
            layers_num=3,
            layer_size=256
        ).to(args.device),
        action_min=t.as_tensor(env.action_space.low).to(args.device),
        action_max=t.as_tensor(env.action_space.high).to(args.device),
        logstd_range=(-2, 5),
    )

    critic = MinEnsamble([
        MLP(
            input_size=state_len + action_len,
            output_size=1,
            layers_num=3,
            layer_size=256
        ).to(args.device)
        for _ in range(2)
    ])

    q_func = QFunc(critic)

    q_target_func = QFunc(
        deepcopy(q_func.model).to(args.device)
    )

    q_optimizer = QOptimizer(
        q_func=q_func,
        q_target_func=q_target_func,
        actor=DeterministicActorWrapper(actor),
        lr=1e-4,
        gamma=0.98,
        update_target_each=1,
        update_target_tau=0.005,
    )

    awac_optimizer = AWACOptimizer(
        actor=actor,
        q_func=q_func,
        alambda=0.5,
        lr=1e-4,
    )

    tb_logger_train = None
    if args.tb_log_dir is not None:
        tb_logger_train = TensorboardLogger(os.path.join(args.tb_log_dir, 'train'))

    # exploration = CorrelatedExploration(action_len=action_len, std=0.2, beta=0.5)
    exploration = NormalExploration(action_len=action_len, std=0.4)
    global_step_ind = 0
    global_ep_ind = 0

    def episode_postproc(ep: Episode):
        # ep.rewards /= 10
        return ep

    def random_actor(inp: t.Tensor):
        return t.as_tensor([env.action_space.sample()])

    def print_ep(episode: Episode, ep_ind):
        ep_reward = t.sum(episode.rewards).item()
        print(
            f'--- ep {ep_ind} '
            f'episode size {episode.size} '
            f'reward {ep_reward} '
            f'action min {episode.actions.min()} '
            f'max {episode.actions.max()}')

    # pre fill replay buffer
    for pre_ep_ind in range(args.replay_buffer_pre_fill):
        episode = collect_episode(
            env=env,
            actor=random_actor,
            ep_len=ep_len,
            visualise=False,
            state_preproc=state_preproc,
            device=args.device,
            action_min=env.action_space.low,
            action_max=env.action_space.high,
            # frame_skip=2,
        )
        episode = episode_postproc(episode)
        replay_buffer.push_episode(episode)
        print_ep(episode, f'pre fill {pre_ep_ind}')

    for ep_ind in range(args.train_epochs):

        # perform rollout in the env
        for epoch_ep_ind in range(args.episodes_per_epoch):

            exploration_enabled = exploration if random.uniform(0, 1) < args.exploration_prob else None
            visualization_enabled = (global_ep_ind % args.visualize_every == 0) and args.visualize

            episode = collect_episode(
                env=env,
                actor=actor,
                ep_len=ep_len,
                exploration=exploration_enabled,
                visualise=visualization_enabled,
                state_preproc=state_preproc,
                device=args.device,
                action_min=env.action_space.low,
                action_max=env.action_space.high,
                # frame_skip=2,
            )
            episode = episode_postproc(episode)
            print_ep(episode, ep_ind)
            replay_buffer.push_episode(episode)

            if tb_logger_train is not None:
                ep_reward = t.sum(episode.rewards).item()
                ep_reward = {'train': ep_reward} if exploration_enabled else {'valid': ep_reward}
                tb_logger_train.log({
                    'env_ep_reward': ep_reward,
                    'env_transitions': replay_buffer.transitions_pushed,
                    'env_eps': replay_buffer.eps_pushed,
                }, global_step_ind)

            global_ep_ind += 1

        max_batch_num = int(replay_buffer.transitions_pushed / args.batch_size / 2)
        num_train_ops = min(args.train_steps_per_epoch, max_batch_num)

        for batch_ind in range(num_train_ops):

            batch = replay_buffer.sample_batch(args.batch_size)
            batch.state = state_preproc(batch.state)#, args.device)
            batch.next_state = state_preproc(batch.next_state)#, args.device)
            batch = batch_to_device(batch, args.device)

            log_data = {}
            log_data.update(q_optimizer.train_step(batch))
            log_data.update(awac_optimizer.train_step(batch))

            # if batch_ind == 0:
            #     print(f'--- ep {ep_ind} batch {batch_ind} {log_data}')

            if tb_logger_train is not None:
                tb_logger_train.log(log_data, global_step_ind)

            global_step_ind += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
