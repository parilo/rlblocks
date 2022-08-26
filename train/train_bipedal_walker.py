import argparse
import os
import random
from copy import deepcopy
from distutils.util import strtobool
from typing import Dict, Any

import gym
import torch as t
import numpy as np
import mujoco_py
from gym import ObservationWrapper

from rlblocks.data.replay_buffer import ReplayBuffer
from rlblocks.model.actor import Actor
from rlblocks.model.mlp import MLP


def parse_args():
    parser = argparse.ArgumentParser(description='Perform training')
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
        default=None,
        type=int,
        help='Episode length'
    )
    parser.add_argument(
        '--episodes-per-epoch', '--eps',
        default=None,
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
    # parser.add_argument(
    #     '--expl-to-obs', '--eto',
    #     default=False,
    #     type=strtobool,
    #     help='Add exploration to the observation'
    # )
    parser.add_argument(
        '--replay-buffer-pre-fill', '--rbpf',
        default=0,
        type=int,
        help='Pre fill replay buffer with random actions. Number of episodes.'
    )

    return parser.parse_args()


class DictObsWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = {
            'obs': env.observation_space,
        }

    def observation(self, observation):
        return {'obs': observation}


def main():

    args = parse_args()
    ep_num = args.replay_buffer_ep_num

    # init env
    env = DictObsWrapper(gym.make('BipedalWalker-v3'))
    ep_len = args.episode_len
    state_len = sum([val.shape[0] for val in env.observation_space.values()])
    action_len = env.action_space.shape[0]

    obs_space = env.observation_space
    # if args.expl_to_obs:
    #     obs_space['noise'] = env.action_space
    #     state_len += action_len

    # init replay buffer
    replay_buffer = ReplayBuffer(
        ep_num=ep_num,
        ep_len=ep_len,
        state_space=obs_space,
        action_len=action_len
    )

    # modalities = ['qpos', 'qvel', 'target_vel_lin', 'target_vel_rot_z', 'touch', 'noise']
    modalities = ['obs']

    def state_preproc(state: Dict[str, t.Tensor]) -> t.Tensor:
        # try:
        cated_state = t.cat([
            t.as_tensor(state[mod], dtype=t.float32).to(args.device)
            for mod in modalities
        ], dim=-1)
        if len(cated_state.shape) == 1:
            cated_state = cated_state.unsqueeze(0)
        return cated_state
        # except Exception as ex:
        #     state_shapes = {key: val.shape for key, val in state.items()}
        #     print(f'--- state_shapes {state_shapes}')
        #     raise ex

    # init models
    actor = Actor(
        model=MLP(
            input_size=state_len,
            output_size=action_len,
            layers_num=3,
            layer_size=256
        ).to(args.device),
        # state_preproc=state_preproc,
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
        actor=actor,
        lr=2e-4,
        gamma=0.99,
        update_target_each=1,
        update_target_tau=0.001,
    )

    awac_optimizer = AWACOptimizer(
        actor=actor,
        q_func=q_func,
        lr=2e-4,
    )

    tb_logger_train = None
    if args.tb_log_dir is not None:
        tb_logger_train = TensorboardLogger(os.path.join(args.tb_log_dir, 'train'))

    # exploration = CorrelatedExploration(action_len=action_len, std=0.2, beta=0.5)
    exploration = NormalExploration(action_len=action_len, std=0.4)
    global_step_ind = 0
    global_ep_ind = 0

    def episode_postproc(ep: Episode):
        ep.rewards /= 10
        return ep

    def random_actor(inp: t.Tensor):
        return t.as_tensor(env.action_space.sample())

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
            add_noise_to_obs=args.expl_to_obs,
        )
        episode = episode_postproc(episode)
        replay_buffer.push_episode(episode)
        print_ep(episode, f'pre fill {pre_ep_ind}')

    for ep_ind in range(args.train_epochs):

        # perform rollout in the env
        for epoch_ep_ind in range(args.episodes_per_epoch):

            exploration_enabled = exploration if random.uniform(0, 1) < args.exploration_prob else None
            visualization_enabled = (global_ep_ind % args.visualize_every == 0) and args.visualize

            try:
                episode = collect_episode(
                    env=env,
                    actor=actor,
                    ep_len=ep_len,
                    exploration=exploration_enabled,
                    visualise=visualization_enabled,
                    state_preproc=state_preproc,
                    add_noise_to_obs=args.expl_to_obs,
                )
            except mujoco_py.builder.MujocoException as ex:
                print(f'--- mujoco exception {ex}')
                continue
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
            batch.state = state_preproc(batch.state)
            batch.next_state = state_preproc(batch.next_state)

            log_data = {}
            log_data.update(q_optimizer.train_step(batch))
            log_data.update(awac_optimizer.train_step(batch))

            # print(f'--- ep {ep_ind} batch {batch_ind}')

            if tb_logger_train is not None:
                tb_logger_train.log(log_data, global_step_ind)

            global_step_ind += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# python lm-rl/train_cheetah.py --rbep 1000 --ep 10000 --st 400 --el 500 --episodes-per-epoch 1 --bs 128 --explp 0.8 --tb ./logs/exp_8 --visev 10
