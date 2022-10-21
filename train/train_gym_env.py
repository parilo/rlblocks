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
from rlblocks.data_collection.episode import collect_episode
from rlblocks.data_collection.exploration import NormalExploration, NormalRandomizedExploration, CorrelatedExploration, \
    NormalStepBasedExploration
from rlblocks.model.actor import Actor
from rlblocks.model.fixed_normalizer import FixedNormalizer
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
        '--exploration-std', '--explstd',
        default=0.2,
        type=float,
        help='Std of the normal noise used as exploration.',
    )
    parser.add_argument(
        '--replay-buffer-pre-fill', '--rbpf',
        default=0,
        type=int,
        help='Pre fill replay buffer with random actions. Number of episodes.'
    )
    parser.add_argument(
        '--frame-skip', '--fs',
        default=1,
        type=int,
        help='Frame skip',
    )
    parser.add_argument(
        '--gamma', '-g',
        default=0.996,
        type=float,
        help='Discount factor',
    )
    parser.add_argument(
        '--update-target-each',
        default=1,
        type=int,
        help='Update target network every n step',
    )
    parser.add_argument(
        '--update-target-rate',
        default=0.001,
        type=float,
        help='Update target rate, tau',
    )
    parser.add_argument(
        '--reward-scale',
        default=1,
        type=float,
        help='Reward scale',
    )
    parser.add_argument(
        '--reward-shift',
        default=0,
        type=float,
        help='Reward shift',
    )
    parser.add_argument(
        '--lambdaa',
        default=1,
        type=float,
        help='AWAC advantage scale (lambda parameter).',
    )
    parser.add_argument(
        '--critic-heat-up',
        default=0,
        type=int,
        help='',
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
        '--save-dir', '--sd',
        default=None,
        type=str,
        help='Model save dir'
    )
    parser.add_argument(
        '--save-each', '--se',
        default=5000,
        type=int,
        help='Save model each iterations'
    )
    parser.add_argument(
        '--save-episodes-dir', '--sed',
        default=None,
        type=str,
        help='Save collected episodes in folder'
    )
    parser.add_argument(
        '--load-dir', '--ld',
        default=None,
        type=str,
        help='Load models from dir'
    )
    parser.add_argument(
        '--load-episodes', '--le',
        default=None,
        type=str,
        help='Load episodes from dir'
    )
    parser.add_argument(
        '--load-state-norm', '--lsn',
        default=None,
        type=str,
        help='Load state normalizer parameters from file'
    )

    return parser.parse_args()


def main():

    args = parse_args()
    ep_num = args.replay_buffer_ep_num

    # init env
    env = VecToDictObsWrapper(gym.make(args.env_name, healthy_reward=0.0))  #, healthy_z_range=(0.8, 3.0), ctrl_cost_weight=0.0
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
        action_len=action_len,
        save_dir=os.path.join(args.save_episodes_dir, 'replay_buffer') if args.save_episodes_dir else None,
    )

    if args.load_episodes:
        load_buffer_episodes_from_dir(replay_buffer, args.load_episodes)

    modalities = ['obs']

    def state_preproc(state: Dict[str, t.Tensor]) -> t.Tensor:
        cated_state = t.cat([
            t.as_tensor(state[mod], dtype=t.float32)
            for mod in modalities
        ], dim=-1)
        return cated_state

    global_step_ind = 0
    global_ep_ind = 0

    # init models
    if args.load_dir:
        actor = t.load(os.path.join(args.load_dir, 'actor.pt'), map_location=args.device)
        q_func = t.load(os.path.join(args.load_dir, 'q_func.pt'), map_location=args.device)
        q_target_func = t.load(os.path.join(args.load_dir, 'q_target_func.pt'), map_location=args.device)
        expinfo_path = os.path.join(args.load_dir, 'expinfo.pkl')
        if os.path.exists(expinfo_path):
            with open(expinfo_path, 'rb') as f:
                expinfo = pickle.load(f)
                global_step_ind = expinfo['global_step_ind']
    else:

        actor = GaussianActor(
            model=MLP(
                input_size=state_len,
                output_size=2 * action_len,
                layers_num=3,
                layer_size=64,  #256,
            ).to(args.device),
            action_min=t.as_tensor(env.action_space.low).to(args.device),
            action_max=t.as_tensor(env.action_space.high).to(args.device),
            logstd_range=(-5, 5),
        )

        critic = MinEnsamble([
            MLP(
                input_size=state_len + action_len,
                output_size=1,
                layers_num=3,
                layer_size=64,  #256,
            ).to(args.device)
            for _ in range(2)
        ])

        q_func = QFunc(critic)

        q_target_func = QFunc(
            deepcopy(q_func.model).to(args.device)
        )

    if args.load_state_norm:
        print(f'Using state normalization from {args.load_state_norm}')
        with open(args.load_state_norm, 'rb') as f:
            norm_data = pickle.load(f)
            normalizer = FixedNormalizer(
                t.as_tensor(norm_data['state_mean']).to(args.device),
                t.as_tensor(norm_data['state_std']).to(args.device),
            )
            actor.state_norm = normalizer
            q_func.state_norm = normalizer
            q_target_func.state_norm = normalizer

    deterministic_actor = DeterministicActorWrapper(actor)

    randomized_leg = 0
    def env_actor(state: t.Tensor, step_ind: int):

        # if step_ind < 5:
        #     action = t.zeros((1, action_len))
        #
        #     # left leg
        #     if randomized_leg == 0:
        #         action[0, 5] = 0.3
        #         action[0, 6] = 0.0
        #         action[0, 9] = -0.3
        #         action[0, 10] = -0.05
        #
        #     # right leg
        #     elif randomized_leg == 1:
        #         action[0, 5] = -0.3
        #         action[0, 6] = -0.05
        #         action[0, 9] = 0.3
        #         action[0, 10] = 0.0
        #
        #     return action
        # else:
        return deterministic_actor(state)


    q_optimizer = QOptimizer(
        q_func=q_func,
        q_target_func=q_target_func,
        # actor=deterministic_actor,
        actor=actor,
        lr=1e-4,
        gamma=args.gamma,
        update_target_each=args.update_target_each,
        update_target_tau=args.update_target_rate,
    )

    awac_optimizer = AWACOptimizer(
        actor=actor,
        q_func=q_func,
        alambda=args.lambdaa,
        lr=1e-4,
    )

    tb_logger_train = None
    if args.tb_log_dir is not None:
        tb_logger_train = TensorboardLogger(os.path.join(args.tb_log_dir, 'train'))

    # exploration = CorrelatedExploration(action_len=action_len, std=0.2, beta=0.1)
    exploration = NormalExploration(action_len=action_len, std=args.exploration_std)
    # exploration = NormalRandomizedExploration(action_len=action_len, min_std=0, max_std=0.3)
    # exploration = NormalStepBasedExploration(action_len=action_len, std=args.exploration_std, switch_steps=50)

    def episode_postproc(ep: Episode):
        ep.rewards += args.reward_shift
        ep.rewards *= args.reward_scale
        return ep

    # def random_actor(inp: t.Tensor):
    #     return t.as_tensor([env.action_space.sample()])

    # pre fill replay buffer
    eps = []
    for pre_ep_ind in range(args.replay_buffer_pre_fill):
        randomized_leg = random.randint(0, 2)
        actor.model.eval()
        episode = collect_episode(
            env=env,
            actor=env_actor,  #deterministic_actor,  #if args.load_dir else random_actor,
            exploration=exploration,  #if args.load_dir else None,
            ep_len=ep_len,
            visualise=False,
            state_preproc=state_preproc,
            device=args.device,
            action_min=env.action_space.low,
            action_max=env.action_space.high,
            frame_skip=args.frame_skip,
        )
        actor.model.train()
        episode = episode_postproc(episode)
        eps.append(episode)
        replay_buffer.push_episode(episode)
        print(f'pre fill {pre_ep_ind} {episode}')

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for ep_ind in range(args.train_epochs):

        if global_step_ind >= args.critic_heat_up:
            # perform rollout in the env
            for epoch_ep_ind in range(args.episodes_per_epoch):

                exploration_enabled = random.uniform(0, 1) < args.exploration_prob
                used_actor = env_actor # deterministic_actor  # if exploration_enabled else actor
                used_exploration = exploration if exploration_enabled else None
                visualization_enabled = (ep_ind % args.visualize_every == 0) and args.visualize

                randomized_leg = random.randint(0, 2)
                actor.model.eval()
                episode = collect_episode(
                    env=env,
                    actor=used_actor,
                    ep_len=ep_len,
                    exploration=used_exploration,
                    visualise=visualization_enabled,
                    state_preproc=state_preproc,
                    device=args.device,
                    action_min=env.action_space.low,
                    action_max=env.action_space.high,
                    frame_skip=args.frame_skip,
                )
                actor.model.train()
                episode = episode_postproc(episode)
                print(f'{global_step_ind} {ep_ind}, {episode}')
                replay_buffer.push_episode(episode)

                if tb_logger_train is not None:
                    ep_reward = t.sum(episode.rewards).item()
                    ep_reward = {'train': ep_reward} if exploration_enabled else {'valid': ep_reward}
                    env_ep_len = {'train': episode.size} if exploration_enabled else {'valid': episode.size}
                    tb_logger_train.log({
                        'env_ep_reward': ep_reward,
                        'env_ep_len': env_ep_len,
                        'env_transitions': replay_buffer.transitions_pushed,
                        'env_eps': replay_buffer.eps_pushed,
                    }, global_step_ind)

        # max_batch_num = int(replay_buffer.transitions_pushed / args.batch_size / 2)
        # num_train_ops = min(args.train_steps_per_epoch, max_batch_num)
        num_train_ops = args.train_steps_per_epoch

        for batch_ind in range(num_train_ops):

            batch = replay_buffer.sample_batch(args.batch_size)
            batch.state = state_preproc(batch.state)
            batch.next_state = state_preproc(batch.next_state)
            batch = batch_to_device(batch, args.device)

            log_data = {}
            log_data.update(q_optimizer.train_step(batch))
            if global_step_ind >= args.critic_heat_up:
                log_data.update(awac_optimizer.train_step(batch))
            else:
                if global_step_ind % 100 == 0:
                    print(f'--- step {global_step_ind} ep {ep_ind}')

            # if batch_ind == 0:
            #     print(f'--- ep {ep_ind} batch {batch_ind} {log_data}')

            if tb_logger_train is not None:
                tb_logger_train.log(log_data, global_step_ind)

            if args.save_dir and global_step_ind % args.save_each == 0:
                t.save(actor, os.path.join(args.save_dir, 'actor.pt'))
                t.save(q_func, os.path.join(args.save_dir, 'q_func.pt'))
                t.save(q_target_func, os.path.join(args.save_dir, 'q_target_func.pt'))
                with open(os.path.join(args.save_dir, 'expinfo.pkl'), 'wb') as f:
                    pickle.dump({
                        'global_step_ind': global_step_ind,
                    }, f)

            global_step_ind += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
