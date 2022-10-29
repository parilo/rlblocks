from typing import Any, Callable, Optional, List, Tuple

import numpy as np
from gym.envs.mujoco import MujocoEnv
import torch as t

from rlblocks.data.replay_buffer import Episode


def collect_rollout_iterator(
        env: MujocoEnv,
        actor: Callable[[t.Tensor], t.Tensor],
        rollout_len: int,
        ep_len: int,
        get_exploration: Callable[[int], Any] = None,
        get_visualize: Callable[[int], bool] = None,
        # visualise: bool = False,
        # state_preproc: Optional[Callable[[Any, str], t.Tensor]] = None,
        # add_noise_to_obs: bool = False,
        action_min: float = -1,
        action_max: float = 1,
        frame_skip: int = 1,
        device: str = 'cpu',
) -> Tuple[Episode, bool, bool]:

    obs = env.reset()
    rol_ind = 0
    ep_ind = 0
    step_ind = 0
    exploration = get_exploration(ep_ind)
    visualise = get_visualize(ep_ind)
    if exploration:
        exploration.reset()

    while True:

        rollout_ended = False
        obs_list = [obs]
        actions_list = []
        reward_list = []
        done = False
        truncated = False

        # if exploration is not None:
        #     exploration.reset()

        for step_ind in range(step_ind, ep_len):
            # if exploration is not None:
            #     noise = exploration.get_noise()
            # else:
            #     noise = np.zeros_like(env.action_space.sample())

            # if add_noise_to_obs:
            #     obs['noise'] = noise

            # actor_obs = obs

            # if state_preproc:
            #     actor_obs = state_preproc(actor_obs).to(device).unsqueeze(0)
            # action = actor(actor_obs, step_ind).detach().cpu().numpy().squeeze(0)
            action = actor(obs, step_ind).detach().cpu().numpy().squeeze(0)
            if exploration:
                action = exploration(action)
            action = np.clip(action, action_min, action_max)

            # if exploration is not None:
            #     action = exploration(action)

            reward = 0
            for _ in range(frame_skip):
                obs, reward_substep, done, truncated, info = env.step(action)
                reward += reward_substep
                if done or truncated:
                    break

            obs_list.append(obs)
            actions_list.append(action)
            reward_list.append([reward])

            if visualise:
                env.render()

            rol_ind += 1
            if rol_ind % rollout_len == 0:
                rollout_ended = True
                break

            if done or truncated:
                break

        step_ind += 1

        episode_ended = (step_ind == ep_len) or done or truncated
        exploration_used = exploration is not None

        if episode_ended:
            obs = env.reset()
            ep_ind += 1
            step_ind = 0
            exploration = get_exploration(ep_ind)
            visualise = get_visualize(ep_ind)
            if exploration:
                exploration.reset()

        if isinstance(obs_list[0], dict):
            states = {}
            for key in obs_list[0].keys():
                states[key] = t.as_tensor(np.array(
                    [obs[key] for obs in obs_list]
                ), dtype=t.float32)
        else:
            states = t.as_tensor(np.array(obs_list), dtype=t.float32)

        yield Episode(
            states=states,
            actions=t.as_tensor(np.array(actions_list), dtype=t.float32),
            rewards=t.as_tensor(np.array(reward_list), dtype=t.float32),
            size=len(actions_list),
            done=done,
        ), rollout_ended, episode_ended, exploration_used
