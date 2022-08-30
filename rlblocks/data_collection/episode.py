from typing import Any, Callable, Optional

import numpy as np
from gym.envs.mujoco import MujocoEnv
import torch as t

from rlblocks.data.replay_buffer import Episode


def collect_episode(
        env: MujocoEnv,
        actor: Callable[[t.Tensor], t.Tensor],
        ep_len: int,
        exploration: Any = None,
        visualise: bool = False,
        state_preproc: Optional[Callable[[Any, str], t.Tensor]] = None,
        # add_noise_to_obs: bool = False,
        action_min: float = -1,
        action_max: float = 1,
        frame_skip: int = 1,
        device: str = 'cpu',
        set_env_start_state: Optional[Callable[[MujocoEnv], None]] = None,
) -> Episode:

    obs = env.reset()
    if set_env_start_state is not None:
        set_env_start_state(env)
    obs_list = [obs]
    actions_list = []
    reward_list = []
    if exploration is not None:
        exploration.reset()

    for step_ind in range(ep_len):
        if exploration is not None:
            noise = exploration.get_noise()
        else:
            noise = np.zeros_like(env.action_space.sample())

        # if add_noise_to_obs:
        #     obs['noise'] = noise

        actor_obs = obs

        if state_preproc:
            actor_obs = state_preproc(actor_obs).to(device).unsqueeze(0)
        action = actor(actor_obs).detach().cpu().numpy().squeeze()
        action += noise
        action = np.clip(action, action_min, action_max)

        # if exploration is not None:
        #     action = exploration(action)

        reward = 0
        for _ in range(frame_skip):
            obs, reward_substep, done, info = env.step(action)
            reward += reward_substep
            if done:
                break

        obs_list.append(obs)
        actions_list.append(action)
        reward_list.append([reward])

        if visualise:
            env.render()

        if done:
            break

    # if add_noise_to_obs:
    #     obs['noise'] = np.zeros_like(noise)

    if isinstance(obs_list[0], dict):
        states = {}
        for key in obs_list[0].keys():
            states[key] = t.as_tensor(np.array(
                [obs[key] for obs in obs_list]
            ), dtype=t.float32)
    else:
        states = t.as_tensor(np.array(obs_list), dtype=t.float32)

    return Episode(
        states=states,
        actions=t.as_tensor(np.array(actions_list), dtype=t.float32),
        rewards=t.as_tensor(np.array(reward_list), dtype=t.float32),
        size=len(actions_list)
    )
