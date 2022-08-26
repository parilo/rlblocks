import gym
from gym import ObservationWrapper


def get_state_shapes(obs_space):
    return {mod_name: tuple(obs_space[mod_name].shape) for mod_name in obs_space}


def get_state_shapes_from_ep(ep_states):
    return {mod_name: tuple(mod_vals[0].shape) for mod_name, mod_vals in ep_states.items()}


class VecToDictObsWrapper(ObservationWrapper):

    def __init__(self, env: gym.Env, key: str = 'obs'):
        super().__init__(env)
        self._key = key
        self.observation_space = {
            self._key: env.observation_space,
        }

    def observation(self, observation):
        return {self._key: observation}
