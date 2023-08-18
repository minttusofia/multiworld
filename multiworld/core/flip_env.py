import numpy as np

from multiworld.core.wrapper_env import ProxyEnv


class FlipEnv(ProxyEnv):
    def __init__(
            self,
            wrapped_env,
            obs_keys=None,
            goal_keys=None,
            append_goal_to_obs=False,
    ):
        self.quick_init(locals())
        super(FlipEnv, self).__init__(wrapped_env)
        # Flipping only implemented for one stacked image type.
        assert len(wrapped_env.obs_keys) == 1

    def _update_obs(self, obs):
        hw = int(np.sqrt(obs.shape[-1] // 3))
        obs = np.reshape(obs, [hw, hw, 3])
        obs = obs[::-1, ::-1]
        obs = obs.flatten()
        return obs

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        obs = self._update_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        obs = self._update_obs(obs)
        return obs

