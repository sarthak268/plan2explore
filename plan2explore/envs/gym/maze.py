import numpy as np

from plan2explore.envs.gym.env import GymEnv
from plan2explore.utils.general_utils import ParamDict


class MazeEnv(GymEnv):
    """Shallow wrapper around gym env for maze envs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _default_hparams(self):
        default_dict = ParamDict({
            'start_rand_range': 2.,     # range of start position randomization, fixed pos if 0.
        })
        return super()._default_hparams().overwrite(default_dict)

    def reset(self):
        super().reset()
        if self.TARGET_POS is not None and self.START_POS is not None:
            start_pos = self.START_POS #+ self._hp.start_rand_range * (np.random.rand(2) * 4 - 2)
            self._env.set_target(self.TARGET_POS)
            self._env.reset_to_location(start_pos)
        self._env.render(mode='rgb_array')  # these are necessary to make sure new state is rendered on first frame
        obs, _, _, _ = self._env.step(np.zeros_like(self._env.action_space.sample()))
        return self._wrap_observation(obs)

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        # Change to allow maze to finish when done
        #if rew > 0:
        #    rew *= 100.
        #    done = True
        return obs, np.float64(rew), done, info     # casting reward to float64 is important for getting shape later


class ACRandMaze0S40Env(MazeEnv):
    START_POS = np.array([10., 24.])
    #TARGET_POS = np.array([18., 8.])
    TARGET_POS = np.array([8., 10.])

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': "ACRandMaze-v0",
        })
        return super()._default_hparams().overwrite(default_dict)