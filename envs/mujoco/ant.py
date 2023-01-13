import numpy as np
from gym import spaces
from .mujoco_env import MujocoEnv
from collections import deque

class AntEnv(MujocoEnv):
    def __init__(self, use_low_gear_ratio=False, max_episode_steps=100):
        if use_low_gear_ratio:
            xml_path = "low_gear_ratio_ant.xml"
        else:
            xml_path = "ant.xml"
        self._goal = np.array([-1.0, 1.0])
        self._max_episode_steps = max_episode_steps
        self.noise_coeff = 1.0
        self.state_queue = deque([np.array([0.0, 0.0])]*10, 10)
        super().__init__(
            xml_path,
            frame_skip=5,
            automatically_set_obs_and_action_space=True,
        )
        self.partial_observation_space = spaces.Box(
                    -np.inf, np.inf, shape=(47,), dtype="float64"
                )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))
        goal_reward = -1.5 * np.sum(
            np.abs(xposafter[:2] - self._goal)
        )  # make it happy, not suicidal

        reached_goal = False
        if goal_reward > -0.15:
            reached_goal = True
            goal_reward += 1.0

        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = 0.0
        #         (
        #         0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # )
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward

        done = False
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                goal_forward=goal_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
                reached_goal=reached_goal
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

    def obscure_state(self, state):
        if state[0] < 0.0:
            new_loc = state[:2] + self.np_random.uniform(
                size=2, low=self.noise_coeff*state[0], high=self.noise_coeff*state[0]
            )
        else:
            new_loc = state[:2]
        self.state_queue.pop()
        self.state_queue.appendleft(new_loc)
        obs = np.concatenate([np.hstack(self.state_queue), state[2:]])
        return obs

    def reset_model(self):
        qpos = np.copy(self.init_qpos)
        qpos[2:] = qpos[2:] + self.np_random.uniform(
            size=self.model.nq-2, low=-0.1, high=0.1
        )
        qpos[:2] = qpos[:2] + self.np_random.uniform(
            size=2, low=-2.0, high=0.0
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def sample_tasks(self, num_tasks):
        a = np.random.random(num_tasks) * 2 * np.pi
        r = 3 * np.random.random(num_tasks) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        return goals


