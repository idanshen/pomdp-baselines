import numpy as np
import torch
from .mujoco_env import MujocoEnv

import pickle
import matplotlib.pyplot as plt


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param

class PusherEnv(MujocoEnv):

    def __init__(self,
        max_episode_steps=60,
        max_train_radius=0.5,
        min_train_radius=0.0,
        goal_radius=0.1,
        **kwargs
    ):
        xml_path = 'pusher_env.xml'

        self._max_episode_steps = max_episode_steps
        self.step_count = 0

        self.max_train_radius = max_train_radius
        self.min_train_radius = min_train_radius
        self.goal_radius = goal_radius

        radius = np.random.uniform(self.min_train_radius, self.max_train_radius, size=1)
        # angles = np.random.uniform(0, 2*np.pi, size=n_tasks)
        angles = np.random.uniform(-np.pi / 4, np.pi / 4, size=1)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        goals = np.stack([xs, ys], axis=1)
        self._goal = [0.5, 0.5]  # goals[0]

        self.obj_idx = None
        super().__init__(
            xml_path,
            frame_skip=10,
            automatically_set_obs_and_action_space=True,
        )

    def _get_obs(self):
        if self.obj_idx is None:
            self.obj_idx = self.sim.model.body_names.index("obj1")

        return np.concatenate([
            self.sim.data.qpos.flat[:3],
            self.sim.data.geom_xpos[-1,:2].flat,
            self.sim.data.qvel.flat,
            self.sim.model.body_mass[self.obj_idx].flat
        ]).reshape(-1)

    def get_current_task(self):
        # for multi-task MDP
        return self._goal.copy()

    def reset_model(self):
        # radius = np.random.uniform(self.min_train_radius, self.max_train_radius, size=1)
        # # angles = np.random.uniform(0, 2*np.pi, size=n_tasks)
        # angles = np.random.uniform(-np.pi / 4, np.pi / 4, size=1)
        # xs = radius * np.cos(angles)
        # ys = radius * np.sin(angles)
        # goals = np.stack([xs, ys], axis=1)
        # self._goal = goals[0]
        # self.sim.data.body_xpos[-1, :2] = self._goal

        self.step_count = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        self.sim.forward()
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()
        curr_block_pos = np.array([next_obs[3], next_obs[4]])
        block_dist = np.linalg.norm(self._goal - curr_block_pos)

        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))

        if block_dist > self.goal_radius:
            # reward = -1.0 - ctrl_cost
            reward = -block_dist
            reached_goal = False
        else:
            reward = -block_dist
            reached_goal = True

        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True
        else:
            done = False

        return next_obs, reward, done, dict(ctrl_cost=ctrl_cost, distance=np.linalg.norm(next_obs[:2] - self._goal), reached_goal=reached_goal)

    def get_all_task_idx(self):
        return range(len(self.goals))

    def is_goal_state(self):
        obs = self._get_obs()
        if np.linalg.norm(obs[3:5] - self._goal) <= self.goal_radius:
            return True
        else:
            return False

    def plot_env(self):
        ax = plt.gca()
        # plot full circle and goal position
        angles = np.linspace(0, 2*np.pi, num=100)
        x, y = self.max_train_radius*np.cos(angles), self.max_train_radius*np.sin(angles)
        plt.plot(x, y, color="k")
        # fix visualization
        plt.axis("scaled")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plt.xticks([])
        plt.yticks([])
        circle = plt.Circle(
            (self._goal[0], self._goal[1]), radius=self.goal_radius, alpha=0.3
        )
        ax.add_artist(circle)

    def plot_behavior(self, observations, plot_env=True, **kwargs):
        # kwargs are color and label
        if plot_env:  # whether to plot circle and goal pos..(maybe already exists)
            self.plot_env()
        # label the starting point
        plt.scatter(observations[[0], 3], observations[[0], 4], marker="x", **kwargs)
        # plot trajectory
        plt.plot(observations[:, 3], observations[:, 4], **kwargs)