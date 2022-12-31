from gym.envs.registration import register

register(
    "PusherEnv-v0",
    entry_point="envs.mujoco.pusher_env:PusherEnv",
    kwargs={"max_episode_steps": 60, "n_tasks": 2, "goal_radius": 0.1, "max_train_radius": 0.5},
)

register(
    id="MountainCarContinuous-v0",
    entry_point="envs.mujoco.continuous_mountain_car:Continuous_MountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)
