from gym.envs.registration import register

register(
    id="AntGoal-v0",
    entry_point="envs.mujoco.ant:AntEnv",
    max_episode_steps=100,
    reward_threshold=1000.0,
)

register(
    id="MountainCarContinuous-v0",
    entry_point="envs.mujoco.continuous_mountain_car:Continuous_MountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)
