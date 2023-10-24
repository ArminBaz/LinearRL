from gymnasium.envs.registration import register

# Simple 5x5
register(
    id="simple-5x5",
    entry_point="gym_env.envs:MazeEnv5x5",
    max_episode_steps=2000,
)

# Hairpin
register(
    id="hairpin-14x14",
    entry_point="gym_env.envs:MazeEnvHairpin",
    max_episode_steps=2000,
)

# 10x10

# Tolman detour task