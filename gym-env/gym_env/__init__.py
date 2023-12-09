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

# Tolman detour task
register(
    id="tolman-9x9-nb",
    entry_point="gym_env.envs:MazeEnvTolmanNB",
    max_episode_steps=2000,
)

register(
    id="tolman-9x9-b",
    entry_point="gym_env.envs:MazeEnvTolmanB",
    max_episode_steps=2000,
)

# Tolman latent task
register(
    id="tolman-10x10-latent",
    entry_point="gym_env.envs:MazeEnvTolmanLatent",
    max_episode_steps=2000,
)

# Tolman latent task
register(
    id="tolman-10x10-latent-new-goal",
    entry_point="gym_env.envs:MazeEnvTolmanLatentNewGoal",
    max_episode_steps=2000,
)

# Four room task
register(
    id="four_room_tr",
    entry_point="gym_env.envs:MazeEnv4RoomTR",
    max_episode_steps=2000,
)

register(
    id="four_room_br",
    entry_point="gym_env.envs:MazeEnv4RoomBR",
    max_episode_steps=2000,
)

# 15x15
register(
    id="simple-15x15",
    entry_point="gym_env.envs:MazeEnv15x15",
    max_episode_steps=2000,
)