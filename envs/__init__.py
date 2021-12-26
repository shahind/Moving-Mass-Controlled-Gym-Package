from mmcgym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()


# Testbed
# ----------------------------------------

register(
    id="MMCTestBed-v0",
    entry_point="mmcgym.envs.classic_control:MMCBiRotorTestBedEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="MMCTestBed-v1",
    entry_point="mmcgym.envs.classic_control:MMCBiRotorTestBedEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)
