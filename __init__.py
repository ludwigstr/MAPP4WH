from gymnasium.envs.registration import register

register(
    id="WireHarness-v0",
    entry_point="wire_harness_env:WireHarnessEnv",
)
