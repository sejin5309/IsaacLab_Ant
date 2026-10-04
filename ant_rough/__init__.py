"""Import after AppLauncher to register the standalone E8A inference tasks."""

import gymnasium as gym

for suffix, cfg in (("SelfEval", "AntSelfEvaluationCfg"), ("Medium", "AntMediumPreviewCfg")):
    gym.register(
        id=f"Isaac-Ant-Rough-E8A-{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"ant_rough.env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": "ant_rough.agent_cfg:AntPPORunnerCfg",
        },
    )
