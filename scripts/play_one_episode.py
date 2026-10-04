# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Portable E8A playback; first-episode accounting follows IsaacLab's assignment runner."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Ant-Rough-E8A-SelfEval-v0",
                    choices=["Isaac-Ant-Rough-E8A-SelfEval-v0", "Isaac-Ant-Rough-E8A-Medium-v0"])
parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/e8a_model_399.pt")
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--output", type=Path, default=ROOT / "outputs/latest")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=960)
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--show-scanner", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.video_length < 1:
    parser.error("num_envs and video_length must be positive")
if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
# Never overwrite an earlier evaluation silently.
args.output.mkdir(parents=True, exist_ok=False)
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import ant_rough  # noqa: F401
from ant_rough.agent_cfg import AntPPORunnerCfg
from ant_rough.env_cfg import AntSelfEvaluationCfg, AntMediumPreviewCfg


def main():
    cfg = AntSelfEvaluationCfg() if "SelfEval" in args.task else AntMediumPreviewCfg()
    agent = AntPPORunnerCfg()
    cfg.seed = agent.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = agent.device = args.device
    cfg.scene.height_scanner.debug_vis = args.show_scanner
    cfg.log_dir = str(args.output)
    cfg.viewer.eye = (-4.0, 4.0, 2.5)
    cfg.viewer.lookat = (0.0, 0.0, 0.5)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(args.output / "videos"), step_trigger=lambda step: step == 0,
            video_length=args.video_length, disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        if obs["policy"].shape[-1] != 382 or env.num_actions != 8:
            raise RuntimeError("E8A requires exactly 382 ordered observations and 8 effort actions")
        returns = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
        lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        per_env = {}
        for _ in range(env.max_episode_length):
            if not app.is_running():
                break
            start = time.perf_counter()
            with torch.inference_mode():
                obs, rewards, dones, extras = env.step(policy(obs))
                active = ~finished
                returns[active] += rewards[active]
                lengths[active] += 1
                newly_finished = active & dones.bool()
                manager = env.unwrapped.termination_manager
                monitor = env.unwrapped.locomotion_monitor
                for idx in newly_finished.nonzero(as_tuple=False).flatten().tolist():
                    per_env[str(idx)] = {
                        "return": returns[idx].item(), "steps": lengths[idx].item(),
                        "failure": bool(manager.get_term("body_orientation")[idx].item()),
                        "boundary_exit": bool(manager.get_term("map_boundary")[idx].item()),
                        "time_out": bool(manager.get_term("time_out")[idx].item()),
                        **{name: values[idx].item() for name, values in monitor.last_episode.items()},
                    }
                finished |= dones.bool()
            if finished.all().item():
                print(f"[INFO] All {env.num_envs} environments finished their first episode.")
                break
            if args.real_time:
                time.sleep(max(0, env.unwrapped.step_dt - (time.perf_counter() - start)))
        complete = bool(finished.all().item())
        result = {
            "complete": complete, "completed": int(finished.sum().item()), "num_envs": env.num_envs,
            "seed": args.seed, "terrain_seed": cfg.scene.terrain.terrain_generator.seed, "task": args.task,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "reward_definition": "Original Isaac-Ant-v0 RewardsCfg, all seven terms unchanged",
            "return_mean": returns.mean().item(), "return_std_population": returns.std(unbiased=False).item(),
            "steps_mean": lengths.double().mean().item(),
            "steps_std_population": lengths.double().std(unbiased=False).item(),
            "failure_count": sum(row["failure"] for row in per_env.values()),
            "boundary_exit_count": sum(row["boundary_exit"] for row in per_env.values()),
            "per_env": per_env,
        }
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[INFO] Completed first episodes: {result['completed']}/{env.num_envs}")
        if not complete:
            raise RuntimeError("Incomplete evaluation; saved statistics include partial episodes")
        print(f"[RESULT] Episode reward total: mean={result['return_mean']:.6f}, std={result['return_std_population']:.6f}")
        print(f"[RESULT] Episode steps: mean={result['steps_mean']:.6f}, std={result['steps_std_population']:.6f}")
        print(f"[RESULT] Falls={result['failure_count']}, boundary exits={result['boundary_exit_count']}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
