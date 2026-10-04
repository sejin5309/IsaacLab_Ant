"""Evaluate this frozen 382-D policy in a pinned external Ant environment.

Loads only the reviewed task modules, not another IsaacLab installation.
Observations/sensing are adapted; terrain, physics, events and terminations stay native.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', choices=['assignment', 'ant_rl'], required=True)
p.add_argument('--repository', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/ant_rough_1000.pt')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--native-reward', action='store_true')
p.add_argument('--seed', type=int, default=24)
p.add_argument('--num_envs', type=int, default=100)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(a).app

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg
from ant_rough.env_cfg import MySceneCfg, ObservationsCfg
from ant_rough.agent_cfg import AntPPORunnerCfg

EXPECTED = {'assignment': 'b312c336161b986bd0db5ef506fb000d4007a397',
            'ant_rl': 'dd35ebc11b01ea57e79cc6dfa03ac6ffcbfac557'}


def load_config():
    commit = subprocess.check_output(['git', '-C', str(a.repository), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED[a.source]:
        raise RuntimeError(f'Expected reviewed commit {EXPECTED[a.source]}, got {commit}')
    directory = a.repository / ('source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant'
                                if a.source == 'assignment' else 'source/ant')
    # Avoid the foreign package's registration and automatic module discovery.
    namespace = 'transfer_' + a.source
    package = types.ModuleType(namespace)
    package.__path__ = [str(directory.resolve())]
    sys.modules[namespace] = package
    module = importlib.import_module(namespace + ('.ant_final_unseen_env_cfg' if a.source == 'assignment' else '.ant_env_cfg'))
    cfg = (module.AntFinalUnseenEnvCfg if a.source == 'assignment' else module.AntEnvCfg)()
    dump_yaml(str(a.output/'native_env.yaml'), cfg)
    cfg.observations = ObservationsCfg()
    cfg.scene.height_scanner = MySceneCfg().height_scanner
    cfg.scene.height_scanner.update_period = cfg.decimation * cfg.sim.dt
    cfg.scene.clone_in_fabric = False
    if a.source == 'ant_rl':
        # The checkpoint has no image encoder. Remove the unused render-only camera.
        cfg.scene.depth_camera = None
        cfg.scene.contact_forces.debug_vis = False
    if not a.native_reward:
        cfg.rewards = RewardsCfg()
    elif a.source == 'ant_rl':
        base = module.rewards.TotalReward
        class MeasuredTotalReward(base):
            """Read-only component capture before RewardManager's auto-reset.

            Native float32 episode-sum differences have small rounding error;
            compare their accumulated sum to the actual env.step return.
            """
            def __call__(self, env):
                before = {k: v.clone() for k, v in self.episode_sums.items()}
                result = super().__call__(env)
                self.last_contributions = {k: self.episode_sums[k].double()-v.double() for k,v in before.items()}
                return result
        cfg.rewards.total_reward.func = MeasuredTotalReward
    cfg.seed = a.seed
    cfg.scene.num_envs = a.num_envs
    if a.device:
        cfg.sim.device = a.device
    # Capture the terminal position BEFORE automatic reset. Always returns False.
    def capture_terminal_state(env):
        env.transfer_pre_reset_pos = env.scene['robot'].data.root_pos_w.clone()
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    cfg.terminations.transfer_measurement = TerminationTermCfg(func=capture_terminal_state)
    cfg.log_dir = str(a.output)
    dump_yaml(str(a.output/'adapted_env.yaml'), cfg)
    return cfg, commit


def main():
    cfg, commit = load_config()
    agent = AntPPORunnerCfg()
    agent.seed = a.seed
    agent.device = cfg.sim.device
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg), clip_actions=agent.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(a.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        obs = env.get_observations()
        assert obs['policy'].shape == (a.num_envs, 382) and env.num_actions == 8
        core = env.unwrapped
        initial = core.scene['robot'].data.root_pos_w.clone()
        np.savez(a.output/'initial_state.npz', root=core.scene['robot'].data.root_state_w.cpu().numpy(),
                 joint_pos=core.scene['robot'].data.joint_pos.cpu().numpy(),
                 joint_vel=core.scene['robot'].data.joint_vel.cpu().numpy(),
                 origins=core.scene.env_origins.cpu().numpy(),
                 observations=obs['policy'].cpu().numpy(),
                 materials=core.scene['robot'].root_physx_view.get_material_properties().cpu().numpy())
        reward_fn = core.reward_manager.get_term_cfg('total_reward').func if a.native_reward and a.source == 'ant_rl' else None
        names = list(reward_fn.episode_sums) if reward_fn else core.reward_manager.active_terms
        returns = torch.zeros(a.num_envs, device=env.device, dtype=torch.float64)
        components = torch.zeros((a.num_envs, len(names)), device=env.device, dtype=torch.float64)
        lengths = torch.zeros(a.num_envs, device=env.device, dtype=torch.int64)
        done = torch.zeros(a.num_envs, device=env.device, dtype=torch.bool)
        rows = {}
        for _ in range(env.max_episode_length):
            with torch.inference_mode():
                obs, reward, ended, _ = env.step(policy(obs))
                active = ~done
                returns[active] += reward[active]
                lengths[active] += 1
                contribution = (torch.stack([reward_fn.last_contributions[k] for k in names], dim=1)
                                if reward_fn else core.reward_manager._step_reward.double()*core.step_dt)
                components[active] += contribution[active]
                for idx in (active & ended.bool()).nonzero().flatten().tolist():
                    rows[str(idx)] = {'return': returns[idx].item(), 'steps': lengths[idx].item(),
                       'displacement_x_m': (core.transfer_pre_reset_pos[idx,0]-initial[idx,0]).item(),
                       'terminated': bool(core.termination_manager.terminated[idx]),
                       'terminations': {k: bool(core.termination_manager.get_term(k)[idx]) for k in core.termination_manager.active_terms if k != 'transfer_measurement'},
                       'reward_components': dict(zip(names, components[idx].tolist()))}
                done |= ended.bool()
                if done.all():
                    break
        error = (components.sum(1)-returns).abs().max().item()
        if not done.all() or error > (1e-3 if reward_fn else 1e-4):
            raise RuntimeError(f'Invalid accounting: completed={done.sum()}, error={error}')
        result = {'source': a.source, 'repository_commit': commit, 'seed': a.seed, 'num_envs': a.num_envs,
           'complete': True, 'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
           'reward_definition': 'native training reward' if a.native_reward else 'original Ant seven terms',
           'terrain_seed': cfg.scene.terrain.terrain_generator.seed,
           'return_mean': returns.mean().item(), 'return_std_population': returns.std(unbiased=False).item(),
           'steps_mean': lengths.double().mean().item(), 'steps_std_population': lengths.double().std(unbiased=False).item(),
           'failure_count': sum(r['terminated'] for r in rows.values()),
           'displacement_x_mean_m': np.mean([r['displacement_x_m'] for r in rows.values()]),
           'reward_component_means': dict(zip(names, components.mean(0).tolist())),
           'max_component_sum_error': error, 'per_env': rows}
        (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print('[RESULT]', json.dumps({k:v for k,v in result.items() if k != 'per_env'}), flush=True)
    finally:
        env.close()

if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
