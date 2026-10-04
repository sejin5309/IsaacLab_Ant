"""Episode diagnostics; they never change rewards or terminate an episode."""

import math
import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


class TorsoContactMonitor(ManagerTermBase):
    """Sample after physics, before automatic reset, including the terminal step.

    Kept under the original class name for saved-config compatibility. Tracks torso
    contact, signed X displacement, horizontal goal progress, and lack of progress.
    Stagnation = mean goal-directed progress speed over the last full second <0.05m/s.
    The first window after reset is warm-up and does not enter that ratio.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.window_steps = max(1, math.ceil(cfg.params.get("stagnation_window_s", 1.0) / self.dt))
        self.target = torch.tensor(cfg.params.get("target_pos", (1000.0, 0.0)), device=env.device)
        self.contact_steps = torch.zeros(env.num_envs, device=env.device)
        self.total_steps = torch.zeros_like(self.contact_steps)
        self.stagnant_steps = torch.zeros_like(self.contact_steps)
        self.valid_stagnation_steps = torch.zeros_like(self.contact_steps)
        self.forward_distance = torch.zeros_like(self.contact_steps)
        self.progress_distance = torch.zeros_like(self.contact_steps)
        self.start_xy = env.scene["robot"].data.root_pos_w[:, :2].clone()
        self.start_distance = torch.linalg.vector_norm(self.target - self.start_xy, dim=-1)
        self.distance_history = self.start_distance.repeat(self.window_steps, 1)
        self.cursor = 0
        self.completed_episodes = 0
        self.last_episode = {
            name: torch.zeros_like(self.contact_steps)
            for name in ("forward_distance_m", "goal_progress_m", "torso_contact_ratio",
                         "stagnation_ratio", "stagnation_sample_seconds", "duration_s")
        }
        # Accessed by preview and training loggers to retrieve per-environment results.
        env.locomotion_monitor = self

    def __call__(
        self, env, sensor_cfg: SceneEntityCfg, force_threshold: float,
        stagnation_window_s: float = 1.0, stagnation_speed_mps: float = 0.05,
        target_pos: tuple = (1000.0, 0.0),
    ):
        forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w
        contact = torch.linalg.vector_norm(forces, dim=-1).amax(dim=-1) > force_threshold
        self.contact_steps += contact.float()
        self.total_steps += 1
        xy = env.scene["robot"].data.root_pos_w[:, :2]
        distance = torch.linalg.vector_norm(self.target - xy, dim=-1)
        self.forward_distance[:] = xy[:, 0] - self.start_xy[:, 0]
        self.progress_distance[:] = self.start_distance - distance
        window_speed = (self.distance_history[self.cursor] - distance) / (self.window_steps * self.dt)
        ready = self.total_steps >= self.window_steps
        self.valid_stagnation_steps += ready.float()
        self.stagnant_steps += (ready & (window_speed < stagnation_speed_mps)).float()
        self.distance_history[self.cursor] = distance
        self.cursor = (self.cursor + 1) % self.window_steps
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            selection = env_ids if isinstance(env_ids, slice) else slice(None)
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
            env_ids = env_ids[selection]
        else:
            env_ids = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
        steps = self.total_steps[env_ids]
        valid = steps > 0
        ids = env_ids[valid]
        if len(ids):
            duration = self.total_steps[ids] * self.dt
            values = {
                "forward_distance_m": self.forward_distance[ids],
                "goal_progress_m": self.progress_distance[ids],
                "torso_contact_ratio": self.contact_steps[ids] / self.total_steps[ids],
                "stagnation_ratio": self.stagnant_steps[ids] / self.valid_stagnation_steps[ids].clamp_min(1),
                "stagnation_sample_seconds": self.valid_stagnation_steps[ids] * self.dt,
                "duration_s": duration,
            }
            self.completed_episodes += len(ids)
            for name, value in values.items():
                self.last_episode[name][ids] = value
                if name == "stagnation_ratio":
                    measured = self.valid_stagnation_steps[ids] > 0
                    # A too-short episode has no measurement; never log it as zero stagnation.
                    if measured.any():
                        self._env.extras["log"][f"Diagnostics/{name}"] = value[measured].mean()
                else:
                    self._env.extras["log"][f"Diagnostics/{name}"] = value.mean()
        self.contact_steps[env_ids] = 0
        self.total_steps[env_ids] = 0
        self.stagnant_steps[env_ids] = 0
        self.valid_stagnation_steps[env_ids] = 0
        self.forward_distance[env_ids] = 0
        self.progress_distance[env_ids] = 0
        # Reset events already wrote the NEW spawn position; do not count teleportation.
        xy = self._env.scene["robot"].data.root_pos_w[env_ids, :2]
        self.start_xy[env_ids] = xy
        distance = torch.linalg.vector_norm(self.target - xy, dim=-1)
        self.start_distance[env_ids] = distance
        self.distance_history[:, env_ids] = distance.unsqueeze(0)
