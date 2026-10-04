"""Reset events, independent of any learning curriculum."""

import torch


def reselect_terrain(env, env_ids, margin_tiles: int = 2):
    """Choose a different tile on each reset, then let reset_base place the robot.

    Sampling is uniform over interior tiles, excluding margin_tiles on every side.
    Terrain geometry is never rebuilt. The previous tile is excluded when possible.
    Multiple parallel environments may share a tile; IsaacLab filters their collisions.
    """
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    rows, cols = terrain.terrain_origins.shape[:2]
    if not isinstance(margin_tiles, int) or margin_tiles < 0:
        raise ValueError("margin_tiles must be a nonnegative integer")
    inner_rows, inner_cols = rows - 2 * margin_tiles, cols - 2 * margin_tiles
    if inner_rows <= 0 or inner_cols <= 0:
        raise ValueError(
            f"Map {rows}x{cols} has no interior tiles with margin_tiles={margin_tiles}. "
            "Increase the map or explicitly lower --spawn_margin_tiles for small previews."
        )
    total = inner_rows * inner_cols
    old_row = terrain.terrain_levels[env_ids] - margin_tiles
    old_col = terrain.terrain_types[env_ids] - margin_tiles
    inside = (old_row >= 0) & (old_row < inner_rows) & (old_col >= 0) & (old_col < inner_cols)
    old = old_row * inner_cols + old_col
    if total > 1:
        # Draw from all tiles except the previous one, without rejection loops.
        selected = torch.randint(total - 1, (len(env_ids),), device=env.device)
        selected += (selected >= old).long()
        selected = torch.where(inside, selected, torch.randint(total, (len(env_ids),), device=env.device))
    else:
        selected = torch.zeros_like(old)
    new_row, new_col = selected // inner_cols + margin_tiles, selected % inner_cols + margin_tiles
    terrain.terrain_levels[env_ids] = new_row
    terrain.terrain_types[env_ids] = new_col
    terrain.env_origins[env_ids] = terrain.terrain_origins[new_row, new_col]
