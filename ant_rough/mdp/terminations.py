"""Map limits are administrative truncations, separate from robot failures."""


def outside_map_safe_area(env, margin_m: float = 2.0):
    """Stop before the map edge; terrain generator centres its mesh at world (0, 0)."""
    cfg = env.scene.terrain.cfg.terrain_generator
    half_x = cfg.num_rows * cfg.size[0] / 2.0 - margin_m
    half_y = cfg.num_cols * cfg.size[1] / 2.0 - margin_m
    if margin_m < 0 or min(half_x, half_y) <= 0:
        raise ValueError("Map boundary margin must be nonnegative and smaller than half the map")
    xy = env.scene["robot"].data.root_pos_w[:, :2]
    return (xy[:, 0].abs() >= half_x) | (xy[:, 1].abs() >= half_y)
