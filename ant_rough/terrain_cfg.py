"""Terrain presets. A fixed difficulty means fixed geometry parameters, not one shape."""

import isaaclab.terrains as terrain_gen


# 720 tiles (30 x 24), versus 200 in the teammate's map. All dimensions are metres.
# No curriculum: all six kinds use the same easy parameter setting throughout.
SINGLE_DIFFICULTY_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=False,
    difficulty_range=(0.0, 0.0),
    size=(10.0, 10.0),
    num_rows=30,
    num_cols=24,
    border_width=2.0,
    # Height-field mesh spacing; independent of the scanner's 0.1m ray spacing.
    # 0.2m keeps the full 720-tile map renderable on the local 8GB GPU.
    horizontal_scale=0.2,
    vertical_scale=0.005,
    use_cache=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.10, 0.10), platform_width=1.0, border_width=0.0,
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.10, 0.10), platform_width=1.0, border_width=0.0,
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.03, 0.03), step_width=0.3,
            platform_width=1.0, border_width=0.0, holes=False,
        ),
        "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.03, 0.03), step_width=0.3,
            platform_width=1.0, border_width=0.0, holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.03, 0.03),
            platform_width=1.0, holes=False,
        ),
    },
)
