import pytest
from hydra import compose, initialize
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from tire_env.base import TireWorld
from tire_env.dataclasses import TireInfo
from tire_env.grid_util import create_tire_grid


@pytest.fixture(scope="session")
def tire_world():
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="env_config")

    tire_info_paths = list(Path("./data/tires/").glob("*/tire_info.yaml"))
    tire_infos = [TireInfo.load(path) for path in tire_info_paths]
    tire_info_dict = {tire_info.name: tire_info for tire_info in tire_infos}
    target_tire_info = tire_info_dict["600"]

    env = TireWorld(**cfg)
    env.set_tire_sequence([target_tire_info] * 12)

    return env

def test_reset(tire_world: TireWorld):
    tire_world.reset()

def test_get_image(tire_world: TireWorld):
    occ, rgb = tire_world.get_images()

def test_calculate_safe_placement(tire_world: TireWorld):
    occ, rgb = tire_world.get_images()
    theta = np.pi/4
    target_tire_info = tire_world.tire_seq_list[0]

    state = tire_world.calculate_safe_placement(occ, target_tire_info, 0.1, theta)
    tire_grid, _ = create_tire_grid(target_tire_info, theta)

    index = tire_world.grid.point_to_index([state[:2]], is_int=True)[0]
    combined = np.zeros_like(occ[0])
    combined[index[1], index[0]] = 1
    overlaid = binary_dilation(combined, structure=tire_grid)
    overlaid = np.maximum(occ[0], overlaid)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(occ[0])
    ax[1].imshow(overlaid)
    plt.show()

def test_step(tire_world: TireWorld):
    target_tire_info = tire_world.tire_seq_list[0]
    
    tire_world.reset()

    for i in range(12):
        occ, _ = tire_world.get_images()
        x_rand = np.random.uniform(*tire_world.bounds[0])
        theta_rand = np.random.uniform(-np.pi/2, np.pi/2)
        tire_pose = tire_world.calculate_safe_placement(occ[0], target_tire_info, x_rand, theta_rand)
        tire = tire_world.load_tire()
        tire.set_tire_pose(tire_pose)
        is_col = tire.is_in_collision(tol=0.01)
        
        if is_col:
            tire_world.remove_last_tire()
        tire_world.world.wait_to_stablize(tol=0.05)
        is_outside = tire_world.is_misaligned_tires()
        if is_outside:
            tire_world.remove_last_tire()