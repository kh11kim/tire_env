import numpy as np
from hydra import compose, initialize
from pathlib import Path
from tire_env.dataclasses import TireInfo
from tire_env.base import TireWorld
from tire_env.grid_util import Grid2D, calculate_all_safe_placements
from omegaconf import OmegaConf
from dataclasses import dataclass
import uuid
import ray
from tqdm import tqdm

@dataclass
class Config:
    save_dir: str
    num_max_tires: int
    num_data: int
    theta_res: int
    gui: bool
    num_cores: int
    num_trial: int
    verify: bool # verify stable poses
    only_placeable: bool

@dataclass
class OccPlacementPairV2:
    occ: np.ndarray
    placeable: np.ndarray # x, theta
    stable: np.ndarray # x, theta
    unstable: np.ndarray # x, theta

    def save(self, path: str):
        data = {
            "occ": self.occ.astype(bool),
            "placeable": self.placeable.astype(bool),
            "stable": self.stable.astype(bool),
            "unstable": self.unstable.astype(bool)
        }
        np.savez_compressed(path, **data)
    
    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        return cls(
            occ=data["occ"], 
            placeable=data["placeable"],
            stable=data["stable"],
            unstable=data["unstable"]
        )

action_theta_res = 15

def make_scene(env:TireWorld, num_tires=8, max_retry=20):
    env.reset()

    retry = 0    
    while env.num_placed_tire < num_tires:
        idx = env.curr_tire_index
        tire_info = env.tire_seq_list[idx]

        env.save_tire_state()
        # stack one tire
        occ, _ = env.get_images()
        tire = env.load_tire()    
        x_rand = np.random.uniform(*env.bounds[0])
        theta_rand = np.random.uniform(-np.pi/2, np.pi/2)
        tire_pose = env.calculate_safe_placement(
            occ[0], tire_info, x_rand, theta_rand
        )
        tire.set_tire_pose(tire_pose)
        env.world.wait_to_stablize(tol=0.05)
        is_outside = env.is_misaligned_tires()
        if is_outside:
            env.remove_last_tire()
            env.restore_tire_state()
            retry += 1
        if retry > max_retry:
            break
    occ, _ = env.get_images()
    return occ

def stability_criteria(tire_pose, tire_pose_post):
    theta_err = abs(tire_pose_post[2] - tire_pose[2])
    xy_err = np.linalg.norm(tire_pose_post[:2] - tire_pose[:2])
    return theta_err < np.pi/8 and xy_err < 0.1

def validate_placement(stable_poses, env:TireWorld):
    if stable_poses is None:
        return None
    tire = env.tires[-1]

    # stability check
    real_stable_poses = []
    for tire_pose in stable_poses:
        env.restore_tire_state()
        tire.set_tire_pose(tire_pose)
        env.world.wait_to_stablize(tol=0.05)

        if not stability_criteria(tire_pose, tire.get_tire_pose()):
            continue
        real_stable_poses.append(tire_pose)
    
    real_stable_poses = np.array(real_stable_poses)
    if len(real_stable_poses) == 0:
        return None
    
    return real_stable_poses

def get_stable_placements(env:TireWorld, xy_grid:Grid2D, theta_grid:np.ndarray, num_trial:int = 50):
    occ, _ = env.get_images()
    
    # calculate placements
    tire_info = env.tire_seq_list[env.curr_tire_index]
    cands = calculate_all_safe_placements(
        xy_grid, occ, tire_info, theta_grid) # candidate
    
    np.random.shuffle(cands)
    cands_sampled = cands[:num_trial]
    
    tire = env.load_tire()
    env.save_tire_state()

    stable_poses = []
    infeasible_poses = []
    for tire_pose in cands_sampled:
        env.restore_tire_state()
        tire.set_tire_pose(tire_pose)
        is_col = tire.is_in_collision()
        env.world.wait_to_stablize(tol=0.05)
        is_outside = env.is_misaligned_tires()
        if is_outside or is_col:
            infeasible_poses.append(tire_pose)
            continue
        
        # check if placement is valid
        tire_pose_post = tire.get_tire_pose()
        if not stability_criteria(tire_pose, tire_pose_post):
            infeasible_poses.append(tire_pose)
            tire_pose_post[1] += 0.05
            tire.set_tire_pose(tire_pose_post)
            if tire.is_in_collision():
                infeasible_poses.append(tire_pose)
                continue
            stable_poses.append(tire_pose_post)
        else:
            stable_poses.append(tire_pose)
    
    poses = []
    for pose in [stable_poses, infeasible_poses, cands]:
        if len(pose) == 0:
            poses.append(None)
        else:
            poses.append(np.array(pose))
    return poses

def get_data(
    occ, 
    stable_poses, 
    infeasible_poses, 
    cands, 
    xy_grid:Grid2D, 
    theta_grid_res:tuple,
    check_placeable:bool = True
):
    resolution = (xy_grid.res[1], xy_grid.res[0], theta_grid_res)
    
    # default: known, not placable
    stable = np.zeros(resolution).astype(bool) # y, x, theta
    unstable = np.zeros(resolution).astype(bool) # y, x, theta
    placeable = np.zeros(resolution).astype(bool)
    
    def get_xyt_indices(poses, resolution):
        theta_grid_size = np.pi / resolution[-1]
        xy = poses[:, :2]
        theta = poses[:, 2] # rad
        theta[np.pi/2 < theta] -= np.pi
        theta[theta<-np.pi/2] += np.pi
        theta = np.clip(theta, -np.pi/2, np.pi/2 - 1e-4)
        theta_indices = ((theta + np.pi/2) / theta_grid_size).astype(int)
        xy_indices = xy_grid.point_to_index(xy, is_int=True)
        return np.concatenate([xy_indices, theta_indices[:,np.newaxis]], -1) # x, y, theta
    
    if stable_poses is not None:
        # check stable poses as known, placeable
        indices = get_xyt_indices(stable_poses, resolution)
        stable[indices[:,1], indices[:,0], indices[:,2]] = True # stable

    if infeasible_poses is not None:
        # check infeasible poses as known
        indices = get_xyt_indices(infeasible_poses, resolution)
        unstable[indices[:,1], indices[:,0], indices[:,2]] = True # unstable
    
    indices = get_xyt_indices(cands, resolution)
    placeable[indices[:,1], indices[:,0], indices[:,2]] = True # placeable

    # remove z axis
    stable = stable.argmax(0).astype(bool) # H, W, theta -> W, theta
    unstable = unstable.argmax(0).astype(bool) # H, W, theta -> W, theta
    placeable = placeable.argmax(0).astype(bool) # H, W, theta -> W, theta
    
    if check_placeable:
        stable = placeable & stable
        unstable = placeable & unstable

    return OccPlacementPairV2(
        occ=occ, 
        placeable=placeable,
        stable=stable,
        unstable=unstable,
    )

def generate_data(
    env:TireWorld, 
    num_tires:int, 
    config:Config
) -> OccPlacementPairV2:
    assert env.tire_seq_list is not None
    
    xy_sizes = env.bounds[:, 1] - env.bounds[:, 0]
    xy_center = env.bounds.mean(axis=1)
    
    xy_grid = Grid2D(xy_sizes, env.resolutions, xy_center)
    theta_grid = np.linspace(-np.pi/2, np.pi/2, config.theta_res)
    
    occ = make_scene(env, num_tires)
    poses = get_stable_placements(
        env, xy_grid, theta_grid,
        num_trial=config.num_trial
    )
    stable, infeasible, cands = poses
    
    if config.verify and stable is not None:
        stable = validate_placement(stable, env)        
    
    data = get_data(
        occ=occ, 
        stable_poses=stable, 
        infeasible_poses=infeasible, 
        cands=cands,
        xy_grid=xy_grid, 
        theta_grid_res=config.theta_res,
        check_placeable=config.only_placeable
    )
    return data
    
    


def main(cfg:Config):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with initialize(config_path="../config", version_base=None):
        env_cfg = compose(config_name="env_config")
        OmegaConf.set_readonly(env_cfg, False)
        env_cfg['gui'] = cfg.gui

    tire_info_paths = list(Path("./data/tires/").glob("*/tire_info.yaml"))
    tire_infos = [TireInfo.load(path) for path in tire_info_paths]
    tire_info_dict = {tire_info.name: tire_info for tire_info in tire_infos}
    target_tire_info = tire_info_dict["600"]

    env = TireWorld(**env_cfg)
    env.set_tire_sequence([target_tire_info] * cfg.num_max_tires)
    
    pbar = tqdm(range(cfg.num_data), total=cfg.num_data)
    for _ in pbar:
        num_tires = np.random.randint(0, cfg.num_max_tires-1)
        pair = generate_data(
            env=env, 
            num_tires=num_tires,
            config=cfg
        )
        if pair is not None:
            save_path = save_dir / f"{uuid.uuid4().hex}.npz"
            pair.save(save_path)

@ray.remote
def main_parallel(cfg:Config):
    main(cfg)

def parse_config() -> Config:
    import argparse
    import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_max_tires", type=int, default=16)
    parser.add_argument("--num_data", type=int, default=100)
    parser.add_argument("--theta_res", type=int, default=15)
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_cores", type=int, default=0)
    parser.add_argument("--num_trial", type=int, default=100)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-placeable", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(vars(args))
    return Config(**vars(args))

if __name__ == "__main__":
    cfg = parse_config()
    
    if cfg.num_cores <= 1:
        main(cfg)
    else:
        # use ray
        num_data_per_core = cfg.num_data // cfg.num_cores
        remainder = cfg.num_data % cfg.num_cores
        split = [num_data_per_core + 1 
            if i < remainder else num_data_per_core for i in range(cfg.num_cores)]

        context = ray.init(num_cpus=cfg.num_cores)
        print(f"dashboard: {context.dashboard_url}")
        print(ray.available_resources())

        jobs = []
        for num_data in split:
            cfg.gui = False
            cfg.num_data = num_data
            jobs.append(main_parallel.remote(cfg))
        ray.get(jobs)