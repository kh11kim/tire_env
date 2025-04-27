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
class OccPlacementPair:
    occ: np.ndarray
    placements: np.ndarray

    def save(self, path: str):
        data = {
            "occ": self.occ,
            "placements": self.placements
        }
        np.savez_compressed(path, **data)
    
    def load(self, path: str):
        data = np.load(path)
        return OccPlacementPair(data["occ"], data["placements"])


action_space_res = [45, 80, 15]

def generate_data(env:TireWorld, num_tires=8, max_retry=10):
    assert env.tire_seq_list is not None
    
    env.reset()
    xy_sizes = env.bounds[:, 1] - env.bounds[:, 0]
    xy_center = env.bounds.mean(axis=1)
    xy_grid = Grid2D(xy_sizes, action_space_res[:2], xy_center)
    theta_grid = np.linspace(-np.pi/2, np.pi/2, 15)

    retry = 0    
    while env.num_placed_tire < num_tires:
        idx = env.curr_tire_index
        tire_info = env.tire_seq_list[idx]

        env.save_tire_state()
        # stack one tire
        occ, rgb = env.get_images()
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
    occ_down = xy_grid.get_occ_grid(env.grid.pointify(occ[0]))
        
    # calculate placements
    tire_info = env.tire_seq_list[env.curr_tire_index]
    cands = calculate_all_safe_placements(
        xy_grid, occ_down, tire_info, theta_grid) # candidate
    
    placements = np.full((80, 45, 15), fill_value=-1).astype(int) # y, x, theta
    indices = xy_grid.point_to_index(cands[:, :2], is_int=True)
    placements[indices[:,1], indices[:,0]] = 0 # unknown
    np.random.shuffle(cands)
    
    tire = env.load_tire()
    env.save_tire_state()
    
    def get_stable_placement(tire_pose, tire_post_pose):
        # check if placement is valid
        theta_err = abs(tire_pose_post[2] - tire_pose[2])
        xy_err = np.linalg.norm(tire_pose_post[:2] - tire_pose[:2])
        if theta_err > np.pi/8 or xy_err > 0.1:
            tire_pose_post[1] += 0.05
            tire.set_tire_pose(tire_pose_post)
            if tire.is_in_collision():
                return None
            return tire_pose_post
        return tire_pose

    stable_poses = []
    for tire_pose in cands[:50]:
        env.restore_tire_state()
        tire.set_tire_pose(tire_pose)
        is_col = tire.is_in_collision()
        env.world.wait_to_stablize(tol=0.05)
        is_outside = env.is_misaligned_tires()
        if is_outside or is_col:
            continue
        # check if placement is valid
        tire_pose_post = tire.get_tire_pose()
        stable_pose = get_stable_placement(tire_pose, tire_pose_post)
        if stable_pose is None:
            continue
        stable_poses.append(stable_pose)
    stable_poses = np.array(stable_poses)
    if len(stable_poses) == 0:
        return None
    
    # stability check
    real_stable_poses = []
    for tire_pose in stable_poses:
        env.restore_tire_state()
        tire.set_tire_pose(tire_pose)
        env.world.wait_to_stablize(tol=0.05)
        tire_pose_post = tire.get_tire_pose()
        stable_pose = get_stable_placement(tire_pose, tire_pose_post)
        if stable_pose is None: continue
        real_stable_poses.append(stable_pose)
    stable_poses = np.array(real_stable_poses)
    
    theta = stable_poses[:, 2]
    theta[np.pi/2 < theta] -= np.pi
    theta[theta<-np.pi/2] += np.pi
    theta_res = np.pi / 15
    theta_indices = ((theta + np.pi/2) / theta_res).astype(int)
    theta_indices = np.clip(theta_indices, 0, action_space_res[2]-1)
    xy_indices = xy_grid.point_to_index(stable_poses[:, :2], is_int=True)
    placements[xy_indices[:,1], xy_indices[:,0], theta_indices] = 1 # feasible
    
    return OccPlacementPair(occ, placements)


def main(save_dir:str, num_max_tires:int, num_data:int=1000, gui=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="env_config")
    OmegaConf.set_readonly(cfg, False)
    cfg['gui'] = gui

    tire_info_paths = list(Path("./data/tires/").glob("*/tire_info.yaml"))
    tire_infos = [TireInfo.load(path) for path in tire_info_paths]
    tire_info_dict = {tire_info.name: tire_info for tire_info in tire_infos}
    target_tire_info = tire_info_dict["600"]

    env = TireWorld(**cfg)
    env.set_tire_sequence([target_tire_info] * num_max_tires)
    
    pbar = tqdm(range(num_data), total=num_data)
    for _ in pbar:
        num_tires = np.random.randint(0, num_max_tires-1)
        pair = generate_data(env, num_tires)
        if pair is not None:
            save_path = save_dir / f"{uuid.uuid4().hex}.npz"
            pair.save(save_path)

@ray.remote
def main_parallel(save_dir:str, num_max_tires:int, num_data:int=1000, gui=False):
    main(save_dir, num_max_tires, num_data, gui)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_max_tires", type=int, default=12)
    parser.add_argument("--num_data", type=int, default=100)
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--num_cores", type=int, default=0)
    args = parser.parse_args()
    
    if args.num_cores > 1:
        # use ray
        num_data_per_core = [args.num_data // args.num_cores] * args.num_cores
        num_data_per_core[-1] += args.num_data % args.num_cores

        context = ray.init(num_cpus=args.num_cores)
        print(f"dashboard: {context.dashboard_url}")
        print(ray.available_resources())

    if args.num_cores > 1:
        ray.get([main_parallel.remote(
            save_dir=args.save_dir, 
            num_max_tires=args.num_max_tires,
            num_data=num_data,
            gui=args.gui
        ) for num_data in num_data_per_core])
    
    else:
        main(
            save_dir=args.save_dir, 
            num_max_tires=args.num_max_tires,
            num_data=args.num_data,
            gui=args.gui
        )