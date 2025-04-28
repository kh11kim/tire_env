from typing import Optional, Union

import ezbullet as eb
import gymnasium as gym
import numpy as np
import trimesh
import pybullet as p
from ezpose import SE3, SO3

from scipy.ndimage import binary_dilation

from .dataclasses import TireInfo
from .grid_util import Grid2D, create_tire_grid
from .tire_util import tire_pose_gen

class Tire(eb.Mesh):
    @classmethod
    def from_body(cls, body: eb.Body):
        return cls(body.world, body.uid, body.name, body.mass, body.ghost)
    
    def get_tire_pose(self):
        pose = self.get_pose()
        xoff = pose.trans[0]
        zoff = pose.trans[2]
        pitch = -pose.rot.as_rotvec()[1] # rad
        return np.array([xoff, zoff, pitch])
    
    def set_tire_pose(self, pose: np.ndarray, degree=False):
        xoff, zoff, pitch = pose
        rot = SO3.from_euler("Y", -pitch, degrees=degree)
        self.set_pose(SE3(trans=[xoff, 0, zoff], rot=rot))




class TireWorld:
    def __init__(
        self,
        gui: bool=True,
        x_bound: tuple[float]=(-1.121, 1.121),
        height: float=4.0,
        num_initial_tires: int=1,
        y_tol: float=0.1,
        **kwargs
    ):
        # state & action bound
        self.gui = gui
        self.bounds = np.array([
            [x_bound[0], x_bound[1]],
            [0.0, height]])
        
        # tire loading config
        self.num_initial_tires = num_initial_tires
        self.y_tol = y_tol
        self.sim_config = {}
        if "simulation_config" in kwargs:
            self.sim_config = kwargs["simulation_config"]
        
        # setup world
        self.world = eb.World(
            gui=self.gui,
            solver_iter=10, 
            dt=0.01, 
            realtime=False)
        self.setup_world(self.world)
        self.world.configureDebugVisualizer(
            p.COV_ENABLE_GUI, False)
        self.world.watch_workspace(
            target_pos=[0, 3., 0.8],
            distance=6, cam_yaw=0, cam_pitch=-15)

        # vision
        pixel_size = 0.025
        sizes = self.bounds[:, 1] - self.bounds[:, 0]
        resolutions = sizes / pixel_size
        center = self.bounds.mean(axis=1)
        self.resolutions = resolutions.astype(int)
        self.grid = Grid2D(sizes, resolutions, center)
        
        self.cameras:dict[str, eb.Camera] = None
        self.camera_poses:dict[str, SE3] = None
        if "cam_config" in kwargs:
            self.cameras, self.camera_poses = \
                self.initialize_camera(
                    self.world, kwargs["cam_config"])
        
        # world states
        self.tires: list[Tire] = []
        self.tire_seq_list: list[TireInfo] = None
    
    def reset(self):
        self.remove_all_tires()
        self.load_initial_tire_heuristic(self.num_initial_tires)

    @property
    def num_placed_tire(self):
        return len(self.tires)
    
    @property
    def curr_tire_index(self):
        return self.num_placed_tire
    
    @property
    def all_stacked(self):
        return self.num_placed_tire == len(self.tire_seq_list)
    
    @property
    def env_objects(self):
        return list(self.env_geoms.values())
    
    """Initialize"""
    def setup_world(self, world:eb.World):
        world.setDefaultContactERP(1.0)
        ground_extents = np.array([4, 4, 0.25]) * 2
        ground_mesh = trimesh.creation.box(ground_extents)
        ground = eb.URDF.from_trimesh(
            "ground", world, ground_mesh, ground_mesh, fixed=True
        )
        ground_pose = SE3(trans=[0, 0, -0.25])
        ground.set_pose(ground_pose)

        wall_t = 0.054 * 2
        wall_width = 1.
        wall_height = self.bounds[1, 1]
        wall_extents = np.array([wall_t, wall_width, wall_height])
        rgba = [0.192, 0.804, 0.463, 1]
        wall_mesh = trimesh.creation.box(wall_extents)
        left_wall = eb.URDF.from_trimesh(
            "left_wall", world, wall_mesh, wall_mesh, fixed=True, rgba=rgba
        )
        right_wall = eb.URDF.from_trimesh(
            "right_wall", world, wall_mesh, wall_mesh, fixed=True, rgba=rgba
        )

        left_wall_pose = SE3(trans=[self.bounds[0, 0]-wall_t, 0, wall_height/2])
        right_wall_pose = SE3(trans=[self.bounds[0, 1]+wall_t, 0, wall_height/2])
        left_wall.set_pose(left_wall_pose)
        right_wall.set_pose(right_wall_pose)

        back_wall_size_x = self.bounds[0, 1] - self.bounds[0, 0] + wall_t * 2
        back_wall_extents = np.array([back_wall_size_x, wall_t, wall_height])
        back_wall_mesh = trimesh.creation.box(back_wall_extents)
        back_wall = eb.URDF.from_trimesh(
            "back_wall",
            world,
            back_wall_mesh,
            back_wall_mesh,
            fixed=True,
            rgba=rgba,
        )
        back_wall_pose = SE3(trans=[0, wall_width/2, wall_height/2])
        back_wall.set_pose(back_wall_pose)

        self.env_geoms = {
            "ground": ground,
            "left_wall": left_wall,
            "right_wall": right_wall,
            "back_wall": back_wall,
        }
        for geom in self.env_geoms.values():
            geom.set_dynamics_info(
                lateralFriction=1.0,
                rollingFriction=0.01,
                spinningFriction=0.001,
                frictionAnchor=True,
                contactProcessingThreshold=0.003)
    
    def initialize_camera(self, world:eb.World, cam_config:dict):
        cameras = dict()
        camera_poses = dict()
        for name, cfg in cam_config.items():
            intr = eb.CameraIntrinsic(
                width=cfg["width"],
                height=cfg["height"],
                fx=cfg["fx"],
                fy=cfg["fy"],
                cx=cfg["cx"],
                cy=cfg["cy"],
                near=0.1,
                far=5.0)
            cameras[name] = eb.Camera(world, intr)
            camera_poses[name] = SE3(
                trans=cfg["pos"], 
                rot=SO3.from_euler("ZYX", cfg["rpy"][::-1], degrees=True))
            
            # draw camera
            if self.gui:
                w, h = cfg["width"]/10000, cfg["height"]/10000
                world.draw_frustum(camera_poses[name], w, h)
                frame = eb.Frame.create("cam", world)
                frame.set_pose(camera_poses[name])

        return cameras, camera_poses
    
    """Load/Remove Tire"""
    def load_tire(self) -> Tire:
        index = self.curr_tire_index
        tire_info: TireInfo = self.tire_seq_list[index]
        index = self.curr_tire_index
        tire = Tire.create(
            f"tire_{index}",
            self.world,
            visual_mesh_path=str(tire_info.mesh_path),
            col_mesh_path=str(tire_info.col_mesh_path),
            rgba=[0.2, 0.2, 0.2, 1],
            mass=10.0,
        )
        
        tire.set_dynamics_info(
            lateralFriction=self.sim_config.get("lateralFriction", 1.0),
            rollingFriction=self.sim_config.get("rollingFriction", 0.01),
            spinningFriction=self.sim_config.get("spinningFriction", 0.001),
            frictionAnchor=True,
            contactProcessingThreshold=0.003,
        )
        self.tires.append(tire)
        return tire
    
    def remove_last_tire(self):
        last_tire_index = self.curr_tire_index - 1
        tire = self.tires[last_tire_index]
        self.world.remove_body(tire)
        self.tires.pop(last_tire_index)
    
    def remove_all_tires(self):
        for tire in self.tires:
            self.world.remove_body(tire)
        self.tires = []
    
    def place_tire(self, tire: eb.Body, pose: SE3):
        tire.set_pose(pose)
    
    def load_initial_tire_heuristic(self, num, x_noise=0.1):        
        trial = 0
        while self.num_placed_tire < num:
            idx = self.curr_tire_index
            tire_info = self.tire_seq_list[idx]
            initial_pose_actions = tire_pose_gen(
                tire_info.outer_radius,
                tire_info.width,
                num,
                overlap_const=np.random.uniform(0.7, 1.0),
            )
            if idx != 0:
                # Small randomization on x
                initial_pose_actions[idx][0] += np.random.uniform(
                    -x_noise, x_noise)
            
            trial += 1
            env_geoms = list(self.env_geoms.values())
            self.load_tire()
            self.tires[idx].set_tire_pose(
                initial_pose_actions[idx], degree=True)
            if self.tires[idx].is_in_collision(exclude=env_geoms):
                self.remove_last_tire()
                continue
            self.world.wait_to_stablize()
            

            # Check if the first tire is placed well
            if self.num_placed_tire == 1:
                # Get tire state 
                tire_pose = self.tires[idx].get_tire_pose()
                pitch = tire_pose[2] / np.pi * 180
                pitch = ((pitch + 90) % 180) - 90 # Convert into -90~90
                
                # Check initial tire fall
                if pitch >= -5 or pitch <= -80: 
                    self.remove_all_tires() # Remove tire if not placed well
            
            if trial >= 100:
                self.remove_all_tires()
                trial = 0

    def get_images(self)->tuple[np.ndarray, np.ndarray]:
        point_list = []
        rgb_list = []
        
        for cam_name, cam in self.cameras.items():
            cam_pose = self.camera_poses[cam_name]
            rgb, depth, _ = cam.render(cam_pose)

            points_cam = cam.intrinsic.depth_to_points(depth)
            points_world = cam_pose.apply(points_cam)

            point_list.append(points_world)
            rgb_list.append(rgb)
        
        points = np.concatenate(point_list, axis=0)
        y = points[:, 1]
        points_2d = points[y < 0.1][:, [0, 2]]

        occ = self.grid.get_occ_grid(points_2d)
        occ = occ[np.newaxis, ...]  # shape (1, H, W)
        rgb = rgb_list[0]
        return occ, rgb
    
    def set_tire_sequence(self, tire_seq_list: list[TireInfo]):
        self.tire_seq_list = tire_seq_list
    
    def is_misaligned_tires(self) -> bool:
        for tire in self.tires:
            pose = tire.get_pose()
            y_err = abs(pose.trans[1])
            
            if y_err > self.y_tol: #  or not in_bounds
                return True
        return False
    
    def calculate_safe_placement(self, occ, tire_info, x, theta, pixel_offset=0):
        if occ.ndim == 3:
            occ = occ[0]
        tire_grid, center = create_tire_grid(tire_info, theta, self.grid.pixel_size)
        expanded_occ = binary_dilation(occ, structure=tire_grid)
        x_index = self.grid.point_to_index([[x, 0]], is_int=True)[0][0]
        cols_with_occ = np.where(expanded_occ[:, x_index])[0]
        
        if len(cols_with_occ) == 0:
            y = 0.
        else:
            y_index = cols_with_occ.min() - 1 - pixel_offset
            points = np.array([[x_index, y_index]]).astype(int)
            y = self.grid.index_to_point(points)[0][1]
        return np.array([x, y, theta])
    
    def save_tire_state(self):
        self._tire_poses = {}
        for tire in self.tires:
            tire_pose = tire.get_pose()
            self._tire_poses[tire.name] = tire_pose
    
    def restore_tire_state(self):
        for tire in self.tires:
            tire.set_pose(self._tire_poses[tire.name])

    # def calculate_safe_placement(self, occ, tire_info, x, theta):
    #     tire_grid, _ = create_tire_grid(tire_info, theta)
    #     expanded_occ = binary_dilation(occ, structure=tire_grid)
    #     x_index = self.grid.point_to_index([[x, 0]], is_int=True)[0][0]
    #     cols_with_occ = np.where(expanded_occ[:, x_index])[0]
        
    #     if len(cols_with_occ) == 0:
    #         y = 0.
    #     else:
    #         y_index = cols_with_occ.min() - 1
    #         points = np.array([[x_index, y_index]]).astype(int)
    #         y = self.grid.index_to_point(points)[0][1]
    #     return np.array([x, y, theta])
    
    def calculate_all_safe_placements(self, occ, tire_info, thetas):
        placements = []
        for theta in thetas:
            tire_grid, center = create_tire_grid(tire_info, theta)
            expanded_occ = binary_dilation(occ, structure=tire_grid)
            y_indices = expanded_occ.argmax(axis=0) - 1
            x_indices = np.arange(expanded_occ.shape[1])
            x_offset = center[1]
            x_indices = x_indices[x_offset:-x_offset]
            y_indices = y_indices[x_offset:-x_offset]
            indices = np.stack([x_indices, y_indices], axis=1)
            xy = self.grid.index_to_point(indices)
            theta = np.ones(xy.shape[0]) * theta    
            placement = np.hstack([xy, theta[:,np.newaxis]])
            placements.append(placement)
        placements = np.vstack(placements)
        return placements



        

if __name__ == "__main__":
    from hydra import compose, initialize
    from pathlib import Path

    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="env_config")

    tire_info_paths = list(Path("./data/tires/").glob("*/tire_info.yaml"))
    tire_infos = [TireInfo.load(path) for path in tire_info_paths]
    tire_info_dict = {tire_info.name: tire_info for tire_info in tire_infos}
    target_tire_info = tire_info_dict["600"]
    
    env = TireWorld(**cfg)
    env.set_tire_sequence([target_tire_info]*16)
    env.reset()

    for i in range(12):
        occ, _ = env.get_images()
        x_rand = np.random.uniform(*env.bounds[0])
        theta_rand = np.random.uniform(-np.pi/2, np.pi/2)
        tire_pose = env.calculate_safe_placement(occ[0], target_tire_info, x_rand, theta_rand)
        tire = env.load_tire()
        tire.set_tire_pose(tire_pose)
        is_col = tire.is_in_collision(tol=0.01)
        
        if is_col:
            env.remove_last_tire()
        env.world.wait_to_stablize(tol=0.05)
        is_outside = env.is_misaligned_tires()
        if is_outside:
            env.remove_last_tire()

            

    print("done")
    