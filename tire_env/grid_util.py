from __future__ import annotations
import numpy as np
from .dataclasses import TireInfo
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

def create_tire_grid(tire_info: TireInfo, theta:float, pixel_size):
    w = tire_info.outer_radius*2
    h = tire_info.width
    R = np.array([[np.cos(theta), np.sin(theta)],
                [-np.sin(theta),  np.cos(theta)]])
    corners = np.array([[ w/2,  h/2],
                        [-w/2,  h/2],
                        [-w/2, -h/2],
                        [ w/2, -h/2]])
    rotated = corners @ R.T
    min_xy, max_xy = rotated.min(0), rotated.max(0)
    
    x = np.arange(min_xy[0], max_xy[0], pixel_size[0])
    y = np.arange(min_xy[1], max_xy[1], pixel_size[1])
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.ravel(), Y.ravel()], -1)
    
    local = pts @ R
    mask = (np.abs(local[:,0]) <= w/2) & (np.abs(local[:,1]) <= h/2)
    tire_grid = mask.reshape(X.shape).astype(np.uint8)
    indices = np.argwhere(tire_grid)
    center = indices.mean(axis=0).astype(int)
    # tire_grid[tuple(center)] = 0 # just to visualize center
    return tire_grid, center

def calculate_all_safe_placements(
    grid: Grid2D, occ, tire_info, theta_grid):
    if occ.ndim == 3:
        occ = occ.squeeze(0)
        
    placements = []
    for theta in theta_grid:
        tire_grid, center = create_tire_grid(
            tire_info, theta, grid.pixel_size)
        occ[-1] = True
        expanded_occ = binary_dilation(occ, structure=tire_grid)

        y_indices = expanded_occ.argmax(axis=0) - 1
        y_indices
        x_indices = np.arange(expanded_occ.shape[1])
        x_offset = center[1]
        x_indices = x_indices[x_offset:-x_offset]
        y_indices = y_indices[x_offset:-x_offset]
        indices = np.stack([x_indices, y_indices], axis=1)
        xy = grid.index_to_point(indices)
        theta = np.ones(xy.shape[0]) * theta    
        placement = np.hstack([xy, theta[:,np.newaxis]])
        placements.append(placement)
    placements = np.vstack(placements)
    return placements

class Grid2D:
    def __init__(self, size: np.ndarray, res: tuple, center: np.ndarray = None):
        assert len(size) == 2
        assert len(res) == 2
        self.size = np.array(size)
        self.res = np.array(res).astype(int)

        self.offset = np.ones(2) * self.pixel_size / 2
        self.min_point = -np.ones(2) * self.size / 2
        self.origin = self.min_point + self.offset
        if center is None:
            center = np.zeros(2)
        self.center = center

    @property
    def pixel_size(self):
        return self.size / self.res

    def is_outside_points(self, points, margins=(0.0, 0.0)):
        margins = np.array(margins)
        points_grid = points - self.center
        boundary_point = self.size / 2 - self.pixel_size / 2
        out_cond1 = (points_grid < -boundary_point + margins).any(axis=1)
        out_cond2 = (boundary_point - margins < points_grid).any(axis=1)
        return out_cond1 | out_cond2

    def point_to_index(self, points, drop=False, is_int=False):
        if drop:
            points = points[~self.is_outside_points(points)]

        points_grid = points - self.center
        indices = (points_grid - self.origin) / self.pixel_size
        if is_int:
            indices = np.round(indices).astype(int)

        idx_min, idx_max = 0, self.res - 1
        indices = np.clip(indices, idx_min, idx_max)
        indices[:, 1] = idx_max[1] - indices[:, 1]  # upside down
        return indices  # xy

    def index_to_point(self, indices):
        # indices_xy = np.vstack([indices[:,1], indices[:,0]]).T
        indices[:, 1] = (self.res[1] - 1) - indices[:, 1]  # upside down
        points = indices * self.pixel_size + self.origin + self.center
        return points

    def get_occ_grid(self, points):
        occ_grid = np.zeros(self.res, dtype=bool)
        if len(points) == 0:
            return occ_grid
        indices = self.point_to_index(points, drop=True, is_int=True)
        upper_lim_viol = np.any(indices >= self.res, axis=1)
        lower_lim_viol = np.any(indices < 0, axis=1)
        indices = indices[~(lower_lim_viol | upper_lim_viol)]
        occ_grid[indices[:, 0], indices[:, 1]] = True
        return occ_grid.T  # h, w

    def pointify(self, occ_grid):
        indices_yx = np.array(np.nonzero(occ_grid.T)).T
        indices = np.vstack([indices_yx[:,0], indices_yx[:,1]]).T
        return self.index_to_point(indices)

    def get_tire_patch(self, tire_info: TireInfo, size : float = 0.02, rounding: bool = False) -> np.ndarray:
        if not rounding:
            tire_rad = tire_info.outer_radius
            tire_wid = tire_info.width

            # Create a rectangular patch for the tire
            rect = np.array([
                [-tire_rad, -tire_wid / 2],
                [ tire_rad, -tire_wid / 2],
                [ tire_rad,  tire_wid / 2],
                [-tire_rad,  tire_wid / 2]
            ])
            
            # Interpolate between points 
            interpolated_points = []
            
            for i in range(len(rect)):
                start_point = rect[i]
                end_point = rect[(i + 1) % len(rect)]
                
                # Compute number of points to interpolate
                distance = np.linalg.norm(end_point - start_point)
                num_points = int(distance / size)
                
                interp_points = np.linspace(start_point, end_point, num_points)
                interpolated_points.append(interp_points)
            
            # Concatenate all interpolated points
            rect = np.concatenate(interpolated_points, axis=0)
            
            # Convert to float32
            rect = rect.astype(np.float32)
            
            return rect
        else:
            L = tire_info.outer_radius
            W = tire_info.width / 2.0
            r = tire_info.get_tire_corner_radius()
            r = min(r, L, W)

            # Corner centers: TL, BL, BR, TR for CW sampling
            centers = [
                (-L + r,  W - r),
                (-L + r, -W + r),
                ( L - r, -W + r),
                ( L - r,  W - r),
            ]

            points = []

            # number of samples
            n_edge_h = max(int((2*L - 2*r) / size), 1)
            n_edge_v = max(int((2*W - 2*r) / size), 1)
            n_arc    = max(int((np.pi/2 * r) / size), 2)

            # Top edge (left to right)
            xs = np.linspace(-L + r, L - r, n_edge_h, endpoint=False)
            ys = np.full_like(xs,  W)
            points.append(np.stack([xs, ys], axis=1))

            # Corners with clockwise quarter-circles
            angle_starts = [0.5*np.pi, np.pi, 1.5*np.pi, 0.0]  # TR, TL, BL, BR starts for CW
            for (cx, cy), theta_start in zip(centers, angle_starts):
                theta = np.linspace(theta_start,
                                    theta_start + 0.5*np.pi,
                                    n_arc, endpoint=False)
                xarc = cx + r * np.cos(theta)
                yarc = cy + r * np.sin(theta)
                points.append(np.stack([xarc, yarc], axis=1))

                # Straight edge after each arc
                if theta_start == 0.5*np.pi:    # top-right to bottom-right (right edge down)
                    ys = np.linspace(W - r, -W + r, n_edge_v, endpoint=False)
                    xs = np.full_like(ys,  L)
                elif theta_start == np.pi:      # top-left to top-right (top edge right)
                    xs = np.linspace(-L + r, L - r, n_edge_h, endpoint=False)
                    ys = np.full_like(xs,  W)
                elif theta_start == 1.5*np.pi:  # bottom-left to top-left (left edge up)
                    ys = np.linspace(-W + r, W - r, n_edge_v, endpoint=False)
                    xs = np.full_like(ys, -L)
                elif theta_start == 0.0:          # bottom-right to bottom-left (bottom edge left)
                    xs = np.linspace(L - r, -L + r, n_edge_h, endpoint=False)
                    ys = np.full_like(xs, -W)
                points.append(np.stack([xs, ys], axis=1))

            patch = np.concatenate(points, axis=0).astype(np.float32)
            
            return patch
    
    # def draw_tire_on_grid(self, grid: np.ndarray, pose: np.ndarray, tire_info: TireInfo, val: int = 1, shape_fill : bool = False):
    #     """
    #     This function takes a grid, tire pose, and tire information, and marks the 
    #     grid cells corresponding to the tire's position and orientation. It also 
    #     calculates the number of overlapping cells where the tire overlaps with 
    #     already marked cells.
    #     Args:
    #         grid (np.ndarray): The 2D grid representing the environment.
    #         pose (np.ndarray): A 3-element array representing the tire's pose 
    #             [x, y, angle], where x and y are the tire's position, and angle 
    #             (in degrees) is its orientation.
    #         tire_info (TireInfo): An object containing information about the tire, 
    #             such as its dimensions and shape.
    #         val (int, optional): The value to assign to the grid cells occupied by 
    #             the tire. Defaults to 1.
    #         shape_fill (bool, optional): A flag indicating whether to fill the 
    #             shape of the tire on the grid. Defaults to False.
    #     Returns:
    #         Tuple[np.ndarray, int]: A tuple containing:
    #             - The updated grid with the tire drawn on it.
    #             - The count of overlapping cells where the tire overlaps with 
    #               already marked cells.
    #     Notes:
    #         - The function assumes that the tire's patch points are provided as 
    #           an N x 2 array, where each row is a point (x, y).
    #         - The `point_to_index` method is used to convert the tire's translated 
    #           patch points into grid indices.
    #         - The grid is indexed as `grid_[row, column]` or `grid_[y, x]`.
    #         - The function ensures that the tire is drawn only within the bounds 
    #           of the grid.
    #     """
    
    #     # Draw tire on the grid by working on a copy
    #     grid_ = grid.copy()
        
    #     # Get the tire patch points (assumed to be a N x 2 array: each row is (x, y))
    #     tire_patch = self.get_tire_patch(tire_info, rounding=True)
        
    #     # Rotate the tire patch based on the pose (angle in pose[2])
    #     angle = pose[2] * np.pi / 180.0  # Convert to radians
    #     cos_a, sin_a = np.cos(angle), np.sin(angle)
    #     rotation_matrix = np.array([
    #         [cos_a, -sin_a],
    #         [sin_a,  cos_a]
    #     ])
    #     rotated_patch = (rotation_matrix @ tire_patch.T).T
        
    #     # Translate the patch to the tire's position
    #     translated_patch = rotated_patch + pose[:2]
        
    #     # Convert to grid indices.
    #     # IMPORTANT: Ensure that point_to_index returns indices in (x, y) order.
    #     indices = self.point_to_index(translated_patch, drop=True, is_int=True)
    #     indices = np.array(indices)  # ensure indices is a NumPy array
        
    #     # Fill the shape of the tire on the grid if shape_fill is True
    #     if shape_fill:
    #         # Create a mask for the tire shape
    #         tire_shape_mask = np.zeros(grid_.shape, dtype=np.uint8)
    #         cv2.fillConvexPoly(tire_shape_mask, indices, 1)
            
    #         indices = np.argwhere(tire_shape_mask)
            
    #         # Convert to (x, y) format
    #         indices = np.array([idx[::-1] for idx in indices])
        
        
    #     # Draw the tire on the grid.
    #     # If indices is in (x, y) format, then for NumPy we need to use grid_[y, x]
    #     overlap_count = 0
    #     for idx in indices:
    #         x, y = idx  # assuming idx = [x, y]
    #         # Check bounds where grid_ is indexed as grid_[row, column] = grid_[y, x]
    #         if 0 <= y < grid_.shape[0] and 0 <= x < grid_.shape[1]:
    #             if grid_[y, x] == 1:
    #                 overlap_count += 1
    #             grid_[y, x] = val

    #     return grid_, overlap_count

    def check_collision_along_z(self, grid, pose, tire_info: TireInfo, state_bound: dict = None):
        # 1. Get tire patch and apply rotation
        tire_patch = self.get_tire_patch(tire_info, rounding=True)  # shape: (N, 2)
        angle = pose[2] * np.pi / 180.0  # Convert to radians
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        rotated_patch = (rotation_matrix @ tire_patch.T).T  # shape: (N, 2)
        
        # 2. Translate in x-direction using pose[0]
        rotated_patch[:, 0] += pose[0]
        base_patch = rotated_patch  # base_patch: (N, 2)

        # 3. Compute tire patch for all height (h) levels by adding y offsets.
        n_heights = self.res[1]
        heights = np.arange(n_heights) * self.pixel_size[1]  # (n_heights,)
        # Create an offsets array: each row is (0, height) → shape: (n_heights, 2)
        offsets = np.column_stack((np.zeros(n_heights), heights))
        # Apply the offsets via broadcasting: result has shape (n_heights, N, 2)
        patches = base_patch[None, :, :] + offsets[:, None, :]

        # 4. Collision Check via grid lookup:
        # Flatten patches to shape (n_heights * N, 2) for conversion.
        patches_flat = patches.reshape(-1, 2)
        # IMPORTANT: Change drop parameter to False so that no points are dropped.
        indices_flat = self.point_to_index(patches_flat, drop=False, is_int=True)
        # Now, reshape to (n_heights, N, 2) since no points have been dropped.
        indices = indices_flat.reshape(n_heights, -1, 2)
        
        # Assume point_to_index returns (x, y) ordering. For grid indexing we use grid[y, x].
        grid_vals = grid[indices[:, :, 1], indices[:, :, 0]]  # shape: (n_heights, N)
        # Collision flag per height level: True if any value equals 1.
        collision = (grid_vals == 1).any(axis=1)
        
        # 6. Determine infeasibility: if out_of-bound OR collision occurs.
        if state_bound is not None:
            # 5 (optional). Check if each height level's tire patch is within state boundaries.
            patch_min = patches.min(axis=1)  # shape: (n_heights, 2)
            patch_max = patches.max(axis=1)  # shape: (n_heights, 2)
            # state_bound is a dict: {"low": [low_x, low_y], "high": [high_x, high_y]}
            out_of_bound = (patch_min[:, 0] < state_bound["low"][0]) | (patch_max[:, 0] > state_bound["high"][0]) | \
                        (patch_min[:, 1] < state_bound["low"][1]) | (patch_max[:, 1] > state_bound["high"][1])

        else:
            # Check only z bound
            patch_min = patches.min(axis=1)  # shape: (n_heights, 2)
            patch_max = patches.max(axis=1)  # shape: (n_heights, 2)
            # state_bound is a dict: {"low": [low_x, low_y], "high": [high_x, high_y]}
            out_of_bound = (patch_min[:, 1] < 0) | (patch_max[:, 1] > self.size[1])
            
        infeasible = out_of_bound | collision

        # 7. Find the lowest feasible height (where infeasible is False).
        feasible_indices = np.where(~infeasible)[0]
        if feasible_indices.size == 0:
            feasible_height = 0
            # print("No feasible height found!")
        else:
            feasible_height_idx = feasible_indices.min()
            feasible_height = feasible_height_idx * self.pixel_size[1] + self.pixel_size[1] / 2
            # print("Feasible height:", feasible_height)
        
        return feasible_height
