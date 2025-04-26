import os
import numpy as np
from scipy.spatial import ConvexHull
import subprocess


from pathlib import Path
from tire_env.dataclasses import TireInfo

ROOT = Path(__file__).parent.parent

##################################################
################### PARAMETERS ###################
##################################################
_TIRE_INNER_RADIUS = 0.40  # Average inner radius
_TIRE_OUTER_RADIUS = 0.60  # Average outer radius
_TIRE_WIDTH = 0.22  # Average tire width
_INNER_RADIUS_VARIATION_STD = 0.008  # Variation range for inner and outer radii
_OUTER_RADIUS_VARIATION_STD = 0.01  # Variation range for inner and outer radii
_WIDTH_VARIATION_STD = 0.006  # Variation range for width
_OUTER_CONER_RADIUS = 0.03
_INNER_CORNER_RADIUS = 0.02
_OUTER_CORNER_RADIUS_VARIATION_STD = 0.003
_INNER_CONER_RADIUS_VARIATION_STD = 0.002
_MIN_GAP = 0.08  # Minimum gap between inner and outer radii
_NUM_ANGLES = 50
_NUM_TIRE_TYPES = 9
_NUM_RANDOMIZED_TIRE_PER_TYPE = 5

outer_diameters = np.array([0.5612, 0.6006, 0.6014, 0.6898, 0.6793, 0.7376, 0.7026, 0.7992, 0.7063])
outer_radii = outer_diameters / 2
widths = np.array([0.1647, 0.1696, 0.2102, 0.2247, 0.2328, 0.2690, 0.2804, 0.2902, 0.3135])
inner_radii = [0.1651, 0.1778, 0.2032, 0.2159, 0.2159, 0.2668, 0.1863, 0.2286, 0.2794]
corner_rad = np.array([0.0384, 0.0429, 0.0329, 0.0412, 0.0430, 0.0499, 0.0309, 0.0529, 0.0217]) * 1.5
TIRE_CONFIG_DICT = dict(
    outer_diameters=outer_diameters,
    outer_radii=outer_radii,
    widths=widths,
    inner_radii=inner_radii,
    corner_rad=corner_rad
)


##################################################

def return_tire_convex_hull(
    inner_radius=_TIRE_INNER_RADIUS,
    outer_radius=_TIRE_OUTER_RADIUS,
    tire_width=_TIRE_WIDTH,
    inner_corner_radius=_INNER_CORNER_RADIUS,
    outer_corner_radius=_OUTER_CONER_RADIUS,
    inner_arc_resolution=1,
    outer_arc_resolution=3,
):
    """
    Generates and returns the points of a 2D convex hull for a tire cross-section.

    :param inner_radius: Inner radius of the tire.
    :param outer_radius: Outer radius of the tire.
    :param tire_width: Width of the tire.
    :return: Numpy array of hull points.
    """
    half_y = tire_width / 2.0

    # Inner 쪽 좌표
    r0 = inner_radius
    r1 = inner_radius + inner_corner_radius

    # Outer 쪽 좌표
    r2 = outer_radius - outer_corner_radius
    r3 = outer_radius

    y0 = half_y
    y1_inner = half_y - inner_corner_radius
    y1_outer = half_y - outer_corner_radius

    y2_outer = -half_y + outer_corner_radius
    y2_inner = -half_y + inner_corner_radius
    y3 = -half_y

    points = []

    # 좌상단 inner fillet (반시계)
    center1 = np.array([r1, y1_inner])
    theta = np.linspace(np.pi, np.pi / 2, inner_arc_resolution)
    for t in theta:
        x = center1[0] + inner_corner_radius * np.cos(t)
        y = center1[1] + inner_corner_radius * np.sin(t)
        points.append([x, y])

    # 위쪽 직선
    points.append([r2, y0])

    # 우상단 outer fillet (반시계)
    center2 = np.array([r2, y1_outer])
    theta = np.linspace(np.pi / 2, 0, outer_arc_resolution)
    for t in theta:
        x = center2[0] + outer_corner_radius * np.cos(t)
        y = center2[1] + outer_corner_radius * np.sin(t)
        points.append([x, y])

    # 우측 수직
    points.append([r3, y1_outer])
    points.append([r3, y2_outer])

    # 우하단 outer fillet (반시계)
    center3 = np.array([r2, y2_outer])
    theta = np.linspace(0, -np.pi / 2, outer_arc_resolution)
    for t in theta:
        x = center3[0] + outer_corner_radius * np.cos(t)
        y = center3[1] + outer_corner_radius * np.sin(t)
        points.append([x, y])

    # 하단 직선
    points.append([r1, y3])

    # 좌하단 inner fillet (반시계)
    center4 = np.array([r1, y2_inner])
    theta = np.linspace(-np.pi / 2, -np.pi, inner_arc_resolution)
    for t in theta:
        x = center4[0] + inner_corner_radius * np.cos(t)
        y = center4[1] + inner_corner_radius * np.sin(t)
        points.append([x, y])

    # 좌측 수직
    points.append([r0, y2_inner])
    points.append([r0, y1_inner])
    points.append([r1, y0])
    points = np.array(points)

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    if not hull_points.shape == points.shape:
        print("WARNING: The generated points do not form a convex hull.")

    return hull_points

def generate_convex_decomposition_files_coacd(tire_dir:Path):
    """
    Performs convex decomposition on all OBJ files in the given directory
    by calling an external script (obj2mjcf).

    :param file_dir: Directory containing the OBJ files for decomposition.
    """
    coacd_path = str(ROOT / "scripts/run_coacd.py")
    env = os.environ.copy()
    
    vis_mesh_path = tire_dir / "visual.obj"
    col_mesh_path = tire_dir / "col.obj"

    command = [
        "uv",
        "run",
        coacd_path,
        "--t",
        "0.08",
        "-i",
        str(vis_mesh_path),
        "-o",
        str(col_mesh_path),
    ]

    print("Convex Decomposition ...")
    retcode = subprocess.call(command, env=env)
    assert retcode == 0


def get_tire_info(tire_type:int, randomize_idx:int=0) -> TireInfo:
    tire_configs = TIRE_CONFIG_DICT
    inner_radius = tire_configs['inner_radii'][tire_type] * 1.1   ## Considering deformation of tire
    outer_radius = tire_configs['outer_radii'][tire_type]
    corner_radius = tire_configs['corner_rad'][tire_type]
    width = tire_configs['widths'][tire_type]
    if randomize_idx > 0:
        inner_radius += np.random.normal(0, _INNER_RADIUS_VARIATION_STD)
        outer_radius += np.random.normal(0, _OUTER_RADIUS_VARIATION_STD)
        corner_radius += np.random.normal(0, _OUTER_CORNER_RADIUS_VARIATION_STD)
        width += np.random.normal(0, _WIDTH_VARIATION_STD)
    name = f"{tire_type+1}0{randomize_idx}"
    
    return TireInfo(
        name=name,
        tire_type=tire_type,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        width=width,
        corner_radius=corner_radius,
    )

def generate_obj_files(
    obj_path:Path,
    tire_info:TireInfo,
    inner_arc_resolution=1,
    outer_arc_resolution=5,
    num_angles=_NUM_ANGLES,
):
    """
    Generates an OBJ file for a tire model by sweeping the 2D cross-section
    around the x-axis.

    :param hull_points: 2D points (y, z) forming the convex hull cross-section.
    :param num_angles: Number of rotational subdivisions around the tire axis.
    :param tire_idx: Identifier for the tire, used in naming the OBJ file.
    """

    hull_points = return_tire_convex_hull(
        inner_radius=tire_info.inner_radius,
        outer_radius=tire_info.outer_radius,
        tire_width=tire_info.width,
        inner_corner_radius=_INNER_CORNER_RADIUS,
        outer_corner_radius=tire_info.corner_radius,
        inner_arc_resolution=inner_arc_resolution,
        outer_arc_resolution=outer_arc_resolution,
    )

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    H = len(hull_points)
    P = num_angles

    # Create vertices by rotating the hull points
    vertices = []
    for theta in angles:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for y_val, z_val in hull_points:
            x = y_val * cos_t
            y_new = y_val * sin_t
            vertices.append([x, y_new, z_val])
    vertices = np.array(vertices)

    # Create faces for the tire mesh
    faces = []
    for i in range(P):
        next_i = (i + 1) % P
        for j in range(H):
            next_j = (j + 1) % H

            v0 = i * H + j
            v1 = next_i * H + j
            v2 = next_i * H + next_j
            v3 = i * H + next_j

            # Append face indices (1-based for OBJ)
            faces.append([v0 + 1, v1 + 1, v2 + 1])
            faces.append([v0 + 1, v2 + 1, v3 + 1])

    # Write vertices and faces to the OBJ file
    with open(obj_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


if __name__ == "__main__":
    tire_save_root = ROOT / "data/tires"
    tire_save_root.mkdir(parents=True, exist_ok=True)

    tire_type = 0
    random_idx = 0
    for tire_type in range(9):
        for random_idx in range(5): 
            # random_idx 0 -> default with no randomization
            tire_info = get_tire_info(tire_type, randomize_idx=random_idx)
            tire_dir = tire_save_root / tire_info.name
            tire_dir.mkdir(parents=True, exist_ok=True)
            tire_obj_visual_path = tire_dir / "visual.obj"
            tire_obj_col_path = tire_dir / "col.obj"
            generate_obj_files(tire_obj_visual_path, tire_info)
            generate_convex_decomposition_files_coacd(tire_dir)
            tire_info.save(tire_dir / "tire_info.yaml")