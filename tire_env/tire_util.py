import numpy as np
from scipy.optimize import fsolve
from pathlib import Path
from tire_env.dataclasses import TireInfo

def get_tire_info_dict(tire_dir:Path):
    tire_dir = Path(tire_dir)
    tire_info_paths = list(tire_dir.glob("*/tire_info.yaml"))
    tire_infos = [TireInfo.load(path) for path in tire_info_paths]
    tire_info_dict = {tire_info.name: tire_info for tire_info in tire_infos}
    return tire_info_dict

def tire_pose_gen(rad, wid, tire_num, overlap_const=0.5, case_width=1.121):
    """
    rad: tire radius
    wid: tire width
    tire_num: number of tires
    overlap_const: overlap constant
    case_width: case width

    return: tire pose list
    """
    tire_pose_list = np.zeros((tire_num, 3))

    def _get_theta_from_tire(diameter, width, overlap_const):
        def eq(theta, width, diameter):
            return (
                1.5 * diameter * np.cos(theta)
                + width * np.sin(theta)
                + width * 3 / np.sin(theta)
                - 2.146
                + width * np.sin(theta) * (1 - overlap_const)
            )

        (theta_sol,) = fsolve(eq, 1.5, args=(width, diameter))

        return theta_sol

    init_theta = _get_theta_from_tire(2 * rad, wid, overlap_const)
    case_width = case_width
    left_corner_x = -case_width
    right_corner_x = case_width

    # initial_tire_z = wid / 2 + 0.015
    # initial_tire_x = left_corner_x + rad + 0.015

    lv_idx = 0

    for lv_idx in range(int(tire_num / 4)):
        if lv_idx % 2 == 0:
            x_pose = (
                left_corner_x
                + rad * np.cos(init_theta)
                + wid / 2 * np.sin(init_theta)
                + 0.01
            )
            z_pose = (
                rad * np.sin(init_theta)
                + wid / 2 * np.cos(init_theta)
                + 0.005
                + 0.295 * lv_idx
            )  ## 0.3, the z distance need to be formulated

            init_theta_degree_reg = 16

            x_dist = 1.1 * wid / np.sin(init_theta)
            init_theta_degree = init_theta * 180 / np.pi

            if lv_idx == 0:
                tire_pose_list[lv_idx * 4] = np.array(
                    [x_pose + 0.03, z_pose + 0.015, 180 - init_theta_degree]
                )
                tire_pose_list[lv_idx * 4 + 1] = np.array(
                    [x_pose + x_dist, z_pose, 180 - init_theta_degree]
                )
                tire_pose_list[lv_idx * 4 + 2] = np.array(
                    [x_pose + x_dist * 2, z_pose, 180 - init_theta_degree]
                )
                tire_pose_list[lv_idx * 4 + 3] = np.array(
                    [x_pose + x_dist * 3, z_pose, 180 - init_theta_degree]
                )
            else:
                tire_pose_list[lv_idx * 4] = np.array(
                    [
                        x_pose - 0.04,
                        z_pose - 0.015,
                        180 - init_theta_degree - init_theta_degree_reg,
                    ]
                )
                tire_pose_list[lv_idx * 4 + 1] = np.array(
                    [x_pose + x_dist, z_pose - 0.01, 180 - init_theta_degree]
                )
                tire_pose_list[lv_idx * 4 + 2] = np.array(
                    [x_pose + x_dist * 2, z_pose, 180 - init_theta_degree]
                )
                tire_pose_list[lv_idx * 4 + 3] = np.array(
                    [x_pose + x_dist * 3, z_pose, 180 - init_theta_degree]
                )
        else:
            init_theta_degree_reg = 15

            x_pose = (
                right_corner_x
                - (rad * np.cos(init_theta) + wid / 2 * np.cos(init_theta))
                - 0.01
            )
            z_pose = (
                rad * np.sin(init_theta)
                + wid / 2 * np.cos(init_theta)
                + 0.005
                + 0.295 * lv_idx
            )  ## 0.3, the z distance need to be formulated

            x_dist = wid / np.sin(init_theta)
            init_theta_degree = init_theta * 180 / np.pi

            tire_pose_list[lv_idx * 4] = np.array(
                [x_pose, z_pose, init_theta_degree + init_theta_degree_reg]
            )
            tire_pose_list[lv_idx * 4 + 1] = np.array(
                [x_pose - x_dist, z_pose, init_theta_degree]
            )
            tire_pose_list[lv_idx * 4 + 2] = np.array(
                [x_pose - x_dist * 2, z_pose, init_theta_degree]
            )
            tire_pose_list[lv_idx * 4 + 3] = np.array(
                [x_pose - x_dist * 3, z_pose, init_theta_degree]
            )

    # Place extra tires
    if int(tire_num % 4) > 0 and lv_idx > 0:
        lv_idx = lv_idx + 1

    if lv_idx % 2 == 0:
        x_pose = (
            left_corner_x
            + rad * np.cos(init_theta)
            + wid / 2 * np.sin(init_theta)
            + 0.01
        )
        z_pose = (
            rad * np.sin(init_theta)
            + wid / 2 * np.cos(init_theta)
            + 0.01
            + 0.3 * lv_idx
        )  ## 0.3, the z distance need to be formulated

        init_theta_degree_reg = 16

        x_dist = 1.1 * wid / np.sin(init_theta)
        init_theta_degree = init_theta * 180 / np.pi

        for i in range(tire_num % 4):
            if i == 0:
                if lv_idx == 0:
                    ## Initial Tire pose
                    tire_pose_list[lv_idx * 4] = np.array(
                        [x_pose + 0.0, z_pose + 0.002, 180 - init_theta_degree - 5]
                    )
                else:
                    tire_pose_list[lv_idx * 4] = np.array(
                        [
                            x_pose - 0.04,
                            z_pose - 0.015,
                            180 - init_theta_degree - init_theta_degree_reg,
                        ]
                    )
            else:
                tire_pose_list[lv_idx * 4 + i] = np.array(
                    [x_pose + x_dist * i, z_pose - 0.01, 180 - init_theta_degree]
                )
    else:
        init_theta_degree_reg = 15

        x_pose = (
            right_corner_x
            - (rad * np.cos(init_theta) + wid / 2 * np.cos(init_theta))
            - 0.01
        )
        z_pose = (
            rad * np.sin(init_theta)
            + wid / 2 * np.cos(init_theta)
            + 0.01
            + 0.3 * lv_idx
        )  ## 0.3, the z distance need to be formulated

        x_dist = wid / np.sin(init_theta)
        init_theta_degree = init_theta * 180 / np.pi

        for i in range(tire_num % 4):
            if i == 0:
                tire_pose_list[lv_idx * 4] = np.array(
                    [x_pose + 0.01, z_pose + 0.01, init_theta_degree + init_theta_degree_reg]
                )
            else:
                tire_pose_list[lv_idx * 4 + i] = np.array(
                    [x_pose - x_dist * i, z_pose, init_theta_degree]
                )

    ## Convert theta range from [0, 180] to [-90, 90]
    theta = tire_pose_list[:, 2]

    # 변환
    mask = theta > 90
    theta[mask] = -(180 - theta[mask])

    # 갱신
    tire_pose_list[:, 2] = theta

    return tire_pose_list