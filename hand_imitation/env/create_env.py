import os
import numpy as np
from hand_imitation.env import task_setting
from hand_imitation.env.rl_env.relocate_env import AllegroRelocateRLEnv
from hand_imitation.env.sim_env.constructor import add_default_scene_light
import pdb


def create_env(name, task_kwargs=None, use_gui=False, is_eval=False, is_vision=False, is_demo_rollout=False, is_real_robot=False, pc_noise=False, point_cs="world", norm_traj=False):
    cur_dir = os.path.dirname(__file__)
    if norm_traj:
        motion_path = os.path.join(cur_dir, '../../', 'norm_trajectories', f'{name}.npz')
    else:
        motion_path = os.path.join(cur_dir, '../../', 'trajectories', f'{name}.npz')

    try:
        motion_file = np.load(motion_path)
        motion_file = {k:v for k, v in motion_file.items()}
        if name == "pour":
            motion_file['task_name'] = "pour"
        elif name == "place":
            motion_file['task_name'] = "place"
        else:
            motion_file['task_name'] = "relocate"
    except:
        motion_file = name

    env_params = dict(motion_file=motion_file, use_gui=use_gui, task_kwargs=task_kwargs, is_eval=is_eval, is_vision=is_vision, is_demo_rollout=is_demo_rollout, is_real_robot=is_real_robot, pc_noise=pc_noise, point_cs=point_cs, norm_traj=norm_traj)

    if is_eval or is_vision or is_demo_rollout:
        env_params["no_rgb"] = False
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    env = AllegroRelocateRLEnv(**env_params)

    if is_eval:
        if is_vision:
            if is_real_robot:
                env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
                add_default_scene_light(env.scene, env.renderer)
                env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_pc_seg"])
            else:
                env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
                add_default_scene_light(env.scene, env.renderer)
                env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance"])
        else:
            env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
            add_default_scene_light(env.scene, env.renderer)

    # flush cache
    env.action_space
    env.observation_space

    return env
