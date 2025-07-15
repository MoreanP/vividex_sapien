from abc import abstractmethod
from typing import Dict, Optional, Callable, List, Union, Tuple

import gym
import numpy as np
import sapien.core as sapien
import transforms3d

from hand_imitation.env.sim_env.base import BaseSimulationEnv
from hand_imitation.env.sim_env.constructor import add_default_scene_light
from hand_imitation.utils.kinematics_helper import PartialKinematicModel
from hand_imitation.utils.common_robot_utils import load_robot, generate_ur5_robot_hand_info, ArmRobotInfo
from hand_imitation.env.rl_env.pc_processing import process_pc
from hand_imitation.utils.random_utils import np_random
import pdb

VISUAL_OBS_RETURN_TORCH = False
MAX_DEPTH_RANGE = 2.5
gl2sapien = sapien.Pose(q=np.array([0.5, 0.5, -0.5, -0.5]))


def recover_action(action, limit):
    action = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]
    return action


class BaseRLEnv(BaseSimulationEnv, gym.Env):
    def __init__(self, use_gui=False, frame_skip=5, renderer: str = "sapien", **renderer_kwargs):
        # Do not write any meaningful in this __init__ function other than type definition,
        # Since multiple parents are presented for the child RLEnv class
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        # Visual staff for offscreen rendering
        self.camera_infos: Dict[str, Dict] = {}
        self.camera_pose_noise: Dict[
            str, Tuple[Optional[float], sapien.Pose]] = {}  # tuple for noise level and original camera pose
        self.imagination_infos: Dict[str, float] = {}
        self.imagination_data: Dict[str, Dict[str, Tuple[sapien.ActorBase, np.ndarray, int]]] = {}
        self.need_flush_when_change_instance = False
        self.imaginations: Dict[str, np.ndarray] = {}
        self.eval_cam_names = renderer_kwargs['eval_cam_names'] if renderer_kwargs.__contains__("eval_cam_names") else None
        self.use_history_obs = renderer_kwargs['use_history_obs'] if renderer_kwargs.__contains__('use_history_obs') else False
        self.last_obs = None

        # RL related attributes
        self.is_robot_free: Optional[bool] = None
        self.arm_dof: Optional[int] = None
        self.rl_step: Optional[Callable] = None
        self.get_observation: Optional[Callable] = None
        self.robot_collision_links: Optional[List[sapien.Actor]] = None
        self.robot_info: Optional[Union[ArmRobotInfo]] = None
        self.velocity_limit: Optional[np.ndarray] = None
        self.kinematic_model: Optional[PartialKinematicModel] = None

        # Robot cache
        self.control_time_step = None
        self.ee_link_name = None
        self.ee_link: Optional[sapien.Actor] = None
        self.cartesian_error = None

        self.pregrasp_success = False
        self.pregrasp_steps = 10
        self.imitate_steps = 60

        self.hand_jpos_err = 0.3
        self.hand_mjpos_err = 0.0
        self.obj_com_err = 0.0

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def get_observation(self):
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, action):
        pass

    def get_info(self):
        info = dict()
        info['pregrasp_success'] = self.pregrasp_success
        info['pregrasp_steps'] = self.pregrasp_steps
        info['imitate_steps'] = self.imitate_steps
        info['hand_jpos_err'] = self.hand_jpos_err
        info['hand_mjpos_err'] = self.hand_mjpos_err
        info['obj_com_err'] = self.obj_com_err
        info['obj_lift'] = self.object_lift
        info['stage'] = self._stage
        info['control_error'] = self.cartesian_error
        if self.is_vision:
            info['obj_tgt_dist'] = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object_pos)
        else:
            info['obj_tgt_dist'] = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object.get_pose().p)

        return info

    def update_cached_state(self):
        return

    @abstractmethod
    def is_done(self):
        pass

    @property
    @abstractmethod
    def obs_dim(self):
        return 0

    @property
    def action_dim(self):
        return self.robot.dof

    @property
    @abstractmethod
    def horizon(self):
        return 0

    def setup(self, robot_name):
        self.robot_name = robot_name
        self.robot = load_robot(self.scene, robot_name, disable_self_collision=False)
        self.robot.set_pose(sapien.Pose(np.array([0, 0, -5])))

        info = generate_ur5_robot_hand_info()[robot_name]
        self.arm_dof = info.arm_dof
        hand_dof = info.hand_dof
        velocity_limit = np.array([1] * 3 + [1] * 3 + [np.pi] * hand_dof)
        self.velocity_limit = np.stack([-velocity_limit, velocity_limit], axis=1)
        start_joint_name = self.robot.get_joints()[1].get_name()
        end_joint_name = self.robot.get_active_joints()[self.arm_dof - 1].get_name()
        self.kinematic_model = PartialKinematicModel(self.robot, start_joint_name, end_joint_name)
        self.ee_link_name = self.kinematic_model.end_link_name
        self.ee_link = [link for link in self.robot.get_links() if link.get_name() == self.ee_link_name][0]

        self.robot_info = info
        self.robot_collision_links = [link for link in self.robot.get_links() if len(link.get_collision_shapes()) > 0]
        self.control_time_step = self.scene.get_timestep() * self.frame_skip

        self.rl_step = self.arm_sim_step

        if self.is_vision and not self.is_demo_rollout:
            self.get_observation = self.get_test_state 
        else:
            self.get_observation = self.get_oracle_state 

    def arm_sim_step(self, action: np.ndarray, check=False, denormalized=False):
        if denormalized:
            target_qpos = action
            target_qvel = np.zeros_like(target_qpos)
            self.robot.set_drive_target(target_qpos)
            self.robot.set_drive_velocity_target(target_qvel)
            if check:
                self.robot_sim_obs = [self.robot.get_qpos().tolist()]

            for i in range(self.frame_skip):
                self.robot.set_qf(self.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
                self.scene.step()
                if check and i < self.frame_skip - 1:
                    self.robot_sim_obs.append(self.robot.get_qpos().tolist())
            self.current_step += 1
            self.cartesian_error = 0.0
        else:
            current_qpos = self.robot.get_qpos()
            ee_link_last_pose = self.ee_link.get_pose()
            action = np.clip(action, -1, 1)
            target_root_velocity = recover_action(action[:6], self.velocity_limit[:6])
            palm_jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(current_qpos[:self.arm_dof])
            arm_qvel = compute_inverse_kinematics(target_root_velocity, palm_jacobian)[:self.arm_dof]
            arm_qvel = np.clip(arm_qvel, -np.pi / 1, np.pi / 1)
            arm_qpos = arm_qvel * self.control_time_step + self.robot.get_qpos()[:self.arm_dof]
            hand_qpos = recover_action(action[6:], self.robot.get_qlimits()[self.arm_dof:])
            target_qpos = np.concatenate([arm_qpos, hand_qpos])
            target_qvel = np.zeros_like(target_qpos)
            target_qvel[:self.arm_dof] = arm_qvel
            self.robot.set_drive_target(target_qpos)
            self.robot.set_drive_velocity_target(target_qvel)
            if check:
                self.robot_sim_obs = [self.robot.get_qpos().tolist()]

            for i in range(self.frame_skip):
                self.robot.set_qf(self.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
                self.scene.step()
                if check and i < self.frame_skip - 1:
                    self.robot_sim_obs.append(self.robot.get_qpos().tolist())
            self.current_step += 1

            ee_link_new_pose = self.ee_link.get_pose()
            relative_pos = ee_link_new_pose.p - ee_link_last_pose.p
            self.cartesian_error = np.linalg.norm(relative_pos - target_root_velocity[:3] * self.control_time_step)

    def reset_internal(self):
        self.current_step = 0
        if self.init_state is not None:
            self.scene.unpack(self.init_state)
        self.reset_env()
        if self.init_state is None:
            self.init_state = self.scene.pack()

    def step(self, action: np.ndarray, check=False, denormalized=False):
        self.rl_step(action, check, denormalized)
        if self.is_vision and not self.is_demo_rollout:
            obs = dict()
            test_state = self.get_test_state()
            point_cloud = self.get_camera_obs()['instance_1-point_cloud']
            trans_mat = test_state[25:].reshape((-1, 4, 4))
            obs['agent_pos'] = test_state[:22]
            if self.point_cs == "target":
                point_cloud = np.concatenate([point_cloud, point_cloud - test_state[22:25]], 1)
            elif self.point_cs == "hand":
                pcs_list = []
                pcs_list.append(point_cloud)
                pcs_list.append(point_cloud - test_state[22:25])
                for idx in range(len(trans_mat)):
                    trans_pc = point_cloud - trans_mat[idx, :3, 3]
                    trans_pc = (trans_mat[idx, :3, :3].T @ trans_pc.transpose(1, 0)).transpose(1, 0)
                    pcs_list.append(trans_pc)
                point_cloud = np.concatenate(pcs_list, 1)
            obs['point_cloud'] = point_cloud
        else:
            obs = self.get_observation()
        reward = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()
        # Reference: https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
        # Need to consider that is_done and timelimit can happen at the same time
        if self.current_step >= self.horizon:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, reward, done, info
    
    def setup_visual_obs_config(self, config: Dict[str, Dict]):
        for name, camera_cfg in config.items():
            if name not in self.cameras.keys():
                raise ValueError(f"Camera {name} not created. Existing {len(self.cameras)} cameras: {self.cameras.keys()}")
            self.camera_infos[name] = {}
            banned_modality_set = {"point_cloud", "depth"}
            if len(banned_modality_set.intersection(set(camera_cfg.keys()))) == len(banned_modality_set):
                raise RuntimeError(f"Request both point_cloud and depth for same camera is not allowed. Point cloud contains all information required by the depth.")

            # Add perturb for camera pose
            cam = self.cameras[name]
            if "pose_perturb_level" in camera_cfg:
                cam_pose_perturb = camera_cfg.pop("pose_perturb_level")
            else:
                cam_pose_perturb = None
            self.camera_pose_noise[name] = (cam_pose_perturb, cam.get_pose())

            for modality, cfg in camera_cfg.items():
                if modality == "point_cloud":
                    if "num_points" not in cfg:
                        raise RuntimeError(f"Missing num_points in camera {name} point_cloud config.")

                self.camera_infos[name][modality] = cfg

        modality = []
        for camera_cfg in config.values():
            modality.extend(camera_cfg.keys())
        modality_set = set(modality)

    def get_robot_state(self):
        raise NotImplementedError

    def get_oracle_state(self):
        raise NotImplementedError

    def get_camera_obs(self):
        self.scene.update_render()
        obs_dict = {}
        for name, camera_cfg in self.camera_infos.items():
            cam = self.cameras[name]
            modalities = list(camera_cfg.keys())
            # ic(modalities)
            texture_names = []
            for modality in modalities:
                if modality == "rgb":
                    texture_names.append("Color")
                elif modality == "depth":
                    texture_names.append("Position")
                elif modality == "point_cloud" and camera_cfg["point_cloud"].get("use_seg") is True:
                    texture_names.append("Segmentation")
                    texture_names.append("Position")
                elif modality == "point_cloud":
                    texture_names.append("Position")
                elif modality == "segmentation":
                    texture_names.append("Segmentation")
                else:
                    raise ValueError(f"Visual modality {modality} not supported.")
            await_dl_list = cam.take_picture_and_get_dl_tensors_async(texture_names)  # how is this done?
            dl_list = await_dl_list.wait()

            i = 0  # because pc_seg use 2 dl_list items, we cannot use enumerate here.
            for modality in modalities:
                key_name = f"{name}-{modality}"  # NOTE
                if modality == "point_cloud" and camera_cfg["point_cloud"].get("use_seg") is True:
                    import torch
                    dl_tensor_seg = dl_list[i]
                    output_array_seg = torch.from_dlpack(dl_tensor_seg).cpu().numpy()
                    i += 1
                    dl_tensor_pos = dl_list[i]
                    shape = sapien.dlpack.dl_shape(dl_tensor_pos)
                    output_array_pos = np.zeros(shape, dtype=np.float32)
                    sapien.dlpack.dl_to_numpy_cuda_async_unchecked(dl_tensor_pos, output_array_pos)
                    sapien.dlpack.dl_cuda_sync()

                    obs_pos = np.reshape(output_array_pos[..., :3], (-1, 3))
                    obs_seg = np.reshape(output_array_seg[..., 1:2], (-1, 1))
                    camera_pose = self.get_camera_to_robot_pose(name)
                    kwargs = camera_cfg["point_cloud"].get("process_fn_kwargs", {})

                    obs = process_pc(cloud=obs_pos, camera_pose=camera_pose, num_points=camera_cfg['point_cloud']['num_points'], np_random=self.np_random, grouping_info=None, segmentation=obs_seg, **kwargs)
                    # obs_dict[f"{name}-seg_gt"] = obs[:, 3:]  # NOTE: add gt segmentation
                    obs_dict[f"{name}-seg_gt"] = obs
                    if obs_dict[f"{name}-seg_gt"].shape != (camera_cfg["point_cloud"]["num_points"], 4):
                        # align the gt segmentation mask
                        obs_dict[f"{name}-seg_gt"] = np.zeros((camera_cfg["point_cloud"]["num_points"], 4))
                    obs = obs[:, :3]
                else:
                    dl_tensor = dl_list[i]
                    shape = sapien.dlpack.dl_shape(dl_tensor)
                    output_array = np.zeros(shape, dtype=np.float32)
                    if modality == "segmentation":
                        import torch
                        output_array = torch.from_dlpack(
                            dl_tensor).cpu().numpy()  # H, W, 4. [..., 0]: mesh-level segmentation; [..., 1]: link-level segmentation
                    elif modality == "rgb":
                        import torch
                        output_array = cam.get_color_rgba()
                    else:
                        output_array = np.zeros(shape, dtype=np.float32)
                        sapien.dlpack.dl_to_numpy_cuda_async_unchecked(dl_tensor, output_array)
                        sapien.dlpack.dl_cuda_sync()
                    if modality == "rgb":
                        obs = output_array[..., :3]
                    elif modality == "depth":
                        obs = -output_array[..., 2:3]
                        obs[obs[..., 0] > MAX_DEPTH_RANGE] = 0  # Set depth out of range to be 0
                    elif modality == "point_cloud":
                        obs = np.reshape(output_array[..., :3], (-1, 3))
                        # ic(obs.shape)
                        camera_pose = self.get_camera_to_robot_pose(name)
                        kwargs = camera_cfg["point_cloud"].get("process_fn_kwargs", {})
                        # obs = process_pc(cloud=obs, camera_pose=camera_pose, num_points=camera_cfg['point_cloud']['num_points'], np_random=self.np_random, grouping_info=None, segmentation=None, noise_level=3 if self.pc_noise else 0, **kwargs)
                        obs = process_pc(cloud=obs, camera_pose=camera_pose, num_points=camera_cfg['point_cloud']['num_points'], np_random=self.np_random, grouping_info=None, segmentation=None, **kwargs)
                        if "additional_process_fn" in camera_cfg["point_cloud"]:
                            for fn in camera_cfg["point_cloud"]["additional_process_fn"]:
                                obs = fn(obs, self.np_random)
                        obs_dict[f"{name}-seg_gt"] = np.zeros((camera_cfg["point_cloud"]["num_points"], 4))
                    elif modality == "segmentation":
                        obs = output_array[..., :2].astype(np.uint8)
                    else:
                        raise RuntimeError("What happen? you should not see this error!")
                obs_dict[key_name] = obs
                i += 1

        if len(self.imaginations) > 0:
            obs_dict.update(self.imaginations)

        return obs_dict
    
    def get_camera_to_robot_pose(self, camera_name):
        gl_pose = self.cameras[camera_name].get_pose()
        camera_pose = gl_pose * gl2sapien
        # camera2robot = self.robot.get_pose().inv() * camera_pose
        return camera_pose.to_transformation_matrix()

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,))

    @property
    def observation_space(self):
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        state_space = gym.spaces.Box(low=low, high=high)
        return state_space


def compute_inverse_kinematics(delta_pose_world, palm_jacobian, damping=0.05):
    lmbda = np.eye(6) * (damping ** 2)
    # When you need the pinv for matrix multiplication, always use np.linalg.solve but not np.linalg.pinv
    delta_qpos = palm_jacobian.T @ np.linalg.lstsq(palm_jacobian.dot(palm_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

    return delta_qpos
