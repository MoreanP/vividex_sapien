from functools import cached_property
from typing import Optional

import os
import copy
import trimesh
import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from hand_imitation.env.rl_env.base import BaseRLEnv
from hand_imitation.utils.ycb_object_utils import INVERSE_YCB_CLASSES, YCB_ROOT, YCB_CLASSES, YCB_HEIGHT_UNSEEN, YCB_ORIENTATION_UNSEEN
from hand_imitation.env.sim_env.relocate_env import LabRelocateEnv
from hand_imitation.real_world import lab
import pdb


def to_quat(arr):
    if isinstance(arr, Quaternion):
        return arr.unit
    if len(arr.shape) == 2:
        return Quaternion(matrix=arr).unit
    elif len(arr.shape) == 1 and arr.shape[0] == 9:
        return Quaternion(matrix=arr.reshape((3,3))).unit
    return Quaternion(array=arr).unit


def rotation_distance(q1, q2):
    delta_quat = to_quat(q2) * to_quat(q1).inverse
    return np.abs(delta_quat.angle)


class AllegroRelocateRLEnv(LabRelocateEnv, BaseRLEnv):
    def __init__(self, is_eval=False, is_vision=False, is_demo_rollout=False, is_real_robot=False, pc_noise=False, point_cs="world", norm_traj=False, task_kwargs=None, use_gui=False, frame_skip=10, motion_file=None, robot_name="allegro_hand_ur5", object_category="YCB", root_frame="robot", **renderer_kwargs):
        try:
            object_name = '_'.join(motion_file['object_name'].tolist().split('_')[1:])
            self._motion_file = motion_file.copy()
            self._reference_motion = motion_file.copy()
            self.init_object_height = self._reference_motion['init_object_height']
            self.task_name = motion_file['task_name']
        except:
            object_name = motion_file.split('-')[0]
            # object_name = motion_file.split('-')[1][4:]
            pose_idx = int(motion_file.split('-')[-1])
            # pose_idx = int(motion_file.split('-')[-2])
            self.init_object_height = YCB_HEIGHT_UNSEEN[object_name][pose_idx]
            self.init_object_quat = YCB_ORIENTATION_UNSEEN[object_name][pose_idx]
            self.task_name = "relocate"
            if "master_chef_can" in object_name or "bleach_cleanser" in object_name:
                self.init_object_height *= 0.8

            if "pitcher_base" in object_name or "wood_block" in object_name:
                self.init_object_height *= 0.6

        self.is_eval = is_eval
        self.is_vision = is_vision
        self.is_demo_rollout = is_demo_rollout
        self.is_real_robot = is_real_robot
        self.pc_noise = pc_noise
        self.point_cs = point_cs
        self.norm_traj = norm_traj
        super().__init__(use_gui, frame_skip, robot_name, object_category, object_name, **renderer_kwargs)

        # Base class
        self.setup(robot_name)

        # Parse link name
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]

        # Base frame for observation
        self.root_frame = root_frame
        self.base_frame_pos = np.zeros(3)

        # Finger tip: thumb, index, middle, ring
        finger_tip_names = ["right_gripper_link_15_tip", "right_gripper_link_03_tip", "right_gripper_link_07_tip", "right_gripper_link_11_tip"]
        finger_contact_link_names = ["right_gripper_link_15_tip", "right_gripper_link_15", "right_gripper_link_14","right_gripper_link_03_tip", "right_gripper_link_03", "right_gripper_link_02", "right_gripper_link_01", "right_gripper_link_07_tip", "right_gripper_link_07", "right_gripper_link_06", "right_gripper_link_05", "right_gripper_link_11_tip", "right_gripper_link_11", "right_gripper_link_10", "right_gripper_link_09"]
        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        robot_joint_names = ["right_shoulder_link", "right_upper_arm_link", "right_forearm_link", "right_wrist_1_link", "right_wrist_2_link", "right_wrist_3_link", "right_gripper_link_00", "right_gripper_link_01", "right_gripper_link_02", "right_gripper_link_03", "right_gripper_link_04", "right_gripper_link_05", "right_gripper_link_06", "right_gripper_link_07", "right_gripper_link_08", "right_gripper_link_09", "right_gripper_link_10", "right_gripper_link_11", "right_gripper_link_12", "right_gripper_link_13", "right_gripper_link_14", "right_gripper_link_15"]
        self.finger_tip_links = [self.robot.get_links()[robot_link_names.index(name)] for name in finger_tip_names]
        self.robot_joint_links = [self.robot.get_links()[robot_link_names.index(name)] for name in robot_joint_names]
        self.finger_contact_links = [self.robot.get_links()[robot_link_names.index(name)] for name in finger_contact_link_names]
        self.finger_contact_ids = np.array([0] * 3 + [1] * 4 + [2] * 4 + [3] * 4 + [4])
        self.finger_tip_pos = np.zeros([len(finger_tip_names), 3])

        # Contact buffer
        self.robot_object_contact = np.zeros(len(finger_tip_names) + 1)  # four fingers

        # Reference trajectory updates
        try:
            self._substeps = int(self._reference_motion['SIM_SUBSTEPS'])                
            self._data_substeps = self._reference_motion.get('DATA_SUBSTEPS', self._substeps)              
            self._step, self.traj_step = 0, 0
            pregrasp_step = self._reference_motion['pregrasp_step']

            self._reference_motion['robot_pregrasp_jpos'] = self._reference_motion['robot_jpos'][:pregrasp_step + 1].copy()
            self._reference_motion['robot_jpos'] = self._reference_motion['robot_jpos'][pregrasp_step:]
            self._reference_motion['object_translation'] = self._reference_motion['object_translation'][pregrasp_step:]
            self._reference_motion['object_orientation'] = self._reference_motion['object_orientation'][pregrasp_step:]
            self._reference_motion['length'] = self._reference_motion['length'] - pregrasp_step
            self.pregrasp_qpos = self._reference_motion['robot_qpos'][pregrasp_step][6:]
            self.cur_reference_motion = copy.deepcopy(self._reference_motion)
            self.start_step = 0
        except:
            self._step, self.traj_step = 0, 0
            self.start_step = 0

        self._stage = 0
        self.init_x = 0.35
        self.init_y = 0.35
        self.target_object_pos = np.zeros(3)

    def get_oracle_state(self):
        robot_state = self.get_robot_state()
        object_state = self.get_object_state()
        goal_state = self.get_goal_state()
        time_state = self.get_time_state()
        return np.concatenate([robot_state, object_state, goal_state, time_state])
    
    def get_object_state(self):
        object_pos = self.manipulated_object.get_pose().p
        object_quat = self.manipulated_object.get_pose().q
        object_lin_vel = self.manipulated_object.get_velocity()
        object_ang_vel = self.manipulated_object.get_angular_velocity()
        return np.concatenate([object_pos, object_quat, object_lin_vel, object_ang_vel])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        robot_qvel_vec = self.robot.get_qvel()
        robot_joint_pos = np.zeros([len(self.robot_joint_links), 3])
        robot_joint_quat = np.zeros([len(self.robot_joint_links), 4])
        robot_joint_lin_vel = np.zeros([len(self.robot_joint_links), 3])
        robot_joint_ang_vel = np.zeros([len(self.robot_joint_links), 3])
        for i, link in enumerate(self.robot_joint_links):
            robot_joint_pos[i] = self.robot_joint_links[i].get_pose().p
            robot_joint_quat[i] = self.robot_joint_links[i].get_pose().q
            robot_joint_lin_vel[i] = self.robot_joint_links[i].get_velocity()
            robot_joint_ang_vel[i] = self.robot_joint_links[i].get_angular_velocity()
        return np.concatenate([robot_qpos_vec, robot_qvel_vec, robot_joint_pos.reshape(-1), robot_joint_quat.reshape(-1), robot_joint_lin_vel.reshape(-1), robot_joint_ang_vel.reshape(-1)])
    
    def get_test_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        if self.is_vision:
            if self.norm_traj:
                robot_target_vec = np.array([self.init_x, self.init_y, 0.2])
            else:
                robot_target_vec = self.target_object_pos
        else:
            if self.norm_traj:
                robot_target_vec = np.array([self.init_x, self.init_y, 0.2])
            else:
                robot_target_vec = self.target_object.get_pose().p

        robot_mat = np.eye(4, dtype=np.float32)[None].repeat(5, 0)
        for i, link in enumerate(self.finger_tip_links):
            robot_mat[i] = self.finger_tip_links[i].get_pose().to_transformation_matrix()
        robot_mat[-1] = self.palm_link.get_pose().to_transformation_matrix()

        return np.concatenate([robot_qpos_vec, robot_target_vec, robot_mat.reshape(-1)])
    
    def get_goal_state(self):
        traj_goals = []
        for i in [1, 5, 10]:
            if self.traj_step + i <= self.pregrasp_steps:
                for k in ('object_orientation', 'object_translation'):
                    traj_goals.append(self.cur_reference_motion[k][0].flatten())
            else:
                i = min(self.traj_step + i, self.imitate_steps - 1) - self.pregrasp_steps
                for k in ('object_orientation', 'object_translation'):
                    traj_goals.append(self.cur_reference_motion[k][i].flatten())
        traj_goals = np.concatenate(traj_goals)
        
        hand_obj_diff = self.palm_link.get_pose().p - self.manipulated_object.get_pose().p
        finger_tip_pos = np.zeros([len(self.finger_tip_links), 3])
        for i, _ in enumerate(self.finger_tip_links):
            finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p
        hand_obj_dense_diff = finger_tip_pos - self.manipulated_object.get_pose().p
        if self.is_vision:
            hand_tgt_diff = self.palm_link.get_pose().p - self.target_object_pos
            obj_tgt_diff = self.manipulated_object.get_pose().p - self.target_object_pos
        else:
            hand_tgt_diff = self.palm_link.get_pose().p - self.target_object.get_pose().p
            obj_tgt_diff = self.manipulated_object.get_pose().p - self.target_object.get_pose().p
        return np.concatenate([traj_goals, hand_obj_diff.reshape(-1), hand_obj_dense_diff.reshape(-1), hand_tgt_diff.reshape(-1), obj_tgt_diff.reshape(-1)])
    
    def get_time_state(self):
        t = self.traj_step / self.imitate_steps
        t = np.array([1, 4, 6, 8]) * t
        time_state = np.concatenate((np.sin(t), np.cos(t)))
        return time_state

    def get_reward(self, action):
        if self.is_vision and not self.is_demo_rollout:
            return 0
        else:
            robot_hand_links = self.finger_tip_links
            robot_hand_pos = np.zeros([len(robot_hand_links) , 3])
            for i, link in enumerate(robot_hand_links):
                robot_hand_pos[i] = robot_hand_links[i].get_pose().p
            object_pos = self.manipulated_object.get_pose().p
            object_rot = self.manipulated_object.get_pose().q

            check_contact_links = self.finger_contact_links + [self.palm_link]
            contact_boolean = self.check_actor_pair_contacts(check_contact_links, self.manipulated_object)
            self.robot_object_contact[:] = np.clip(np.bincount(self.finger_contact_ids, weights=contact_boolean), 0, 1)
            self.is_contact = sum(self.robot_object_contact[:]) >= 1
            self.object_lift = max(object_pos[2] - self.init_object_height, 0)
        
            if self.current_step <= self.pregrasp_steps:
                tgt_robot_hand_pos = self.cur_reference_motion['robot_pregrasp_jpos'][-1, 1:]
                self.hand_jpos_err = np.mean(np.linalg.norm(robot_hand_pos - tgt_robot_hand_pos, axis=1))
                reward = 10 * np.exp(-10 * self.hand_jpos_err)
            else:
                reward = sum(self.robot_object_contact) * 0.5

                tgt_object_pos = self.cur_reference_motion['object_translation'][self.current_step - self.pregrasp_steps]
                tgt_object_rot = self.cur_reference_motion['object_orientation'][self.current_step - self.pregrasp_steps]
                tgt_robot_hand_pos = self.cur_reference_motion['robot_jpos'][self.current_step - self.pregrasp_steps, 1:]

                self.obj_com_err = np.linalg.norm(object_pos - tgt_object_pos)
                self.obj_rot_err = rotation_distance(object_rot, tgt_object_rot) / np.pi
                self.hand_mjpos_err = np.mean(np.linalg.norm(robot_hand_pos - tgt_robot_hand_pos, axis=1))

                reward += 10 * np.exp(-50 * (self.obj_com_err + 0.1 * self.obj_rot_err))
                reward += 4.0 * np.exp(-10 * self.hand_mjpos_err)

                if self.object_lift > 0.02:
                    reward += 2.5
        
            if not self.pregrasp_success and self.current_step == self.pregrasp_steps:
                if self.hand_jpos_err < 0.05:
                    self.pregrasp_success = True

            controller_penalty = (self.cartesian_error ** 2) * -1e3
            action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1) ** 2) * -0.01

            return (reward + action_penalty + controller_penalty) / 10

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, vis_pregrasp = False):
        # Gym reset function
        if seed is not None:
            self.seed(seed)
        
        if self._stage == 0:
            traj_x = 0.35
            traj_y = 0.35
            traj_rot = 0.0
        elif self._stage == 1:
            traj_x = self.np_random.uniform(low=0.3, high=0.4)
            traj_y = self.np_random.uniform(low=0.3, high=0.4)
            traj_rot = 0.0
        elif self._stage == 2:
            traj_x = self.np_random.uniform(low=0.3, high=0.4)
            traj_y = self.np_random.uniform(low=0.3, high=0.4)
            traj_rot = self.np_random.uniform(low=-1/12, high=1/12) * np.pi

        self.init_x = traj_x
        self.init_y = traj_y

        try:
            self.cur_reference_motion = copy.deepcopy(self._reference_motion)
            # canonicalize the trajectory
            self.cur_reference_motion['robot_jpos'] -= self.cur_reference_motion['object_translation'][0]
            self.cur_reference_motion['object_translation'] -= self.cur_reference_motion['object_translation'][0]
            self.cur_reference_motion['robot_jpos'][:, :, 2] += self.init_object_height
            self.cur_reference_motion['object_translation'][:, 2] += self.init_object_height
            
            # Rotate the trajectory
            rot_matrix = np.array([[np.cos(traj_rot), -np.sin(traj_rot), 0], [np.sin(traj_rot), np.cos(traj_rot), 0], [0, 0, 1]])
            self.cur_reference_motion['object_translation'] = (rot_matrix @ self.cur_reference_motion['object_translation'].transpose(1, 0)).transpose(1, 0)
            for idx in range(self.cur_reference_motion['object_orientation'].shape[0]):
                self.cur_reference_motion['object_orientation'][idx] = Quaternion(matrix=(rot_matrix @ Quaternion(self.cur_reference_motion['object_orientation'][idx]).rotation_matrix)).elements
            self.cur_reference_motion['robot_jpos'] = (rot_matrix[None] @ self.cur_reference_motion['robot_jpos'].transpose(0, 2, 1)).transpose(0, 2, 1)
            self.cur_reference_motion['robot_pregrasp_jpos'] = (rot_matrix[None] @ self.cur_reference_motion['robot_pregrasp_jpos'].transpose(0, 2, 1)).transpose(0, 2, 1)

            # Translate the trajectory
            self.cur_reference_motion['object_translation'][:, :2] += np.array([traj_x, traj_y])
            self.cur_reference_motion['robot_jpos'][:, :, :2] += np.array([traj_x, traj_y])
            self.cur_reference_motion['robot_pregrasp_jpos'][:, :, :2] += np.array([traj_x, traj_y])
            
            # Imitate until lifting by 0.1m
            if not self.norm_traj:
                keyframe = np.where((self.cur_reference_motion['object_translation'][:, 2] - self.cur_reference_motion['object_translation'][0, 2]) > 0.1)[0][0] + 1
                self.cur_reference_motion['object_translation'] = self.cur_reference_motion['object_translation'][:keyframe]
                self.cur_reference_motion['object_orientation'] = self.cur_reference_motion['object_orientation'][:keyframe]
                self.cur_reference_motion['robot_jpos'] = self.cur_reference_motion['robot_jpos'][:keyframe]
            else:
                if self.task_name == "pour":
                    target_pos = np.array([0.15, 0.35, 0.12])
                    target_orn = (R.from_rotvec(-2 * np.pi / 3 * np.array([0, 1, 0])) * R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

                    dist_vec = target_pos - self.cur_reference_motion['object_translation'][-1]
                    unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
                    relocate_step = 0.01
                    num_step = int(np.linalg.norm(dist_vec) // relocate_step)
                    init_rot = R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])
                    rotation_step = -2 * np.pi / 3 / num_step

                    syn_object_translation = []
                    syn_object_orientation = []
                    syn_robot_jpos = []
                    for idx in range(num_step):
                        step_size = (idx + 1) * relocate_step
                        syn_object_translation.append(self.cur_reference_motion['object_translation'][-1] + step_size * unit_dist_vec)
                        rot_size = (idx + 1) * rotation_step
                        syn_object_orientation.append((R.from_rotvec(rot_size * np.array([0, 1, 0])) * init_rot).as_quat()[[3, 0, 1, 2]])
                        cur_joint = self.cur_reference_motion['robot_jpos'][-1] - self.cur_reference_motion['object_translation'][-1]
                        rotmat = R.from_rotvec(rot_size * np.array([0, 1, 0])).as_matrix()
                        cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0) + syn_object_translation[-1]
                        syn_robot_jpos.append(cur_joint)
                    
                    rot_size = -2 * np.pi / 3
                    syn_object_translation.append(target_pos)
                    syn_object_orientation.append(target_orn)
                    cur_joint = self.cur_reference_motion['robot_jpos'][-1] - self.cur_reference_motion['object_translation'][-1]
                    rotmat = R.from_rotvec(rot_size * np.array([0, 1, 0])).as_matrix()
                    cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0) + syn_object_translation[-1]
                    syn_robot_jpos.append(cur_joint)

                    self.cur_reference_motion['object_translation'] = np.concatenate((self.cur_reference_motion['object_translation'], np.array(syn_object_translation)))
                    self.cur_reference_motion['object_orientation'] = np.concatenate((self.cur_reference_motion['object_orientation'], np.array(syn_object_orientation)))
                    self.cur_reference_motion['robot_jpos'] = np.concatenate((self.cur_reference_motion['robot_jpos'], np.array(syn_robot_jpos)))
                elif self.task_name == "place":
                    keyframe = np.where((self.cur_reference_motion['object_translation'][:, 2] - self.cur_reference_motion['object_translation'][0, 2]) > 0.25)[0][0] + 1
                    self.cur_reference_motion['object_translation'] = self.cur_reference_motion['object_translation'][:keyframe]
                    self.cur_reference_motion['object_orientation'] = self.cur_reference_motion['object_orientation'][:keyframe]
                    self.cur_reference_motion['robot_jpos'] = self.cur_reference_motion['robot_jpos'][:keyframe]

                    target_pos = np.array([0.15, 0.35, 0.12])
                    target_orn = (R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])) * R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

                    dist_vec = target_pos - self.cur_reference_motion['object_translation'][-1]
                    unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
                    relocate_step = 0.01
                    num_step = int(np.linalg.norm(dist_vec) // relocate_step)
                    init_rot = R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])
                    rotation_step = -np.pi / 2 / num_step

                    syn_object_translation = []
                    syn_object_orientation = []
                    syn_robot_jpos = []
                    for idx in range(num_step):
                        step_size = (idx + 1) * relocate_step
                        syn_object_translation.append(self.cur_reference_motion['object_translation'][-1] + step_size * unit_dist_vec)
                        rot_size = (idx + 1) * rotation_step
                        syn_object_orientation.append((R.from_rotvec(rot_size * np.array([1, 0, 0])) * init_rot).as_quat()[[3, 0, 1, 2]])
                        cur_joint = self.cur_reference_motion['robot_jpos'][-1] - self.cur_reference_motion['object_translation'][-1]
                        rotmat = R.from_rotvec(rot_size * np.array([1, 0, 0])).as_matrix()
                        cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0) + syn_object_translation[-1]
                        syn_robot_jpos.append(cur_joint)
                    
                    descent_vec = np.array([0, 0, -0.01])
                    for idx in range(2):
                        syn_object_translation.append(syn_object_translation[-1] + descent_vec)
                        syn_object_orientation.append(syn_object_orientation[-1])
                        syn_robot_jpos.append(syn_robot_jpos[-1] + descent_vec[None, :])

                    self.cur_reference_motion['object_translation'] = np.concatenate((self.cur_reference_motion['object_translation'], np.array(syn_object_translation)))
                    self.cur_reference_motion['object_orientation'] = np.concatenate((self.cur_reference_motion['object_orientation'], np.array(syn_object_orientation)))
                    self.cur_reference_motion['robot_jpos'] = np.concatenate((self.cur_reference_motion['robot_jpos'], np.array(syn_robot_jpos)))

            self.traj_len = len(self.cur_reference_motion['object_translation'])
            self.traj_step = 0
            self._step = 0
            
            # Set the final goal
            self._final_goal = self.cur_reference_motion['object_translation'][-1]
        except:
            self.traj_step = 0
            self._step = 0
        
        self.reset_internal()
        # Set robot qpos
        if vis_pregrasp:
            qpos = self._motion_file['robot_qpos'][self._motion_file['pregrasp_step']]
        else:
            qpos = np.zeros(self.robot.dof)
            xarm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = xarm_qpos
            qpos[18] = 0.5

        self.robot.set_qpos(qpos)
        self.robot.set_drive_target(qpos)

        # Set object pose
        try:
            object_pose = sapien.Pose(p=self.cur_reference_motion['object_translation'][0, :2].tolist() + [self.init_object_height], q=self.cur_reference_motion['object_orientation'][0].tolist())
            self.manipulated_object.set_pose(object_pose)
            if self.is_vision:
                self.target_object_pos = self.cur_reference_motion['object_translation'][-1]
            else:
                target_object_pose = sapien.Pose(p=self.cur_reference_motion['object_translation'][-1], q=self.cur_reference_motion['object_orientation'][-1])
                self.target_object.set_pose(target_object_pose)
        except:
            rot_matrix = np.array([[np.cos(traj_rot), -np.sin(traj_rot), 0], [np.sin(traj_rot), np.cos(traj_rot), 0], [0, 0, 1]])
            cur_object_quat = Quaternion(matrix=(rot_matrix @ Quaternion(self.init_object_quat).rotation_matrix)).elements
            object_pose = sapien.Pose(p=[self.init_x, self.init_y, self.init_object_height], q=cur_object_quat)
            self.manipulated_object.set_pose(object_pose)
            self.target_object_pos = np.array([self.init_x, self.init_y, 0.2])

        init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
        init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        self.robot.set_pose(init_pose)
        self.base_frame_pos = np.zeros(3)

        try:
            cur_robot_palm_pos = self.palm_link.get_pose().p
            self.start_step = np.argmin(np.linalg.norm(self.cur_reference_motion['robot_pregrasp_jpos'][:, 0, :] - cur_robot_palm_pos, axis=1))
            self.pregrasp_steps = 15
            self.imitate_steps = self.pregrasp_steps + len(self.cur_reference_motion['object_translation']) - 1
            if self.task_name == "pour" or self.task_name == "place":
                constant_steps = 80
            else:
                constant_steps = 60
                
            if self.imitate_steps < constant_steps:
                self.cur_reference_motion['object_translation'] = self.cur_reference_motion['object_translation'].tolist()
                self.cur_reference_motion['object_orientation'] = self.cur_reference_motion['object_orientation'].tolist()
                self.cur_reference_motion['robot_jpos'] = self.cur_reference_motion['robot_jpos'].tolist()
                self.cur_reference_motion['object_translation'] += [self.cur_reference_motion['object_translation'][-1]] * (constant_steps - self.imitate_steps)
                self.cur_reference_motion['object_orientation'] += [self.cur_reference_motion['object_orientation'][-1]] * (constant_steps - self.imitate_steps)
                self.cur_reference_motion['robot_jpos'] += [self.cur_reference_motion['robot_jpos'][-1]] * (constant_steps - self.imitate_steps)
                self.cur_reference_motion['object_translation'] = np.array(self.cur_reference_motion['object_translation'])
                self.cur_reference_motion['object_orientation'] = np.array(self.cur_reference_motion['object_orientation'])
                self.cur_reference_motion['robot_jpos'] = np.array(self.cur_reference_motion['robot_jpos'])
                self.imitate_steps = constant_steps
        except:
            self.imitate_steps = 60

        self.pregrasp_success = False
        self.hand_jpos_err = 0.3
        self.hand_mjpos_err = 0.0
        self.obj_com_err = 0.0
        self.obj_rot_err = 0.0
        self.object_lift = 0.0
        self.robot_object_contact = np.zeros(5, dtype=np.float32)

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
            return obs
        else:
            return self.get_observation()

    def is_done(self):
        if self.is_vision and not self.is_demo_rollout:
            if self.current_step >= self.imitate_steps:
                return True
            else:
                return False
        else:
            if not self.pregrasp_success:
                if self.current_step >= self.pregrasp_steps:
                    return True
                else:
                    self.traj_step += 1
                    return False
            else:
                if self.obj_com_err >= 0.15 or (not self.is_contact and self.current_step > self.pregrasp_steps):
                    return True
                else:
                    if self.current_step >= self.imitate_steps:
                        return True
                    else:
                        self.traj_step += 1
                        return False

    def is_success(self):
        if self.norm_traj:
            success_3 = np.linalg.norm(self.manipulated_object.get_pose().p - np.array([self.init_x, self.init_y, 0.2])) < 0.03
            success_10 = self.manipulated_object.get_pose().p[2] - self.init_object_height > 0.05
            check_contact_links = self.finger_contact_links + [self.palm_link]
            contact_boolean = self.check_actor_pair_contacts(check_contact_links, self.manipulated_object)
            self.robot_object_contact[:] = np.clip(np.bincount(self.finger_contact_ids, weights=contact_boolean), 0, 1)
            is_contact = self.robot_object_contact[0] and sum(self.robot_object_contact[1:]) >= 1
        else:
            if self.is_vision:
                success_3 = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object_pos) < 0.03
                success_10 = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object_pos) < 0.1
            else:
                success_3 = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object.get_pose().p) < 0.03
                success_10 = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object.get_pose().p) < 0.1
            check_contact_links = self.finger_contact_links + [self.palm_link]
            contact_boolean = self.check_actor_pair_contacts(check_contact_links, self.manipulated_object)
            self.robot_object_contact[:] = np.clip(np.bincount(self.finger_contact_ids, weights=contact_boolean), 0, 1)
            is_contact = self.robot_object_contact[0] and sum(self.robot_object_contact[1:]) >= 1
        return (is_contact and success_3), (is_contact and success_10)

    @cached_property
    def obs_dim(self):
        if self.is_vision and not self.is_demo_rollout:
            return len(self.get_test_state())
        else:
            return len(self.get_oracle_state())

    @cached_property
    def horizon(self):
        return self.imitate_steps
