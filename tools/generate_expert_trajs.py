import os
import sys
import pickle
import argparse
import numpy as np
import yaml
import zarr
import _init_paths
from tqdm import tqdm
from stable_baselines3 import PPO
from termcolor import cprint
from hand_imitation.env.create_env import create_env
import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pc(pc_obs):
    pc_obs_hand = pc_obs[np.argwhere(pc_obs[:,-1]==0).reshape(-1)]
    pc_obs_obj = pc_obs[np.argwhere(pc_obs[:,-1]==1).reshape(-1)]
    # 方式1：设置三维图形模式
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_obs_hand[:,0],pc_obs_hand[:,1],pc_obs_hand[:,2],c='b', s=5) # 画出hand的散点图
    ax.scatter(pc_obs_obj[:,0],pc_obs_obj[:,1],pc_obs_obj[:,2],c='r', s=10) # 画出obj的散点图
    ax.set_xlabel('X label') # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    ax.view_init(elev=45, azim=90)
    plt.savefig('point_cloud_seg.png')
    plt.cla()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--checkpoint_dir', type=str, required=True)
    parser.add_argument('-n', '--num_trajs', type=int, default=100)
    parser.add_argument('-o', '--data_name', type=str, required=True)
    parser.add_argument('-t', '--norm_traj', action='store_true')
    parser.add_argument('-r', '--real_robot', action='store_true')
    args = parser.parse_args()

    cur_path = os.path.dirname(__file__)
    if args.real_robot:
        save_dir = os.path.join(f'{cur_path}/../data/{args.data_name}_real.zarr')
    else:
        save_dir = os.path.join(f'{cur_path}/../data/{args.data_name}.zarr')
    os.makedirs(save_dir, exist_ok=True)

    tot_count = 0
    state_arrays = []
    point_cloud_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    for type_dir in os.listdir(args.checkpoint_dir):
        if type_dir not in ["mustard_bottle", "tomato_soup_can", "sugar_box", "extra_large_clamp", "mug"]:
            continue
        for model_dir in os.listdir(os.path.join(args.checkpoint_dir, type_dir)):
            print(f"{type_dir}-{model_dir}")
            # if model_dir not in ['ycb-005_tomato_soup_can-20201015-subject-09-20201015_143403']:
            #     continue
            model_path = os.path.join(args.checkpoint_dir, type_dir, model_dir)
            checkpoint_path = os.path.join(model_path, "restore_checkpoint.zip")
            config_path = os.path.join(model_path, "exp_config.yaml")
            config = yaml.safe_load(open(config_path, 'r'))

            policy = PPO.load(checkpoint_path)
            env_name = config['params']['env']['name']
            task_kwargs = config['params']['env']['task_kwargs']
            env = create_env(name=env_name, task_kwargs=task_kwargs, use_gui=False, is_eval=True, is_vision=True, is_demo_rollout=True, is_real_robot=args.real_robot, norm_traj=args.norm_traj)
            env._stage = 2

            with tqdm(total=args.num_trajs) as pbar:
                count_success = 0

                while count_success < args.num_trajs:
                    state_arrays_sub = []
                    point_cloud_arrays_sub = []
                    action_arrays_sub = []

                    obs = env.reset()
                    tot_reward = 0
                    tot_count_sub = 0
                    for j in range(env.horizon):
                        robot_obs = env.get_test_state()

                        # pc_obs = env.get_camera_obs()['instance_1-point_cloud']

                        # pc_obs = env.get_camera_obs()['instance_1-seg_gt']
                        # plot_pc(pc_obs)
                        # pdb.set_trace()

                        if env.camera_infos['instance_1']["point_cloud"].get("use_seg") is True:
                            pc_obs_seg = env.get_camera_obs()['instance_1-seg_gt']
                            pc_obs = pc_obs_seg[np.argwhere(pc_obs_seg[:,-1]==1).reshape(-1)][:,:3]
                            try:
                                index = np.random.choice(np.arange(pc_obs.shape[0]), size=128, replace=False)
                            except:
                                # print("obj pc num: ", pc_obs.shape[0])
                                # plot_pc(pc_obs_seg)
                                # pdb.set_trace()
                                index = np.random.choice(np.arange(pc_obs.shape[0]), size=128, replace=True)
                            pc_obs = pc_obs[index]
                        else:
                            pc_obs = env.get_camera_obs()['instance_1-point_cloud']

                        if isinstance(obs, dict):
                            for key, value in obs.items():
                                obs[key] = value[None, :]
                        else:
                            obs = obs[None, :]

                        try:
                            action = policy.predict(observation=obs, deterministic=True)[0]
                        except:
                            fix_obs = np.zeros((1, 396), dtype=np.float32)
                            fix_obs[:, :367] = obs[:, :367]
                            fix_obs[:, 367:370] = obs[:, 364:367]
                            fix_obs[:, 370:] = obs[:, 367:]
                            action = policy.predict(observation=fix_obs, deterministic=True)[0]

                        if len(action.shape) > 0:
                            action = action[0]

                        tot_count_sub += 1
                        state_arrays_sub.append(robot_obs)
                        point_cloud_arrays_sub.append(pc_obs)
                        action_arrays_sub.append(action)

                        obs, reward, done, info = env.step(action)
                        tot_reward += reward

                        if done:
                            break

                    if j == env.horizon - 1 and env.is_success():
                        count_success += 1
                        tot_count += tot_count_sub

                        episode_ends_arrays.append(tot_count) # the index of the last step of the episode    
                        state_arrays.extend(state_arrays_sub)
                        point_cloud_arrays.extend(point_cloud_arrays_sub)
                        action_arrays.extend(action_arrays_sub)

                        pbar.update(1)
                        pbar.set_description(f"The {count_success}th episode's reward is {tot_reward}")
    
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    point_cloud_chunk_size = (env.horizon, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    state_chunk_size = (env.horizon, state_arrays.shape[1])
    action_chunk_size = (env.horizon, action_arrays.shape[1])
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    # print shape
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}/data.zarr', 'green')
