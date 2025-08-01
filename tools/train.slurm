#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J train             # The job name
#SBATCH -o train-%j.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e train-%j.err        # Write the standard error to file named 'ret-<job_number>.err'


#- Needed resources

# (TODO)
# Please modify your requirements

#SBATCH -p r8nv-gpu-hw                # Submit to 'nvidia-gpu' Partitiion or queue
#SBATCH -t 1-06:00:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:8                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --constraint=A100            # Request GPU Type
#SBATCH --exclude=gpu-v02,r8a30-a07
###SBATCH --nodelist=gpu-a07
###SBATCH --ntasks-per-node=17

###
### The system will alloc 8 cores per gpu by default.
### If you need more or less, use following:
### SBATCH --cpus-per-task=32        # Request K cores 1card8 2card16 4card32 8card64
###

#SBATCH --qos=gpu-normal                 # Request QOS Type

#- Operstions
echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

#- Load environments
#source /tools/module_env.sh
#module list                       # list modules loaded by default

##- tools
#module load cmake/3.15.7
#module load git/2.17.1
#module load vim/8.1.2424
##- language
#module load python3/3.6.8

##- cuda
#module load cuda-cudnn/11.0-8.0.4

##- virtualenv
# source xxxxx/activate

#- Log information

#module list                      # list modules loaded by default
echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(which python3)
nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit
echo "Use GPU ${CUDA_VISIBLE_DEVICES}$"
TEMP_DEVICE=$CUDA_VISIBLE_DEVICES
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mujoco200
echo $MUJOCO_PY_MJPRO_PATH
IFS=','
read -ra ADDR <<< "$TEMP_DEVICE"
#- Warning! Please not change your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script

#- Job step
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-006_mustard_bottle-20200709-subject-01-20200709_143211 env.norm_traj=True ;
# CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-006_mustard_bottle-20200813-subject-02-20200813_150608 env.norm_traj=True ;
# CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-006_mustard_bottle-20200820-subject-03-20200820_140454 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-006_mustard_bottle-20200908-subject-05-20200908_144439 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-006_mustard_bottle-20200928-subject-07-20200928_144226 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-005_tomato_soup_can-20200709-subject-01-20200709_142853 env.norm_traj=True ;
# CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-005_tomato_soup_can-20200813-subject-02-20200813_150308 env.norm_traj=True ;
# CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-005_tomato_soup_can-20200820-subject-03-20200820_140158 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-005_tomato_soup_can-20201015-subject-09-20201015_143403 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-005_tomato_soup_can-20200709-subject-01-20200709_142926 env.norm_traj=True ;
# CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-004_sugar_box-20200709-subject-01-20200709_142517 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-004_sugar_box-20200813-subject-02-20200813_145951 env.norm_traj=True ;
# CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-004_sugar_box-20200820-subject-03-20200820_135841 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-004_sugar_box-20200918-subject-06-20200918_113441 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-004_sugar_box-20200903-subject-04-20200903_104157 env.norm_traj=True ;
# CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-004_sugar_box-20200908-subject-05-20200908_143931 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-052_extra_large_clamp-20200709-subject-01-20200709_152843 env.norm_traj=True ;
# CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-052_extra_large_clamp-20200813-subject-02-20200813_155356 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-052_extra_large_clamp-20200820-subject-03-20200820_144829 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-052_extra_large_clamp-20201002-subject-08-20201002_112816 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-025_mug-20200709-subject-01-20200709_150949 env.norm_traj=True ;
# CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-025_mug-20200813-subject-02-20200813_153839 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-025_mug-20200820-subject-03-20200820_143304 env.norm_traj=True ;
CUDA_VISIBLE_DEVICES=${ADDR[0]} python train.py env.name=ycb-025_mug-20200928-subject-07-20200928_154547 env.norm_traj=True

wait




#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
