#!/bin/bash

# SLURM Resource Parameters

#SBATCH -n 5                       # Number of cores
#SBATCH -N 1                        # Number of nodes always 1
#SBATCH -t 4:00:00 # D-HH:MM        # Time using the nodes
#SBATCH -p GPU-shared               # Partition you submit to
#SBATCH --gres=gpu:v100-32:1               # GPUs
#SBATCH --mem=32G                   # Memory you need
#SBATCH --job-name=DROIDSLAM     # Job name
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END             # Type of notification BEGIN/FAIL/END/ALL
#SBATCH --mail-user=satanama.ring@gmail.com
# Executable
EXE=/bin/bash

singularity exec --nv --bind /data2/datasets/yuhengq:/zihao/datasets:ro,/data2/datasets/yuhengq/zihao/DROID-SLAM:/zihao/DROID-SLAM /data2/datasets/yuhengq/zihao/DroidSLAM.sif bash /zihao/DROID-SLAM/script_v2_test.sh