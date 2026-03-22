#!/bin/bash
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_caim
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --job-name=seg_fold0
#SBATCH --output=outputs/%x/output.log
#SBATCH --error=outputs/%x/error.log

module load CUDA/12.6
source /software.9/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate torch

export WANDB_API_KEY="706b7fac4cabad0096600f592e3f3373f145ef86"

accelerate launch --config_file ./configs/accelerate/fp16.yaml run.py \
    name=seg_fold0 
