# Medical Image Segmentation

## Run the following commend to run code
`sbatch run.slurm`

## Run the following commend to watch queue
`squeue -u jl25i045`

## Watch all the GPU types
`sinfo -o "%20N %10P %G"`

## Apply for computing node
`srun --account=invest --partition=gpu-invest --qos=job_gpu_caim --nodes=1 --gres=gpu:rtx4090:1 --cpus-per-task=8 --mem=80G --time=2:00:00 --pty bash`

`srun --account=gratis --partition=gpu --qos=job_gratis --nodes=1 --gres=gpu:rtx4090:1 --cpus-per-task=8 --mem=80G --time=2:00:00 --pty bash`

`srun --account=gratis --partition=gpu --qos=job_debug --nodes=1 --gres=gpu:rtx4090:1 --cpus-per-task=4 --mem=40G --time=20:00 --pty bash`
