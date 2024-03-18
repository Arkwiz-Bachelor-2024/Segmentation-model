#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=small_seg_model

#SBATCH --time=0-01:00:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1              # Number of GPUS
#SBATCH --constraint="gpu40g"     # Type of GPU
#SBATCH --mem=20G                 # Asking for 20GB RAM
#SBATCH --nodes=1

#SBATCH --output=output.txt       # Specifying 'stdout'
#SBATCH --error=errors.txt        # Specifying 'stderr'

#SBATCH --mail-user=petteed@stud.ntnu.no
#SBATCH --mail-type=ALL

#* Running jobs 
#? Queues the job
# sbatch <job_name>
# Returns job_id    e.g <1891923>

#? Check jobs queued
# squeue -u <username> 
# Returns table of jobs queued

#? Cancel job
# scancel <job_id>

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "---------------------------------------------------------"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "---------------------------------------------------------"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "---------------------------------------------------------"

#* Modules

module load TensorFlow/2.13.0-foss-2023a
module load CUDA/11.8.0 

pip install --user natsort

echo "---------------------------------------------------------"
echo "GPU specifications:"
nvidia-smi
echo "---------------------------------------------------------"

echo "Running script.."
echo "---------------------------------------------------------"
python app.py
echo "---------------------------------------------------------"

echo "Script completed"
# Resets the enviroment maintaining idempotency
module purge











































