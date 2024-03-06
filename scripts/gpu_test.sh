#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=Test_job_petteed
#SBATCH --time=0-00:15:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1              # Number of GPUS
#SBATCH --constraint="a100"  # Type of GPU
#SBATCH --mem=1G                  # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output.txt       # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'
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
echo "Number of GPUs: $SLURM_GPUS"


#* Modules
echo "---------------------------------------------------------"
echo "Loading modules..."

module load Python/3.10.4-GCCcore-11.3.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# # Activates the virtual enviroment for tensorflow 12
# conda activate tf_12

echo "---------------------------------------------------------"
echo "CUDA test:"
nvidia-smi

echo "---------------------------------------------------------"
echo "Running script.."
python ./utils/gpu_test.py
echo "---------------------------------------------------------"

# Resets the enviroment maintaining idempotency
module purge

























