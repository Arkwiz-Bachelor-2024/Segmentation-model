#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=Test_job_petteed
#SBATCH --time=0-00:15:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --constraint="v100|a100|p100" # Asks for a specific types of GPU
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output.txt      # Specifying 'stdout'
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
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

#* Modules

# GPU acceleration modules
module load Anaconda3/2023.09-0
module load CUDA/11.8.0

# Creates an Anaconda enviroment 
conda create -n mlenv python=3.10
# Activates the virtual enviroment 
conda activate mytensorflowenv
pip install tensorflow=2.12.0

python ./utils/gpu_test.py

# Prints the specifications of the enviroment the job was run on
uname -a

# Resets the enviroment maintaining idempotency
module purge

















