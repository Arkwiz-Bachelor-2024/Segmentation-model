#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=Test_job_petteed
#SBATCH --time=0-00:15:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output.txt      # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'

#SBATCH --mail-user=petteed@stud.ntnu.no
#SBATCH --mail-type=ALL

#* Required modules for ML 
# TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
# Python/3.10.8-GCCcore-12.2.0

#* Running job

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

module purge

# Running your python file
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load Python/4.10.8-GCCcore-12.2.0

python ./utils/gpu_test.py

# Prints the specifications of the enviroment the job was run on
uname -a
















