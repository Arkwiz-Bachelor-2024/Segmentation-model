#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=seg_model_training

#SBATCH --time=0-01:00:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1              # Number of GPUS
#SBATCH --constraint="gpu40g"     # Type of GPU
#SBATCH --mem=1G                  # Asking for 16GB RAM
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
echo "Number of GPUs: $SLURM_GPUS"

#* Modules
echo "---------------------------------------------------------"
echo "Loading modules..."

module load Python/3.10.4-GCCcore-11.3.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load Anaconda3/2023.09-0 

# Initializing virtual enviroment
ENV_NAME=seg_model_env

# Checks if the enviroment is present
#? | : "pipeline" which takes the previous command as input to the new one 
#? grep : Searches the input given from pipeline (previous command) by the given pattern
#? ^$ENV_NAME\s : regex as the pattern
#? > : Takes the input on the left and writes it as an output to the file on the right where /dev/null is for discarding

# Name of the environment, extracted from the YAML file
ENV_NAME=$(grep 'name:' env.yaml | cut -d ' ' -f 2)

# Check if the environment exists
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' exists. Updating it based on YAML file."
    conda env update --file env.yaml --prune
else
    echo "Environment '$ENV_NAME' does not exist. Creating it from YAML file."
    conda env create -f env.yaml
fi

# Activate the environment
echo "Activating environment '$ENV_NAME'."
source activate $ENV_NAME  # Or use 'conda activate $ENV_NAME' if it works in your shell

echo "---------------------------------------------------------"
echo "GPU specifications:"
nvidia-smi
echo "---------------------------------------------------------"

echo "Running script.."
python ./
echo "---------------------------------------------------------"

# Resets the enviroment moddules maintaining idempotency
module purge

























