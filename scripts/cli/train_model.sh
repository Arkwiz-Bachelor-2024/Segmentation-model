#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=DEA_Deeplabv3Plus_quarter_200e_4b_Centropy_adam_sgd_hybrid+DA_mild

#SBATCH --time=0-10:00:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:3              # Number of GPUS
#SBATCH --mem=40G                 # Asking for RAM
#SBATCH --nodes=1

#SBATCH --output=output.txt       # Specifying 'stdout'
#SBATCH --error=errors.txt        # Specifying 'stderr'
#SBATCH --mail-user=petteed@stud.ntnu.no
#SBATCH --mail-type=ALL

#* Running jobs 
#? Queues the job
# sbatch <job_name> test
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
echo "Number of GPUs : $SLURM_JOB_GPUS"
echo "---------------------------------------------------------"
echo "Assert Enviroment modules are loaded..."
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
echo "---------------------------------------------------------"
echo "Assert python modules are loaded...."
pip install scikit-learn --user
pip install matplotlib --user
pip install cython --user
pip install git+https://github.com/lucasb-eyer/pydensecrf.git --user
echo "---------------------------------------------------------"
echo "GPU specifications:"
nvidia-smi
echo "---------------------------------------------------------"
echo "Training model:"
echo "---------------------------------------------------------"
python ./scripts/train_model.py
echo "---------------------------------------------------------"
# echo "Evaluating model:"
# python ./scripts/evaluate_model.py
echo "---------------------------------------------------------"
echo "Script completed"

# Resets the enviroment maintaining idempotency
module purge











































