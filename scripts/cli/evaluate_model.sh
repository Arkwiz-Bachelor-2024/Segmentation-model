#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --job-name=Evaluate_final_model

#SBATCH --time=0-00:15:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --constraint="gpu80g" # Type of GPUs
#SBATCH --gres=gpu:1              # Number of GPUS
#SBATCH --mem=64G                 # Asking for RAM
#SBATCH --nodes=1

#SBATCH --output=output.txt       # Specifying 'stdout'
#SBATCH --error=errors.txt        # Specifying 'stderr'
#SBATCH --mail-user=petteed@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "---------------------------------------------------------"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "---------------------------------------------------------"
echo "Assert Enviroment modules are loaded..."
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load Python/3.10.8-GCCcore-12.2.0
echo "---------------------------------------------------------"
echo "Assert python modules are loaded...."
pip install scikit-learn --user
pip install matplotlib --user
pip install cython --user
pip install git+https://github.com/lucasb-eyer/pydensecrf.git --user
echo "---------------------------------------------------------"
echo "Evaluating model: "
python ./scripts/evaluate_model.py
echo "---------------------------------------------------------"

# Resets the enviroment maintaining idempotency
module purge











































