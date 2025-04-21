#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6        # Adjust based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
#SBATCH --time=4-00:00           # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=slurm_logs/slurm-%j.out  # Use Job ID for unique output files


if [ "$(pwd)" = "/home/kaim/projects/def-ichiro/kaim/SCNet" ]; then
    echo "Do not run this script from the SCNet directory. Exiting."
    exit 1
fi

rm -r metadata
rm -r result



module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

CURRENT_DATE=$(date +"%Y-%m-%d_%H-%M-%S")
SCRATCH_DIR=$SLURM_TMPDIR

git pull

#rm -r metadata
rm -r result

# Variables
MODEL_TYPE="scnet"
CONFIG_PATH="configs/config_musdb18_scnet.yaml"
DATASET_NAME="MUSDB18HQ_TEST"
DATASET_ZIP="/home/kaim/scratch/$DATASET_NAME.zip" # Specify the dataset ZIP name
SLURM_LOGS_PATH="slurm_logs/${MODEL_TYPE}_${CURRENT_DATE}"
CHECKPOINTS_PATH="/home/kaim/scratch/checkpoints/${MODEL_TYPE}_${CURRENT_DATE}"


source /home/kaim/projects/def-ichiro/kaim/SCNet/scnet_env/bin/activate


echo "Dataset path set to: $DATA_PATH"


echo "Running training script for model: $MODEL_TYPE with dataset at $DATA_PATH"

accelerate launch -m scnet.train_optuna 


