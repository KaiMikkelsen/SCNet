#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6        # Adjust based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
#SBATCH --time=5-00:00           # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=slurm_logs/slurm-%j.out  # Use Job ID for unique output files

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

CURRENT_DATE=$(date +"%Y-%m-%d_%H-%M-%S")
SCRATCH_DIR=$SLURM_TMPDIR

# Variables
MODEL_TYPE="scnet"
CONFIG_PATH="configs/config_musdb18_scnet.yaml"
DATASET_NAME="MUSDB18HQ"
DATASET_ZIP="../data/$DATASET_NAME.zip" # Specify the dataset ZIP name
SLURM_LOGS_PATH="slurm_logs/${MODEL_TYPE}_${CURRENT_DATE}"
CHECKPOINTS_PATH="/home/kaim/scratch/checkpoints/${MODEL_TYPE}_${CURRENT_DATE}"

# Create necessary directories
#mkdir -p "$SCRATCH_DIR"
mkdir -p "$SLURM_LOGS_PATH"

mkdir -p "$CHECKPOINTS_PATH"

# Redirect SLURM output dynamically
exec > >(tee -a "$SLURM_LOGS_PATH/slurm-${SLURM_JOB_ID}.out") 2>&1

# Activate the environment
source /home/kaim/projects/def-ichiro/kaim/Music-Source-Separation-Training/separation_env/bin/activate

RUNNING_ON_MAC=False
if [ "$RUNNING_ON_MAC" = False ]; then

    # Move and unzip dataset to scratch directory
    if [ ! -f "$SCRATCH_DIR/$(basename "$DATASET_ZIP")" ]; then
        echo "Moving $DATASET_ZIP to $SCRATCH_DIR for faster access"
        cp "$DATASET_ZIP" "$SCRATCH_DIR"
    else
        echo "Dataset already exists in $SCRATCH_DIR, skipping copy."
    fi

    DATASET_ZIP_BASENAME=$(basename "$DATASET_ZIP")
    SCRATCH_ZIP="$SCRATCH_DIR/$DATASET_ZIP_BASENAME"

    # mkdir -p "$SCRATCH_DIR/$DATASET_NAME"
    # echo "created directory $SCRATCH_DIR/$DATASET_NAME"
    # echo "Unzipping dataset in $SCRATCH_DIR/$DATASET_NAME"

    echo "unzipping $SCRATCH_DIR/$DATASET_NAME.zip"
    unzip "$SCRATCH_DIR/$DATASET_NAME.zip" -d "$SCRATCH_DIR"


    # if ! unzip -q "$SCRATCH_ZIP" -d "$SCRATCH_DIR/$DATASET_NAME"; then
    #     echo "zip failed"
    # fi

    echo "Dataset successfully unzipped."

    if [ "$DATASET_NAME" = "MOISESDB" ]; then
        DATA_PATH="$SCRATCH_DIR/$DATASET_NAME/moisesdb/moisesdb_v0.1"
    elif [ "$DATASET_NAME" = "MUSDB18HQ" ]; then
        DATA_PATH="$SCRATCH_DIR/$DATASET_NAME"
    elif [ "$DATASET_NAME" = "SDXDB23_Bleeding" ]; then
        DATA_PATH="$SCRATCH_DIR/$DATASET_NAME/sdxdb12_bleeding"
    elif [ "$DATASET_NAME" = "SDXDB23_LabelNoise" ]; then
        DATA_PATH="$SCRATCH_DIR/$DATASET_NAME/sdxdb23_labelnoise"
    else
        echo "Unknown dataset: $DATASET_NAME"
        exit 1
    fi

else
    DATA_PATH="../data/$DATASET_NAME"
    echo "Running on Mac. Skipping dataset unzipping."
fi

echo "Dataset path set to: $DATA_PATH"


echo "Running training script for model: $MODEL_TYPE with dataset at $DATA_PATH"

accelerate launch -m scnet.train 

# python train.py \
#     --model_type "$MODEL_TYPE" \
#     --config_path "$CONFIG_PATH" \
#     --results_path "$CHECKPOINTS_PATH" \
#     --data_path "$DATA_PATH/train" \
#     --valid_path "$DATA_PATH/validation" \
#     --metrics sdr \
#     --num_workers 4 \
#     --start_check_point "" \
#     --device_ids 0 \
#     --wandb_key 689bb384f0f7e0a9dbe275c4ba6458d13265990d

