#!/bin/bash

# Default Variables
CONFIG_PATH="configs/config.yml"
DATA_PATH="processed_dataset" 
REP_PATH="."
EPOCHS=10 # reduced for debugging
TEACHER=""
FLAG=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --teacher) TEACHER="$2"; shift ;;
        --resume) FLAG="--continue_train" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Function to run training with the specified teacher
run_training() {
    local TEACHER="$1"
    echo "Starting training with teacher: $TEACHER ..."
    accelerate launch train_modified.py \
        --config_path "$CONFIG_PATH" \
        --data_path "$DATA_PATH" \
        --rep_path "$REP_PATH" \
        --teacher "$TEACHER" \
        --epochs "$EPOCHS" \
        $FLAG
    echo "$TEACHER training completed."
}

# Main logic
if [ -z "$TEACHER" ]; then
    # If no teacher is specified, run sequentially through the options
    for teacher_option in "hubert" "llm" "combined"; do
        run_training "$teacher_option"
    done
else
    # If teacher is specified, run the training for that type
    run_training "$TEACHER"
fi

