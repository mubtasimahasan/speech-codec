#!/bin/bash

# Default Variables
CONFIG_PATH="config_debug.yml"
DATA_PATH="processed_dataset"
REP_PATH="."
EPOCHS=20
REP_TYPE=""
FLAG=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rep_type) REP_TYPE="$2"; shift ;;
        --resume) FLAG="--continue_train" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Function to run training with the specified rep_type
run_training() {
    local REP_TYPE="$1"
    echo "Starting training with rep_type: $REP_TYPE..."
    accelerate launch train_modified.py \
        --config_path "$CONFIG_PATH" \
        --data_path "$DATA_PATH" \
        --rep_path "$REP_PATH" \
        --rep_type "$REP_TYPE" \
        --epochs "$EPOCHS" \
        $FLAG
    echo "$REP_TYPE training completed."
}

# Main logic
if [ -z "$REP_TYPE" ]; then
    # If no rep_type is specified, run sequentially through the options
    for rep_type_option in "hubert" "llm" "combined"; do
        run_training "$rep_type_option"
    done
else
    # If rep_type is specified, run the training for that type
    run_training "$REP_TYPE"
fi
