#!/bin/bash

# Variables
CONFIG="config_modified.json"
AUDIO_DIR="processed_dataset"
EXTS="flac,wav"
SPLIT_SEED=42
VALID_SET_SIZE=0.00467
# VALID_SET_SIZE=0.34

# Assign the first argument to SEMANTIC
SEMANTIC="${1:-all}"  # If no argument is provided, default to "all"

# Function to extract based on the semantic type
run_extraction() {
    local SEMANTIC_TYPE="$1"
    echo "Extracting $SEMANTIC_TYPE representation."
    python extract_semantic_rep.py --audio_dir "$AUDIO_DIR" --rep_typ "$SEMANTIC_TYPE" --exts "$EXTS" --split_seed "$SPLIT_SEED" --valid_set_size "$VALID_SET_SIZE"
    echo "$SEMANTIC_TYPE representation extraction completed."
}

# Main logic
if [[ "$SEMANTIC" == "all" ]]; then
    for semantic_type in "hubert" "llm" "combined"; do
        run_extraction "$semantic_type"
    done
elif [[ "$SEMANTIC" == "hubert" || "$SEMANTIC" == "llm" || "$SEMANTIC" == "combined" ]]; then
    run_extraction "$SEMANTIC"
else
    echo "Unknown semantic type: $SEMANTIC. Please use 'hubert', 'llm', 'combined', or leave empty for 'all'."
    exit 1
fi
