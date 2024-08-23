#!/bin/bash

# Variables
DATASET="zachary24/librispeech_train_clean_100"
SPLIT="train"
SEGMENT_LENGTH=3.0
RATIO=1.0
# RATIO=0.00084
    
# Script to process the dataset
echo "Running process_dataset.py script with the following parameters:"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Ratio: $RATIO"
echo "Segment Length: $SEGMENT_LENGTH"

python process_dataset.py --dataset "$DATASET" --split "$SPLIT" --ratio "$RATIO" --segment_length "$SEGMENT_LENGTH"

echo "Dataset processing completed."