#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Variables
INPUT_FILE=""
OUTPUT_DIR="data/output"
MODEL_PATH=""
INSTRUMENTS=""

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --input-file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --instruments)
            INSTRUMENTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if required options are provided
if [ -z "$INPUT_FILE" ] || [ -z "$MODEL_PATH" ] || [ -z "$INSTRUMENTS" ]; then
    echo "Usage: $0 --input-file <path> --model-path <path> --instruments <list> [--output-dir <path>]"
    exit 1
fi

# Run inference
python3 inference.py \
    --model "$MODEL_PATH" \
    --input "$INPUT_FILE" \
    --outdir "$OUTPUT_DIR/$INSTRUMENTS" \
    --no-cuda

echo "Inference complete. Separated tracks are in $OUTPUT_DIR/$INSTRUMENTS"
