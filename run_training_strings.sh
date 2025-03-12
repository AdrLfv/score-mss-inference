#!/bin/bash

# Modified from the MUSDB18 example of the asteroid library.

# Exit on error
set -e
set -o pipefail

python_path=python

# Example usage
# ./run_training_strings.sh --architecture input_concat

# General
train_on=synthsod     # Controls the dataset used for training (only synthsod supported for now)
architecture=noaudio
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0 #$CUDA_VISIBLE_DEVICES

. parse_options.sh

tag=synthsod_${architecture}_strings  # Controls the directory name associated to the experiment

uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
expdir=exp/train_xumx_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

mkdir -p logs
CUDA_VISIBLE_DEVICES=$id $python_path train.py \
  --architecture ${architecture} \
  --train_on $train_on \
  --targets Violin Viola Cello Bass \
  --output ${expdir} | tee logs/train_${tag}.log \
cp logs/train_${tag}.log $expdir/train.log

echo "Training finished. Results are stored in $expdir"
