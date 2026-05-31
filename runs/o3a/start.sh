#!/bin/bash

# Optional: activate virtual environment if you have one
# source /path/to/venv/bin/activate

# Make the dataset required for training
# python3 -c "from dataset import make_dataset; make_dataset()"

# Retry any failed/missing segments
# python3 -c "from dataset import retry_dataset; retry_dataset()"
# python3 -c "from dataset import retry_dataset; retry_dataset(detectors=['H1'], num_workers=8)"

# Run Sage
python3 -c "from train import run_sage; run_sage()"