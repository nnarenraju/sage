#!/bin/bash

# Optional: activate virtual environment if you have one
# source /path/to/venv/bin/activate

# Make the dataset required for training
# python3 -c "from dataset import make_dataset; make_dataset()"

# Run Sage
python3 -c "from train import run_sage; run_sage()"