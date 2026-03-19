#!/bin/bash

# Optional: activate virtual environment if you have one
# source /path/to/venv/bin/activate

# Register the Sage run configs
python3 -c "from config import set_configs; set_configs()"

# Make the dataset required for training
python3 -c "from dataset import make_dataset; make_dataset()"

# Run Sage
python3 -c "from train import run_sage; run_sage()"