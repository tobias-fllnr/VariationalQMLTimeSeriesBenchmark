#!/bin/bash

module load spack/default #Modify to your use

module load gcc/12.3.0

module load python/3.12.1

# Activate the virtual environment
source /path/to/your/venv/bin/activate

python3 submit_multiple_jobs.py