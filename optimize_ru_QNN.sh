#!/bin/bash

python3 ./utils/optimize_ru_qnn_ansatz.py
python3 ./utils/submit_multiple_jobs.py -model "best_ru_QNN" -slurm "no"