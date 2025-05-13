#!/bin/bash

models=("d_QNN" "ru_QNN" "QRNN" "QLSTM" "le_QLSTM" "MLP" "RNN" "LSTM")

for model in "${models[@]}"; do
    python3 ./utils/submit_multiple_jobs.py -model "$model" -slurm "yes"
done