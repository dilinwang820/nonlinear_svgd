# Nonlinear-SVGD for Unsupervised  Abnormal Detection
Use `run.sh` to reproduce our results. Example:

    python trainer.py --method svgd --n_components 4 --kernel rbf --seed 100 --temperature 5e-1 --n_epochs 200 --clean --gpu 0 --lr_weight_decay

## Requirements
Tensorflow 1.8

## References
Our code is based on the following pytorch implementation:
https://github.com/danieltan07/dagmm

