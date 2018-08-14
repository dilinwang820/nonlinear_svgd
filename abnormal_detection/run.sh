#! /bin/bash

n_epochs=200
gpu=0
n_components=4
temperature=5e-1

for((seed=100;seed<120;seed++))
do
    python trainer.py  --method svgd --n_components ${n_components} --kernel rbf --seed ${seed} --temperature ${temperature} --n_epochs ${n_epochs} --clean --gpu ${gpu} --lr_weight_decay 
done



