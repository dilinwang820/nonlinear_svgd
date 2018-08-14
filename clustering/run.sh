#! /bin/bash

set -e 

kernel='rbf'
temperature=0.5

for((i=100;i<110;i++))
do
    if [ $i -eq 100 ];then
        python trainer.py --method svgd --dataset mnist --seed $i  --gpu 0  --kernel ${kernel} --temperature ${temperature} 
    else
        python trainer.py --method svgd --dataset mnist --seed $i  --skip_pretrain --gpu 0  --kernel ${kernel} --temperature ${temperature} 
    fi
done
