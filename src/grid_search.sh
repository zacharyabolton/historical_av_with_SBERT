#!/bin/bash

# Hyperparameters to vary
margins=(0.45 0.5 0.55)
learning_rates=(0.00001 0.00002 0.00003)
clip_values=(0.5 1.0 1.5)

# Fixed hyperparameters (based on prior experiments)
batch_size=8
accum_steps=4
chunk_size=512
epsilon=0.000001
num_pairs=4096
num_folds=2
num_epochs=1
seed=0

for margin in "${margins[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for clip in "${clip_values[@]}"; do
            margin_str=$(echo $margin | tr '.' '_')
            lr_str=$(echo $lr | tr '.' '_')
            clip_str=$(echo $clip | tr '.' '_')
            exp_name="margin${margin_str}_lr${lr_str}_clip${clip_str}"
            
            echo "%%%%%%%%%%%% starting ${exp_name}"
            
            python3 main.py \
                "$exp_name" \
                "../data/normalized" \
                $batch_size \
                $accum_steps \
                $chunk_size \
                $margin \
                $epsilon \
                $num_pairs \
                $num_folds \
                $num_epochs \
                $lr \
                -s $seed \
                -m $clip
                
            echo "%%%%%%%%%%%% ending ${exp_name}"
            echo ""
        done
    done
done

echo "Done with all experiments"