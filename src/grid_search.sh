#!/bin/bash

# Hyperparameters to vary
margin_ss=(0.5 0.6 0.7 0.8 0.9)
margin_ds=(0.1 0.2 0.3 0.4 0.5)

# Fixed hyperparameters (based on prior experiments)
batch_size=8
accum_steps=4
chunk_size=512
epsilon=0.000001
num_pairs=4096
num_folds=2
num_epochs=1
initial_lr=0.00002
max_norm=1.0

for margin_s in "${margin_ss[@]}"; do
    for margin_d in "${margin_ds[@]}"; do
        margin_s_str=$(echo $margin_s | tr '.' '_')
        margin_d_str=$(echo $margin_d | tr '.' '_')
        exp_name="margin_s${margin_s_str}_margin_d${margin_d_str}"
        
        echo "%%%%%%%%%%%% starting ${exp_name}"
        
        python3 main.py \
            "$exp_name" \
            "../data/normalized" \
            $batch_size \
            $accum_steps \
            $chunk_size \
            $margin_s \
            $margin_d \
            $epsilon \
            $num_pairs \
            $num_folds \
            $num_epochs \
            $initial_lr \
            -m $max_norm
            
        echo "%%%%%%%%%%%% ending ${exp_name}"
        echo ""
    done
done

echo "Done with all experiments"
