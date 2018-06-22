#!/bin/sh
export CUDA_VISIBLE_DEVICES=7
train_dir="./DCN-fc_mnist_10c"

python train_DCN.py --train_dir $train_dir \
    --network "DCN-fc" \
    --dataset "mnist-aug" \
    --data_dir "data/mnist/" \
    --num_train_instance 60000 \
    --num_classes 10 \
    --num_clusters 10 \
    --feature_dim 10 \
    --batch_size 100 \
    --test_interval 600 \
    --test_iter 100 \
    --fc_bias False \
    --clustering_loss 0.1 \
    --l2_weight 0.0001 \
    --initial_lr 0.001 \
    --lr_step_epoch 50.0 \
    --lr_decay 5.0 \
    --max_steps 60000 \
    --checkpoint_interval 12000 \
    --gpu_fraction 0.96 \
    --display 100 \
