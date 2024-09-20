#!/bin/bash

# 设置参数
NUM_GPUS_PER_NODE=1  # 每台服务器的 GPU 数量
NODE_RANK=0          # 当前节点的排名，第一台服务器为 0，第二台服务器为 1
RDZV_ENDPOINT="192.168.0.56:8000"  # 主进程的 IP 地址和端口

# 运行训练命令
torchrun \
  --nnodes=2 \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --rdzv_endpoint=$RDZV_ENDPOINT \
  ./train_mutil.py \
  --audio-dir ./output_audio \
  --image-dir ./output_images \
  --train-label-file ./label.csv \
  --val-label-file ./val_label.csv \
  --batch-size 16 \
  --epochs 10 \