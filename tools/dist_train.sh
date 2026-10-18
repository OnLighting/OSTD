#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORK_DIR=${3:-work_dirs/$(basename ${CONFIG%.*})}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG $WORK_DIR --launcher pytorch ${@:4}