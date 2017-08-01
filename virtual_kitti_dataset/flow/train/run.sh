#!/bin/bash

set -e
TOOLS=/home/bingbing/git/convlstm_anomaly_detection/build/tools

$TOOLS/caffe train \
    --solver=solver.prototxt \
    --gpu 1 \
    2>&1 | tee log/train000000001.log
