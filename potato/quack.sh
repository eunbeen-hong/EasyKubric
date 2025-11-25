#!/bin/bash

set -e

# Run render.sh with CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 ./potato/render_rgb_track.sh \
    ./potato/configs/rgb_track.json \
    /datasets/kubric/rgb_track \
    5 \
    5


# ./potato/render.sh ./potato/configs/etap_simple.json /datasets/kubric/examples/etap_simple 3
# ./potato/render.sh ./potato/configs/etap_complex.json /datasets/kubric/examples/etap_complex 1
# ./potato/render.sh ./potato/configs/etap_low_steprate.json /datasets/kubric/examples/etap_low_steprate 1
# ./potato/render.sh ./potato/configs/dynamic_low_steprate.json /datasets/kubric/examples/dynamic_low_steprate 3
