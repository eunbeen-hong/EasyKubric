#!/bin/bash
export PYTHONPATH=/kubric:$PYTHONPATH

# --- Parameters
min_num_static_objects=10
max_num_static_objects=20

min_num_dynamic_objects=1
max_num_dynamic_objects=3

camera="linear_movement"  # choose from fixed_random, linear_movement, linear_movement_linear_lookat
max_camera_movement=4.0
max_motion_blur=2.0

frame_rate=48
step_rate=240

frame_start=1
frame_end=96

resolution="512x512"

min_amb_light=0.01
max_amb_light=1.0

iter=3

# --- Function to run kub_render.py
run_render() {
    local motion_blur=$1
    local job_dir=$2

    CMD="python potato/scripts/multi_rendering.py \
        --min_num_static_object $min_num_static_objects \
        --max_num_static_object $max_num_static_objects \
        --min_num_dynamic_object $min_num_dynamic_objects \
        --max_num_dynamic_object $max_num_dynamic_objects \
        --camera $camera \
        --max_camera_movement $max_camera_movement \
        --max_motion_blur $motion_blur \
        --frame_rate $frame_rate \
        --step_rate $step_rate \
        --frame_start $frame_start \
        --frame_end $frame_end \
        --resolution $resolution \
        --min_amb_light $min_amb_light \
        --max_amb_light $max_amb_light \
        --job-dir $job_dir"

    eval $CMD
}

# --- Loop over iterations
for i in $(seq 0 $((iter-1))); do
    output_dir=/datasets/kubric/examples/etap_simple/${i}
    run_render $max_motion_blur ${output_dir}
done
