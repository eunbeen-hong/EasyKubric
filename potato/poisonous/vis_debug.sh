data_root="./outputs/"

vis_dir="./outputs/vis/"

exp_names=(
    blur_debug
)

python potato/scripts/visualizer.py \
    --data_dir ${data_root}${exp} \
    --output_dir "$vis_dir$exp" \
    --pair_vis
