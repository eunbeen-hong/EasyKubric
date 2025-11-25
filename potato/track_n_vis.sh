data_root="/mnt/data6/eunbeen/4d-recon/kubric/multiview/"

tracks_dir="/mnt/data6/eunbeen/4d-recon/kubric/multiview/tracks/"

exp_names=(
    "scene0"
)

for exp in "${exp_names[@]}"; do
    # Extract points
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 potato/scripts/point_extracter.py \
        --data_dir "$data_root$exp" \
        --out_dir "$tracks_dir$exp" \
        --all_pixels

    # Visualize tracks
    # !! Remove fps option to use input video's fps !!
    python3 potato/scripts/visualizer.py \
        --fps 12 \
        --data_dir "$data_root$exp" \
        --tracks_dir "$tracks_dir$exp" \
        --output_dir "$vis_dir$exp"
done
