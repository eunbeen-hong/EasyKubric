data_root="/mnt/nvme1n1/event_data/kubric/"

tracks_dir="/home/cvlab21/project/eunbeen/kubric/outputs/tracks/"

exp_names=(
    "0919"
)

for exp in "${exp_names[@]}"; do
    # Extract points
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python potato/scripts/point_extracter.py \
        --data_dir "$data_root$exp" \
        --out_dir "$tracks_dir$exp"

    # Visualize tracks
    # !! Remove fps option to use input video's fps !!
    python potato/scripts/visualizer.py \
        --fps 12 \
        --data_dir "$data_root$exp" \
        --tracks_dir "$tracks_dir$exp" \
        --output_dir "$vis_dir$exp"
done
