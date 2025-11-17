data_root="/mnt/nvme1n1/event_data/kubric/"

tracks_dir="/mnt/nvme1n1/event_data/kubric/"

exp_names=(
    "240fps_no_blur_18proc"
    "240fps_no_blur_18proc_2"
)

for exp in "${exp_names[@]}"; do
    # Extract points
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python potato/scripts/point_extracter.py \
        --data_dir "$data_root$exp" \
        --out_dir "$tracks_dir$exp" \
        --n_procs 16
done
