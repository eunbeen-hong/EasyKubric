data_root="/mnt/data6/eunbeen/4d-recon/kubric/multiview/"

tracks_dir="/mnt/data6/eunbeen/4d-recon/kubric/multiview/tracks/"

exp_names=(
    "scene0"
)

for exp in "${exp_names[@]}"; do
    # Extract points
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python potato/scripts/point_extracter.py \
        --data_dir "$data_root$exp" \
        --out_dir "$tracks_dir$exp" \
        --n_procs 16 \
        --all_pixels
done
