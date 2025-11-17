data_root="/mnt/nvme1n1/event_data/kubric/examples/"

vis_dir="/mnt/nvme1n1/event_data/kubric/examples/vis2/"

exp_names=(
    etap_complex
    etap_simple
)

for exp in "${exp_names[@]}"; do

    # Visualize pngs to mp4
    # !! Remove fps option to use input video's fps !!
    # !! add --pair_vis for side-by-side view (Optional) !!
    # !! add --tracks_dir ${tracks_dir} for tracking vis !!
    python potato/scripts/visualizer.py \
        --data_dir ${data_root}${exp} \
        --output_dir "$vis_dir$exp" \
        --fps 5
        --pair_vis

done
