data_root="/mnt/data6/eunbeen/4d-recon/kubric/multiview/"

tracks_dir="/mnt/data6/eunbeen/4d-recon/kubric/multiview/tracks/"

exp_names=(
    "scene0"
)


for exp in "${exp_names[@]}"; do

    # Visualize pngs to mp4
    # !! Remove fps option to use input video's fps !!
    # !! add --pair_vis for side-by-side view (Optional) !!
    # !! add --tracks_dir ${tracks_dir} for tracking vis !!
    python potato/scripts/visualizer.py \
        --data_dir ${data_root}${exp} \
        --tracks_dir ${tracks_dir}${exp} \
        --output_dir "$vis_dir$exp" \
        --fps 5

done
