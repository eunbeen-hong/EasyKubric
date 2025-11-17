
data_root="/mnt/nvme1n1/event_data/kubric/240fps_no_blur"

vis_dir="./outputs/examples/240fps/"


# Visualize pngs to mp4
# !! Remove fps option to use input video's fps !!
# !! add --pair_vis for side-by-side view (Optional) !!
# !! add --tracks_dir ${tracks_dir} for tracking vis !!
python potato/scripts/visualizer.py \
    --data_dir ${data_root} \
    --output_dir "$vis_dir$exp" \
    --fps 24 \
    --tracks_dir ${data_root} \
