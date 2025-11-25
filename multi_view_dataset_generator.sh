docker run --rm --interactive \
  --user $(id -u):$(id -g)    \
  --volume "$(pwd):/kubric"   \
  --volume "/mnt/data6/eunbeen:/mnt/data6/eunbeen"   \
  kubricdockerhub/kubruntu    \
  /usr/bin/python3 multi_view_dataset_generator.py \
  --output_dir /mnt/data6/eunbeen/4d-recon/kubric/scene0 \
  --resolution 512 512 \
  --max_camera_movement 8.0 \
  --frame_rate 24 \
  --frame_start 0 \
  --frame_end 24