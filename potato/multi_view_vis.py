

import cv2
import os
import tqdm
import glob

########### Parameters ############
fps = 8
root_dir = f"/mnt/data6/eunbeen/4d-recon/kubric/scene0"
###################################


out_path = f"./video_vis/scene0/"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
cam_dirs = sorted(glob.glob(os.path.join(root_dir, "camera_*")))

for cam_dir in cam_dirs:
    # Get sorted list of frames
    frames = [f for f in os.listdir(cam_dir) if f.endswith(".png") and "rgba" in f]
    frames = sorted(frames)

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(cam_dir, frames[0]))
    height, width, channels = first_frame.shape

    # Define VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
    out = cv2.VideoWriter(os.path.join(out_path, f"cam_{cam_dir[-3:]}.mp4"), fourcc, fps, (width, height))

    # Write frames
    for f in tqdm.tqdm(frames):
        img_path = os.path.join(cam_dir, f)
        img = cv2.imread(img_path)
        out.write(img)

    out.release()
    print("Video saved to:", os.path.join(out_path, f"cam_{cam_dir[-3:]}.mp4"))

frames = [os.path.join(cam_dirs[i], f"rgba_{i:05d}.png") for i in range(min(len(cam_dirs), len(frames)))]
frames = sorted(frames)

# Read first frame to get dimensions
first_frame = cv2.imread(os.path.join(cam_dir, frames[0]))
height, width, channels = first_frame.shape

# Define VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
out = cv2.VideoWriter(os.path.join(out_path, f"global.mp4"), fourcc, fps, (width, height))

# Write frames
for f in tqdm.tqdm(frames):
    img_path = os.path.join(cam_dir, f)
    img = cv2.imread(img_path)
    out.write(img)

out.release()
print("Video saved to:", os.path.join(out_path, f"global.mp4"))