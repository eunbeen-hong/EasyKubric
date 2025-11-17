import numpy as np
import cv2
import glob
import os
import argparse
import json


def in_bounds(x, y, width, height):
    return 0 <= x < width and 0 <= y < height


def load_tracks(track_file):
    data = np.load(track_file, allow_pickle=True)
    return data['coords'], data['occluded']


def load_frames(frame_dir):
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "rgba_*.png")))
    frames = [cv2.imread(f) for f in frame_files]
    height, width, _ = frames[0].shape
    return frames, width, height


def get_fps(metadata_file, override_fps=None):
    fps = json.load(open(metadata_file))['flags']['frame_rate']
    return fps if override_fps is None else override_fps


def create_video_writer(output_file, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_file, fourcc, fps, (width, height))


def draw_tracks_on_frame(frame, tracks, occluded, colors, frame_idx, circle_radius):
    height, width, _ = frame.shape
    frame_copy = frame.copy()
    num_tracks = tracks.shape[0]

    for t in range(num_tracks):
        x, y = tracks[t, frame_idx, :2]

        # Scale normalized coordinates
        if np.max([x, y]) <= 1.0:
            x = int(x * width)
            y = int(y * height)
        else:
            x = int(x)
            y = int(y)

        if in_bounds(x, y, width, height):
            thickness = 1 if occluded[t, frame_idx] else -1
            cv2.circle(frame_copy, (x, y), circle_radius, colors[t], thickness)

        # Draw line from previous frame
        if frame_idx > 0:
            px, py = tracks[t, frame_idx - 1, :2]
            if np.max([px, py]) <= 1.0:
                px = int(px * width)
                py = int(py * height)
            else:
                px = int(px)
                py = int(py)

            if in_bounds(px, py, width, height) and in_bounds(x, y, width, height):
                cv2.line(frame_copy, (px, py), (x, y), colors[t], 1)

    return frame_copy


def process_scene(scene_id, tracks_dir, data_dir, output_dir, circle_radius, fps_override=None):
    print(f"[INFO] Processing scene: {scene_id}")
    
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    if tracks_dir == None:
        print("No tracks were given. Visualizing only RGB images.")
        frame_dir = os.path.join(data_dir, scene_id)
        frames, width, height = load_frames(frame_dir)

        fps = get_fps(os.path.join(frame_dir, "metadata.json"), fps_override)
        print(f"[INFO] Generating video at {fps} fps.")

        # Prepare output directories
        out_frames = create_video_writer(os.path.join(output_dir, f"{scene_id}_frames.mp4"), fps, width, height)

        # Process frames
        for f_idx, frame in enumerate(frames):
            out_frames.write(frame)
        out_frames.release()
        print(f"[INFO] Videos saved to {output_dir}")
        return

    # Load tracks and frames
    track_file = os.path.join(tracks_dir, scene_id, "tracks.npz")
    tracks, occluded = load_tracks(track_file)

    frame_dir = os.path.join(data_dir, scene_id)
    frames, width, height = load_frames(frame_dir)

    fps = get_fps(os.path.join(frame_dir, "metadata.json"), fps_override)
    print(f"[INFO] Generating video at {fps} fps.")

    # Prepare colors
    num_tracks = tracks.shape[0]
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_tracks)]

    # Prepare output directories
    out_tracks = create_video_writer(os.path.join(output_dir, f"{scene_id}_tracks.mp4"), fps, width, height)
    out_frames = create_video_writer(os.path.join(output_dir, f"{scene_id}_frames.mp4"), fps, width, height)

    # Process frames
    for f_idx, frame in enumerate(frames):
        out_frames.write(frame)
        frame_with_tracks = draw_tracks_on_frame(frame, tracks, occluded, colors, f_idx, circle_radius)
        out_tracks.write(frame_with_tracks)

    out_tracks.release()
    out_frames.release()
    print(f"[INFO] Videos saved to {output_dir}")

def process_side_by_side(scene_id1, scene_id2, tracks_dir, data_dir, output_dir, circle_radius, fps_override=None):
    
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    
    if tracks_dir == None:
        print("No tracks were given. Visualizing only RGB images.")
        frames1, width1, height1 = load_frames(os.path.join(data_dir, scene_id1))
        fps1 = get_fps(os.path.join(data_dir, scene_id1, "metadata.json"), fps_override)
        frames2, width2, height2 = load_frames(os.path.join(data_dir, scene_id2))
        fps2 = get_fps(os.path.join(data_dir, scene_id2, "metadata.json"), fps_override)
        fps = min(fps1, fps2)
        out_frames = create_video_writer(os.path.join(output_dir, f"{scene_id1}_{scene_id2}_frames.mp4"), fps, width1 + width2, max(height1, height2))

        num_frames = min(len(frames1), len(frames2))

        for f in range(num_frames):
            out_frames.write(np.hstack([frames1[f], frames2[f]]))
        out_frames.release()
        print(f"[INFO] Side-by-side video saved to {output_dir}")
        
        return
    
    # Load first scene
    track_file1 = os.path.join(tracks_dir, scene_id1, "tracks.npz")
    tracks1, occluded1 = load_tracks(track_file1)
    frames1, width1, height1 = load_frames(os.path.join(data_dir, scene_id1))
    fps1 = get_fps(os.path.join(data_dir, scene_id1, "metadata.json"), fps_override)

    # Load second scene
    track_file2 = os.path.join(tracks_dir, scene_id2, "tracks.npz")
    tracks2, occluded2 = load_tracks(track_file2)
    frames2, width2, height2 = load_frames(os.path.join(data_dir, scene_id2))
    fps2 = get_fps(os.path.join(data_dir, scene_id2, "metadata.json"), fps_override)

    fps = min(fps1, fps2)

    colors1 = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(tracks1.shape[0])]
    colors2 = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(tracks2.shape[0])]

    out_tracks = create_video_writer(os.path.join(output_dir, f"{scene_id1}_{scene_id2}_tracks.mp4"), fps, width1 + width2, max(height1, height2))
    out_frames = create_video_writer(os.path.join(output_dir, f"{scene_id1}_{scene_id2}_frames.mp4"), fps, width1 + width2, max(height1, height2))

    num_frames = min(len(frames1), len(frames2))

    for f in range(num_frames):
        out_frames.write(np.hstack([frames1[f], frames2[f]]))
        f1 = draw_tracks_on_frame(frames1[f], tracks1, occluded1, colors1, f, circle_radius)
        f2 = draw_tracks_on_frame(frames2[f], tracks2, occluded2, colors2, f, circle_radius)

        if height1 != height2:
            f2 = cv2.resize(f2, (width2, height1))

        side_by_side = np.hstack([f1, f2])
        out_tracks.write(side_by_side)

    out_tracks.release()
    out_frames.release()
    print(f"[INFO] Side-by-side video saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Kubric tracks and save videos")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing frame PNGs")
    parser.add_argument("--tracks_dir", type=str, default=None, help="Path to tracks .npz file")
    parser.add_argument("--output_dir", type=str, default="./outputs/vis/", help="Output MP4 directory")
    parser.add_argument("--circle_radius", type=int, default=1, help="Radius of the track points")
    parser.add_argument("--fps", type=int, default=None, help="FPS of output video")
    parser.add_argument("--pair_vis", action="store_true", help="Visualize videos side-by-side. Will pair directories based on lexicographical order.")
    args = parser.parse_args()

    scene_ids = sorted(os.listdir(args.data_dir))
    print(f"[INFO] Found {len(scene_ids)} scenes in {args.data_dir}")

    if args.pair_vis:
        scene_ids = sorted(os.listdir(args.data_dir))
        
        # assert len(scene_ids) % 2 == 0, "For paired visualizations, input scenes should be even number."
        
        for i in range(len(scene_ids) // 2):
            process_side_by_side(
                scene_ids[2 * i], scene_ids[2 * i + 1],
                args.tracks_dir,
                args.data_dir,
                args.output_dir,
                args.circle_radius,
                args.fps
            )
    else:
        scene_ids = sorted(os.listdir(args.data_dir))
        for scene_id in scene_ids:
            process_scene(
                scene_id,
                args.tracks_dir,
                args.data_dir,
                args.output_dir,
                args.circle_radius,
                args.fps
            )


if __name__ == "__main__":
    main()
