from __future__ import annotations
import os
import warnings

# turn off warning. too noisy (imageio, tensorflow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
from tqdm import tqdm
from extractor_utils import add_tracks
from tqdm import tqdm
from PIL import Image

import ast
import json
import glob
import cv2 
import time
import imageio.v2 as imageio
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import argparse
import matplotlib.colors as mcolors 
import multiprocessing as mp

def read_rgb_stack(view_dir, rgba_key="rgba"):
    pngs = sorted([p for p in Path(view_dir).iterdir() if rgba_key in p.name])
    frames = [imageio.imread(p) for p in pngs]
    return np.stack(frames, axis=0)  # (F,H,W,4)

# cv2 is faster!
def read_png_cv2(path: str | Path, rng=None):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:  # BGR(A) -> RGB(A)
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA
                          if im.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        
    bit16 = im.dtype == np.uint16
    im = im.astype(np.float32)
    
    if rng is not None:
        im /= 65535.0 if bit16 else 255.0
        im = im * (rng[1] - rng[0]) + rng[0]
        
    return im

# load attributes from metadata json file
def _extract_instances(meta_instances: list[dict]):
    inst_attr = {}
    if meta_instances:
        for inst in meta_instances:
            for k in inst.keys():
                if k not in inst_attr:
                    inst_attr[k] = []
                inst_attr[k].append(inst[k])

    for k in inst_attr.keys():
        # actually, these are not used in postprocessing
        if k in ["asset_id", "category", "description"]:
            inst_attr[k] = np.asarray(inst_attr[k], str)
        elif k in ["bbox_frames", "bboxes"]:
            inst_attr[k] = inst_attr[k]
        else:
            inst_attr[k] = np.asarray(inst_attr[k], np.float32)

    keys = list(inst_attr.keys())
    tmp = inst_attr[keys[0]]
    
    # sanity check: all attributes should have the same length
    for k in keys[1:]:
        assert len(tmp) == len(inst_attr[k]), \
            f"Length mismatch for {k}: {len(tmp)} vs {len(inst_attr[k])}"

    return inst_attr

def load_kubric_sequence(seq_dir: str | Path, as_tf=True):
    p = Path(seq_dir)
    meta = json.loads((p / "metadata.json").read_text())

    # file lists
    rgba_files   = sorted(p.glob("rgba_*.png"))
    depth_files  = sorted(p.glob("depth_*.tiff"))
    normal_files = sorted(p.glob("normal_*.png"))
    objc_files   = sorted(p.glob("object_coordinates_*.png"))
    segm_files   = sorted(p.glob("segmentation_*.png"))
    bcwd_files   = sorted(p.glob("backward_flow_*.png"))
    fwd_files    = sorted(p.glob("forward_flow_*.png"))

    rgba   = np.stack([read_png_cv2(f) for f in rgba_files]).astype(np.uint8)
    depth  = np.stack([read_png_cv2(f) for f in depth_files])[..., None]
    normal = np.stack([read_png_cv2(f) for f in normal_files]).astype(np.uint16)
    objc   = np.stack([read_png_cv2(f) for f in objc_files]).astype(np.uint16)
    
    if len(bcwd_files) > 0:
        bcwd = np.stack([read_png_cv2(f) for f in bcwd_files])[..., :2].astype(np.uint16)
    else:
        bcwd = np.zeros_like(rgba[..., :2])
    if len(fwd_files) > 0:
        fwd = np.stack([read_png_cv2(f) for f in fwd_files])[..., :2].astype(np.uint16)
    else:
        fwd = np.zeros_like(rgba[..., :2])

    # compress segmentation
    # e.g. 0, 10, 20 ... -> 0, 1, 2 ...
    # !!!!!! kubric renderer code maps the index of objects in rainbow order !!!!!!!!
    # if the order of objects are not matched to metadata file, postprocessing will fail.
    if segm_files:
        segm_raw = np.stack([read_png_cv2(f) for f in segm_files])
        # rgb to grayscale 
        # segm_raw = segm_raw[:, :, :, 0:1] + segm_raw[:, :, :, 1:2] * 1000 + segm_raw[:, :, :, 2:3] * 1000000 
        rgb = segm_raw.astype(np.float32) / 255.0          # 0~1 normalized RGB
        hsv = mcolors.rgb_to_hsv(rgb)                      

        H = (hsv[..., 0] * 360).astype(np.uint32)          # 0~359
        S = (hsv[..., 1] * 100).astype(np.uint32)          # 0~100
        V = (hsv[..., 2] * 100).astype(np.uint32)          # 0~100

        #   (Hue)   * 10000  +  (Sat) * 100  + (Val) * 1
        segm_id = H * 10000 + S * 100 + V                  
        segm_id = segm_id[:, :, :, None]

        uniq = np.sort(np.unique(segm_id))                 
        id_mapper = {id_: i for i, id_ in enumerate(uniq)} 

        segm = np.zeros_like(segm_id, dtype=np.uint8)
        
        for id_, i in id_mapper.items():
            if id_ == 0:  # background
                continue
            segm[segm_id == id_] = i
        if len(id_mapper) == 0:
            raise ValueError("No valid segmentations found in the sequence.")
    else:
        segm = (rgba[..., 3:4] > 0).astype(np.uint8)
        
    # normalize depth to 0 to 65535, uint16
    # !!!! preserve the range for later use !!!!!
    depth_range = (depth.min(), depth.max())
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 65535).astype(np.uint16)  - 1

    # ---------- instances (bbox, quat) ---------- #
    meta_instances = meta.get("instances", [])
    assert meta_instances, "No instances found in the metadata."
    
    inst_attributes = _extract_instances(meta_instances)

    # ---------- camera dict ---------- #
    cam = meta["camera"]
    cam_dict = dict(
        focal_length  = np.asarray(cam["focal_length"],   np.float32),
        positions     = np.asarray(cam["positions"],      np.float32),
        quaternions   = np.asarray(cam["quaternions"],    np.float32),
        sensor_width  = np.asarray(cam["sensor_width"],   np.float32),
        field_of_view = np.asarray(cam["field_of_view"], np.float32),
    )
    
    # range meta
    meta = json.loads((p / "data_ranges.json").read_text())
    fwd_rng   = meta.get("forward_flow", None)
    bcwd_rng  = meta.get("backward_flow", None)

    data = dict(
        video              = rgba[..., :3],
        depth              = depth,
        normal             = normal,
        object_coordinates = objc,
        segmentations      = segm,
        metadata           = dict(depth_range=depth_range, 
                                  forward_flow_range=(fwd_rng['min'], fwd_rng['max']),
                                  backward_flow_range=(bcwd_rng['min'], bcwd_rng['max'])
                                 ),
        camera             = cam_dict,
        instances          = inst_attributes,
        backward_flow      = bcwd,
        forward_flow       = fwd,
    )

    if as_tf:
        data = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x) if isinstance(x, np.ndarray) else x,
            data,
        )
    return data

def process_scene(scene_id, args, views):
    s = time.time()
    for view in views:
        seq_dir = os.path.join(args.data_dir, scene_id)
        if not os.path.isdir(seq_dir):
            print(f"[WARNING] {seq_dir} is not a valid directory. Skipping.")

        data = load_kubric_sequence(seq_dir, as_tf=True)
        image_res = data['video'].shape[1:3]
        
        args.tracks_to_sample = image_res[0] * image_res[1]
        results = add_tracks(
            data=data,
            train_size=image_res,
            vflip=False,
            random_crop=False,
            tracks_to_sample=args.tracks_to_sample,
            sampling_stride=1,
            max_seg_id=25,
            max_sampled_frac=1.0,
            snap_to_occluder=False,
            grid_frames=(0, 4, 8, 12, 16, 20),
            grid_size=args.grid_size
        )

        out_dir = os.path.join(args.out_dir, scene_id)
        os.makedirs(out_dir, exist_ok=True)

        # print(f"[INFO] {results['query_points'].shape[0]} points extracted from {scene_id}-{view}")
        
        save_dict = {
            'coords':           results['target_points'].numpy(),
            'abs_depth':        results['absolute_depth'].numpy(),
            'occluded':         results['occluded'].numpy()
            # 'query_points':   results['query_points'].numpy(),
            # 'relative_depth': results['relative_depth'].numpy(),
            # 'normals':        results['normals'].numpy(),
        }
        if args.camera:
            save_dict.update({
                'intrinsics':  results['intrinsics'].numpy(),
                'cam_rot':     results['cam_rot'].numpy(),
                'cam_trans':   results['cam_trans'].numpy(),
                # 'world_coords':results['world_coords'].numpy(),
            })
            
        if args.bkgd_flag:
            save_dict.update({
                'is_bkgd':   results['is_bkgd'].numpy()[:, None],
            })
            
        if args.add_video:
            rgb_stack = read_rgb_stack(seq_dir)
            assert len(rgb_stack.shape) == 4
            
            print(f"[INFO] Loaded video frames: {rgb_stack.shape}")
            save_dict.update({
                'video':     rgb_stack[..., :3]  # (F,H,W,3)
            })
            
        # for k, v in save_dict.items():
        #     print(f"  {k}: {v.shape}, {v.dtype}, min: {v.min()}, max: {v.max()}")
        
        # np.save(os.path.join(out_dir, 'tracks.npy'), save_dict)
        np.savez_compressed(os.path.join(out_dir, 'tracks.npz'), **save_dict)
        
    print(f"[INFO] Processed {scene_id} in {time.time() - s:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tracks from Kubric sequence")
    parser.add_argument("--data_dir", type=str, help="Path to the Kubric sequence directory")
    parser.add_argument("--out_dir", type=str, help="Path to save the extracted tracks")
    parser.add_argument("--grid_size", type=int, default=24, help="Grid size for sampling tracks")
    parser.add_argument("--track_to_sample", type=int, default=4096, help="Number of tracks to sample per sequence")
    parser.add_argument("--tracks_to_sample", type=int, default=2048, help="Number of tracks to sample")
    parser.add_argument("--n_procs", type=int, default=1)
    # parser.add_argument("--target_views", type=str, default='["view_0001"]', help="List of views to process, e.g. '[view_0001, view_0002]'")
    parser.add_argument("--target_views", type=str, nargs="*", default=["view_0001"], help="List of views to process, e.g. '[view_0001, view_0002]'")
    parser.add_argument("--camera", action='store_true', help="Whether to include camera parameters in the output")
    parser.add_argument("--bkgd_flag", action='store_true', help="Whether to include is_bkgd flag in the output")
    parser.add_argument("--add_video", action='store_true', help="Whether to include video in the output")
    parser.add_argument("--all_pixels", action='store_true', help="Extract tracks from all pixels.")
    
    args = parser.parse_args()
    
    scene_ids = os.listdir(args.data_dir)
    # scene_ids.sort(key=lambda x: int(x))
    scene_ids.sort()
    
    print(f"[INFO] Found {len(scene_ids)} scenes in {args.data_dir}")
    
    # views = ast.literal_eval(str(args.target_views).strip())
    views = args.target_views
    assert isinstance(views, list), "target_views should be a list of view names"
    assert all(isinstance(v, str) for v in views), "All elements in target_views should be strings"
    print(f"[INFO] Target views: {views}")
    
    # views = ['view_0001']
    # views = os.listdir(os.path.join(args.data_dir, scene_ids[0]))
    # views.sort()
    
    print(f"[INFO] Found {len(views)} views: {views}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.n_procs > 1:
        print(f"[INFO] Using multiprocessing with {args.n_procs} workers.")
        with mp.Pool(processes=args.n_procs) as pool:
            list(tqdm(
                pool.starmap(process_scene,
                             [(scene_id, args, views) for scene_id in scene_ids]),
                total=len(scene_ids),
                desc="Processing scenes (parallel)"
            ))
    else:
        print("[WARNING] Running in single‑process mode. It can be extremely slow!")
        for scene_id in tqdm(scene_ids, desc="Processing scenes"):
            process_scene(scene_id, args, views)
            