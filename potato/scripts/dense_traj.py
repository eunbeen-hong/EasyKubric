
import functools
import itertools

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d


def project_point(cam, point3d, num_frames):
  """Compute the image space coordinates [0, 1] for a set of points.

  Args:
    cam: The camera parameters, as returned by kubric.  'matrix_world' and
      'intrinsics' have a leading axis num_frames.
    point3d: Points in 3D world coordinates.  it has shape [num_frames,
      num_points, 3].
    num_frames: The number of frames in the video.

  Returns:
    Image coordinates in 2D.  The last coordinate is an indicator of whether
      the point is behind the camera.
  """

  homo_transform = tf.linalg.inv(cam['matrix_world'])
  homo_intrinsics = tf.zeros((num_frames, 3, 1), dtype=tf.float32)
  homo_intrinsics = tf.concat([cam['intrinsics'], homo_intrinsics], axis=2)

  point4d = tf.concat([point3d, tf.ones_like(point3d[:, :, 0:1])], axis=2)
  projected = tf.matmul(point4d, tf.transpose(homo_transform, (0, 2, 1)))
  projected = tf.matmul(projected, tf.transpose(homo_intrinsics, (0, 2, 1)))
  image_coords = projected / projected[:, :, 2:3]
  image_coords = tf.concat(
      [image_coords[:, :, :2],
       tf.sign(projected[:, :, 2:])], axis=2)
  return image_coords


def unproject(coord, cam, depth):
  """Unproject points.

  Args:
    coord: Points in 2D coordinates.  it has shape [num_points, 2].  Coord is in
      integer (y,x) because of the way meshgrid happens.
    cam: The camera parameters, as returned by kubric.  'matrix_world' and
      'intrinsics' have a leading axis num_frames.
    depth: Depth map for the scene.

  Returns:
    Image coordinates in 3D.
  """
  shp = tf.convert_to_tensor(tf.shape(depth))
  idx = coord[:, 0] * shp[1] + coord[:, 1]
  coord = tf.cast(coord[..., ::-1], tf.float32)
  shp = tf.cast(shp[1::-1], tf.float32)[tf.newaxis, ...]

  # Need to convert from pixel to raster coordinate.
  projected_pt = (coord + 0.5) / shp

  projected_pt = tf.concat(
      [
          projected_pt,
          tf.ones_like(projected_pt[:, -1:]),
      ],
      axis=-1,
  )

  camera_plane = projected_pt @ tf.linalg.inv(tf.transpose(cam['intrinsics']))
  camera_ball = camera_plane / tf.sqrt(
      tf.reduce_sum(
          tf.square(camera_plane),
          axis=1,
          keepdims=True,
      ),)
  camera_ball *= tf.gather(tf.reshape(depth, [-1]), idx)[:, tf.newaxis]

  camera_ball = tf.concat(
      [
          camera_ball,
          tf.ones_like(camera_plane[:, 2:]),
      ],
      axis=1,
  )
  points_3d = camera_ball @ tf.transpose(cam['matrix_world'])
  return points_3d[:, :3] / points_3d[:, 3:]


def reproject(coords, camera, camera_pos, num_frames, bbox=None):
  """Reconstruct points in 3D and reproject them to pixels.

  Args:
    coords: Points in 3D.  It has shape [num_points, 3].  If bbox is specified,
      these are assumed to be in local box coordinates (as specified by kubric),
      and bbox will be used to put them into world coordinates; otherwise they
      are assumed to be in world coordinates.
    camera: the camera intrinsic parameters, as returned by kubric.
      'matrix_world' and 'intrinsics' have a leading axis num_frames.
    camera_pos: the camera positions.  It has shape [num_frames, 3]
    num_frames: the number of frames in the video.
    bbox: The kubric bounding box for the object.  Its first axis is num_frames.

  Returns:
    Image coordinates in 2D and their respective depths.  For the points,
    the last coordinate is an indicator of whether the point is behind the
    camera.  They are of shape [num_points, num_frames, 3] and
    [num_points, num_frames] respectively.
  """
  # First, reconstruct points in the local object coordinate system.
  if bbox is not None:
    coord_box = list(itertools.product([-.5, .5], [-.5, .5], [-.5, .5]))
    coord_box = np.array([np.array(x) for x in coord_box])
    coord_box = np.concatenate(
        [coord_box, np.ones_like(coord_box[:, 0:1])], axis=1)
    coord_box = tf.tile(coord_box[tf.newaxis, ...], [num_frames, 1, 1])
    bbox_homo = tf.concat([bbox, tf.ones_like(bbox[:, :, 0:1])], axis=2)

    local_to_world = tf.linalg.lstsq(tf.cast(coord_box, tf.float32), bbox_homo)
    world_coords = tf.matmul(
        tf.cast(
            tf.concat([coords, tf.ones_like(coords[:, 0:1])], axis=1),
            tf.float32)[tf.newaxis, :, :], local_to_world)
    world_coords = world_coords[:, :, 0:3] / world_coords[:, :, 3:]
  else:
    world_coords = tf.tile(coords[tf.newaxis, :, :], [num_frames, 1, 1])

  # Compute depths by taking the distance between the points and the camera
  # center.
  depths = tf.sqrt(
      tf.reduce_sum(
          tf.square(world_coords - camera_pos[:, np.newaxis, :]),
          axis=2,
      ),)

  # Project each point back to the image using the camera.
  projections = project_point(camera, world_coords, num_frames)

  return (
      tf.transpose(projections, (1, 0, 2)),
      tf.transpose(depths),
      tf.transpose(world_coords, (1, 0, 2)),
  )


def estimate_occlusion_by_depth_and_segment(
    data,
    segments,
    x,
    y,
    num_frames,
    thresh,
    seg_id,
):
  """Estimate depth at a (floating point) x,y position.

  We prefer overestimating depth at the point, so we take the max over the 4
  neightoring pixels.

  Args:
    data: depth map. First axis is num_frames.
    segments: segmentation map. First axis is num_frames.
    x: x coordinate. First axis is num_frames.
    y: y coordinate. First axis is num_frames.
    num_frames: number of frames.
    thresh: Depth threshold at which we consider the point occluded.
    seg_id: Original segment id.  Assume occlusion if there's a mismatch.

  Returns:
    Depth for each point.
  """

  # need to convert from raster to pixel coordinates
  x = x - 0.5
  y = y - 0.5

  x0 = tf.cast(tf.floor(x), tf.int32)
  x1 = x0 + 1
  y0 = tf.cast(tf.floor(y), tf.int32)
  y1 = y0 + 1

  shp = tf.shape(data)
  assert len(data.shape) == 3
  x0 = tf.clip_by_value(x0, 0, shp[2] - 1)
  x1 = tf.clip_by_value(x1, 0, shp[2] - 1)
  y0 = tf.clip_by_value(y0, 0, shp[1] - 1)
  y1 = tf.clip_by_value(y1, 0, shp[1] - 1)

  data = tf.reshape(data, [-1])
  rng = tf.range(num_frames)[:, tf.newaxis]
  i1 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
  i2 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
  i3 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
  i4 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

  depth = tf.maximum(tf.maximum(tf.maximum(i1, i2), i3), i4)

  segments = tf.reshape(segments, [-1])
  i1 = tf.gather(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
  i2 = tf.gather(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
  i3 = tf.gather(segments, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
  i4 = tf.gather(segments, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

  depth_occluded = tf.less(tf.transpose(depth), thresh)
  seg_occluded = True
  for i in [i1, i2, i3, i4]:
    i = tf.cast(i, tf.int32)
    seg_occluded = tf.logical_and(seg_occluded, tf.not_equal(seg_id, i))

  return tf.logical_or(depth_occluded, tf.transpose(seg_occluded))


def get_camera_matrices(
    cam_focal_length,
    cam_positions,
    cam_quaternions,
    cam_sensor_width,
    input_size,
    num_frames=None,
):
  """Tf function that converts camera positions into projection matrices."""
  intrinsics = []
  matrix_world = []
  assert cam_quaternions.shape[0] == num_frames
  for frame_idx in range(cam_quaternions.shape[0]):
    focal_length = tf.cast(cam_focal_length, tf.float32)
    sensor_width = tf.cast(cam_sensor_width, tf.float32)
    f_x = focal_length / sensor_width
    f_y = focal_length / sensor_width * input_size[0] / input_size[1]
    p_x = 0.5
    p_y = 0.5
    intrinsics.append(
        tf.stack([
            tf.stack([f_x, 0., -p_x]),
            tf.stack([0., -f_y, -p_y]),
            tf.stack([0., 0., -1.]),
        ]))

    position = cam_positions[frame_idx]
    quat = cam_quaternions[frame_idx]
    rotation_matrix = rotation_matrix_3d.from_quaternion(
        tf.concat([quat[1:], quat[0:1]], axis=0))
    transformation = tf.concat(
        [rotation_matrix, position[:, tf.newaxis]],
        axis=1,
    )
    transformation = tf.concat(
        [transformation,
         tf.constant([0.0, 0.0, 0.0, 1.0])[tf.newaxis, :]],
        axis=0,
    )
    matrix_world.append(transformation)

  return (
      tf.cast(tf.stack(intrinsics), tf.float32),
      tf.cast(tf.stack(matrix_world), tf.float32),
  )


def quat2rot(quats):
  """Convert a list of quaternions to rotation matrices."""
  rotation_matrices = []
  for frame_idx in range(quats.shape[0]):
    quat = quats[frame_idx]
    rotation_matrix = rotation_matrix_3d.from_quaternion(
        tf.concat([quat[1:], quat[0:1]], axis=0))
    rotation_matrices.append(rotation_matrix)
  return tf.cast(tf.stack(rotation_matrices), tf.float32)


def rotate_surface_normals(
    world_frame_normals,
    point_3d,
    cam_pos,
    obj_rot_mats,
    frame_for_query,
):
  """Points are occluded if the surface normal points away from the camera."""
  query_obj_rot_mat = tf.gather(obj_rot_mats, frame_for_query)
  obj_frame_normals = tf.einsum(
      'boi,bi->bo',
      tf.linalg.inv(query_obj_rot_mat),
      world_frame_normals,
  )
  world_frame_normals_frames = tf.einsum(
      'foi,bi->bfo',
      obj_rot_mats,
      obj_frame_normals,
  )
  cam_to_pt = point_3d - cam_pos[tf.newaxis, :, :]
  dots = tf.reduce_sum(world_frame_normals_frames * cam_to_pt, axis=-1)
  faces_away = dots > 0

  # If the query point also faces away, it's probably a bug in the meshes, so
  # ignore the result of the test.
  faces_away_query = tf.reduce_sum(
      tf.cast(faces_away, tf.int32)
      * tf.one_hot(frame_for_query, tf.shape(faces_away)[1], dtype=tf.int32),
      axis=1,
      keepdims=True,
  )
  faces_away = tf.logical_and(faces_away, tf.logical_not(faces_away_query > 0))
  return faces_away


def single_object_reproject(
    bbox_3d=None,
    pt=None,
    pt_segments=None,
    camera=None,
    cam_positions=None,
    num_frames=None,
    depth_map=None,
    segments=None,
    window=None,
    input_size=None,
    quat=None,
    normals=None,
    frame_for_pt=None,
    trust_normals=None,
):
  """Reproject points for a single object.

  Args:
    bbox_3d: The object bounding box from Kubric.  If none, assume it's
      background.
    pt: The set of points in 3D, with shape [num_points, 3]
    pt_segments: The segment each point came from, with shape [num_points]
    camera: Camera intrinsic parameters
    cam_positions: Camera positions, with shape [num_frames, 3]
    num_frames: Number of frames
    depth_map: Depth map video for the camera
    segments: Segmentation map video for the camera
    window: the window inside which we're sampling points
    input_size: [height, width] of the input images.
    quat: Object quaternion [num_frames, 4]
    normals: Point normals on the query frame [num_points, 3]
    frame_for_pt: Integer frame where the query point came from [num_points]
    trust_normals: Boolean flag for whether the surface normals for each query
      are trustworthy [num_points]

  Returns:
    Position for each point, of shape [num_points, num_frames, 2], in pixel
    coordinates, and an occlusion flag for each point, of shape
    [num_points, num_frames].  These are respect to the image frame, not the
    window.

  """
  # Finally, reproject
  reproj, depth_proj, world_pos = reproject(
      pt,
      camera,
      cam_positions,
      num_frames,
      bbox=bbox_3d,
  )

  occluded = tf.less(reproj[:, :, 2], 0)
  reproj = reproj[:, :, 0:2] * np.array(input_size[::-1])[np.newaxis,
                                                          np.newaxis, :]
  occluded = tf.logical_or(
      occluded,
      estimate_occlusion_by_depth_and_segment(
          depth_map[:, :, :, 0],
          segments[:, :, :, 0],
          tf.transpose(reproj[:, :, 0]),
          tf.transpose(reproj[:, :, 1]),
          num_frames,
          depth_proj * .99,
          pt_segments,
      ),
  )
  obj_occ = occluded
  obj_reproj = reproj

  obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 1], window[0]))
  obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 0], window[1]))
  obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 1], window[2]))
  obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 0], window[3]))

  if quat is not None:
    faces_away = rotate_surface_normals(
        normals,
        world_pos,
        cam_positions,
        quat2rot(quat),
        frame_for_pt,
    )
    faces_away = tf.logical_and(faces_away, trust_normals)
  else:
    # world is convex; can't face away from cam.
    faces_away = tf.zeros([tf.shape(pt)[0], num_frames], dtype=tf.bool)

  return obj_reproj, tf.logical_or(faces_away, obj_occ), depth_proj


def track_points_first_frame_dense(
    object_coordinates,
    depth,
    depth_range,
    segmentations,
    surface_normals,
    bboxes_3d,
    obj_quat,
    cam_focal_length,
    cam_positions,
    cam_quaternions,
    cam_sensor_width,
    window,
    snap_to_occluder=False,
):
    """
    Dense tracking: every pixel in the first frame is a query.
    """

    # ------------------------------------------------------------
    # Basic shapes
    # ------------------------------------------------------------
    num_frames = object_coordinates.shape.as_list()[0]
    H, W = object_coordinates.shape.as_list()[1:3]
    window = tf.cast(window, tf.int32)

    # ------------------------------------------------------------
    # Depth to metric
    # ------------------------------------------------------------
    depth_range_f32 = tf.cast(depth_range, tf.float32)
    depth_min, depth_max = depth_range_f32[0], depth_range_f32[1]
    depth_map = (
        depth_min
        + tf.cast(depth, tf.float32) * (depth_max - depth_min) / 65535.0
    )

    surface_normal_map = surface_normals / 65535.0 * 2.0 - 1.0

    # ------------------------------------------------------------
    # Camera matrices
    # ------------------------------------------------------------
    intrinsics, matrix_world = get_camera_matrices(
        cam_focal_length,
        cam_positions,
        cam_quaternions,
        cam_sensor_width,
        [H, W],
        num_frames=num_frames,
    )

    def get_camera(fr):
        return {
            "intrinsics": intrinsics[fr],
            "matrix_world": matrix_world[fr],
        }

    # ------------------------------------------------------------
    # Construct dense queries at t = 0
    # ------------------------------------------------------------
    y, x = tf.meshgrid(
        tf.range(window[1], window[3]),
        tf.range(window[0], window[2]),
        indexing="ij",
    )
    num_points = tf.size(y)

    t = tf.zeros_like(y)
    chosen_points = tf.reshape(
        tf.stack([t, y, x], axis=-1), [-1, 3]
    )  # [N, 3]

    # ------------------------------------------------------------
    # Flatten per-pixel data at frame 0
    # ------------------------------------------------------------
    seg0 = tf.reshape(
        segmentations[0, window[1]:window[3], window[0]:window[2]],
        [-1],
    )

    objcoord0 = tf.reshape(
        object_coordinates[0, window[1]:window[3], window[0]:window[2]],
        [-1, 3],
    )

    normals0 = tf.reshape(
        surface_normal_map[0, window[1]:window[3], window[0]:window[2]],
        [-1, 3],
    )

    depth0 = tf.reshape(
        depth_map[0, window[1]:window[3], window[0]:window[2]],
        [-1],
    )

    # ------------------------------------------------------------
    # Output buffers
    # ------------------------------------------------------------
    all_reproj = []
    all_occ = []
    all_reproj_depth = []
    chosen_points_depth = []

    # ------------------------------------------------------------
    # Process per object id
    # ------------------------------------------------------------
    max_seg_id = tf.reduce_max(seg0) + 1

    for seg_id in tf.range(max_seg_id):
        mask = tf.equal(seg0, seg_id)
        idx = tf.where(mask)[:, 0]

        if tf.size(idx) == 0:
            continue

        pt_coords = tf.gather(chosen_points, idx)
        pt_depth = tf.gather(depth0, idx)
        chosen_points_depth.append(pt_depth)

        obj_id = seg_id - 1

        # --------------------------------------------------------
        # Background
        # --------------------------------------------------------
        if obj_id == -1:
            pt_3d = unproject(
                pt_coords[:, 1:],
                get_camera(0),
                depth_map[0],
            )
            bbox = None
            quat = None
            frame_for_pt = None

        # --------------------------------------------------------
        # Foreground object
        # --------------------------------------------------------
        else:
            pt_3d = tf.gather(objcoord0, idx)
            pt_3d = pt_3d / tf.cast(tf.reduce_max(pt_3d), tf.float32) - 0.5

            bbox = bboxes_3d[obj_id]
            quat = obj_quat[obj_id]
            frame_for_pt = tf.zeros([tf.shape(pt_3d)[0]], tf.int32)

        # --------------------------------------------------------
        # Reprojection
        # --------------------------------------------------------
        reproj, occ, reproj_depth = single_object_reproject(
            bbox_3d=bbox,
            pt=pt_3d,
            pt_segments=seg_id,
            camera={"intrinsics": intrinsics, "matrix_world": matrix_world},
            cam_positions=cam_positions,
            num_frames=num_frames,
            depth_map=depth_map,
            segments=segmentations,
            window=tf.cast(window, tf.float32),
            input_size=[H, W],
            quat=quat,
            normals=tf.gather(normals0, idx),
            frame_for_pt=frame_for_pt,
            trust_normals=tf.ones([tf.shape(idx)[0], 1], tf.bool),
        )

        all_reproj.append(reproj)
        all_occ.append(occ)
        all_reproj_depth.append(reproj_depth)

    # ------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------
    all_reproj = tf.concat(all_reproj, axis=0)
    all_occ = tf.concat(all_occ, axis=0)
    all_reproj_depth = tf.concat(all_reproj_depth, axis=0)
    chosen_points_depth = tf.concat(chosen_points_depth, axis=0)

    # Pixel center convention
    chosen_points = tf.cast(chosen_points, tf.float32)
    chosen_points += tf.constant([0.0, 0.5, 0.5])[None]

    all_relative_depth = all_reproj_depth / chosen_points_depth[:, None]

    return chosen_points, all_reproj, all_occ, all_relative_depth



def add_tracks(data,
               train_size=(256, 256),
               snap_to_occluder=False):
  """Track points in 2D using Kubric data.

  Args:
    data: Kubric data, including RGB/depth/object coordinate/segmentation
      videos and camera parameters.
    train_size: Cropped output will be at this resolution.  Ignored if
      random_crop is False.
    snap_to_occluder: If true, query points within 1 pixel of occlusion 
      boundaries will track the occluding surface rather than the background.
      This results in models which are biased to track foreground objects
      instead of background.  Whether this is desirable depends on downstream
      applications.

  Returns:
    A dict with the following keys:
    query_points:
      A set of queries, randomly sampled from the video (with a bias toward
      objects), of shape [num_points, 3].  Each point is [t, y, x], where
      t is time.  Points are in pixel/frame coordinates.
      [num_frames, height, width].
    target_points:
      The trajectory for each query point, of shape [num_points, num_frames, 3].
      Each point is [x, y].  Points are in pixel/frame coordinates.
    occlusion:
      Occlusion flag for each point, of shape [num_points, num_frames].  This is
      a boolean, where True means the point is occluded.
    video:
      The cropped video, normalized into the range [-1, 1]

  """
  shp = data['video'].shape.as_list()
  num_frames = shp[0]

  crop_window = tf.constant([0, 0, shp[1], shp[2]],
                            dtype=tf.int32,
                            shape=[4])


  query_points, target_points, occluded, relative_depth = track_points_first_frame_dense(
      data['object_coordinates'], data['depth'],
      data['metadata']['depth_range'], data['segmentations'],
      data['normal'],
      data['instances']['bboxes_3d'], data['instances']['quaternions'],
      data['camera']['focal_length'],
      data['camera']['positions'], data['camera']['quaternions'],
      data['camera']['sensor_width'], crop_window, snap_to_occluder)
  video = data['video']

  tracks_to_sample = train_size[0] * train_size[1]

  shp = video.shape.as_list()
  query_points.set_shape([tracks_to_sample, 3])
  target_points.set_shape([tracks_to_sample, num_frames, 2])
  relative_depth.set_shape([tracks_to_sample, num_frames])
  occluded.set_shape([tracks_to_sample, num_frames])

  # Crop the video to the sampled window, in a way which matches the coordinate
  # frame produced the track_points functions.
  start = tf.tensor_scatter_nd_update(
      [0, 0, 0, 0], [[1], [2]], crop_window[0:2]
  )
  size = tf.tensor_scatter_nd_update(
      tf.shape(video), [[1], [2]], crop_window[2:4] - crop_window[0:2]
  )
  video = tf.slice(video, start, size)
  video = tf.image.resize(tf.cast(video, tf.float32), train_size)
  video.set_shape([num_frames, train_size[0], train_size[1], 3])
  
  res = {
      'query_points': query_points,
      'target_points': target_points,
      'relative_depth': relative_depth,
      'occluded': occluded,
      'video': video / (255. / 2.) - 1.,
  }
  return res

import os
import json
import glob
import imageio.v2 as imageio
import numpy as np
import tensorflow as tf


def _sorted_glob(pattern):
    return sorted(glob.glob(pattern))


def load_scene_from_dir(scene_dir, camera_id="camera_000"):
    cam_dir = os.path.join(scene_dir, camera_id)

    # --------------------------------------------------
    # Load metadata
    # --------------------------------------------------
    with open(os.path.join(cam_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # --------------------------------------------------
    # Load per-frame data
    # --------------------------------------------------
    rgba_files = _sorted_glob(os.path.join(cam_dir, "rgba_*.png"))
    depth_files = _sorted_glob(os.path.join(cam_dir, "depth_*.tiff"))
    normal_files = _sorted_glob(os.path.join(cam_dir, "normal_*.png"))
    objcoord_files = _sorted_glob(os.path.join(cam_dir, "object_coordinates_*.png"))
    seg_files = _sorted_glob(os.path.join(cam_dir, "segmentation_*.png"))

    assert len(rgba_files) == len(depth_files)
    num_frames = len(rgba_files)

    # --------------------------------------------------
    # Video (RGB only)
    # --------------------------------------------------
    video = []
    for f in rgba_files:
        img = imageio.imread(f)[..., :3]  # drop alpha
        video.append(img)
    video = np.stack(video, axis=0)  # [T, H, W, 3]

    # --------------------------------------------------
    # Depth (metric float32)
    # --------------------------------------------------
    depth = []
    for f in depth_files:
        d = imageio.imread(f).astype(np.float32)
        if d.ndim == 2:
            d = d[..., None]
        depth.append(d)
    depth = np.stack(depth, axis=0)  # [T, H, W, 1]

    # --------------------------------------------------
    # Normals (uint16 → [-1,1])
    # --------------------------------------------------
    normal = []
    for f in normal_files:
        n = imageio.imread(f).astype(np.float32)
        normal.append(n)
    normal = np.stack(normal, axis=0)

    # --------------------------------------------------
    # Object coordinates
    # --------------------------------------------------
    object_coordinates = []
    for f in objcoord_files:
        oc = imageio.imread(f).astype(np.float32)
        object_coordinates.append(oc)
    object_coordinates = np.stack(object_coordinates, axis=0)

    # --------------------------------------------------
    # Segmentation
    # --------------------------------------------------
    segmentations = []
    for f in seg_files:
        seg = imageio.imread(f).astype(np.int32)
        segmentations.append(seg)
    segmentations = np.stack(segmentations, axis=0)

    # --------------------------------------------------
    # Camera info
    # --------------------------------------------------
    camera = metadata["camera"]
    instances = metadata["instances"]

    data = {
        "video": tf.convert_to_tensor(video, tf.uint8), # F, 512, 512, 3
        "depth": tf.convert_to_tensor(depth, tf.float32), # F, 512, 512, 1
        "normal": tf.convert_to_tensor(normal, tf.float32), # F, 512, 512, 3
        "object_coordinates": tf.convert_to_tensor(object_coordinates, tf.float32),
        "segmentations": tf.convert_to_tensor(segmentations, tf.int32),
        "camera": {
            "focal_length": tf.convert_to_tensor(camera["focal_length"], tf.float32),
            "positions": tf.convert_to_tensor(camera["positions"], tf.float32),
            "quaternions": tf.convert_to_tensor(camera["quaternions"], tf.float32),
            "sensor_width": tf.convert_to_tensor(camera["sensor_width"], tf.float32),
        },
        "instances": {
            "bboxes_3d": tf.convert_to_tensor([inst["bboxes_3d"] for inst in instances], tf.float32),
            "quaternions": tf.convert_to_tensor([inst["quaternions"] for inst in instances], tf.float32),
        },
        "metadata": {
            # Not actually used anymore, but keep API-compatible
            "depth_range": tf.constant([0.0, 1.0], tf.float32),
        },
    }

    return data


def plot_tracks(rgb, points, occluded, trackgroup=None):
  """Plot tracks with matplotlib."""
  disp = []
  cmap = plt.cm.hsv

  z_list = np.arange(
      points.shape[0]) if trackgroup is None else np.array(trackgroup)
  # random permutation of the colors so nearby points in the list can get
  # different colors
  z_list = np.random.permutation(np.max(z_list) + 1)[z_list]
  colors = cmap(z_list / (np.max(z_list) + 1))
  figure_dpi = 64

  for i in range(rgb.shape[0]):
    fig = plt.figure(
        figsize=(256 / figure_dpi, 256 / figure_dpi),
        dpi=figure_dpi,
        frameon=False,
        facecolor='w')
    ax = fig.add_subplot()
    ax.axis('off')
    ax.imshow(rgb[i])

    valid = points[:, i, 0] > 0
    valid = np.logical_and(valid, points[:, i, 0] < rgb.shape[2] - 1)
    valid = np.logical_and(valid, points[:, i, 1] > 0)
    valid = np.logical_and(valid, points[:, i, 1] < rgb.shape[1] - 1)

    colalpha = np.concatenate([colors[:, :-1], 1 - occluded[:, i:i + 1]],
                              axis=1)
    # Note: matplotlib uses pixel corrdinates, not raster.
    plt.scatter(
        points[valid, i, 0] - 0.5,
        points[valid, i, 1] - 0.5,
        s=3,
        c=colalpha[valid],
    )

    occ2 = occluded[:, i:i + 1]

    colalpha = np.concatenate([colors[:, :-1], occ2], axis=1)

    plt.scatter(
        points[valid, i, 0],
        points[valid, i, 1],
        s=20,
        facecolors='none',
        edgecolors=colalpha[valid],
    )

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(
        fig.canvas.tostring_rgb(),
        dtype='uint8').reshape(int(height), int(width), 3)
    disp.append(np.copy(img))
    plt.close(fig)

  return np.stack(disp, axis=0)



def main():
    scene_dir = "/mnt/ssd3/eunbeen/kubric/multiview/1212/scene0"

    # Load single scene
    data = load_scene_from_dir(scene_dir)

    # Run dense tracking
    example = add_tracks(data, train_size=(512, 512))

    # Convert to numpy (plot_tracks expects numpy)
    video = example['video'].numpy() * 0.5 + 0.5
    target_points = example['target_points'].numpy()
    occluded = example['occluded'].numpy()

    # Plot once
    disp = plot_tracks(video, target_points, occluded)

    os.makedirs("./outouts", exist_ok=True)
    media.write_video("./outputs/scene0_tracks.mp4", disp, fps=10)

if __name__ == '__main__':
  main()
