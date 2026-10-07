"""Build per-camera 4x4 ego→image projection matrices for System1.

Inputs:
  - calibration/camN.yaml          : OpenCV YAML, intrinsics at calibration res
  - calibration/camera_N_calibration.json : ego→camera extrinsics
                                     (pos_x/y/z meters, roll/pitch/yaw degrees)

System1 expects a [1, 6, 4, 4] tensor that maps homogeneous 3D points in the
ego frame to homogeneous image coordinates AFTER preprocessing (resize + crop).

Conventions assumed:
  ego frame:    x = forward, y = left,  z = up  (right-handed, REP-103 / ROS).
  camera optical frame: x = right, y = down, z = forward  (OpenCV / nuScenes).
  euler angles in JSON: roll (about ego-x), pitch (about ego-y), yaw (about ego-z),
                       applied in that order to orient the camera body in ego frame.

If the actual mounting convention differs, the projection matrix will be wrong
in a systematic way and detection / scene-context inputs will be wrong — but
since system1_runner currently passes scene_ctx=None, the trajectory head still
runs (it only needs visual features). The matrices are passed to deformable
attention only; bad-but-non-zero values yield degraded but finite outputs.
"""

import json
import os
import argparse
from pathlib import Path

import numpy as np

CALIB_DIR_DEFAULT = "/home/tonyho/driveragent/calibration"
NUM_CAMERAS = 6

# Stream resolution out of /tmp/cam* (RGBA from camtest). cam*.yaml is calibrated
# at 1920x1080 but the shmsrc delivers 1280x720, so intrinsics must be scaled.
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720


def _load_yaml_intrinsics(yaml_path):
    """Parse OpenCV %YAML:1.0 file and return (K_3x3, image_width, image_height).
    OpenCV YAML uses a non-standard !!opencv-matrix tag; we parse manually."""
    text = Path(yaml_path).read_text()
    image_width = None
    image_height = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("image_width:"):
            image_width = int(line.split(":", 1)[1].strip())
        elif line.startswith("image_height:"):
            image_height = int(line.split(":", 1)[1].strip())

    # camera_matrix block: find "camera_matrix:" then grab the data: [...] list
    if "camera_matrix:" not in text:
        raise ValueError(f"{yaml_path}: no camera_matrix block")
    after = text.split("camera_matrix:", 1)[1]
    # Stop at next top-level key ("dist_coeffs:" etc.)
    after = after.split("\ndist_coeffs:", 1)[0]
    if "data:" not in after:
        raise ValueError(f"{yaml_path}: camera_matrix has no data")
    data_str = after.split("data:", 1)[1]
    # Take everything between the first '[' and matching ']'
    lb = data_str.index("[")
    rb = data_str.index("]", lb)
    nums = [float(x) for x in data_str[lb + 1:rb].replace(",", " ").split()]
    if len(nums) != 9:
        raise ValueError(f"{yaml_path}: camera_matrix must be 9 floats, got {len(nums)}")
    K = np.array(nums, dtype=np.float64).reshape(3, 3)
    return K, image_width, image_height


def _load_json_extrinsics(json_path):
    """Returns dict with pos_x, pos_y, pos_z (m), roll, pitch, yaw (deg)."""
    with open(json_path) as f:
        d = json.load(f)
    ext = d["extrinsic"]
    return {
        "pos_x": float(ext["pos_x"]),
        "pos_y": float(ext["pos_y"]),
        "pos_z": float(ext["pos_z"]),
        "roll":  float(ext["roll"]),
        "pitch": float(ext["pitch"]),
        "yaw":   float(ext["yaw"]),
    }


def _euler_xyz_to_R(roll_deg, pitch_deg, yaw_deg):
    """Roll-pitch-yaw (degrees) → 3x3 rotation matrix.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (intrinsic Z-Y-X)."""
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ego frame (x=fwd, y=left, z=up) → camera body frame (still aligned with ego
# until we apply the JSON-defined rotation). After that we still need to swap
# axes into the OpenCV optical frame: x_opt = -y_body, y_opt = -z_body, z_opt = x_body.
_R_BODY_TO_OPT = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
])


def build_projection_for_camera(yaml_path, json_path,
                                stream_w=STREAM_WIDTH, stream_h=STREAM_HEIGHT,
                                final_w=704, final_h=256, model_input_h=256):
    """Return a 4x4 ego→image projection matrix sized for the model's
    post-resize/crop pixel grid. Also returns (image_w, image_h) used.

    Pipeline applied to the intrinsics:
      1) calibration K (at image_width x image_height)
      2) scale to stream resolution (1280 x 720 from shmsrc)
      3) scale to fit final_w x final_h after resize (matches preprocess_images)
      4) subtract crop offsets (top crop_y, horizontal center crop)
    """
    K_calib, calib_w, calib_h = _load_yaml_intrinsics(yaml_path)
    if calib_w is None or calib_h is None:
        # Default to 1920x1080 if unspecified
        calib_w, calib_h = 1920, 1080

    # Scale intrinsics from calibration → stream resolution
    sx = stream_w / calib_w
    sy = stream_h / calib_h
    K_stream = K_calib.copy()
    K_stream[0, 0] *= sx  # fx
    K_stream[0, 2] *= sx  # cx
    K_stream[1, 1] *= sy  # fy
    K_stream[1, 2] *= sy  # cy

    # Resize to make final_w wide, then top-crop to final_h tall
    # Match run_system1.preprocess_images: scale picks max(w/stream_w, h/stream_h)
    # so both dimensions reach the final_w x ? envelope.
    scale_w = final_w / stream_w
    scale_h = final_h / stream_h
    scale = max(scale_w, scale_h)  # ensure both axes ≥ target after resize
    new_w = int(stream_w * scale)
    new_h = int(stream_h * scale)
    crop_y = 0  # match run_system1 top-crop
    crop_x = max(0, (new_w - final_w) // 2)

    K_model = K_stream.copy()
    K_model[0, 0] *= scale
    K_model[0, 2] = K_model[0, 2] * scale - crop_x
    K_model[1, 1] *= scale
    K_model[1, 2] = K_model[1, 2] * scale - crop_y

    # Extrinsics: ego → camera optical frame (4x4)
    ext = _load_json_extrinsics(json_path)
    R_body = _euler_xyz_to_R(ext["roll"], ext["pitch"], ext["yaw"])
    R_ego_opt = _R_BODY_TO_OPT @ R_body.T  # rotate ego point into opt-aligned axes
    t_ego = np.array([ext["pos_x"], ext["pos_y"], ext["pos_z"]], dtype=np.float64)
    # If camera position in ego is t_ego, then point p_ego in optical frame is
    # p_opt = R_ego_opt @ (p_ego - t_ego)
    T_ego_opt = np.eye(4, dtype=np.float64)
    T_ego_opt[:3, :3] = R_ego_opt
    T_ego_opt[:3, 3] = -R_ego_opt @ t_ego

    K4 = np.eye(4, dtype=np.float64)
    K4[:3, :3] = K_model

    proj = K4 @ T_ego_opt
    return proj.astype(np.float32), (final_w, final_h)


def load_all(calib_dir=CALIB_DIR_DEFAULT, num_cameras=NUM_CAMERAS,
             final_w=704, final_h=256):
    """Load 6 cameras. Returns:
        projection_mat: torch.Tensor [1, 6, 4, 4] float32
        image_wh:       torch.Tensor [1, 6, 2]    float32
    Missing cameras (no JSON file) get an identity-like fallback so the
    runner can still start; users are warned in stderr.
    """
    import torch
    projs = np.zeros((num_cameras, 4, 4), dtype=np.float32)
    whs = np.zeros((num_cameras, 2), dtype=np.float32)
    for i in range(num_cameras):
        yaml_p = os.path.join(calib_dir, f"cam{i}.yaml")
        json_p = os.path.join(calib_dir, f"camera_{i}_calibration.json")
        if not os.path.exists(yaml_p) or not os.path.exists(json_p):
            # Plausible fallback: pinhole at center, identity extrinsic
            print(f"[calibration] WARNING: cam{i} missing "
                  f"(yaml={os.path.exists(yaml_p)}, json={os.path.exists(json_p)}); "
                  f"using identity placeholder")
            K = np.array([
                [700.0,   0.0, final_w / 2, 0.0],
                [  0.0, 700.0, final_h / 2, 0.0],
                [  0.0,   0.0,         1.0, 0.0],
                [  0.0,   0.0,         0.0, 1.0],
            ], dtype=np.float32)
            projs[i] = K
        else:
            proj, _ = build_projection_for_camera(
                yaml_p, json_p, final_w=final_w, final_h=final_h)
            projs[i] = proj
        whs[i] = [final_w, final_h]

    proj_t = torch.from_numpy(projs).unsqueeze(0)   # [1, 6, 4, 4]
    wh_t = torch.from_numpy(whs).unsqueeze(0)        # [1, 6, 2]
    return proj_t, wh_t


def _print_main():
    """python -m system1_runner.calibration --print"""
    proj, wh = load_all()
    print(f"projection_mat shape: {tuple(proj.shape)} dtype: {proj.dtype}")
    print(f"image_wh shape:       {tuple(wh.shape)}")
    for i in range(proj.shape[1]):
        P = proj[0, i].numpy()
        finite = np.isfinite(P).all()
        nonzero = (P != 0).any()
        # Project a point 10m forward, 0 lateral, 1m up to confirm it lands
        # roughly in image bounds for the front camera.
        p_test = np.array([10.0, 0.0, 1.0, 1.0])
        h = P @ p_test
        u = h[0] / h[2] if abs(h[2]) > 1e-9 else float("nan")
        v = h[1] / h[2] if abs(h[2]) > 1e-9 else float("nan")
        print(f"  cam{i}: finite={finite} nonzero={nonzero} "
              f"point(10m fwd) → pixel=({u:.0f}, {v:.0f}) z={h[2]:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--print", action="store_true",
                   help="Load calibration for all 6 cams and print sanity checks")
    args = p.parse_args()
    if args.print:
        _print_main()
    else:
        p.print_help()
