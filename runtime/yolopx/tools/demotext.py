#!/usr/bin/env python3
"""
YOLOPX live daemon: /tmp/camX -> JSON + ZMQ (gated by SelfDrivingStatus)

IMPROVED VERSION V2: Extracts polygon contours for filled visualization
- Drivable area polygon (for green fill)
- Lane marking polygons (for red fill)
- Matches demo1.py visualization exactly

Uses cv2.findContours to extract polygon boundaries from segmentation masks.

FIXED: Proper start/stop handling with robust cleanup
"""
import argparse
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
import glob
import json
import signal

import cv2
import torch
import torch.nn.functional as F
from numpy import random
import numpy as np
import torchvision.transforms as transforms

import zmq
import capnp

# --- YOLOPX imports ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.models import get_net
from lib.utils.utils import select_device
from lib.core.general import non_max_suppression, scale_coords

# --- preprocessing ---
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)


# ═══════════════════════════════════════════════════════════════════
# ROBUST CLEANUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def safe_cleanup_directory(out_dir, max_retries=5, retry_delay=0.5):
    """
    Safely clean up a directory with retry logic.
    
    Handles cases where files might still be locked or being written.
    
    Args:
        out_dir: Directory to clean
        max_retries: Maximum number of cleanup attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if cleanup succeeded, False otherwise
    """
    if not os.path.exists(out_dir):
        return True
    
    for attempt in range(max_retries):
        try:
            # First, try to remove all files individually
            for f in glob.glob(os.path.join(out_dir, "*")):
                try:
                    os.remove(f)
                except (OSError, PermissionError):
                    pass
            
            # Then try to remove the directory
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir, ignore_errors=True)
            
            # Verify it's gone or empty
            if not os.path.exists(out_dir):
                print(f"[CLEANUP] Directory {out_dir} cleaned successfully")
                return True
            
            # If directory still exists, check if it's empty
            remaining = os.listdir(out_dir)
            if not remaining:
                os.rmdir(out_dir)
                print(f"[CLEANUP] Directory {out_dir} cleaned successfully")
                return True
            
            print(f"[CLEANUP] Attempt {attempt + 1}/{max_retries}: {len(remaining)} files remaining")
            
        except Exception as e:
            print(f"[CLEANUP] Attempt {attempt + 1}/{max_retries} failed: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    # Final fallback: just try to clean files, leave directory
    print(f"[CLEANUP] Warning: Could not fully clean {out_dir}, clearing files only")
    try:
        for f in glob.glob(os.path.join(out_dir, "*")):
            try:
                os.remove(f)
            except:
                pass
        return True
    except:
        return False


def stop_gst_process(gst_proc, out_dir, timeout=3.0):
    """
    Properly stop a GStreamer process and clean up.
    
    Args:
        gst_proc: The subprocess.Popen object for GStreamer
        out_dir: The output directory to clean
        timeout: Timeout for process termination
        
    Returns:
        True if cleanup succeeded
    """
    if gst_proc is None:
        return safe_cleanup_directory(out_dir)
    
    print("[GST] Stopping GStreamer pipeline...")
    
    # Step 1: Send SIGTERM for graceful shutdown
    if gst_proc.poll() is None:
        try:
            gst_proc.terminate()
            print("[GST] Sent SIGTERM to GStreamer")
        except Exception as e:
            print(f"[GST] Error sending SIGTERM: {e}")
    
    # Step 2: Wait for process to terminate
    try:
        gst_proc.wait(timeout=timeout)
        print("[GST] GStreamer terminated gracefully")
    except subprocess.TimeoutExpired:
        print("[GST] GStreamer didn't respond to SIGTERM, sending SIGKILL")
        try:
            gst_proc.kill()
            gst_proc.wait(timeout=2.0)
            print("[GST] GStreamer killed")
        except Exception as e:
            print(f"[GST] Error killing GStreamer: {e}")
    
    # Step 3: Small delay to allow file handles to release
    time.sleep(0.3)
    
    # Step 4: Clean up the directory
    return safe_cleanup_directory(out_dir)


def start_gst_capture(socket_path: str, out_dir: str, width: int, height: int):
    """Start a gst-launch-1.0 pipeline that reads from /tmp/camX and
    writes JPEG frames to out_dir/frame_XXXXX.jpg.
    
    Returns:
        subprocess.Popen object or None if failed
    """
    # Clean up existing directory first
    if not safe_cleanup_directory(out_dir):
        print(f"[SHM] Warning: Could not fully clean {out_dir}")
    
    # Create fresh directory
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(out_dir, "frame_%05d.jpg")

    pipeline = (
        "gst-launch-1.0 -q -e "
        f"shmsrc socket-path={socket_path} is-live=true do-timestamp=true ! "
        f"video/x-raw,format=RGBA,width={width},height={height},framerate=30/1 ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "nvvidconv interpolation-method=1 ! "
        "videoconvert ! "
        "jpegenc quality=70 ! "
        f"multifilesink location=\"{pattern}\""
    )

    print(f"[SHM] Starting GStreamer pipeline:\n{pipeline}")
    
    try:
        proc = subprocess.Popen(pipeline, shell=True, preexec_fn=os.setsid)
        # Give it a moment to start
        time.sleep(0.5)
        
        if proc.poll() is not None:
            print("[SHM] GStreamer process failed to start")
            return None
        
        print("[SHM] GStreamer pipeline started successfully")
        return proc
    except Exception as e:
        print(f"[SHM] Error starting GStreamer: {e}")
        return None


def extract_contours_from_mask(mask_np, simplify_epsilon=2.0, min_area=100):
    """
    Extract polygon contours from a binary mask using OpenCV.
    
    Args:
        mask_np: Binary mask (numpy array, values 0 or 1)
        simplify_epsilon: Epsilon for polygon simplification (higher = fewer points)
        min_area: Minimum contour area to include
        
    Returns:
        List of polygons, where each polygon is a list of [x, y] points
    """
    # Convert to uint8 for OpenCV
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Filter small contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplify polygon to reduce point count
        epsilon = simplify_epsilon
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to list of [x, y] points
        polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
        
        if len(polygon) >= 3:  # Need at least 3 points for a polygon
            polygons.append(polygon)
    
    # Sort by area (largest first)
    polygons.sort(key=lambda p: cv2.contourArea(np.array(p).reshape(-1, 1, 2)), reverse=True)
    
    return polygons


def extract_centerline_from_mask(da_np, orig_h, orig_w, num_samples=25):
    """
    Extract drivable area centerline by sampling rows.
    IMPROVED: Only uses the drivable region that contains the car (center-bottom).
    
    Args:
        da_np: Drivable area mask
        orig_h: Original image height
        orig_w: Original image width
        num_samples: Number of row samples
        
    Returns:
        List of [x, y] centerline points
    """
    centerline = []
    ys = np.linspace(orig_h * 0.35, orig_h - 1, num_samples).astype(int)
    
    # Find the car's lane at the bottom of the image
    # The car is at center-bottom, so we look for the drivable region containing center_x
    center_x = orig_w // 2
    
    for y in ys:
        if y < 0 or y >= da_np.shape[0]:
            continue
        
        row = np.where(da_np[y, :] == 1)[0]
        if row.size == 0:
            continue
        
        # Find clusters (separate drivable regions) in this row
        clusters = []
        cluster_start = row[0]
        cluster_end = row[0]
        
        for i in range(1, len(row)):
            if row[i] - row[i-1] > 50:  # Gap > 50 pixels = new cluster
                clusters.append((cluster_start, cluster_end))
                cluster_start = row[i]
            cluster_end = row[i]
        clusters.append((cluster_start, cluster_end))
        
        # Find the cluster that contains or is closest to center_x
        best_cluster = None
        best_dist = float('inf')
        
        for c_start, c_end in clusters:
            if c_start <= center_x <= c_end:
                # This cluster contains the center - use it
                best_cluster = (c_start, c_end)
                break
            else:
                # Find closest cluster to center
                c_center = (c_start + c_end) / 2
                dist = abs(c_center - center_x)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = (c_start, c_end)
        
        if best_cluster:
            x_center = int((best_cluster[0] + best_cluster[1]) / 2)
            centerline.append([x_center, int(y)])
    
    return centerline


def identify_main_polygon(polygons, orig_h, orig_w):
    """
    Identify which polygon is the "main" one (the lane the car is in).
    
    The main polygon should:
    1. Contain the center-bottom of the image (where the car is)
    2. Or be the closest/largest polygon near the bottom-center
    
    Args:
        polygons: List of polygons
        orig_h, orig_w: Image dimensions
        
    Returns:
        Index of the main polygon, or -1 if none found
    """
    if not polygons:
        return -1
    
    # Car position is at center-bottom
    car_x = orig_w // 2
    car_y = orig_h - 50  # Near bottom
    
    # Check which polygon contains the car position
    for i, poly in enumerate(polygons):
        if point_in_polygon(car_x, car_y, poly):
            return i
    
    # If no polygon contains the car, find the closest one
    best_idx = -1
    best_score = float('inf')
    
    for i, poly in enumerate(polygons):
        # Calculate centroid
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        
        # Score: distance from car + penalty for being far from bottom
        dist = ((cx - car_x) ** 2 + (cy - car_y) ** 2) ** 0.5
        
        # Bonus for polygons that extend to the bottom
        max_y = max(p[1] for p in poly)
        bottom_bonus = (orig_h - max_y)  # Lower is better
        
        score = dist + bottom_bonus * 2
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    return best_idx


def point_in_polygon(x, y, polygon):
    """
    Check if point (x, y) is inside a polygon using ray casting.
    """
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def polygon_intersects(poly1, poly2, margin=20):
    """
    Check if two polygons are adjacent or overlapping.
    Uses bounding box check with margin for efficiency.
    """
    # Get bounding boxes
    x1_min = min(p[0] for p in poly1)
    x1_max = max(p[0] for p in poly1)
    y1_min = min(p[1] for p in poly1)
    y1_max = max(p[1] for p in poly1)
    
    x2_min = min(p[0] for p in poly2)
    x2_max = max(p[0] for p in poly2)
    y2_min = min(p[1] for p in poly2)
    y2_max = max(p[1] for p in poly2)
    
    # Check if bounding boxes overlap (with margin)
    return not (x1_max + margin < x2_min or x2_max + margin < x1_min or
                y1_max + margin < y2_min or y2_max + margin < y1_min)


def identify_main_lane_markings(lane_polygons, main_drivable_polygon, orig_w):
    """
    Identify which lane marking polygons belong to the main lane.
    
    Lane markings adjacent to the main drivable area are considered "main".
    
    Args:
        lane_polygons: List of lane marking polygons
        main_drivable_polygon: The main drivable area polygon
        orig_w: Image width
        
    Returns:
        List of indices of lane polygons that belong to the main lane
    """
    if not lane_polygons or not main_drivable_polygon:
        return []
    
    main_indices = []
    center_x = orig_w // 2
    
    # Get the x-range of the main drivable area
    da_x_min = min(p[0] for p in main_drivable_polygon)
    da_x_max = max(p[0] for p in main_drivable_polygon)
    
    for i, lane_poly in enumerate(lane_polygons):
        # Get lane polygon center
        lane_cx = sum(p[0] for p in lane_poly) / len(lane_poly)
        
        # Check if this lane marking is within or adjacent to the main drivable area x-range
        lane_x_min = min(p[0] for p in lane_poly)
        lane_x_max = max(p[0] for p in lane_poly)
        
        # Lane marking should be near the edges of the drivable area
        # Left lane marking: near da_x_min
        # Right lane marking: near da_x_max
        margin = 100  # pixels
        
        is_left_lane = abs(lane_x_max - da_x_min) < margin or (lane_x_min < da_x_min < lane_x_max)
        is_right_lane = abs(lane_x_min - da_x_max) < margin or (lane_x_min < da_x_max < lane_x_max)
        
        # Also check if it overlaps with drivable area
        overlaps = polygon_intersects(lane_poly, main_drivable_polygon, margin=50)
        
        if is_left_lane or is_right_lane or overlaps:
            main_indices.append(i)
    
    return main_indices


def extract_lane_boundaries_from_mask(ll_np, da_np, main_da_polygon, orig_h, orig_w, num_samples=25):
    """
    Extract left and right lane boundaries for the main lane.
    
    Looks for lane markings on the left and right sides of the main drivable area.
    
    Returns:
        left_boundary: List of [x, y] points for left lane marking
        right_boundary: List of [x, y] points for right lane marking
    """
    left_boundary = []
    right_boundary = []
    
    if main_da_polygon is None or len(main_da_polygon) == 0:
        return left_boundary, right_boundary
    
    # Get x-range of main drivable area at each y level
    ys = np.linspace(orig_h * 0.35, orig_h - 1, num_samples).astype(int)
    
    center_x = orig_w // 2
    
    for y in ys:
        if y < 0 or y >= orig_h:
            continue
        
        # Get drivable area span at this row
        row_da = np.where(da_np[y, :] == 1)[0]
        if row_da.size == 0:
            continue
        
        # Find the drivable region containing center (main lane)
        clusters_da = []
        cluster_start = row_da[0]
        cluster_end = row_da[0]
        
        for i in range(1, len(row_da)):
            if row_da[i] - row_da[i-1] > 50:
                clusters_da.append((cluster_start, cluster_end))
                cluster_start = row_da[i]
            cluster_end = row_da[i]
        clusters_da.append((cluster_start, cluster_end))
        
        # Find cluster containing center_x
        main_da_range = None
        for c_start, c_end in clusters_da:
            if c_start <= center_x <= c_end:
                main_da_range = (c_start, c_end)
                break
        
        if main_da_range is None:
            # Use closest cluster
            best_dist = float('inf')
            for c_start, c_end in clusters_da:
                c_center = (c_start + c_end) / 2
                dist = abs(c_center - center_x)
                if dist < best_dist:
                    best_dist = dist
                    main_da_range = (c_start, c_end)
        
        if main_da_range is None:
            continue
        
        da_left, da_right = main_da_range
        
        # Get lane markings at this row
        row_ll = np.where(ll_np[y, :] == 1)[0]
        if row_ll.size == 0:
            continue
        
        # Find lane marking clusters
        clusters_ll = []
        cluster_start = row_ll[0]
        cluster_end = row_ll[0]
        
        for i in range(1, len(row_ll)):
            if row_ll[i] - row_ll[i-1] > 30:
                clusters_ll.append((cluster_start, cluster_end))
                cluster_start = row_ll[i]
            cluster_end = row_ll[i]
        clusters_ll.append((cluster_start, cluster_end))
        
        # Find left lane marking (closest to da_left, on the left side)
        best_left = None
        best_left_dist = float('inf')
        
        # Find right lane marking (closest to da_right, on the right side)
        best_right = None
        best_right_dist = float('inf')
        
        for c_start, c_end in clusters_ll:
            c_center = (c_start + c_end) / 2
            
            # Check if this is a left lane marking (left of or at the left edge of drivable area)
            if c_center < da_left + 50:  # Left side
                dist = abs(c_end - da_left)
                if dist < best_left_dist:
                    best_left_dist = dist
                    best_left = (c_start, c_end)
            
            # Check if this is a right lane marking (right of or at the right edge)
            if c_center > da_right - 50:  # Right side
                dist = abs(c_start - da_right)
                if dist < best_right_dist:
                    best_right_dist = dist
                    best_right = (c_start, c_end)
        
        # Add boundary points (use inner edge of lane marking)
        if best_left is not None:
            # Use the right edge of left lane marking (inner edge)
            left_boundary.append([int(best_left[1]), int(y)])
        
        if best_right is not None:
            # Use the left edge of right lane marking (inner edge)
            right_boundary.append([int(best_right[0]), int(y)])
    
    return left_boundary, right_boundary


def compute_centerline_from_boundaries(left_boundary, right_boundary, da_np, orig_h, orig_w, num_samples=25):
    """
    Compute centerline as the midpoint between left and right lane boundaries.
    
    Falls back to drivable area center if boundaries are incomplete.
    """
    centerline = []
    
    # Create lookup by y-coordinate
    left_by_y = {p[1]: p[0] for p in left_boundary}
    right_by_y = {p[1]: p[0] for p in right_boundary}
    
    ys = np.linspace(orig_h * 0.35, orig_h - 1, num_samples).astype(int)
    center_x = orig_w // 2
    
    for y in ys:
        y = int(y)
        
        left_x = left_by_y.get(y)
        right_x = right_by_y.get(y)
        
        if left_x is not None and right_x is not None:
            # Both boundaries available - use midpoint
            cx = int((left_x + right_x) / 2)
            centerline.append([cx, y])
        elif left_x is not None:
            # Only left boundary - estimate from drivable area
            row_da = np.where(da_np[y, :] == 1)[0]
            if row_da.size > 0:
                # Find the drivable region near center
                clusters = []
                c_start = row_da[0]
                c_end = row_da[0]
                for i in range(1, len(row_da)):
                    if row_da[i] - row_da[i-1] > 50:
                        clusters.append((c_start, c_end))
                        c_start = row_da[i]
                    c_end = row_da[i]
                clusters.append((c_start, c_end))
                
                for c_s, c_e in clusters:
                    if c_s <= center_x <= c_e:
                        cx = int((left_x + c_e) / 2)
                        centerline.append([cx, y])
                        break
        elif right_x is not None:
            # Only right boundary
            row_da = np.where(da_np[y, :] == 1)[0]
            if row_da.size > 0:
                clusters = []
                c_start = row_da[0]
                c_end = row_da[0]
                for i in range(1, len(row_da)):
                    if row_da[i] - row_da[i-1] > 50:
                        clusters.append((c_start, c_end))
                        c_start = row_da[i]
                    c_end = row_da[i]
                clusters.append((c_start, c_end))
                
                for c_s, c_e in clusters:
                    if c_s <= center_x <= c_e:
                        cx = int((c_s + right_x) / 2)
                        centerline.append([cx, y])
                        break
        else:
            # No lane boundaries - use drivable area center
            row_da = np.where(da_np[y, :] == 1)[0]
            if row_da.size > 0:
                clusters = []
                c_start = row_da[0]
                c_end = row_da[0]
                for i in range(1, len(row_da)):
                    if row_da[i] - row_da[i-1] > 50:
                        clusters.append((c_start, c_end))
                        c_start = row_da[i]
                    c_end = row_da[i]
                clusters.append((c_start, c_end))
                
                for c_s, c_e in clusters:
                    if c_s <= center_x <= c_e:
                        cx = int((c_s + c_e) / 2)
                        centerline.append([cx, y])
                        break
    
    return centerline


def main():
    parser = argparse.ArgumentParser(
        description="YOLOPX live daemon: /tmp/camX -> JSON + ZMQ (gated by SelfDrivingStatus)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/epoch-195.pth",
        help="model .pth path",
    )
    parser.add_argument(
        "--socket",
        type=str,
        default="/tmp/cam0",
        help="shm socket path, e.g. /tmp/cam0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="expected shm frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="expected shm frame height",
    )
    parser.add_argument(
        "--net-width",
        type=int,
        default=640,
        help="network input width (default: 640)",
    )
    parser.add_argument(
        "--net-height",
        type=int,
        default=320,
        help="network input height (default: 320)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.3,
        help="object confidence threshold",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="IOU threshold for NMS",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    parser.add_argument(
        "--live-dir",
        type=str,
        default="inference/shm_live_in",
        help="directory where gst-launch writes frames",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="logs/yolopx_live.jsonl",
        help="output JSON lines file",
    )
    parser.add_argument(
        "--zmq-pub",
        type=str,
        default="tcp://*:8002",
        help="ZMQ PUB bind address for perception output (default: tcp://*:8002)",
    )
    parser.add_argument(
        "--capnp-schema",
        type=str,
        default="/home/tonyho/driveragent/message/message.capnp",
        help="path to message.capnp",
    )
    parser.add_argument(
        "--ctrl-sub",
        type=str,
        default="tcp://127.0.0.1:5595",
        help="ZMQ SUB connect address for self-driving control (SelfDrivingStatus)",
    )
    parser.add_argument(
        "--simplify-epsilon",
        type=float,
        default=3.0,
        help="Polygon simplification epsilon (higher = fewer points, default: 3.0)",
    )

    args = parser.parse_args()
    net_w, net_h = args.net_width, args.net_height

    # --- Load Cap'n Proto schema ---
    schema_path = args.capnp_schema
    if not os.path.isabs(schema_path):
        schema_path = os.path.join(BASE_DIR, schema_path)
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Cap'n Proto schema not found: {schema_path}")
    message_capnp = capnp.load(schema_path)

    PerceptionMsg = message_capnp.SelfDrivingPrediction

    try:
        SelfDrivingStatusType = message_capnp.SelfDrivingStatus
    except AttributeError:
        SelfDrivingStatusType = None
        print("[WARN] SelfDrivingStatus not found in schema")

    # ---- ZMQ setup ----
    ctx = zmq.Context.instance()
    zmq_socket = None
    if args.zmq_pub and args.zmq_pub.strip():
        zmq_socket = ctx.socket(zmq.PUB)
        zmq_socket.bind(args.zmq_pub)
        print(f"[ZMQ] Publishing on {args.zmq_pub}")

    ctrl_socket = None
    ctrl_poller = None
    if args.ctrl_sub and args.ctrl_sub.strip():
        ctrl_socket = ctx.socket(zmq.SUB)
        ctrl_socket.connect(args.ctrl_sub)
        ctrl_socket.setsockopt(zmq.SUBSCRIBE, b"")
        ctrl_poller = zmq.Poller()
        ctrl_poller.register(ctrl_socket, zmq.POLLIN)
        print(f"[ZMQ] Subscribing on {args.ctrl_sub}")

    # ---- Device and model ----
    device = select_device(None, args.device)
    half = device.type != "cpu"

    model = get_net(cfg)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[int(random.randint(0, 255)) for _ in range(3)] for _ in range(len(names))]

    # Warmup
    dummy = torch.zeros((1, 3, net_h, net_w), device=device)
    _ = model(dummy.half() if half else dummy)

    # ---- JSON output file ----
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    jf = open(args.json_out, "a")

    frame_id = 0
    running = True
    self_driving_enabled = False
    show_ground_truth = False
    
    # Track GStreamer process state
    gst_proc = None
    was_enabled = False  # Track previous state for clean transitions

    def check_control():
        nonlocal self_driving_enabled, show_ground_truth
        if ctrl_poller is None:
            return
        socks = dict(ctrl_poller.poll(timeout=0))
        if ctrl_socket not in socks:
            return
        while True:
            try:
                raw = ctrl_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                parsed = False
                
                # Try Cap'n Proto parsing first
                if SelfDrivingStatusType is not None:
                    try:
                        # Try from_bytes_packed first (if sender uses to_bytes_packed)
                        try:
                            with SelfDrivingStatusType.from_bytes_packed(raw) as msg:
                                self_driving_enabled = bool(msg.enabled)
                                show_ground_truth = bool(msg.showGroundTruth)
                                parsed = True
                                print(f"[CTRL] Cap'n Proto (packed): enabled={self_driving_enabled}")
                        except:
                            # Try from_bytes with context manager
                            with SelfDrivingStatusType.from_bytes(raw) as msg:
                                self_driving_enabled = bool(msg.enabled)
                                show_ground_truth = bool(msg.showGroundTruth)
                                parsed = True
                                print(f"[CTRL] Cap'n Proto: enabled={self_driving_enabled}")
                    except Exception as capnp_err:
                        # Cap'n Proto parsing failed, try JSON
                        pass
                
                # Fallback to JSON
                if not parsed:
                    obj = json.loads(raw.decode("utf-8"))
                    self_driving_enabled = bool(obj.get("enabled", False))
                    show_ground_truth = bool(obj.get("showGroundTruth", False))
                    print(f"[CTRL] JSON: enabled={self_driving_enabled}")
                    
            except Exception as e:
                print(f"[CTRL] Error parsing message: {e}")

    print("[INFO] Waiting for SelfDrivingStatus enabled=true...")
    print("[INFO] Press Ctrl+C to exit")

    try:
        while running:
            check_control()

            # ═══════════════════════════════════════════════════════════════
            # STATE TRANSITION: ENABLED -> DISABLED
            # ═══════════════════════════════════════════════════════════════
            if was_enabled and not self_driving_enabled:
                print("\n" + "="*60)
                print("[STATE] Self-driving DISABLED - Stopping pipeline...")
                print("="*60)
                
                # Stop GStreamer and clean up
                stop_gst_process(gst_proc, args.live_dir)
                gst_proc = None
                was_enabled = False
                
                print("[STATE] Pipeline stopped. Waiting for enable signal...")
                print("="*60 + "\n")
                continue

            # ═══════════════════════════════════════════════════════════════
            # STATE: DISABLED - Wait for enable
            # ═══════════════════════════════════════════════════════════════
            if not self_driving_enabled:
                was_enabled = False
                time.sleep(0.1)
                continue

            # ═══════════════════════════════════════════════════════════════
            # STATE: ENABLED - Check prerequisites
            # ═══════════════════════════════════════════════════════════════
            if not os.path.exists(args.socket):
                print(f"[SHM] Socket {args.socket} does not exist. Waiting...")
                time.sleep(1.0)
                continue

            # ═══════════════════════════════════════════════════════════════
            # STATE TRANSITION: DISABLED -> ENABLED
            # Start GStreamer if not already running
            # ═══════════════════════════════════════════════════════════════
            if gst_proc is None or gst_proc.poll() is not None:
                print("\n" + "="*60)
                print("[STATE] Self-driving ENABLED - Starting pipeline...")
                print("="*60)
                
                # Clean start
                gst_proc = start_gst_capture(
                    socket_path=args.socket,
                    out_dir=args.live_dir,
                    width=args.width,
                    height=args.height,
                )
                
                if gst_proc is None:
                    print("[ERROR] Failed to start GStreamer, retrying in 2s...")
                    time.sleep(2.0)
                    continue
                
                was_enabled = True
                print("[STATE] Pipeline running. Processing frames...")
                print("="*60 + "\n")

            last_processed = None

            # ═══════════════════════════════════════════════════════════════
            # MAIN PROCESSING LOOP
            # ═══════════════════════════════════════════════════════════════
            while running and self_driving_enabled:
                check_control()
                
                # Check if self-driving was disabled
                if not self_driving_enabled:
                    print("[CTRL] Self-driving disabled - breaking processing loop")
                    break

                # Check if GStreamer is still running
                if gst_proc is None or gst_proc.poll() is not None:
                    print("[SHM] GStreamer pipeline exited unexpectedly")
                    gst_proc = None
                    break

                # Find available frames
                files = sorted(glob.glob(os.path.join(args.live_dir, "frame_*.jpg")))
                if not files:
                    time.sleep(0.01)
                    continue

                latest = files[-1]

                if latest == last_processed:
                    time.sleep(0.01)
                    continue

                # Clean up old frames
                for f in files[:-1]:
                    try:
                        os.remove(f)
                    except OSError:
                        pass

                # Read the frame
                frame = cv2.imread(latest)
                if frame is None:
                    time.sleep(0.005)
                    frame = cv2.imread(latest)
                    if frame is None:
                        last_processed = latest
                        continue

                try:
                    os.remove(latest)
                except OSError:
                    pass
                last_processed = latest

                orig_h, orig_w = frame.shape[:2]

                img_det = cv2.resize(frame, (net_w, net_h))
                img_rgb = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
                img = transform(img_rgb).to(device)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                img = img.half() if half else img.float()

                ts = time.time()

                # Inference
                with torch.no_grad():
                    det_out, da_seg_out, ll_seg_out = model(img)
                inf_out, _ = det_out

                # NMS
                det_pred = non_max_suppression(
                    inf_out,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    classes=None,
                    agnostic=False,
                )
                det = det_pred[0]

                # Upscale segmentation masks to ORIGINAL frame size
                da_seg_mask = F.interpolate(
                    da_seg_out,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )
                _, da_seg_mask = torch.max(da_seg_mask, 1)
                da_seg_mask = da_seg_mask.int().squeeze()

                ll_seg_mask = F.interpolate(
                    ll_seg_out,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )
                _, ll_seg_mask = torch.max(ll_seg_mask, 1)
                ll_seg_mask = ll_seg_mask.int().squeeze()

                # Remove lanes from drivable area (same as demo1.py)
                da_seg_mask_clean = da_seg_mask - ll_seg_mask
                road_1 = torch.zeros_like(da_seg_mask_clean)
                road_1[da_seg_mask_clean == 1] = 1
                da_seg_mask_clean = road_1

                da_np = da_seg_mask_clean.cpu().numpy()
                ll_np = ll_seg_mask.cpu().numpy()

                # ---- Compute ratios ----
                total_pixels = float(da_np.size)
                drivable_ratio = float((da_np == 1).sum() / total_pixels) if total_pixels > 0 else 0.0
                lane_ratio = float((ll_np == 1).sum() / total_pixels) if total_pixels > 0 else 0.0

                # ---- Extract polygon contours ----
                # Drivable area polygons (for green fill)
                drivable_polygons = extract_contours_from_mask(
                    da_np, 
                    simplify_epsilon=args.simplify_epsilon,
                    min_area=500
                )
                
                # Lane marking polygons (for red fill)
                lane_polygons = extract_contours_from_mask(
                    ll_np,
                    simplify_epsilon=args.simplify_epsilon,
                    min_area=100
                )
                
                # Identify which drivable polygon is the MAIN lane (car's lane)
                main_polygon_idx = identify_main_polygon(drivable_polygons, orig_h, orig_w)
                
                # Get the main drivable polygon
                main_da_polygon = drivable_polygons[main_polygon_idx] if 0 <= main_polygon_idx < len(drivable_polygons) else None
                
                # Identify which lane markings belong to the main lane
                main_lane_indices = identify_main_lane_markings(lane_polygons, main_da_polygon, orig_w)
                
                # Extract left and right lane boundaries from the mask
                left_boundary, right_boundary = extract_lane_boundaries_from_mask(
                    ll_np, da_np, main_da_polygon, orig_h, orig_w
                )
                
                # Compute centerline from lane boundaries (midpoint between left and right)
                drivable_centerline = compute_centerline_from_boundaries(
                    left_boundary, right_boundary, da_np, orig_h, orig_w
                )

                # ---- Detections ----
                detections = []
                if det is not None and len(det):
                    det[:, :4] = scale_coords(
                        img.shape[2:],
                        det[:, :4],
                        frame.shape
                    ).round()

                    det_cpu = det.detach().cpu().numpy()
                    for row in det_cpu:
                        x1, y1, x2, y2, conf, cls_id = row[:6]
                        detections.append(
                            {
                                "class_id": int(cls_id),
                                "class_name": str(names[int(cls_id)]) if names is not None else "",
                                "confidence": float(conf),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            }
                        )

                # ---- Build JSON record ----
                record = {
                    "frame_id": frame_id,
                    "timestamp": ts,
                    "image_width": orig_w,
                    "image_height": orig_h,
                    "drivable_area_ratio": drivable_ratio,
                    "lane_area_ratio": lane_ratio,
                    # Polygon data for filled visualization
                    "drivable_polygons": drivable_polygons,
                    "lane_polygons": lane_polygons,
                    # Main lane identification
                    "main_polygon_idx": main_polygon_idx,
                    "main_lane_indices": main_lane_indices,  # Which lane markings are for main lane
                    # Lane boundaries (for centerline calculation)
                    "left_boundary": left_boundary,
                    "right_boundary": right_boundary,
                    # Centerline for steering (computed from lane boundaries)
                    "drivable_centerline": drivable_centerline,
                    # Detections
                    "detections": detections,
                }

                json_str = json.dumps(record)

                jf.write(json_str + "\n")
                jf.flush()

                if zmq_socket is not None:
                    msg = PerceptionMsg.new_message()
                    msg.content = json_str
                    msg.timestamp = int(ts * 1000)
                    zmq_socket.send(msg.to_bytes())

                # Count total polygon points
                da_pts = sum(len(p) for p in drivable_polygons)
                ll_pts = sum(len(p) for p in lane_polygons)
                
                print(
                    f"frame {frame_id:05d} | det={len(detections):2d} | "
                    f"da={drivable_ratio:.3f} (main={main_polygon_idx}) | "
                    f"lane={len(lane_polygons)} (main={len(main_lane_indices)}) | "
                    f"L={len(left_boundary)} R={len(right_boundary)}"
                )

                frame_id += 1

            # ═══════════════════════════════════════════════════════════════
            # End of inner loop - cleanup if needed
            # ═══════════════════════════════════════════════════════════════
            # Note: The cleanup is now handled at the top of the outer loop
            # when we detect the state transition from enabled -> disabled

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received")
        running = False

    finally:
        # ═══════════════════════════════════════════════════════════════
        # FINAL CLEANUP
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("[SHUTDOWN] Cleaning up...")
        print("="*60)
        
        # Stop GStreamer process
        if gst_proc is not None:
            stop_gst_process(gst_proc, args.live_dir)
        else:
            # Just clean the directory
            safe_cleanup_directory(args.live_dir)
        
        # Close files and sockets
        jf.close()
        
        if zmq_socket is not None:
            zmq_socket.close()
        if ctrl_socket is not None:
            ctrl_socket.close()
        
        print("[YOLOPX] Stopped cleanly")
        print("="*60)


if __name__ == "__main__":
    main()