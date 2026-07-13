#!/usr/bin/env python3
"""
Convert OVM6211 eye images + VR calibration data into TEyeD-compatible H5+MAT format.

Assumptions:
  - Each recording session: user fixates N calibration points on VR screen
  - For each point, you capture M frames from OVM6211
  - You know:
      * VR screen physical size (mm) and resolution (px)
      * Eyeball-to-screen distance in camera coordinates (from VR optics design)
      * The fixed camera-to-screen transform matrix (from hardware calibration)

Directory structure expected:
  raw_data/
    Subject001/
      calib_01/           ← fixation point 1
        frame_000.png
        frame_001.png
        ...
        screen_xy.txt     ← "960 540"  (the screen pixel being fixated)
      calib_02/
        ...
      ...
"""

import os, sys, glob, json
import numpy as np
import h5py
import scipy.io as scio
import cv2

# ═══════════════════════════════════════════════════════════════
# CONFIG — adjust these to your VR hardware
# ═══════════════════════════════════════════════════════════════

IMG_W, IMG_H = 320, 240            # model input size
SCREEN_W_MM, SCREEN_H_MM = 50, 40  # VR screen physical size
SCREEN_W_PX, SCREEN_H_PX = 1920, 1080  # VR screen resolution
EYE_TO_SCREEN_Z = 30.0             # eyeball to screen distance (mm) along camera Z
EYE_CENTER_3D = [0.0, 0.0, 0.0]   # eyeball center in camera coords (mm)

# ═══════════════════════════════════════════════════════════════
# STEP 1: screen pixel → 3D point in camera coordinates
# ═══════════════════════════════════════════════════════════════

def screen_to_camera_3d(screen_x_px, screen_y_px):
    """
    Convert VR screen pixel to 3D point in camera coordinate system.

    Simplified: assumes screen plane is at z=EYE_TO_SCREEN_Z,
    centered in camera view, with simple scaling.

    Replace with your actual hardware calibration transform!
    """
    # Normalize screen coords to [-1, 1]
    nx = (screen_x_px / SCREEN_W_PX - 0.5) * 2.0
    ny = (screen_y_px / SCREEN_H_PX - 0.5) * 2.0

    # Map to physical mm
    x_mm = nx * SCREEN_W_MM / 2
    y_mm = -ny * SCREEN_H_MM / 2  # flip Y (screen Y down, camera Y up)
    z_mm = EYE_TO_SCREEN_Z

    return np.array([x_mm, y_mm, z_mm])

# ═══════════════════════════════════════════════════════════════
# STEP 2: screen pixel → gaze_3D vector
# ═══════════════════════════════════════════════════════════════

def screen_px_to_gaze(screen_x_px, screen_y_px):
    """Compute ground-truth 3D gaze vector from screen fixation point."""
    target_3d = screen_to_camera_3d(screen_x_px, screen_y_px)
    gaze = target_3d - np.array(EYE_CENTER_3D)  # vector from eye to target
    gaze = gaze / (np.linalg.norm(gaze) + 1e-9)  # normalize
    return gaze

# ═══════════════════════════════════════════════════════════════
# STEP 3: fit ellipse to pupil region (automatic weak annotation)
# ═══════════════════════════════════════════════════════════════

def fit_pupil_ellipse(gray_img):
    """
    Crude pupil detection via thresholding + ellipse fitting.
    Replace with better method (MediaPipe, EllSeg, etc.) for real use.
    Returns (cx, cy, semi_major, semi_minor, angle_rad)
    """
    # Threshold: pupil is darkest region
    thresh = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour = pupil
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None

    ellipse = cv2.fitEllipse(largest)
    # ellipse = ((cx, cy), (MA, ma), angle)
    (cx, cy), (MA, ma), angle = ellipse
    # Convert angle: OpenCV uses degrees, clockwise from horizontal
    angle_rad = np.radians(angle)
    return np.array([cx, cy, MA/2, ma/2, angle_rad])  # semi-axes, not full

def fit_iris_ellipse(gray_img, pupil_ellipse):
    """
    Crude iris detection — refine pupil ellipse outward.
    For real use, use a proper iris segmentation model.
    """
    if pupil_ellipse is None:
        return None
    cx, cy, pa, pb, theta = pupil_ellipse
    # Iris is roughly 2-3x pupil size
    return np.array([cx, cy, pa * 2.5, pb * 2.5, theta])

# ═══════════════════════════════════════════════════════════════
# STEP 4: preprocess OVM6211 frame
# ═══════════════════════════════════════════════════════════════

def preprocess_frame(raw_frame, roi=None):
    """
    raw_frame: (H, W) or (H, W, 3) from OVM6211
    roi: (x, y, w, h) crop region. If None, use full frame.
    Returns: (240, 320) uint8 grayscale
    """
    if len(raw_frame.shape) == 3:
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw_frame

    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y+h, x:x+w]

    return cv2.resize(gray, (IMG_W, IMG_H))  # force W×H = 320×240

# ═══════════════════════════════════════════════════════════════
# STEP 5: process one recording session → H5 + MAT
# ═══════════════════════════════════════════════════════════════

def process_session(raw_dir, subject_name, output_dir):
    """
    raw_dir: path to raw data for one subject
             Contains subdirs calib_01/, calib_02/, ...
             Each subdir has frame_*.png and screen_xy.txt

    subject_name: e.g., 'Subject001'
    output_dir: where to write Datasets/
    """
    mk_dir = os.path.join(output_dir, 'MasterKey')
    h5_dir = os.path.join(output_dir, 'All')
    os.makedirs(mk_dir, exist_ok=True)
    os.makedirs(h5_dir, exist_ok=True)

    h5_name = f'OVM6211-{subject_name}'
    h5_path = os.path.join(h5_dir, f'{h5_name}.h5')

    # Collect all frames
    all_images = []
    all_pupil_loc = []
    all_pupil_fits = []
    all_iris_fits = []
    all_iris_loc = []
    all_eyeball = []
    all_gaze = []

    calib_dirs = sorted(glob.glob(os.path.join(raw_dir, 'calib_*')))
    print(f'Found {len(calib_dirs)} calibration points for {subject_name}')

    for calib_dir in calib_dirs:
        # Read screen fixation point
        xy_file = os.path.join(calib_dir, 'screen_xy.txt')
        if os.path.exists(xy_file):
            with open(xy_file) as f:
                sx, sy = map(int, f.read().strip().split())
        else:
            print(f'  WARNING: {xy_file} not found, skipping {calib_dir}')
            continue

        gaze_gt = screen_px_to_gaze(sx, sy)

        # Process each frame
        frame_files = sorted(glob.glob(os.path.join(calib_dir, '*.png')))
        for ff in frame_files:
            raw = cv2.imread(ff, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue

            img = preprocess_frame(raw)
            h, w = img.shape

            # Fit ellipses
            pupil_fit = fit_pupil_ellipse(img)
            iris_fit  = fit_iris_ellipse(img, pupil_fit)

            if pupil_fit is None:
                pupil_fit = np.full(5, -1.0)
                pupil_loc  = np.full(2, -1.0)
            else:
                pupil_loc  = pupil_fit[:2]

            if iris_fit is None:
                iris_fit  = np.full(5, -1.0)
                iris_loc  = np.full(2, -1.0)
            else:
                iris_loc  = iris_fit[:2]

            all_images.append(img)
            all_pupil_loc.append(pupil_loc)
            all_pupil_fits.append(pupil_fit)
            all_iris_fits.append(iris_fit)
            all_iris_loc.append(iris_loc)
            all_eyeball.append([EYE_CENTER_3D[0], EYE_CENTER_3D[1],
                                EYE_CENTER_3D[2], EYE_TO_SCREEN_Z])
            all_gaze.append(gaze_gt)

        print(f'  {os.path.basename(calib_dir)}: '
              f'screen=({sx},{sy}) gaze=[{gaze_gt[0]:.3f},{gaze_gt[1]:.3f},{gaze_gt[2]:.3f}] '
              f'→ {len(frame_files)} frames')

    N = len(all_images)
    print(f'  Total: {N} frames')

    images_arr   = np.array(all_images, dtype=np.uint8)     # (N, 240, 320)
    pupil_arr    = np.array(all_pupil_loc, dtype=float)     # (N, 2)
    pupil_fit_arr = np.array(all_pupil_fits, dtype=float)   # (N, 5)
    iris_fit_arr = np.array(all_iris_fits, dtype=float)     # (N, 5)
    iris_loc_arr = np.array(all_iris_loc, dtype=float)      # (N, 2)
    eyeball_arr  = np.array(all_eyeball, dtype=float)       # (N, 4)
    gaze_arr     = np.array(all_gaze, dtype=float)          # (N, 3)

    # ═══ Write H5 ═══════════════════════════════════════════
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('Images', data=images_arr)
        f.create_dataset('pupil_loc', data=pupil_arr)
        g = f.create_group('Fits')
        g.create_dataset('pupil', data=pupil_fit_arr)
        g.create_dataset('iris', data=iris_fit_arr)
        f.create_dataset('Eyeball', data=eyeball_arr)
        f.create_dataset('Gaze_vector', data=gaze_arr)
        # Optional fields — empty to trigger fallback
        f.create_dataset('Masks_noSkin', data=np.array([]))
        f.create_dataset('pupil_lm_2D', data=np.zeros((N, 17)))
        f.create_dataset('pupil_lm_3D', data=np.zeros((N, 25)))
        f.create_dataset('iris_lm_2D',  data=np.zeros((N, 17)))
        f.create_dataset('iris_lm_3D',  data=np.zeros((N, 25)))

    # ═══ Write MAT ═══════════════════════════════════════════
    archives = [h5_name] * N
    iris_inner = iris_loc_arr
    fits_dtype = np.dtype([('iris', 'O')])
    Fits = np.zeros((1, 1), dtype=fits_dtype)
    Fits['iris'][0, 0] = iris_inner

    mdict = {
        'archive':    np.array(archives, dtype=object),
        'dataset':    'OVM6211',
        'subset':     subject_name,
        'subject_id': np.full((N, 1), subject_name, dtype=object),
        'pupil_loc':  pupil_arr,
        'Fits':       Fits,
        'resolution': np.tile([IMG_H, IMG_W], (N, 1)),
    }
    mat_path = os.path.join(mk_dir, f'OVM6211_{subject_name}.mat')
    scio.savemat(mat_path, mdict, do_compression=True)

    print(f'  Wrote: {h5_path}')
    print(f'  Wrote: {mat_path}')
    return N


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True, help='Path to raw OVM6211 session data')
    ap.add_argument('--subject', required=True, help='Subject name, e.g. Subject001')
    ap.add_argument('--out_dir', default='./Datasets', help='Output directory')
    a = ap.parse_args()

    N = process_session(a.raw_dir, a.subject, a.out_dir)
    print(f'\nDone! {N} frames processed.')
    print(f'Next: add "{a.subject}" to cur_objs/datasetSelections.py train/test split.')
    print(f'Then:  python run.py --exp_name="ovm6211_train" --model="res_50_3" '
          f'--cur_obj="OVM6211" --weights_path="Results/last.pt"')
