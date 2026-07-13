#!/usr/bin/env python3
"""
Generate synthetic TEyeD-like data for DEBUG mode end-to-end testing.
Creates minimal MasterKey (.mat) and H5 archives in ./Datasets/.
"""

import os
import numpy as np
import h5py
import scipy.io as scio

# ── Config ──────────────────────────────────────────────────────────
IMAGES_PER = 200          # per subject
H, W       = 240, 320     # must match what the code expects
SEED       = 42
# Subset names: SS_1..SS_6 in train set; SS_31 in test set (enough to survive stratification + split)
SUBSETS = ['DikablisSS_1', 'DikablisSS_2', 'DikablisSS_3', 'DikablisSS_4', 'DikablisSS_5', 'DikablisSS_6', 'DikablisSS_31']

np.random.seed(SEED)

BASE   = os.path.dirname(os.path.abspath(__file__))
MK_DIR = os.path.join(BASE, 'Datasets', 'MasterKey')
H5_DIR = os.path.join(BASE, 'Datasets', 'All')
os.makedirs(MK_DIR, exist_ok=True)
os.makedirs(H5_DIR, exist_ok=True)

# ── Helpers ─────────────────────────────────────────────────────────
def make_eye_image():
    """Return a (H,W) uint8 image with a crude iris+pupil disk and their params."""
    img = np.random.randint(100, 180, (H, W), dtype=np.uint8)
    cx = W // 2 + np.random.randint(-40, 40)
    cy = H // 2 + np.random.randint(-20, 20)
    iris_r = 50 + np.random.randint(-8, 8)
    pupil_r = 20 + np.random.randint(-4, 4)

    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    img[dist < iris_r]  = (img[dist < iris_r]  * 0.35).astype(np.uint8)
    img[dist < pupil_r] = (img[dist < pupil_r] * 0.15).astype(np.uint8)
    return img, cx, cy, iris_r, pupil_r

# ── Generate one .mat + one .h5 per subject ─────────────────────────
for si, subset in enumerate(SUBSETS):
    h5_name = f'TEyeD-{subset}'
    h5_path = os.path.join(H5_DIR, f'{h5_name}.h5')

    images     = np.zeros((IMAGES_PER, H, W), dtype=np.uint8)
    pupil_locs = np.zeros((IMAGES_PER, 2))
    pupil_fits = np.zeros((IMAGES_PER, 5))
    iris_fits  = np.zeros((IMAGES_PER, 5))
    iris_locs  = np.zeros((IMAGES_PER, 2))
    eyeballs   = np.zeros((IMAGES_PER, 4))

    archives = []  # archive names for .mat

    for i in range(IMAGES_PER):
        img, cx, cy, ir, pr = make_eye_image()
        images[i] = img

        # Spread pupil positions across [0.15, 0.85] of image to fill stratification bins
        px = int(W * np.random.uniform(0.15, 0.85))
        py = int(H * np.random.uniform(0.15, 0.85))

        pupil_locs[i] = [px, py]
        pupil_fits[i] = [px, py, pr*0.9, pr*0.8, np.random.randn()*0.15]
        iris_fits[i]  = [cx, cy, ir*1.0, ir*0.9,  np.random.randn()*0.1]
        iris_locs[i]  = [cx, cy]
        eyeballs[i]   = [W/2, H/2, 12.0, 24.0]

        archives.append(h5_name)  # same archive for all images in this H5

    # ── write HDF5 ──────────────────────────────────────────────
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('Images',  data=images)
        f.create_dataset('pupil_loc', data=pupil_locs)
        g = f.create_group('Fits')
        g.create_dataset('pupil', data=pupil_fits)
        g.create_dataset('iris',  data=iris_fits)
        f.create_dataset('Eyeball', data=eyeballs)
        f.create_dataset('Masks_noSkin', data=np.array([]))
        # Dikablis-specific fields (non-empty so the code doesn't crash)
        f.create_dataset('Gaze_vector', data=np.tile([0.0, 0.0, -1.0], (IMAGES_PER, 1)))
        f.create_dataset('pupil_lm_2D', data=np.zeros((IMAGES_PER, 17)))
        f.create_dataset('pupil_lm_3D', data=np.zeros((IMAGES_PER, 25)))
        f.create_dataset('iris_lm_2D',  data=np.zeros((IMAGES_PER, 17)))
        f.create_dataset('iris_lm_3D',  data=np.zeros((IMAGES_PER, 25)))

    # ── write .mat (one per subject → scalar dataset/subset) ────
    # Fits struct with iris data
    fits_dtype = np.dtype([('iris', 'O')])
    Fits = np.zeros((1, 1), dtype=fits_dtype)
    Fits['iris'][0, 0] = iris_locs

    mdict = {
        'archive':    np.array(archives, dtype=object),
        'dataset':    'TEyeD',                     # SCALAR, not array
        'subset':     subset,                      # SCALAR, not array
        'subject_id': np.array([f's{si:03d}'] * IMAGES_PER, dtype=object),
        'pupil_loc':  pupil_locs,
        'Fits':       Fits,
        'resolution': np.tile([H, W], (IMAGES_PER, 1)),  # (N,2)
    }
    mat_path = os.path.join(MK_DIR, f'TEyeD_{subset}.mat')
    scio.savemat(mat_path, mdict, do_compression=True)

    print(f'[OK]  {h5_path}  +  {os.path.basename(mat_path)}  ({IMAGES_PER} images)')

total = len(SUBSETS) * IMAGES_PER
print(f'\nDone — synthetic TEyeD dataset ready ({total} images, {len(SUBSETS)} subjects).')
print(f'Now run:  python run.py --exp_name="DEBUG" --model="res_50_3"')
