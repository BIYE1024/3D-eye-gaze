#!/usr/bin/env python3
"""Annotate eye image with gaze direction, iris/pupil, and eyeball center."""
import torch, sys, numpy as np, cv2, os
sys.path.insert(0, '/workspace')
from models_mux import res_50_3

# ── 1. Load & preprocess image ────────────────────────────────
img_path = '/workspace/images/eye.png'
img_bgr = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
H_orig, W_orig = img_gray.shape

# Resize to model input
IMG_W, IMG_H = 320, 240
img_resized = cv2.resize(img_gray, (IMG_W, IMG_H))
img_float = img_resized.astype(np.float32)
img_norm = (img_float - img_float.mean()) / (img_float.std() + 1e-9)

# Model expects (B=1, N=4, H, W)
tensor = torch.from_numpy(np.stack([img_norm]*4)).float().unsqueeze(0)

# ── 2. Load model ─────────────────────────────────────────────
ckpt = torch.load('/workspace/Results/last.pt', map_location='cpu')
args = dict(ckpt['args'])
for k, d in [('frames',4),('extra_depth',0),('base_channel_size',32),
             ('pretrained_resnet',False),('dropout',0.0)]:
    args.setdefault(k, d)

net = res_50_3(args)
net.load_state_dict(ckpt['state_dict'])
net.eval()

# ── 3. Inference ──────────────────────────────────────────────
with torch.no_grad():
    out_dict, _ = net({'image': tensor}, args)

gaze  = out_dict['gaze_vector_3D'][0,0].numpy()   # (3,) normalized
T     = out_dict['T'][0,0].numpy()                 # (3,) eyeball center
R     = out_dict['R'][0,0].numpy()                 # (3,) eye rotation
L_val = out_dict['L'][0,0].item()                  # pupil depth
r_iris  = out_dict['r_iris'][0,0].item()           # tanh-space iris radius
r_pupil = out_dict['r_pupil'][0,0].item()          # tanh-space pupil radius
focal   = out_dict['focal'][0,0].numpy()           # (2,) focal length

# ── 4. Map model outputs to image pixel coordinates ────────────
# Model outputs are tanh-activated, range ~[-1, 1]. Map to image.
# T[0:2] -> eyeball center UV in normalized space
# r_iris, r_pupil in tanh space -> convert to pixel radii via focal

center_x = int(IMG_W/2 + T[0] * IMG_W/2)   # eyeball center X
center_y = int(IMG_H/2 - T[1] * IMG_H/2)   # eyeball center Y (flip Y)

# Radii: tanh value * image scale (rough approximation)
# r_iris and r_pupil are tanh outputs, ~[-1, 1]
# A typical iris occupies ~15-25% of eye image width
iris_r_px  = int((r_iris  + 1) / 2 * IMG_W * 0.25 + IMG_W * 0.05)
pupil_r_px = int((r_pupil + 1) / 2 * IMG_W * 0.12 + IMG_W * 0.02)

# Gaze arrow: from eyeball center, direction = gaze vector
arrow_len = 80  # pixels
gaze_dx = int(gaze[0] * arrow_len)  # X
gaze_dy = int(-gaze[1] * arrow_len) # Y (flip for image coords)

# ── 5. Draw annotations on image ──────────────────────────────
# Work on a color version (BGR) at model input size for drawing
vis = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

# Draw iris circle (green)
cv2.circle(vis, (center_x, center_y), iris_r_px, (0, 255, 0), 2)
cv2.putText(vis, 'iris', (center_x + iris_r_px + 3, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Draw pupil circle (blue)
cv2.circle(vis, (center_x, center_y), pupil_r_px, (255, 0, 0), 2)
cv2.putText(vis, 'pupil', (center_x + pupil_r_px + 3, center_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# Draw eyeball center (red dot)
cv2.circle(vis, (center_x, center_y), 3, (0, 0, 255), -1)
cv2.putText(vis, 'center', (center_x + 8, center_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Draw gaze arrow (red)
arrow_end = (center_x + gaze_dx, center_y + gaze_dy)
cv2.arrowedLine(vis, (center_x, center_y), arrow_end, (0, 0, 255), 2, tipLength=0.2)
cv2.putText(vis, 'gaze', (arrow_end[0] + 5, arrow_end[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Draw legend
h_ang = np.degrees(np.arctan2(gaze[0], gaze[2]))
v_ang = np.degrees(np.arcsin(np.clip(gaze[1], -1, 1)))
cv2.putText(vis, f'Gaze: H={h_ang:+.1f}deg V={v_ang:+.1f}deg',
            (8, IMG_H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
cv2.putText(vis, f'Vector=[{gaze[0]:+.3f}, {gaze[1]:+.3f}, {gaze[2]:+.3f}]',
            (8, IMG_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1)
cv2.putText(vis, f'Eyeball T=[{T[0]:+.3f}, {T[1]:+.3f}, {T[2]:+.3f}]',
            (8, IMG_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1)

# ── 6. Save ───────────────────────────────────────────────────
out_path = '/workspace/images/eye_annotated.png'
cv2.imwrite(out_path, vis)
print(f'Annotated image saved to: {out_path}')
print(f'Eyeball center (px): ({center_x}, {center_y})')
print(f'Iris radius: {iris_r_px}px, Pupil radius: {pupil_r_px}px')
print(f'Gaze: H={h_ang:+.1f}deg (left-right), V={v_ang:+.1f}deg (up-down)')
print(f'Gaze vector: [{gaze[0]:.4f}, {gaze[1]:.4f}, {gaze[2]:.4f}]')
print(f'Eyeball T:   [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}]')
print(f'Done.')
