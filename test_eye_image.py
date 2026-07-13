#!/usr/bin/env python3
"""Test pre-trained model on a real eye image."""
import torch, sys, numpy as np, cv2, os
sys.path.insert(0, '/workspace')
from models_mux import res_50_3

# ── 1. Load image and preprocess ──────────────────────────────
img_path = '/workspace/images/eye.png'
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    print(f'ERROR: Cannot read {img_path}')
    sys.exit(1)

h, w = img_bgr.shape[:2]
print(f'Original image: {w}x{h}, channels={img_bgr.shape[2] if len(img_bgr.shape)>2 else 1}')

# Convert to grayscale
if len(img_bgr.shape) == 3:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img_bgr

# Resize to model input size: 320x240 (W x H)
img_resized = cv2.resize(img_gray, (320, 240))
print(f'Resized to: {img_resized.shape[1]}x{img_resized.shape[0]} (WxH)')

# Show image stats
print(f'Pixel range: [{img_resized.min()}, {img_resized.max()}], dtype={img_resized.dtype}')

# ── 2. Prepare model input ────────────────────────────────────
# Model expects (B=1, N=4, H=240, W=320), same frame repeated 4 times
img_float = img_resized.astype(np.float32)

# Normalize (z-score per image, as in CurriculumLib)
img_norm = (img_float - img_float.mean()) / (img_float.std() + 1e-9)

frames = np.stack([img_norm] * 4, axis=0)  # (4, 240, 320)
tensor = torch.from_numpy(frames).float().unsqueeze(0)  # (1, 4, 240, 320)
print(f'Tensor shape: {list(tensor.shape)}, mean={tensor.mean():.4f}, std={tensor.std():.4f}')

# ── 3. Load model ────────────────────────────────────────────
ckpt = torch.load('/workspace/Results/last.pt', map_location='cpu')
args = dict(ckpt['args'])
for k, d in [('frames',4),('extra_depth',0),('base_channel_size',32),
             ('pretrained_resnet',False),('dropout',0.0)]:
    args.setdefault(k, d)

net = res_50_3(args)
net.load_state_dict(ckpt['state_dict'])
net.eval()
print(f'Model loaded: epoch {ckpt["epoch"]}, {sum(p.numel() for p in net.parameters()):,} params')

# ── 4. Inference ──────────────────────────────────────────────
data_dict = {'image': tensor}
with torch.no_grad():
    out_dict, is_valid = net(data_dict, args)

# ── 5. Results ────────────────────────────────────────────────
gaze = out_dict['gaze_vector_3D'][0, 0].numpy()   # first batch, first frame
T    = out_dict['T'][0, 0].numpy()
R    = out_dict['R'][0, 0].numpy()

# Compute spherical angles for easier interpretation
gaze_norm = gaze / (np.linalg.norm(gaze) + 1e-9)
theta = np.arctan2(gaze_norm[0], gaze_norm[2])  # horizontal: left-right
phi   = np.arcsin(np.clip(gaze_norm[1], -1, 1))  # vertical: up-down
theta_deg = np.degrees(theta)
phi_deg   = np.degrees(phi)

print(f'\n{"="*55}')
print(f'  EYE IMAGE INFERENCE RESULT')
print(f'{"="*55}')
print(f'  Valid: {is_valid}')
print(f'')
print(f'  Gaze 3D vector:  [{gaze[0]:+.4f}, {gaze[1]:+.4f}, {gaze[2]:+.4f}]')
print(f'  Gaze norm:        {np.linalg.norm(gaze):.4f}')
print(f'')
print(f'  Horizontal angle: {theta_deg:+.1f} deg  (+ = right, - = left)')
print(f'  Vertical angle:   {phi_deg:+.1f} deg  (+ = up,   - = down)')
print(f'')
print(f'  Eyeball center:   [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}]')
print(f'  Eye rotation R:   [{R[0]:.4f}, {R[1]:.4f}, {R[2]:.4f}]')
print(f'  Pupil depth L:    {out_dict["L"][0,0].item():.4f}')
print(f'  Iris radius:      {out_dict["r_iris"][0,0].item():.4f}')
print(f'  Pupil radius:     {out_dict["r_pupil"][0,0].item():.4f}')
print(f'{"="*55}')
