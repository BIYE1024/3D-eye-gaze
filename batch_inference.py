#!/usr/bin/env python3
"""Batch inference on OVM6211 images with annotated output for demo."""
import torch, sys, numpy as np, cv2, os, glob
sys.path.insert(0, '/workspace')
from models_mux import res_50_3

# Config
IMG_DIR  = '/workspace/images/0'
OUT_DIR  = '/workspace/demo_output'
os.makedirs(OUT_DIR, exist_ok=True)

# Load model
print('Loading model...')
ckpt = torch.load('/workspace/Results/last.pt', map_location='cpu')
args = dict(ckpt['args'])
for k, d in [('frames',4),('extra_depth',0),('base_channel_size',32),
             ('pretrained_resnet',False),('dropout',0.0)]:
    args.setdefault(k, d)
net = res_50_3(args)
net.load_state_dict(ckpt['state_dict'])
net.eval()

files = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')))[:20]  # first 20

results = []
for i, fpath in enumerate(files):
    fname = os.path.basename(fpath)

    # Preprocess
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (320, 240))
    img_f = img.astype(np.float32)
    img_n = (img_f - img_f.mean()) / (img_f.std() + 1e-9)
    tensor = torch.from_numpy(np.stack([img_n]*4)).float().unsqueeze(0)

    # Inference
    with torch.no_grad():
        out, _ = net({'image': tensor}, args)

    gaze = out['gaze_vector_3D'][0,0].numpy()
    h_ang = np.degrees(np.arctan2(gaze[0], gaze[2]))
    v_ang = np.degrees(np.arcsin(np.clip(gaze[1], -1, 1)))

    # Draw
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cx, cy = 160, 120
    arrow_len = 60
    dx, dy = int(gaze[0]*arrow_len), int(-gaze[1]*arrow_len)
    cv2.arrowedLine(vis, (cx, cy), (cx+dx, cy+dy), (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(vis, f'H={h_ang:+.1f} V={v_ang:+.1f}', (cx+dx+5, cy+dy-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, vis)

    results.append({'file': fname, 'h_deg': round(h_ang, 1), 'v_deg': round(v_ang, 1),
                    'gaze': [round(float(gaze[0]),3), round(float(gaze[1]),3), round(float(gaze[2]),3)]})
    print(f'[{i+1}/{len(files)}] {fname} -> H={h_ang:+.1f}deg V={v_ang:+.1f}deg')

# Summary
import json
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

h_vals = [r['h_deg'] for r in results]
v_vals = [r['v_deg'] for r in results]
print(f'\n--- Summary ---')
print(f'{len(results)} images processed')
print(f'Horizontal: {min(h_vals):.1f} ~ {max(h_vals):.1f} deg (mean {np.mean(h_vals):.1f})')
print(f'Vertical:   {min(v_vals):.1f} ~ {max(v_vals):.1f} deg (mean {np.mean(v_vals):.1f})')
print(f'Output: {OUT_DIR}/')
