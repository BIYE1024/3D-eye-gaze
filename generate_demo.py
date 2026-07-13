#!/usr/bin/env python3
"""Generate eye-tracking demo materials for presentation."""
import torch, sys, numpy as np, cv2, os, glob, json
sys.path.insert(0, '/workspace')
from models_mux import res_50_3

# ── Config ────────────────────────────────────────────────────
IMG_DIR  = '/workspace/images/0'
OUT_DIR  = '/workspace/demo_output'
os.makedirs(OUT_DIR, exist_ok=True)
IMG_W, IMG_H = 320, 240

# ── Load model ────────────────────────────────────────────────
print('Loading pre-trained model...')
ckpt = torch.load('/workspace/Results/last.pt', map_location='cpu')
args = dict(ckpt['args'])
for k, d in [('frames',4),('extra_depth',0),('base_channel_size',32),
             ('pretrained_resnet',False),('dropout',0.0)]:
    args.setdefault(k, d)
net = res_50_3(args)
net.load_state_dict(ckpt['state_dict'])
net.eval()
print(f'Model loaded: epoch {ckpt["epoch"]}, {sum(p.numel() for p in net.parameters()):,} params\n')

# ── Helper: run inference on one image ────────────────────────
def predict(img_gray):
    """img_gray: (240, 320) uint8 → returns (gaze_3d, h_deg, v_deg)"""
    img_f = img_gray.astype(np.float32)
    img_n = (img_f - img_f.mean()) / (img_f.std() + 1e-9)
    tensor = torch.from_numpy(np.stack([img_n]*4)).float().unsqueeze(0)
    with torch.no_grad():
        out, _ = net({'image': tensor}, args)
    gaze = out['gaze_vector_3D'][0,0].numpy()
    h = np.degrees(np.arctan2(gaze[0], gaze[2]))
    v = np.degrees(np.arcsin(np.clip(gaze[1], -1, 1)))
    return gaze, h, v

# ── Helper: draw gaze annotation ──────────────────────────────
def draw_gaze(img_gray, gaze, h_deg, v_deg):
    """Draw gaze arrow + info overlay on BGR image."""
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cx, cy = IMG_W // 2, IMG_H // 2
    alen = 50
    dx, dy = int(gaze[0] * alen), int(-gaze[1] * alen)
    cv2.arrowedLine(vis, (cx, cy), (cx+dx, cy+dy), (0, 0, 255), 2, tipLength=0.25)
    return vis

# ── Load images ───────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')))
print(f'Found {len(files)} images')

# ═══════════════════════════════════════════════════════════════
# DEMO 1: Multi-view comparison (4 different images + gaze arrows)
# ═══════════════════════════════════════════════════════════════
print('Generating Demo 1: multi-view gaze comparison...')

# Pick 4 diverse images (different recording timestamps)
diverse_idx = [0, 12, 30, min(45, len(files)-1)]
diverse_files = [files[i] for i in diverse_idx if i < len(files)]

panel = np.zeros((IMG_H, IMG_W * len(diverse_files), 3), dtype=np.uint8)
for j, fpath in enumerate(diverse_files):
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    gaze, h, v = predict(img)
    vis = draw_gaze(img, gaze, h, v)
    # Label
    cv2.putText(vis, f'H={h:+.1f}deg V={v:+.1f}deg', (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, f'Gaze=[{gaze[0]:.3f},{gaze[1]:.3f},{gaze[2]:.3f}]', (5, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    panel[:, j*IMG_W:(j+1)*IMG_W, :] = vis

cv2.putText(panel, 'Demo 1: Multi-view Gaze Prediction', (10, IMG_H-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
demo1_path = os.path.join(OUT_DIR, 'demo1_multiview.png')
cv2.imwrite(demo1_path, panel)
print(f'  Saved: {demo1_path}')

# ═══════════════════════════════════════════════════════════════
# DEMO 2: Single image deep-dive (raw → preprocessed → zoom → gaze)
# ═══════════════════════════════════════════════════════════════
print('Generating Demo 2: single image analysis pipeline...')

# Pick one clear image
fpath = files[len(files)//2]
img_orig = cv2.imread(fpath)
img_orig_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) if len(img_orig.shape)==3 else img_orig
h_orig, w_orig = img_orig_gray.shape

# Preprocessed
img_pp = cv2.resize(img_orig_gray, (IMG_W, IMG_H))
gaze, h_deg, v_deg = predict(img_pp)

# Build 2x2 panel
cell_h, cell_w = 280, 340
panel2 = np.zeros((cell_h * 2, cell_w * 2, 3), dtype=np.uint8)

def place(img, row, col, title, color=(255,255,255)):
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rsz = cv2.resize(img, (cell_w-10, cell_h-30))
    panel2[row*cell_h+25:row*cell_h+25+rsz.shape[0],
           col*cell_w+5:col*cell_w+5+rsz.shape[1]] = rsz
    cv2.putText(panel2, title, (col*cell_w+5, row*cell_h+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Cell 0: raw input
place(img_orig_gray, 0, 0, f'Raw ({w_orig}x{h_orig})', (0,255,255))

# Cell 1: preprocessed
place(img_pp, 0, 1, f'Preprocessed ({IMG_W}x{IMG_H})', (0,255,255))

# Cell 2: gaze annotated
vis = draw_gaze(img_pp, gaze, h_deg, v_deg)
cv2.putText(vis, f'H={h_deg:+.1f}deg  V={v_deg:+.1f}deg', (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
place(vis, 1, 0, 'Gaze Prediction', (0,255,0))

# Cell 3: model info
info_img = np.ones((cell_h-30, cell_w-10, 3), dtype=np.uint8) * 30
lines = [
    'Model Architecture',
    'res_50_3',
    '',
    'Encoder: ResNet-50',
    'Head: Transformer x3',
    'Output: 3D Gaze Vector',
    '',
    f'Params: 27,661,238',
    f'Pre-trained: TEyeD epoch 79',
    '',
    f'Predicted:',
    f'  H={h_deg:+.1f} deg',
    f'  V={v_deg:+.1f} deg',
    f'  [{gaze[0]:.3f}, {gaze[1]:.3f},',
    f'   {gaze[2]:.3f}]',
]
for i, line in enumerate(lines):
    cv2.putText(info_img, line, (10, 20 + i*16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
panel2[1*cell_h+25:1*cell_h+25+info_img.shape[0],
       1*cell_w+5:1*cell_w+5+info_img.shape[1]] = info_img
cv2.putText(panel2, 'Model Info', (1*cell_w+5, 1*cell_h+18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

demo2_path = os.path.join(OUT_DIR, 'demo2_pipeline.png')
cv2.imwrite(demo2_path, panel2)
print(f'  Saved: {demo2_path}')

# ═══════════════════════════════════════════════════════════════
# DEMO 3: Gaze direction distribution (scatter plot)
# ═══════════════════════════════════════════════════════════════
print('Generating Demo 3: gaze distribution analysis...')

all_h, all_v = [], []
for i, fpath in enumerate(files[:30]):  # first 30 images
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    _, h, v = predict(img)
    all_h.append(h)
    all_v.append(v)

# Draw scatter-like plot on image
plot_w, plot_h = 400, 300
plot = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 40

# Axes
cx_plt, cy_plt = plot_w//2, plot_h//2
scale = 2.5  # pixels per degree
cv2.line(plot, (0, cy_plt), (plot_w, cy_plt), (80,80,80), 1)  # H axis
cv2.line(plot, (cx_plt, 0), (cx_plt, plot_h), (80,80,80), 1)  # V axis

# Points
for h, v in zip(all_h, all_v):
    px = int(cx_plt + h * scale)
    py = int(cy_plt - v * scale)
    if 0 <= px < plot_w and 0 <= py < plot_h:
        cv2.circle(plot, (px, py), 4, (0, 255, 0), -1)

# Mean point
mean_h, mean_v = np.mean(all_h), np.mean(all_v)
cv2.circle(plot, (int(cx_plt+mean_h*scale), int(cy_plt-mean_v*scale)), 6, (0,0,255), -1)
cv2.putText(plot, f'Mean: H={mean_h:.1f} V={mean_v:.1f}', (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

# Labels
cv2.putText(plot, 'Left', (5, cy_plt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)
cv2.putText(plot, 'Right', (plot_w-35, cy_plt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)
cv2.putText(plot, 'Up', (cx_plt+5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)
cv2.putText(plot, 'Down', (cx_plt+5, plot_h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)
cv2.putText(plot, f'Consistency: {np.std(all_h):.1f}deg std (H), {np.std(all_v):.1f}deg std (V)', (10, plot_h-20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

demo3_path = os.path.join(OUT_DIR, 'demo3_distribution.png')
cv2.imwrite(demo3_path, plot)
print(f'  Saved: {demo3_path}')

# ═══════════════════════════════════════════════════════════════
# DEMO 4: System summary card
# ═══════════════════════════════════════════════════════════════
print('Generating Demo 4: system summary card...')

card_w, card_h = 700, 420
card = np.ones((card_h, card_w, 3), dtype=np.uint8) * 20

y = 25
for line, color in [
    ('3D Eye Gaze Estimation System', (0, 255, 255)),
    ('', (200,200,200)),
    ('Hardware:  OVM6211 400x400 CMOS + VR headset', (200,200,200)),
    ('Model:     res_50_3 (ResNet-50 + 3-layer Transformer)', (200,200,200)),
    ('Params:    27,661,238', (200,200,200)),
    ('Pre-train: TEyeD dataset, epoch 79/80, lr=0.002', (200,200,200)),
    ('', (200,200,200)),
    ('Pipeline:', (0, 255, 0)),
    ('  [OVM6211] → crop/resize 320x240 → z-score norm → ResNet50 → Transformer → Gaze3D', (150,150,150)),
    ('', (200,200,200)),
    ('Output:', (0, 255, 0)),
    ('  gaze_vector_3D: unit vector in camera coordinates', (150,150,150)),
    ('  eyeball_center: 3D eyeball position (T)', (150,150,150)),
    ('  iris_radius, pupil_radius, focal_length', (150,150,150)),
    ('', (200,200,200)),
    ('Inference speed: <10ms/frame (GTX 1660 Ti) | Container: Docker CUDA 11.8', (150,150,150)),
]:
    cv2.putText(card, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    y += 22

demo4_path = os.path.join(OUT_DIR, 'demo4_summary.png')
cv2.imwrite(demo4_path, card)
print(f'  Saved: {demo4_path}')

# ── Summary ───────────────────────────────────────────────────
print(f'\n{"="*55}')
print(f'All demo materials generated in: {OUT_DIR}')
print(f'  demo1_multiview.png    — 4-image multi-view comparison')
print(f'  demo2_pipeline.png     — Single image full pipeline')
print(f'  demo3_distribution.png — Gaze direction distribution')
print(f'  demo4_summary.png      — System summary card')
print(f'{"="*55}')
