#!/usr/bin/env python3
"""Load pre-trained res_50_3 and run inference on dummy input."""
import torch, sys, numpy as np
sys.path.insert(0, '/workspace')
from models_mux import res_50_3

# 1. Load checkpoint
ckpt = torch.load('/workspace/Results/last.pt', map_location='cpu')
train_args = ckpt['args']

# 2. Build model using original training args (needs to be dict-like)
args = dict(train_args)
for key, default in [('frames', 4), ('extra_depth', 0), ('base_channel_size', 32),
                      ('pretrained_resnet', False), ('dropout', 0.0)]:
    args.setdefault(key, default)

net = res_50_3(args)
net.load_state_dict(ckpt['state_dict'])
net.eval()
print(f'Loaded epoch {ckpt["epoch"]} checkpoint')
print(f'Parameters: {sum(p.numel() for p in net.parameters()):,}')
print(f'Training config: model={args["model"]}, '
      f'lr={args["lr"]}, epochs={args.get("epochs","?")}')

# 3. Simulated OVM6211 input
# Model expects data_dict['image']: (B, N, H, W) = (1, 4, 240, 320)
dummy_img = torch.randn(1, 4, 240, 320)
data_dict = {'image': dummy_img}

# 4. Forward pass (returns tuple: out_dict, out_dict_valid)
with torch.no_grad():
    out_dict, is_valid = net(data_dict, args)

# 5. Show key predictions
print(f'\n{"="*50}')
print(f'Pre-trained model inference results')
print(f'{"="*50}')
print(f'  Valid output: {is_valid}')

gaze = out_dict['gaze_vector_3D'][0, 0].numpy()  # first batch, first frame
print(f'\n--- Gaze Prediction (normalized 3D vector) ---')
print(f'  X={gaze[0]:.4f}  Y={gaze[1]:.4f}  Z={gaze[2]:.4f}')
print(f'  Direction: ({gaze[0]:.2f}, {gaze[1]:.2f}, {gaze[2]:.2f})')
print(f'  Norm: {np.linalg.norm(gaze):.4f} (should be ~1.0)')

print(f'\n--- Eyeball 3D center (T) ---')
T = out_dict['T'][0, 0].numpy()
print(f'  [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}]')

print(f'\n--- Rendering params ---')
print(f'  L (pupil depth):    {out_dict["L"][0,0].item():.4f}')
print(f'  r_iris:             {out_dict["r_iris"][0,0].item():.4f}')
print(f'  r_pupil:            {out_dict["r_pupil"][0,0].item():.4f}')
print(f'  focal:              {out_dict["focal"][0,0].tolist()}')

print(f'\n{"="*50}')
print(f'Inference pipeline works. Model loaded from epoch 79 checkpoint.')
print(f'Note: values above are from RANDOM noise input, not meaningful.')
print(f'To use real OVM6211 images, replace dummy_img with real data.')
