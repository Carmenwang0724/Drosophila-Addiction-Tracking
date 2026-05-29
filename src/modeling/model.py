import numpy as np
import matplotlib.pyplot as plt
import sleap_nn
import sleap_io as sio
import torch

# 1. Load Model and Data
MODEL_PATH = "/content/drive/MyDrive/saved_models/fly_parts" # Path to folder with best.ckpt
VAL_LABELS_PATH = "/content/drive/MyDrive/saved_models/fly_parts/labels_val_gt_0.slp"

labels = sio.load_slp(VAL_LABELS_PATH)
predictor = sleap_nn.load_model(MODEL_PATH)

# 2. Run Inference on one frame to get the Heatmap (Confmap)
# We pick frame 0 from your validation set
lf = labels.labeled_frames[0]
img = lf.image
# Convert to tensor and run through model
img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0
with torch.no_grad():
    confmaps = predictor.model(img_tensor.cuda()).cpu().numpy()[0]

# 3. Plot the Overlay
# Channel 0 is usually the Head (check your skeleton order)
head_heatmap = confmaps[..., 0] 

plt.figure(figsize=(10, 10), dpi=300)
plt.imshow(img, cmap='gray')
# Overlay the heatmap with a 'hot' or 'jet' colormap and transparency
plt.imshow(head_heatmap, cmap='jet', alpha=0.5, interpolation='bilinear')

plt.title("Figure A1: Neural Network Confidence Map (Head Node)", fontsize=14)
plt.axis('off')
plt.savefig("Confidence_Map_Head.png", bbox_inches='tight')
plt.show()

print("✅ 'Confidence_Map_Head.png' saved. This shows the 'probabilistic glow' for your poster.")