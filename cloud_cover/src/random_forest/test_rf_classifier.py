import numpy as np
import matplotlib.pyplot as plt

# ---- Load scene & run RFCD ----
from rfcd_l8_dataloader import L8RFCDDataLoader
from random_forest_cloud_detection import RandomForestCloudDetection

# scene = "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Testing_L8_Data/LC81360302014162LGN00/BC/LC81360302014162LGN00"
scene = "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_Testing_L8_Data/LC81620432014072LGN00/BC/LC81620432014072LGN00"

loader = L8RFCDDataLoader(scene)
blue, green, red, nir, cloud_gt, valid = loader.load()

img = np.stack([blue, green, red, nir], axis=0)

rfcd = RandomForestCloudDetection(
    input_img=img,
    model_path="/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_model_25k.joblib"
)

from time import time
start_time = time()
soft, cloud_pred = rfcd.rfcd_refined()
end_time = time()
print("RFCD processing time: %.2f seconds" % (end_time - start_time))

# ---- RGB composite (for display) ----
rgb = np.stack([red, green, blue], axis=-1)

# Normalize for visualization
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

# ---- Raw RF probability ----
prob_rf = rfcd.predict_proba_map()

# ---- Refined output ----
soft, refined_mask = rfcd.rfcd_refined()

# ---- Mask invalid pixels ----
raw_mask = (prob_rf > 0.5).astype(np.float32)
refined_mask = refined_mask.astype(np.float32)
cloud_gt = cloud_gt.astype(np.float32)

raw_mask[~valid] = np.nan
refined_mask[~valid] = np.nan
cloud_gt[~valid] = np.nan


# ---- Plot ----
plt.figure(figsize=(18, 12))

plt.subplot(2,3,1)
plt.title("RGB")
plt.imshow(rgb)
plt.axis("off")

plt.subplot(2,3,2)
plt.title("Ground Truth")
plt.imshow(cloud_gt, cmap="gray")
plt.axis("off")

plt.subplot(2,3,3)
plt.title("RF Probability")
plt.imshow(prob_rf, cmap="jet")
plt.colorbar()
plt.axis("off")

plt.subplot(2,3,4)
plt.title("Raw RF Mask")
plt.imshow(raw_mask, cmap="gray")
plt.axis("off")

plt.subplot(2,3,5)
plt.title("RFCD Soft Output")
plt.imshow(soft, cmap="jet")
plt.colorbar()
plt.axis("off")

plt.subplot(2,3,6)
plt.title("RFCD Refined Mask")
plt.imshow(refined_mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

gt = cloud_gt[valid].ravel()
raw = raw_mask[valid].ravel()
rfcd = refined_mask[valid].ravel()

print("RAW RF:")
print("P:", precision_score(gt, raw))
print("R:", recall_score(gt, raw))
print("F1:", f1_score(gt, raw))

print("\nRFCD:")
print("P:", precision_score(gt, rfcd))
print("R:", recall_score(gt, rfcd))
print("F1:", f1_score(gt, rfcd))
