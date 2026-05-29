import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. COLOR PALETTE DEFINITION (Strict RGB from your requirements)
CLR_NAVY = (47/255, 79/255, 127/255)    # High-fidelity data (Parts)
CLR_MUTED = (135/255, 160/255, 199/255)  # Baseline/Structural (Centroids)
CLR_GRAY = (229/255, 229/255, 229/255)   # Grid lines
CLR_BLACK = (0, 0, 0)

# Data reconstruction from SLEAP logs
centroid_val_loss = [0.0079, 0.0077, 0.0054, 0.0033, 0.0024, 0.0018, 0.0017, 0.0019, 0.0017, 0.0011, 
                     0.0010, 0.0009, 0.0011, 0.0009, 0.0009, 0.0008, 0.0008, 0.0007, 0.0007, 0.0008, 
                     0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0006, 0.0006, 0.0006, 0.0010, 
                     0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006]

parts_val_loss = [0.0107, 0.0034, 0.0029, 0.0026, 0.0025, 0.0025, 0.0025, 0.0024, 0.0025, 0.0025, 
                  0.0025, 0.0025, 0.0024, 0.0025, 0.0026, 0.0025, 0.0023, 0.0023, 0.0022, 0.0027, 
                  0.0023, 0.0024, 0.0019, 0.0021, 0.0022, 0.0021, 0.0021, 0.0023, 0.0022, 0.0022, 
                  0.0022, 0.0022]

# Professional Styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
sns.set_context("paper", font_scale=1.5)

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting with your specific colors
plt.plot(centroid_val_loss, label='Centroid Model (Identification)', color=CLR_MUTED, linewidth=4, alpha=0.9)
plt.plot(parts_val_loss, label='Parts Model (Pose Estimation)', color=CLR_NAVY, linewidth=4, alpha=0.9)

# Standard Deep Learning Y-Axis: Logarithmic Scale
plt.yscale('log')

# Customizing the Grid with CLR_GRAY
ax.grid(True, which="both", ls="-", alpha=0.5, color=CLR_GRAY)
ax.set_facecolor('white')
sns.despine(trim=True, offset=10)

# Labels and Title
plt.title("SLEAP U-Net Convergence: Model Validation", fontsize=20, fontweight='bold', pad=25, color=CLR_BLACK)
plt.xlabel("Training Epochs", fontsize=14, fontweight='bold')
plt.ylabel("Validation Loss (MSE)", fontsize=14, fontweight='bold')

# Legend styling
plt.legend(frameon=False, loc='upper right', fontsize=12)

plt.tight_layout()

# Save for poster
plt.savefig("SLEAP_Validation_Curve.png", dpi=600, transparent=True)
plt.show()