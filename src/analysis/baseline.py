import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PROFESSIONAL COLOR PALETTE & CONFIG
# ==========================================
CLR_NAVY = (47/255, 79/255, 127/255)    # Day 2
CLR_MUTED = (135/255, 160/255, 199/255)  # Day 1
CLR_GRAY = (229/255, 229/255, 229/255)   # Grid
CLR_BLACK = (0, 0, 0)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.linewidth': 1.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

FIG_SIZE = (3.15, 2.8) # ~8cm wide

# ==========================================
# 2. GENERATE CONTROL DATA (As per your logic)
# ==========================================
np.random.seed(42)
control_flies = [f'Ctrl_{i}' for i in range(6)]
all_data = []

for fly in control_flies:
    fly_baseline = np.random.uniform(7.5, 9.0)
    
    # Day 1: Hovering around 1.0 (No trend)
    d1_velocities = np.array([9.2, 7.8, 8.9, 8.1, 8.4]) + np.random.normal(0, 0.6, 5)
    # Day 2: Still flat (No sensitization)
    d2_velocities = np.array([8.4, 7.9, 9.6, 8.2, 8.7]) + np.random.normal(0, 0.7, 5)
    
    for trial in range(1, 6):
        all_data.append({'Fly_ID': fly, 'Day': 1, 'Trial': trial, 'Peak_V': d1_velocities[trial-1], 'Base': fly_baseline})
        all_data.append({'Fly_ID': fly, 'Day': 2, 'Trial': trial, 'Peak_V': d2_velocities[trial-1], 'Base': fly_baseline})

df_control = pd.DataFrame(all_data)
df_control['Normalized_Spark'] = df_control['Peak_V'] / df_control['Base']
df_control['Exposure_Index'] = df_control.apply(lambda x: x['Trial'] if x['Day'] == 1 else x['Trial'] + 5, axis=1)

# ==========================================
# 3. VISUALIZATION
# ==========================================
plt.figure(figsize=(FIG_SIZE[0]*1.3, FIG_SIZE[1])) # Keep width consistent with Experimental

# Plot Day 1 (Muted) and Day 2 (Navy)
sns.lineplot(data=df_control[df_control['Day']==1], x='Exposure_Index', y='Normalized_Spark', 
             color=CLR_MUTED, marker='o', markersize=6, linewidth=3, errorbar=('ci', 68), label='Day 1 (Ctrl)')

sns.lineplot(data=df_control[df_control['Day']==2], x='Exposure_Index', y='Normalized_Spark', 
             color=CLR_NAVY, marker='o', markersize=6, linewidth=3, errorbar=('ci', 68), label='Day 2 (Ctrl)')

# Reference Baseline Line at 1.0
plt.axhline(1.0, color=CLR_BLACK, linestyle='--', linewidth=1.5, alpha=0.4)
plt.text(1, 1.05, "Naive Baseline (1.0)", fontsize=7, color=CLR_BLACK, alpha=0.6)

# 24h Incubation Marker
plt.axvline(5.5, color=CLR_BLACK, linestyle=':', linewidth=1, alpha=0.3)

# Formatting per prompt
plt.title("Control Group: Baseline Locomotor Stability\nAcross a 48-Hour Period", fontsize=9, weight='bold')
plt.ylabel("Arousal (Fold-change)", fontsize=8)
plt.xlabel("Exposure Number", fontsize=8)
plt.xticks(range(1, 11))
plt.ylim(0, 5.5) # Match the Y-scale of your Sensitization graph for direct comparison
plt.legend(frameon=False, fontsize=7, loc='upper left')

sns.despine()
plt.tight_layout()
plt.savefig("Fig1B_Control_Stability_8cm.png", dpi=600)
plt.show()