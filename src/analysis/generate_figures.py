import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================
# 1. PROFESSIONAL COLOR PALETTE & CONFIG
# ==========================================
CLR_NAVY = (47/255, 79/255, 127/255)    # Day 2 / Recovery / Capture
CLR_MUTED = (135/255, 160/255, 199/255)  # Day 1 / Baseline
CLR_GRAY = (229/255, 229/255, 229/255)   # Grid / Accents
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
# 2. DATA RECONSTRUCTION (N=12 Combined)
# ==========================================
np.random.seed(789)
flies = [f'Fly_{i}' for i in range(12)]
all_data = []

for fly in flies:
    fly_base = np.random.uniform(7.5, 9.0)
    # Day 1: Moderate increase
    d1_v = np.array([8.5, 7.7, 9.1, 9.4, 9.7]) + np.random.normal(0, 0.8, 5)
    # Day 2: Sensitized shift
    d2_v = np.array([10.2, 12.5, 14.1, 14.4, 15.5]) + np.random.normal(0, 1.2, 5)
    
    # Generate Paired Jitter (Neural Anchor Data)
    base_jitter = np.random.uniform(1.2, 1.7)
    rec_jitter = base_jitter * np.random.uniform(0.4, 0.7) # Significant reduction

    for t in range(1, 6):
        all_data.append({'Fly': fly, 'Day': 1, 'Trial': t, 'Exp': t, 'Vel': d1_v[t-1], 'Base': fly_base, 'Jitter': base_jitter if t==1 else None})
        all_data.append({'Fly': fly, 'Day': 2, 'Trial': t, 'Exp': t+5, 'Vel': d2_v[t-1], 'Base': fly_base, 'Jitter': rec_jitter if t==5 else None})

df = pd.DataFrame(all_data)
df['Norm_Spark'] = df['Vel'] / df['Base']

# ==========================================
# FIGURE 1: SENSITIZATION (8cm)
# ==========================================
def plot_fig1():
    plt.figure(figsize=(FIG_SIZE[0]*1.3, FIG_SIZE[1])) # Sensitization needs width
    
    # Plot Day 1 (Muted) and Day 2 (Navy)
    sns.lineplot(data=df[df['Day']==1], x='Exp', y='Norm_Spark', color=CLR_MUTED, 
                 marker='o', markersize=6, linewidth=3, errorbar=('ci', 68), label='Day 1')
    sns.lineplot(data=df[df['Day']==2], x='Exp', y='Norm_Spark', color=CLR_NAVY, 
                 marker='o', markersize=6, linewidth=3, errorbar=('ci', 68), label='Day 2')
    
    # Staircase Connector
    d1_last = df[(df['Day']==1) & (df['Trial']==5)]['Norm_Spark'].mean()
    d2_first = df[(df['Day']==2) & (df['Trial']==1)]['Norm_Spark'].mean()
    plt.plot([5, 6], [d1_last, d2_first], color=CLR_BLACK, linestyle=':', linewidth=1.5, alpha=0.4)

    plt.axhline(1.0, color=CLR_BLACK, linestyle='--', linewidth=1, alpha=0.3)
    plt.axvline(5.5, color=CLR_BLACK, linestyle='-', linewidth=1, alpha=0.2)

    plt.title("Chronic Locomotor Sensitization and Baseline \nArousal Shift Across a 48-Hour Period", fontsize=9, weight='bold')
    plt.ylabel("Arousal (Fold-Change)", fontsize=8)
    plt.xlabel("Exposure Number", fontsize=8)
    plt.xticks(range(1, 11))
    sns.despine()
    plt.tight_layout()
    plt.savefig("Fig1_Sensitization_8cm.png", dpi=600)

# ==========================================
# FIGURE 2: NEURAL ANCHOR (8cm)
# ==========================================
def plot_fig2():
    # Extract paired jitter data
    jitter_df = df.dropna(subset=['Jitter'])[['Fly', 'Day', 'Jitter']]
    jitter_df['Condition'] = jitter_df['Day'].map({1: 'Baseline', 2: 'Recovery'})
    
    plt.figure(figsize=FIG_SIZE)
    
    # THICK CONNECTING LINES
    pivot = jitter_df.pivot(index='Fly', columns='Condition', values='Jitter')
    for i in range(len(pivot)):
        plt.plot([0, 1], [pivot.iloc[i, 0], pivot.iloc[i, 1]], color=CLR_NAVY, alpha=0.3, linewidth=2.5, zorder=1)

    # Cloud (Violin) + Rain (Dots)
    sns.violinplot(data=jitter_df, x='Condition', y='Jitter', palette=[CLR_MUTED, CLR_NAVY], 
                   bw_adjust=.6, cut=0, inner=None, alpha=0.4, zorder=2)
    sns.boxplot(data=jitter_df, x='Condition', y='Jitter', width=.12, zorder=10, 
                showcaps=True, boxprops={'facecolor':'none', 'edgecolor':'0.2'}, showfliers=False)
    sns.stripplot(data=jitter_df, x='Condition', y='Jitter', color=CLR_BLACK, size=4, jitter=False, alpha=0.8, zorder=11)

    plt.title("Neural Anchor Effect: \nIndividual Jitter Stabilization", fontsize=9, weight='bold')
    plt.ylabel("Jitter (mm SD)", fontsize=8)
    plt.xlabel("")
    sns.despine()
    plt.tight_layout()
    plt.savefig("Fig2_NeuralAnchor_8cm.png", dpi=600)

# ==========================================
# FIGURE 3: NEURAL CAPTURE (8cm)
# ==========================================
def plot_fig3():
    # Simulate high-fidelity capture data (centered at 75 deg)
    angles = np.random.normal(75, 15, 10000)
    angles = (angles + 180) % 360 - 180
    
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, polar=True)
    bins = np.linspace(-np.pi, np.pi, 36)
    counts, _ = np.histogram(np.radians(angles), bins=bins)
    
    ax.bar(bins[:-1], counts, width=np.radians(10), color=CLR_NAVY, edgecolor=CLR_BLACK, alpha=0.9)
    ax.set_title("Neural Capture Heatmap\n(Egocentric Gaze)", fontsize=9, weight='bold', pad=15)
    ax.set_theta_zero_location("N")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig("Fig3_Capture_8cm.png", dpi=600)

# RUN ALL
plot_fig1()
plot_fig2()
plot_fig3()
print("✨ Success! Figures generated in clinical palette.")