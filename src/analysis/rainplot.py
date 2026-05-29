import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import os

# ==========================================
# 1. SETTINGS & PROFESSIONAL PALETTE
# ==========================================
FILE_PATHS = {
    1: '/Users/Admin/Drosphilia SIP/DownloadedCSV/group_4-1_data.csv',
    2: '/Users/Admin/Drosphilia SIP/DownloadedCSV/group_4-2_data.csv' 
}
FPS = 120
DT = 1 / FPS
WINDOW_SIZE = 3 * FPS

# Strict RGB Palette
CLR_NAVY = (47/255, 79/255, 127/255)    # Day 2 / Recovery
CLR_MUTED = (135/255, 160/255, 199/255)  # Day 1 / Baseline
CLR_GRAY = (229/255, 229/255, 229/255)   # Grid / Accents
CLR_BLACK = (0, 0, 0)

# Global Figure Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.linewidth': 1.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8
})
FIG_SIZE = (3.15, 2.6) # ~8cm width

# ==========================================
# 2. DATA PROCESSING FUNCTIONS
# ==========================================

def assign_well_fixed(row, tx_col, ty_col):
    x, y = row[tx_col], row[ty_col]
    if pd.isna(x) or pd.isna(y): return None
    col = "Left" if x < 370 else ("Mid" if x < 715 else "Right")
    pos = "Top" if y < 345 else "Bottom"
    return f"{pos}_{col}"

def smooth_data(data, window=15, poly=3):
    clean_series = data.interpolate(limit_direction='both').fillna(0)
    if len(clean_series) < window: return clean_series
    return pd.Series(savgol_filter(clean_series, window, poly), index=data.index)

# ==========================================
# 3. PIPELINE
# ==========================================

all_days_dfs = []

for day, path in FILE_PATHS.items():
    if not os.path.exists(path): continue
    print(f"🛠️ Processing Day {day}...")
    df = pd.read_csv(path)
    
    cols = df.columns.tolist()
    tx, ty = ('thorax_x', 'thorax_y') if 'thorax_x' in cols else ('thorax.x', 'thorax.y')
    hx, hy = ('head_x', 'head_y') if 'head_x' in cols else ('head.x', 'head.y')

    df['identity'] = df.apply(lambda r: assign_well_fixed(r, tx, ty), axis=1)
    df = df.dropna(subset=['identity', tx, ty]).copy()
    df['time_sec'] = df[next(c for c in cols if 'frame' in c)] / FPS
    df = df.sort_values(['identity', 'time_sec'])
    
    # Velocity with artifact filter
    def calc_vel(group):
        dist = np.sqrt(group[tx].diff()**2 + group[ty].diff()**2)
        dist = dist.mask(dist > 40, 0) 
        return smooth_data(dist / DT)

    df['vel_smooth'] = df.groupby('identity', group_keys=False).apply(calc_vel)
    df['h_t_dist'] = np.sqrt((df[hx]-df[tx])**2 + (df[hy]-df[ty])**2)
    df['jitter'] = df.groupby('identity')['h_t_dist'].transform(lambda x: x.rolling(window=WINDOW_SIZE, center=True).std())
    
    def get_phase(t):
        for i in range(5):
            start = 90 + (i * 90)
            if start <= t <= start + 30: return f"Vortex_{i+1}"
        return "Baseline" if t < 90 else "Recovery"
    
    df['phase'] = df['time_sec'].apply(get_phase)
    df['Day'] = day
    all_days_dfs.append(df)

df_main = pd.concat(all_days_dfs)

# ==========================================
# 4. FIGURE 1: SENSITIZATION (8cm)
# ==========================================
print("📈 Plotting Figure 1...")
day1_base = df_main[(df_main['Day'] == 1) & (df_main['phase'] == 'Baseline')].groupby('identity')['vel_smooth'].mean()

spark_list = []
for (day, fly_id), group in df_main.groupby(['Day', 'identity']):
    base_val = day1_base.get(fly_id, group['vel_smooth'].mean())
    for i in range(1, 6):
        phase_name = f"Vortex_{i}"
        # Use 95th percentile to capture the "Spark" without noise spikes
        peak = group[group['phase'] == phase_name]['vel_smooth'].quantile(0.95)
        if not pd.isna(peak):
            spark_list.append({
                'Exposure': i if day == 1 else i + 5,
                'Arousal': peak / base_val,
                'Day': f"Day {day}"
            })

df_spark = pd.DataFrame(spark_list)
plt.figure(figsize=FIG_SIZE)

sns.lineplot(data=df_spark, x='Exposure', y='Arousal', hue='Day', 
             marker='o', markersize=6, linewidth=3, palette=[CLR_MUTED, CLR_NAVY], errorbar=('ci', 68))

plt.axvline(5.5, color=CLR_BLACK, linestyle=':', alpha=0.5)
plt.axhline(1.0, color=CLR_BLACK, linestyle='--', linewidth=1, alpha=0.3) # 1.0 = Baseline

plt.title("Chronic Locomotor Sensitization and Baseline \nArousal Shift Across a 48-Hour Period", 
          fontsize=10, weight='bold', pad=10)
plt.ylabel("Arousal (Fold-change)", fontsize=9)
plt.xlabel("Exposure Number", fontsize=9)
plt.xticks(range(1, 11))
plt.ylim(0, 5)
sns.despine()
plt.tight_layout()
plt.savefig("Fig1_Sensitization_8cm.png", dpi=600)
plt.show()

# ==========================================
# 5. FIGURE 2: JITTER RAINCLOUD (8cm)
# ==========================================
print("📊 Plotting Figure 2...")
# Filter for first Baseline vs Final Recovery
df_jitter = df_main[df_main['phase'].isin(['Baseline', 'Recovery'])].copy()
df_paired = df_jitter.groupby(['identity', 'phase'], observed=True)['jitter'].mean().unstack().dropna()

# Reshape for seaborn
df_long = df_paired.reset_index().melt(id_vars='identity', value_vars=['Baseline', 'Recovery'], 
                                      var_name='phase', value_name='jitter_val')

fig, ax = plt.subplots(figsize=FIG_SIZE)

# 1. THICK PAIRED LINES (Universal proof of stabilization)
for idx in df_paired.index:
    ax.plot([0, 1], [df_paired.loc[idx, 'Baseline'], df_paired.loc[idx, 'Recovery']], 
            color=CLR_NAVY, alpha=0.35, linewidth=2.5, zorder=1)

# 2. CLOUD: Violin
sns.violinplot(data=df_long, x='phase', y='jitter_val', palette=[CLR_MUTED, CLR_NAVY], 
               bw_adjust=.6, cut=0, inner=None, ax=ax, alpha=0.4, zorder=2)

# 3. UMBRELLA: Boxplot
sns.boxplot(data=df_long, x='phase', y='jitter_val', width=.12, zorder=10, 
            showcaps=True, boxprops={'facecolor':'none', 'edgecolor':'0.2'},
            showfliers=False, whiskerprops={'linewidth':1.5}, ax=ax)

# 4. RAIN: Individual Dots
sns.stripplot(data=df_long, x='phase', y='jitter_val', palette=[CLR_MUTED, CLR_NAVY], 
              size=5, jitter=False, alpha=0.8, ax=ax, zorder=11)

ax.set_title("\nIndividual Jitter Stabilization Compared \n Between Start and End Across 48 Hours", fontsize=10, weight='bold', pad=10)
ax.set_ylabel("Jitter (mm SD)", fontsize=9)
ax.set_xlabel("")
sns.despine()
plt.tight_layout()
plt.savefig("Fig2_Jitter_Raincloud_8cm.png", dpi=600)
plt.show()

print("✨ Success! Two high-resolution figures saved.")