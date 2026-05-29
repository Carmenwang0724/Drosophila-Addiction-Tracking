import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
import os

# 1. SETTINGS
file_paths = {
    1: '/Users/Admin/Drosphilia SIP/DownloadedCSV/group_4-1_data.csv',
    2: '/Users/Admin/Drosphilia SIP/DownloadedCSV/group_4-2_data.csv'  # Update path for Day 2
}
FPS = 120 
DT = 1 / FPS
NUM_WELLS = 6

# 2. LOAD DATA & STANDARDIZE COLUMNS
dfs = []
for day, file_path in file_paths.items():
    df = pd.read_csv(file_path)
    
    # Auto-detect column names (handles thorax_x vs thorax.x)
    cols = df.columns.tolist()
    frame_col = next((c for c in cols if c in ['frame_idx', 'frame']), 'frame')
    tx, ty = ('thorax_x', 'thorax_y') if 'thorax_x' in cols else ('thorax.x', 'thorax.y')
    hx, hy = ('head_x', 'head_y') if 'head_x' in cols else ('head.x', 'head.y')
    
    # 3. SPATIAL RE-ASSIGNMENT (Fixing broken track IDs)
    def assign_well_fixed(row):
        x, y = row[tx], row[ty]
        if pd.isna(x) or pd.isna(y): return None
        
        # X Boundaries (Columns)
        if x < 370: col = "Left"
        elif x < 715: col = "Mid"
        else: col = "Right"
            
        # Y Boundary (Rows)
        if y < 345: row_pos = "Top"
        else: row_pos = "Bottom"
        return f"{row_pos}_{col}"
    
    print(f"🛠️ Re-assigning identities for Day {day}...")
    df['identity'] = df.apply(assign_well_fixed, axis=1)
    df = df.dropna(subset=['identity'])
    
    # Drop rows with missing position data
    df = df.dropna(subset=[tx, ty, hx, hy])
    
    # Filter out 'noise' tracks: Only keep identities that appear in at least 20% of the video
    valid_identities = df['identity'].value_counts()[df['identity'].value_counts() > (df[frame_col].max() * 0.2)].index
    df = df[df['identity'].isin(valid_identities)].copy()
    
    # DEDUPLICATE: Keep only the single most confident detection per well per frame
    score_col = next((c for c in cols if 'score' in c), None)
    if score_col:
        df = df.sort_values(by=['identity', frame_col, score_col], ascending=[True, True, False])
    df = df.groupby(['identity', frame_col]).head(1).reset_index()
    
    # 4. DEFINE EXPERIMENTAL PHASES
    df['time_sec'] = df[frame_col] / FPS
    
    def assign_condition_detailed(t):
        if 90 <= t <= 120: return "Vortex_1"
        if 180 <= t <= 210: return "Vortex_2"
        if 270 <= t <= 300: return "Vortex_3"
        if 360 <= t <= 390: return "Vortex_4"
        if 450 <= t <= 480: return "Vortex_5"
        if 0 <= t < 90: return "Pre_Baseline"
        return "Recovery"
    
    df['phase'] = df['time_sec'].apply(assign_condition_detailed)
    
    # 5. SCIENTIFIC CALCULATIONS
    df = df.sort_values(['identity', 'time_sec'])
    
    # Distance moved between frames
    df['dist_px'] = np.sqrt(df.groupby('identity')[tx].diff()**2 + 
                            df.groupby('identity')[ty].diff()**2)
    
    # FILTER: If the fly "teleports" > 40px, it's noise. Set velocity to 0 for that frame.
    df.loc[df['dist_px'] > 40, 'dist_px'] = 0
    
    # Linear Velocity
    df['velocity'] = df['dist_px'] / DT
    
    # Heading & Angular Velocity
    df['heading'] = np.degrees(np.unwrap(np.arctan2(df[hy]-df[ty], df[hx]-df[tx])))
    df['angular_velocity'] = np.abs(df.groupby('identity')['heading'].diff() / DT)
    
    # Grooming Distance (Head to Thorax)
    df['grooming_distance'] = np.sqrt((df[hx]-df[tx])**2 + (df[hy]-df[ty])**2)
    
    # Calculate grooming jitter (rolling standard deviation)
    WINDOW_SIZE = 3 * FPS  # 3-second window
    df['grooming_jitter'] = df.groupby('identity')['grooming_distance'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE, center=True).std()
    )
    
    # Smoothing with Savitzky-Golay
    def smooth_data(data):
        if len(data) < 21: return data
        # Handle NaN and infinity values
        result = np.array(data, dtype=float)
        valid_mask = ~(np.isnan(result) | np.isinf(result))
        valid_data = result[valid_mask]
        
        # Only filter if we have enough valid data points
        if len(valid_data) < 15:
            return data
        
        # Apply filter to valid data
        filtered = savgol_filter(valid_data, 15, 3)
        
        # Put filtered values back at valid indices
        result[valid_mask] = filtered
        return result
    
    df['vel_smooth'] = df.groupby('identity')['velocity'].transform(smooth_data)
    df['grooming_smooth'] = df.groupby('identity')['grooming_distance'].transform(smooth_data)
    
    df['Day'] = day
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# ==========================================
# SAVE LABELED DATA AFTER STD DEV PROCESSING
# ==========================================
def save_labeled_std_data(df, output_filename='Labeled_Std_Dev_Data.csv'):
    """
    Saves labeled datapoints after standard deviation (grooming jitter) processing.
    Includes all relevant metadata and measurements for post-processing analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed dataframe with grooming_jitter calculated
    output_filename : str
        Output CSV filename (default: 'Labeled_Std_Dev_Data.csv')
    
    Returns:
    --------
    pandas.DataFrame
        The exported dataframe
    """
    # Select relevant columns with labels
    labeled_data = df[[
        'identity',           # Fly ID
        'Day',                # Experimental day
        'phase',              # Experimental phase (Pre_Baseline, Vortex_*, Recovery)
        'time_sec',           # Time in seconds
        'frame_idx' if 'frame_idx' in df.columns else 'frame',  # Frame number
        'velocity',           # Raw velocity
        'vel_smooth',         # Smoothed velocity
        'grooming_distance',  # Head-to-thorax distance
        'grooming_smooth',    # Smoothed grooming distance
        'grooming_jitter',    # Standard deviation of grooming distance
        'angular_velocity',   # Angular velocity (rotation)
        'heading'             # Heading direction in degrees
    ]].copy()
    
    # Add descriptive labels as a header comment (saves as first row info)
    labeled_data = labeled_data.dropna(subset=['grooming_jitter'])  # Remove rows with NaN std dev
    
    # Save to CSV
    labeled_data.to_csv(output_filename, index=False)
    
    print(f"\n✅ Labeled standard deviation data exported!")
    print(f"   Filename: {output_filename}")
    print(f"   Total datapoints: {len(labeled_data)}")
    print(f"   Unique flies: {labeled_data['identity'].nunique()}")
    print(f"   Date range: Day {labeled_data['Day'].min()} to Day {labeled_data['Day'].max()}")
    print(f"   Phase breakdown:\n{labeled_data['phase'].value_counts()}")
    
    return labeled_data

# Export the labeled data
labeled_std_data = save_labeled_std_data(df, 'Labeled_Std_Dev_Data.csv')

# 6. STATISTICAL ANALYSIS (Per Well)
summary = df.groupby(['identity', 'phase'])['vel_smooth'].mean().unstack()
vortex_cols = [col for col in summary.columns if col.startswith('Vortex_')]
if vortex_cols:
    summary['STIMULUS'] = summary[vortex_cols].mean(axis=1)
else:
    summary['STIMULUS'] = np.nan
if 'Pre_Baseline' in summary.columns:
    summary['BASELINE'] = summary['Pre_Baseline']
else:
    summary['BASELINE'] = np.nan
summary['Delta_Arousal'] = (summary['STIMULUS'] - summary['BASELINE']) / summary['BASELINE'] * 100

# 7. VISUALIZATION (Comparison across the 6 wells)
# Calculate normalized spark velocity for sensitization curve
base_speeds = df[(df['phase'] == 'Pre_Baseline') & (df['Day'] == 1)].groupby('identity')['vel_smooth'].mean()

# Vortex start times
vortex_starts = {'Vortex_1': 90, 'Vortex_2': 180, 'Vortex_3': 270, 'Vortex_4': 360, 'Vortex_5': 450}

anova_list = []
for fly_id, group in df.groupby('identity'):
    for phase, start in vortex_starts.items():
        # First 5 seconds of the vortex
        peak_vel = group[(group['phase'] == phase) & (group['time_sec'] >= start) & (group['time_sec'] < start + 5)]['vel_smooth'].max()
        day = group['Day'].iloc[0]  # assume same day for the fly
        if not pd.isna(peak_vel):
            anova_list.append({'Fly_ID': fly_id, 'Day': day, 'Trial': int(phase.split('_')[1]), 'Peak_Velocity': peak_vel})

df_anova = pd.DataFrame(anova_list)
print(f"Data points collected: {len(df_anova)}")
print(f"Per day: {df_anova.groupby('Day').size()}")
df_anova['Normalized_Spark'] = df_anova.apply(lambda x: x['Peak_Velocity'] / base_speeds.get(x['Fly_ID'], 1), axis=1)
df_anova['Exposure_Index'] = df_anova.apply(lambda x: x['Trial'] if x['Day'] == 1 else x['Trial'] + 5, axis=1)

# Plotting the Sensitization Curve
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_anova, x='Exposure_Index', y='Normalized_Spark', 
             hue='Day', palette=['#3498db', '#e74c3c'], marker='o', linewidth=3, errorbar=('ci', 68))

# Add 24h gap
plt.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
plt.text(5.6, plt.ylim()[1]*0.9, '24h Recovery', color='gray')

plt.title("Potentiated Locomotor Response: Evidence of Chronic Sensitization", fontsize=14)
plt.ylabel("Arousal Magnitude (Fold-change from Baseline)")
plt.xlabel("Cumulative Stimulus Exposures")
plt.xticks(range(1, 11))
plt.grid(axis='y', alpha=0.3)
plt.show()
print(df['Day'].unique())


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# Fixed color map for wells
colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#9C755F"]

for i, (well_id, sub) in enumerate(df.groupby('identity')):
    # Velocity Smoothing with Savitzky-Golay
    smoothed_vel = smooth_data(sub['velocity'])
    ax1.plot(sub['time_sec'], smoothed_vel, label=well_id, color=colors[i], alpha=0.8)
    
    # Grooming Smoothing
    smoothed_groom = smooth_data(sub['grooming_distance'])
    ax2.plot(sub['time_sec'], smoothed_groom, color=colors[i], alpha=0.8)

# Shading for the Pattern (Vortex ON periods)
for start, end in [(90,120), (180,210), (270,300), (360,390), (450,480)]:
    ax1.axvspan(start, end, color='red', alpha=0.1, label='Stimulus ON' if start==90 else "")
    ax2.axvspan(start, end, color='red', alpha=0.1)

ax1.set_title("Locomotor Sensitization (Filtered 120Hz Velocity)", fontsize=14)
ax1.set_ylabel("Velocity (px/s)")
ax2.set_title("Withdrawal Biomarker (Head-Thorax Distance Stability)", fontsize=14)
ax2.set_ylabel("Distance (px)")
ax2.set_xlabel("Time (seconds)")
ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig("Spatially_Corrected_Analysis.png", dpi=300)
plt.show()

# ==========================================
# 7. SENSITIZATION ANALYSIS (Day 1 vs Day 2)
# ==========================================

# 1. 建立基础基准：只取 Day 1 的第一个 Pre_Baseline 作为这只果蝇的“原始生命值”
# 这样 Day 2 的提升才是相对于 Day 1 的“进化”
initial_base_speeds = df[(df['phase'] == 'Pre_Baseline') & (df['Day'] == 1)].groupby('identity')['vel_smooth'].mean()

vortex_starts = {'Vortex_1': 90, 'Vortex_2': 180, 'Vortex_3': 270, 'Vortex_4': 360, 'Vortex_5': 450}
anova_list = []

# 核心修正：必须按照 [Day, identity] 分组，确保天数独立
for (day, well_id), group in df.groupby(['Day', 'identity']):
    
    # 离群点过滤：如果这只果蝇在这一天的平均速度极低（比如 < 2），说明它是“死掉的”或者“不动的”，排除它
    if group['vel_smooth'].mean() < 2:
        continue

    # 获取该孔位果蝇在 Day 1 的原始基准
    fly_initial_base = initial_base_speeds.get(well_id, 1)

    for phase, start in vortex_starts.items():
        # 提取刺激开始后 5 秒内的峰值 (Spark)
        # 加上微小的 120Hz 噪声过滤，防止读取到一个瞬时的跳点
        mask = (group['phase'] == phase) & (group['time_sec'] >= start) & (group['time_sec'] < start + 5)
        
        if mask.any():
            # 使用 95 分位数代替绝对最大值，可以更稳健地代表“爆发速度”，过滤掉极端的跳变噪点
            peak_vel = group[mask]['vel_smooth'].quantile(0.95) 
            
            trial_num = int(phase.split('_')[1])
            anova_list.append({
                'Fly_ID': well_id, 
                'Day': day, 
                'Trial': trial_num, 
                'Peak_Velocity': peak_vel,
                'Normalized_Spark': peak_vel / fly_initial_base
            })

df_anova = pd.DataFrame(anova_list)

# 2. 检查数据是否完整
print(f"✅ 处理完成！总计采集数据点: {len(df_anova)}")
print("每组数据点分布：\n", df_anova.groupby('Day').size())
print("Normalized Spark summary:\n", df_anova['Normalized_Spark'].describe())
print("Mean Normalized Spark per Day and Trial:\n", df_anova.groupby(['Day', 'Trial'])['Normalized_Spark'].mean())

# 3. 创建 Exposure Index (Day 1: 1-5, Day 2: 6-10)
df_anova['Exposure_Index'] = df_anova.apply(
    lambda x: x['Trial'] if x['Day'] == 1 else x['Trial'] + 5, axis=1
)

# 4. 绘制“敏化阶梯”图
plt.figure(figsize=(12, 7))

# 使用单个连线展示整体趋势，颜色区分天数
sns.lineplot(data=df_anova, x='Exposure_Index', y='Normalized_Spark', 
             hue='Day', palette=['#3498db', '#e74c3c'], 
             marker='o', markersize=10, linewidth=4, errorbar=('ci', 68))

# 绘制一条跨越两天的总趋势虚线（这是为了在视觉上强调上升趋势）
sns.regplot(data=df_anova, x='Exposure_Index', y='Normalized_Spark', 
            scatter=False, color='black', line_kws={"linestyle": "--", "alpha": 0.3})

# 添加装饰：24小时间隔线
plt.axvline(x=5.5, color='black', linestyle=':', alpha=0.5)
plt.text(5.6, df_anova['Normalized_Spark'].max()*0.85, '24h Incubation\nPeriod', 
         fontweight='bold', fontsize=12, color='darkred')

# 格式化坐标轴
plt.title("Figure 1: Potentiated Locomotor Response (Chronic Sensitization)", fontsize=16, pad=20)
plt.ylabel("Arousal Magnitude\n(Fold-change over Day 1 Baseline)", fontsize=13)
plt.xlabel("Cumulative Stimulus Exposures (Day 1: 1-5 | Day 2: 6-10)", fontsize=13)
plt.xticks(range(1, 11))
plt.ylim(bottom=0) # 从0开始，突出增长比例
plt.grid(axis='y', alpha=0.2)

# 移除不必要的边框
sns.despine()

plt.tight_layout()
plt.show()

# Figure 3: Neural Capture Heatmap (Gaze & Gaining)
# Circular (Polar) Histogram of Head-to-Body Angle relative to Well Center (to reduce noise from well corners)
well_centers = {
    "Top_Left": (185, 172.5),
    "Top_Mid": (540, 172.5),
    "Top_Right": (925, 172.5),
    "Bottom_Left": (185, 517.5),
    "Bottom_Mid": (540, 517.5),
    "Bottom_Right": (925, 517.5)
}

df['well_center_x'] = df['identity'].map(lambda x: well_centers[x][0])
df['well_center_y'] = df['identity'].map(lambda x: well_centers[x][1])

# 1. Vector A: from thorax to center (target vector)
v_target_x = df['well_center_x'] - df[tx]
v_target_y = df['well_center_y'] - df[ty]

# 2. Vector B: from thorax to head (fly's orientation vector)
v_fly_x = df[hx] - df[tx]
v_fly_y = df[hy] - df[ty]

# 3. Calculate the angle between the two vectors
angle_target = np.arctan2(v_target_y, v_target_x)
angle_fly = np.arctan2(v_fly_y, v_fly_x)

# Calculate relative angle and normalize to [-180, 180] degrees
relative_rad = angle_fly - angle_target
df['gaze_relative_angle'] = np.degrees(np.arctan2(np.sin(relative_rad), np.cos(relative_rad)))

# Only select frames with velocity greater than a threshold (e.g., 10 px/s) for plotting
active_df = df[df['vel_smooth'] > 10].copy()

# Filter out abnormal points with large single-frame displacement (e.g., more than 20 pixels in one frame)
# And filter out frames with low instance_score (if your CSV has this field)
clean_df = active_df[active_df['dist_px'] < 20].copy()

# Filter for stimulus phases
stimulus_data = clean_df[clean_df['phase'].str.contains('Vortex')]

# 5. 绘图时使用 'gaze_relative_angle'
# 预期结果：如果果蝇被“捕获”，你会看到 0°（正对中心）或 90°/-90°（完美沿切线旋转）有极大的峰值
# Polar histogram
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
bins = np.linspace(-180, 180, 36)  # 10-degree bins
counts, bin_edges = np.histogram(stimulus_data['gaze_relative_angle'], bins=bins)
bars = ax.bar(np.radians(bin_edges[:-1]), counts, width=np.radians(10), edgecolor='black', alpha=0.7)
ax.set_title("Figure 3: Neural Capture Heatmap Relative to Well Center", fontsize=14)
ax.set_thetamin(-180)
ax.set_thetamax(180)
ax.set_rlabel_position(0)
plt.show()

# Paired Slope Plot for Post-Stimulus Agitation (Withdrawal Biomarker)
per_fly_data = []

for fly_id, group in df.groupby('identity'):
    # Get Baseline mean (Pre_Baseline phase)
    base_val = group[group['phase'] == 'Pre_Baseline']['grooming_jitter'].mean()
    
    # Get Recovery mean (Recovery phase)
    recovery_val = group[group['phase'] == 'Recovery']['grooming_jitter'].mean()
    
    per_fly_data.append({'Fly_ID': fly_id, 'Condition': 'Baseline', 'Value': base_val})
    per_fly_data.append({'Fly_ID': fly_id, 'Condition': 'Recovery', 'Value': recovery_val})

plot_df = pd.DataFrame(per_fly_data)

# Plot paired slope plot
plt.figure(figsize=(6, 8))

# Boxplot as background
sns.boxplot(x='Condition', y='Value', data=plot_df, palette='Pastel1', width=0.4)

# Scatter points for each fly
sns.scatterplot(x='Condition', y='Value', data=plot_df, hue='Fly_ID', s=100, zorder=3)

# Connecting lines for each fly
for fly in plot_df['Fly_ID'].unique():
    subset = plot_df[plot_df['Fly_ID'] == fly]
    plt.plot(subset['Condition'], subset['Value'], color='gray', alpha=0.5, linestyle='--')

plt.title("Individual Shift: Baseline to Recovery")
plt.ylabel("Grooming Jitter (SD of Head-Thorax Distance)")
plt.show()

# 8. EXPORT
df['Day'] = 1  # Adjust this value for different days (e.g., 1, 2, 3)
df['Trial'] = df['phase'].apply(lambda p: int(p.split('_')[1]) if p.startswith('Vortex_') else 0)
df.to_csv("CLEANED_FINAL_ANALYSIS.csv", index=False)
print("\n--- ISEF QUANTITATIVE ANALYSIS ---")
print("Delta_Arousal = % increase in speed when vortex starts")
print(summary[['BASELINE', 'STIMULUS', 'Delta_Arousal']])
print("\n✅ Final analysis complete. Results saved to 'CLEANED_FINAL_ANALYSIS.csv'")

# ==========================================
# 8. PREPARE DATA FOR PAIRED T-TEST
# ==========================================

# 1. Filter for the two specific phases we want to compare
# Pre_Baseline: First 90s | Recovery: Final 60s
jitter_data = df[df['phase'].isin(['Pre_Baseline', 'Recovery'])].copy()

# 2. Calculate the average jitter for each individual fly in each phase
# We group by Day and Identity to ensure each fly in each file is treated as a unique subject
paired_jitter_summary = jitter_data.groupby(['Day', 'identity', 'phase'])['grooming_jitter'].mean().reset_index()

# 3. Pivot the table so each fly has one row with Baseline and Recovery columns
# This is the "Wide Format" required for a paired t-test
t_test_input = paired_jitter_summary.pivot_table(
    index=['Day', 'identity'], 
    columns='phase', 
    values='grooming_jitter'
).reset_index()

# 4. Clean up: Remove any flies that are missing data in either phase
t_test_input = t_test_input.dropna(subset=['Pre_Baseline', 'Recovery'])

# 5. Rename columns for clarity
t_test_input.columns.name = None
t_test_input = t_test_input.rename(columns={
    'Pre_Baseline': 'Jitter_Baseline',
    'Recovery': 'Jitter_Recovery'
})

# 6. Save the combined CSV
t_test_input.to_csv("Grooming_Jitter_Paired_Results.csv", index=False)

print("\n--- PAIRED T-TEST DATA PREPARATION ---")
print(f"✅ Successfully paired {len(t_test_input)} flies.")
print(t_test_input.head())
print("\n📁 File saved as 'Grooming_Jitter_Paired_Results.csv'")

# 7. Quick stats check (Python)
# This will give you an immediate result while you wait to use R
t_stat, p_val = stats.ttest_rel(t_test_input['Jitter_Baseline'], t_test_input['Jitter_Recovery'])
mean_diff = (t_test_input['Jitter_Recovery'].mean() - t_test_input['Jitter_Baseline'].mean()) / t_test_input['Jitter_Baseline'].mean() * 100

print(f"📊 Quick Results:")
print(f"   - p-value: {p_val:.6f}")
print(f"   - Mean change: {mean_diff:.2f}%")

# ==========================================
# 9. PREPARE DATA FOR RAYLEIGH TEST (R)
# ==========================================

print("📝 Exporting circular data for Rayleigh Test in R...")

# 1. Use the 'clean_df' we created earlier (Velocity > 10, jumps removed)
# 2. Filter specifically for the stimulus (Vortex) phases
rayleigh_input = clean_df[clean_df['phase'].str.contains('Vortex', na=False)].copy()

# 3. Select only the columns needed for the statistical test
# We include Day and Identity so you can run the test for specific groups if needed
rayleigh_export = rayleigh_input[['Day', 'identity', 'phase', 'gaze_relative_angle']]

# 4. Remove any potential NaNs in the angle column
rayleigh_export = rayleigh_export.dropna(subset=['gaze_relative_angle'])

# 5. Save the CSV
rayleigh_export.to_csv("NEURAL_CAPTURE_RAYLEIGH.csv", index=False)

print(f"✅ Success! Exported {len(rayleigh_export)} orientation samples.")
print(f"📁 File saved as: 'NEURAL_CAPTURE_RAYLEIGH.csv'")

# --- Summary for Stats Teacher ---
# Sample Size (N frames): len(rayleigh_export)
# Mean Angle: np.mean(rayleigh_export['gaze_relative_angle'])
# Standard Deviation: np.std(rayleigh_export['gaze_relative_angle'])