import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Load the master data in chunks
chunk_size = 100000
chunks = pd.read_csv('MASTER_LONGITUDINAL_DATA.csv', chunksize=chunk_size)

df_list = []
for chunk in chunks:
    df_list.append(chunk)
df = pd.concat(df_list, ignore_index=True)

print("Data loaded, shape:", df.shape)

# Settings
FPS = 120
DT = 1 / FPS

# Columns
tx, ty = 'thorax_x', 'thorax_y'
hx, hy = 'head_x', 'head_y'

# Assign conditions
def assign_condition_detailed(t):
    if 90 <= t <= 120: return 'Vortex_1'
    if 180 <= t <= 210: return 'Vortex_2'
    if 270 <= t <= 300: return 'Vortex_3'
    if 360 <= t <= 390: return 'Vortex_4'
    if 450 <= t <= 480: return 'Vortex_5'
    if 0 <= t < 90: return 'Pre_Baseline'
    return 'Recovery'

df['phase'] = df['time_sec'].apply(assign_condition_detailed)

# Calculations
df = df.sort_values(['identity', 'time_sec'])

# Distance moved
df['dist_px'] = np.sqrt(df.groupby('identity')[tx].diff()**2 + df.groupby('identity')[ty].diff()**2)
df.loc[df['dist_px'] > 40, 'dist_px'] = 0
df['velocity'] = df['dist_px'] / DT

# Smoothing
def smooth_data(data):
    if len(data) < 21: return data
    return savgol_filter(data, 15, 3)

df['vel_smooth'] = df.groupby('identity')['velocity'].transform(smooth_data)

# Extract trial data
initial_base_speeds = df[(df['phase'] == 'Pre_Baseline') & (df['Day'] == 1)].groupby('identity')['vel_smooth'].mean()

vortex_starts = {'Vortex_1': 90, 'Vortex_2': 180, 'Vortex_3': 270, 'Vortex_4': 360, 'Vortex_5': 450}
anova_list = []

for (day, well_id), group in df.groupby(['Day', 'identity']):
    if group['vel_smooth'].mean() < 2:
        continue
    fly_initial_base = initial_base_speeds.get(well_id, 1)
    for phase, start in vortex_starts.items():
        mask = (group['phase'] == phase) & (group['time_sec'] >= start) & (group['time_sec'] < start + 5)
        if mask.any():
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
df_anova['Exposure_Index'] = df_anova.apply(lambda x: x['Trial'] if x['Day'] == 1 else x['Trial'] + 5, axis=1)

print(df_anova.to_csv(index=False))