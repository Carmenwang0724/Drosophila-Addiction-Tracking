import pandas as pd
import glob
import os

# 1. PATH TO YOUR CSVs
# Make sure all your generated CSVs are in one folder
path = "/Users/Admin/Drosphilia SIP/DownloadedCSV/" 
all_files = glob.glob(os.path.join(path, "*.csv"))

full_list = []

for filename in all_files:
    temp_df = pd.read_csv(filename, index_col=None, header=0)
    
    # 自动识别天数 (根据你的文件名逻辑修改)
    if '4-1' in filename or '1' in filename:
        temp_df['Day'] = 1
    elif '4-2' in filename or '2' in filename:
        temp_df['Day'] = 2
    else:
        temp_df['Day'] = 0  # or skip
    
    full_list.append(temp_df)

# 核心步骤：把所有天的 CSV 纵向合并成一个大表
df = pd.concat(full_list, ignore_index=True)

# 如果是原始 CSV，没有 time_sec，则从 frame 计算
if 'time_sec' not in df.columns:
    if 'frame' in df.columns:
        df['time_sec'] = df['frame'] / 120  # 假设 FPS=120
    else:
        raise ValueError("No 'time_sec' or 'frame' column found in CSVs")

# 打印检查：这里必须看到 Day 1 和 Day 2 都有数据
print("Detected Days in dataset:", df['Day'].unique())
print("Total rows loaded:", len(df))

# 4. ASSIGN TRIAL NUMBERS (Epoch Mapping)
def get_trial_num(t):
    if 90 <= t <= 120: return 1
    if 180 <= t <= 210: return 2
    if 270 <= t <= 300: return 3
    if 360 <= t <= 390: return 4
    if 450 <= t <= 480: return 5
    return 0 # 0 means Baseline/Recovery

df['Trial'] = df['time_sec'].apply(get_trial_num)

# 5. EXPORT FOR ANOVA
df.to_csv("MASTER_LONGITUDINAL_DATA.csv", index=False)
print("✅ Master File Created: MASTER_LONGITUDINAL_DATA.csv")