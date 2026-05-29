import h5py
import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


H5_PATH = "/Volumes/Quasar/Carmen/flies_data_2026/derivedCoor/0308_Group3_final.h5"
OUT_PATH = "X_tensor_group3.npz"

FPS = 120
WIN = 120          # 1.0秒窗口
STRIDE = 30        # 步长
SCORE_THR = 0.5    # 置信度阈值

# 你的 6 个圆心坐标
WELL_CENTERS = np.array([
    [439.4, 460.7],[698.6, 475.2], [959.7, 466.7], 
[420.6, 217.8], [699.4, 202.0], [951.1, 202.0] # Bottom: 3, 4, 5
])
WELL_RADIUS = 220  # 根据你的坐标间距，半径设为 220 像素比较合适

# ==========================================
# 2. 读取并初步清洗
# ==========================================
print("🚀 Loading H5 and Cleaning Data...")
with h5py.File(H5_PATH, "r") as f:
    # 你的维度: (469, 2, 3, 199766) -> (ID, XY, Node, Frame)
    tracks = f["tracks"][:].astype(np.float32)
    scores = f["point_scores"][:].astype(np.float32)

n_ids, _, n_nodes, n_frames = tracks.shape

# 置信度过滤
low_conf = scores < SCORE_THR
for d in range(2):
    tracks[:, d, :, :][low_conf] = np.nan

# ==========================================
# 3. 空间身份锁定 (Spatial Re-ID)
# ==========================================
print("📍 Re-indexing 469 tracks into 6 real identities...")

# 存储每一帧每个格子的“最佳 ID”
# best_id_map[frame, well_id]
best_id_map = np.full((n_frames, 6), -1, dtype=np.int16)

# 重点考察 Thorax (Index 1) 的位置
thorax_x = tracks[:, 0, 1, :] # (ID, Frame)
thorax_y = tracks[:, 1, 1, :] # (ID, Frame)
thorax_s = scores[:, 1, :]    # (ID, Frame)

for w_id in range(6):
    center = WELL_CENTERS[w_id]
    # 计算所有 ID 在所有帧到该圆心的距离
    # 利用 NumPy 广播加速计算
    dist = np.sqrt((thorax_x - center[0])**2 + (thorax_y - center[1])**2) # (ID, Frame)
    
    # 判定哪些 ID 属于这个圆圈
    is_in_well = dist < WELL_RADIUS
    
    for f in range(n_frames):
        candidates = np.where(is_in_well[:, f])[0]
        if len(candidates) > 0:
            # 如果有多个 ID 都在这个圆圈里，选置信度最高的那个
            best_id_map[f, w_id] = candidates[np.argmax(thorax_s[candidates, f])]

# ==========================================
# 4. 特征计算 (Vectorized)
# ==========================================
def process_well(w_id):
    # 提取该格子随时间连续的轨迹
    f_idx = np.arange(n_frames)
    t_idx = best_id_map[:, w_id]
    valid = t_idx >= 0
    
    # 获取坐标序列 (frames, nodes, 2)
    pos = np.full((n_frames, 3, 2), np.nan)
    for f in np.where(valid)[0]:
        pos[f] = tracks[t_idx[f], :, :, f].T
        
    # 插值填补短时间缺失 (利于 VAE 学习)
    for n in range(3):
        for d in range(2):
            y = pos[:, n, d]
            v = ~np.isnan(y)
            if v.sum() > 10:
                f_int = interp1d(f_idx[v], y[v], kind='linear', fill_value="extrapolate")
                pos[:, n, d] = f_int(f_idx)
    
    thx, hed, abd = pos[:, 1], pos[:, 0], pos[:, 2]
    
    # 特征 1: 速度
    vel = np.concatenate([[0], np.linalg.norm(np.diff(thx, axis=0), axis=1)])
    
    # 特征 2 & 3: 自中心旋转与姿态 (Head-up Alignment)
    vec_h = hed - thx
    angle_h = np.arctan2(vec_h[:, 1], vec_h[:, 0])
    cos_a, sin_a = np.cos(-angle_h), np.sin(-angle_h)
    
    # 旋转腹部坐标
    rel_abd = abd - thx
    abd_ego_y = rel_abd[:, 0] * sin_a + rel_abd[:, 1] * cos_a
    abd_ego_x = rel_abd[:, 0] * cos_a - rel_abd[:, 1] * sin_a
    spine_angle = np.arctan2(abd_ego_y, abd_ego_x)
    ang_speed = np.concatenate([[0], np.diff(angle_h)])
    
    # 特征 4: 趋壁性 (到圆心的归一化距离)
    dist_to_c = np.sqrt((thx[:, 0]-WELL_CENTERS[w_id,0])**2 + (thx[:, 1]-WELL_CENTERS[w_id,1])**2)
    thigmo = np.clip(dist_to_c / WELL_RADIUS, 0, 1)
    
    # 特征 5: Jitter (微小位移波动)
    body_len = np.linalg.norm(hed - thx, axis=1)
    ms = uniform_filter1d(body_len**2, size=15)
    sm = uniform_filter1d(body_len, size=15)**2
    jitter = np.sqrt(np.maximum(ms - sm, 1e-10))
    jitter[vel > np.percentile(vel, 70)] = 0 # 剔除运动时的伪震颤
    
    def smooth(v): return savgol_filter(np.nan_to_num(v), 7, 3)
    return np.stack([smooth(vel), smooth(spine_angle), smooth(ang_speed), smooth(thigmo), smooth(jitter)], axis=0), valid

# ==========================================
# 5. 窗口切分与保存
# ==========================================
all_windows = []
all_meta = []

print("🪟 Slicing and validating windows...")
for w_id in range(6):
    feats, valid_mask = process_well(w_id)
    for start in range(0, n_frames - WIN, STRIDE):
        if np.mean(valid_mask[start:start+WIN]) > 0.7: # 窗口内 70% 数据真实即可
            all_windows.append(feats[:, start:start+WIN])
            all_meta.append([w_id, start])

X = np.array(all_windows, dtype=np.float32)
meta = np.array(all_meta, dtype=np.int32)

np.savez_compressed(OUT_PATH, X=X, meta=meta)
print(f"✅ DONE! Saved {X.shape[0]} windows to {OUT_PATH}")