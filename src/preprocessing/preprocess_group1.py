import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# ══════════════════════════════════════════════════════════════════════════════
# Step 0: 读取数据 + Confidence Gating（修复一）
# ══════════════════════════════════════════════════════════════════════════════
SCORE_THRESHOLD = 0.5

with h5py.File("/Volumes/Quasar/Carmen/flies_data_2026/derivedCoor/0308_Group1_final.h5", "r") as f:
    print("H5 keys:", list(f.keys()))
    tracks       = f["tracks"][:].T                   # (199766, 3, 2, 469)
    node_names   = [n.decode() for n in f["node_names"][:]]
    point_scores = f["point_scores"][:].T             # (199766, 3, 469)  ← 必须在这里读

print("tracks shape:",       tracks.shape)
print("point_scores shape:", point_scores.shape)

tracks_clean = tracks.copy().astype(np.float32)

low_conf = point_scores < SCORE_THRESHOLD

# 删掉 mask 相关的 broadcast_to 逻辑，直接对第2维度(x和y坐标)分别切片赋值
tracks_clean[:, :, 0, :][low_conf] = np.nan
tracks_clean[:, :, 1, :][low_conf] = np.nan

print(f"Points gated out: {low_conf.sum():,} / {low_conf.size:,}  "
      f"({100*low_conf.mean():.1f}%)")

print(f"Points gated out: {low_conf.sum():,} / {low_conf.size:,}  "
      f"({100*low_conf.mean():.1f}%)")
    

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 插值填 NaN（必须在 gating 之后）
# ══════════════════════════════════════════════════════════════════════════════
frames = tracks_clean.shape[0]
idx  = np.arange(frames)
flat = tracks_clean.reshape(frames, -1)

print(f"NaN before interpolation: {np.isnan(flat).sum()}")
for col in range(flat.shape[1]):
    col_data = flat[:, col]
    valid = ~np.isnan(col_data)
    if valid.sum() < 2:
        flat[:, col] = 0.0
        continue
    if not valid.all():
        f_interp = interp1d(idx[valid], col_data[valid], kind="linear",
                            bounds_error=False,
                            fill_value=(col_data[valid][0], col_data[valid][-1]))
        flat[:, col] = f_interp(idx)

tracks_clean = flat.reshape(tracks_clean.shape)
print(f"NaN after interpolation:  {np.isnan(tracks_clean).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 提取节点原始坐标（Arena坐标系，用于 Thigmotaxis）
# ══════════════════════════════════════════════════════════════════════════════
head    = tracks_clean[:, 0, :, :]   # (frames, 2, flies)  — Arena坐标
thorax  = tracks_clean[:, 1, :, :]
abdomen = tracks_clean[:, 2, :, :]

# ── Thigmotaxis（修复二）────────────────────────────────────────────────────
# 用 thorax 在 Arena 中的位置计算离墙距离
# 假设 arena 是圆形，先估算圆心和半径
arena_center_x = np.nanmedian(thorax[:, 0, :])   # 用所有数据的中位数估算圆心
arena_center_y = np.nanmedian(thorax[:, 1, :])
arena_radius   = np.percentile(
    np.sqrt((thorax[:, 0, :] - arena_center_x)**2 +
            (thorax[:, 1, :] - arena_center_y)**2),
    95                                            # 95th percentile 作为半径估算
)
print(f"Arena center: ({arena_center_x:.1f}, {arena_center_y:.1f}), radius: {arena_radius:.1f} px")

dist_from_center = np.sqrt(
    (thorax[:, 0, :] - arena_center_x)**2 +
    (thorax[:, 1, :] - arena_center_y)**2
)                                                 # (frames, flies)

# 归一化到 [0,1]：0=中心，1=紧贴墙壁
thigmotaxis = dist_from_center / arena_radius
thigmotaxis = np.clip(thigmotaxis, 0, 1)         # (frames, flies)

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 自我中心坐标系 Ego-centric（修复三）
# ══════════════════════════════════════════════════════════════════════════════
# 目标：胸部 → (0,0)，头部 → (0, +y)，消除位置和朝向的影响

N_FLIES = tracks_clean.shape[3]

# 平移：所有节点减去 thorax
head_c    = head    - thorax      # (frames, 2, flies)
abdomen_c = abdomen - thorax      # 胸部现在是原点

# 旋转：计算胸→头的角度，旋转使头部落在 +y 轴
angle_to_head = np.arctan2(head_c[:, 0, :], head_c[:, 1, :])  # (frames, flies)

cos_a = np.cos(-angle_to_head)   # (frames, flies)
sin_a = np.sin(-angle_to_head)

def rotate_coords(xy, cos_a, sin_a):
    """
    xy:    (frames, 2, flies)
    cos_a: (frames, flies)
    sin_a: (frames, flies)
    """
    x_rot = xy[:, 0, :] * cos_a - xy[:, 1, :] * sin_a
    y_rot = xy[:, 0, :] * sin_a + xy[:, 1, :] * cos_a
    return np.stack([x_rot, y_rot], axis=1)      # (frames, 2, flies)

head_ego    = rotate_coords(head_c,    cos_a, sin_a)  # 应接近 (0, +y)
abdomen_ego = rotate_coords(abdomen_c, cos_a, sin_a)  # 自我中心的腹部位置

# 验证：head_ego 的 x 分量应该接近 0
print(f"Ego-centric sanity: head_x mean={head_ego[:,0,:].mean():.4f} (should be ~0)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 运动学特征（全部在 Ego-centric 坐标系下）
# ══════════════════════════════════════════════════════════════════════════════
# Velocity：用 Arena 坐标的 thorax（位移是全局的，ego-centric 下位移为0）
velocity    = np.linalg.norm(np.diff(thorax, axis=0), axis=1)
velocity    = np.vstack([velocity[[0]], velocity])          # (frames, flies)

# Spine Angle：ego-centric 下腹部的角度 = 真实弯曲姿态
spine_angles = np.arctan2(abdomen_ego[:, 1, :], abdomen_ego[:, 0, :])

# Angular Speed：头部朝向的变化率（Arena坐标下，因为是朝向变化）
head_angles_arena = np.arctan2(head_c[:, 1, :], head_c[:, 0, :])
ang_speed = np.diff(head_angles_arena, axis=0)
ang_speed = np.vstack([ang_speed[[0]], ang_speed])

# ── Postural Jitter（修复四）────────────────────────────────────────────────
# 不再用速度方差，改用：速度接近0时，节点间距的局部方差
dist_head_thorax    = np.linalg.norm(head_ego,    axis=1)   # (frames, flies)
dist_abd_thorax     = np.linalg.norm(abdomen_ego, axis=1)

from numpy.lib.stride_tricks import sliding_window_view
WIN_JITTER = 15

# 只在低速窗口内计算姿态抖动
low_velocity_mask = velocity < np.percentile(velocity, 30)  # 最安静的30%帧

dist_combined = (dist_head_thorax + dist_abd_thorax) / 2    # (frames, flies)

# 对低速帧，计算节点间距的局部方差
dist_for_jitter = dist_combined * low_velocity_mask.astype(float)
from scipy.ndimage import uniform_filter1d

# 确保转换为 float32（原代码的 .astype(float) 默认是 float64，会使内存翻倍）
dist_for_jitter = dist_combined * low_velocity_mask.astype(np.float32)


# 利用公式 Var(X) = E(X^2) - E(X)^2 计算局部方差
# mode='nearest' 会自动等效处理你原代码中 np.pad 的 mode="edge" 边缘逻辑
mean_sq = uniform_filter1d(dist_for_jitter**2, size=WIN_JITTER, axis=0, mode='nearest')
sq_mean = uniform_filter1d(dist_for_jitter, size=WIN_JITTER, axis=0, mode='nearest')**2

jitter = mean_sq - sq_mean
jitter = np.maximum(jitter, 0)  # 防止浮点精度误差产生微小负数

# 现在直接输出了正确的 (frames, flies) 形状，连 np.pad 都不需要写了

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: 平滑
# ══════════════════════════════════════════════════════════════════════════════
print("Smoothing...")
velocity_s   = savgol_filter(velocity,     7, polyorder=3, axis=0)
spine_s      = savgol_filter(spine_angles, 7, polyorder=3, axis=0)
ang_speed_s  = savgol_filter(ang_speed,    7, polyorder=3, axis=0)
thigmotaxis_s= savgol_filter(thigmotaxis,  7, polyorder=3, axis=0)
# jitter 本身已经是窗口统计，不再平滑

# ══════════════════════════════════════════════════════════════════════════════
# Step 6: 存到硬盘（5 个特征，含 Thigmotaxis）
# ══════════════════════════════════════════════════════════════════════════════
np.save("velocity_s.npy",    velocity_s.astype(np.float32))
np.save("spine_s.npy",       spine_s.astype(np.float32))
np.save("ang_speed_s.npy",   ang_speed_s.astype(np.float32))
np.save("jitter.npy",        jitter.astype(np.float32))
np.save("thigmotaxis_s.npy", thigmotaxis_s.astype(np.float32))
print("Done! 5 files saved.")
print(f"  thigmotaxis range: [{thigmotaxis_s.min():.3f}, {thigmotaxis_s.max():.3f}]")
