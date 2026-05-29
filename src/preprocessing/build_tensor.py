import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# 从硬盘加载，跳过所有慢步骤
velocity_s    = np.load("velocity_s.npy")
spine_s       = np.load("spine_s.npy")
ang_speed_s   = np.load("ang_speed_s.npy")
jitter        = np.load("jitter.npy")
thigmotaxis_s = np.load("thigmotaxis_s.npy")
print("Loaded from disk!")

def zscore(arr):
    return (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-8)

vel_z   = zscore(velocity_s)
spine_z = zscore(spine_s)
ang_z   = zscore(ang_speed_s)
jit_z   = zscore(jitter)
thig_z  = zscore(thigmotaxis_s)

bad_frames = (np.abs(spine_z) > 5).any(axis=1)
print(f"Bad frames: {bad_frames.sum()}")

# 5 通道
X_all   = np.stack([vel_z, spine_z, ang_z, jit_z, thig_z], axis=1)  # (frames, 5, 469)
WINDOW  = 30
STEP    = 15
N_FLIES = X_all.shape[2]

all_windows = []
for fly_i in range(N_FLIES):
    fly_data     = X_all[:, :, fly_i]
    starts       = np.arange(0, len(fly_data) - WINDOW, STEP)
    valid_starts = [s for s in starts if not bad_frames[s:s+WINDOW].any()]
    if len(valid_starts) == 0:
        continue
    wins = np.stack([fly_data[s:s+WINDOW].T for s in valid_starts])
    all_windows.append(wins)

X_tensor = np.concatenate(all_windows, axis=0).astype(np.float32)

X_tensor = X_tensor - X_tensor.mean()
X_tensor = X_tensor / (X_tensor.std() + 1e-8)

print(f"After global renorm — mean: {X_tensor.mean():.4f}, std: {X_tensor.std():.4f}")
np.save("X_tensor_group2.npy", X_tensor)

print(f"Final tensor: {X_tensor.shape}")   # (N, 5, 30)
print(f"mean: {X_tensor.mean():.4f}, std: {X_tensor.std():.4f}")
