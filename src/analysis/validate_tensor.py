import matplotlib.pyplot as plt
import numpy as np

# 加载你刚生成的 meta 和 X
data = np.load("X_tensor_group3.npz")
X, meta = data["X"], data["meta"]

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for w_id in range(6):
    # 找到属于该格子的所有窗口
    idx = np.where(meta[:, 0] == w_id)[0]
    # 取每个窗口的第一帧坐标（假设你在生成 X 时保留了原始坐标，或者直接用特征里的 thigmo 反推）
    # 这里我们简单地画出该格子的样本分布数量
    plt.bar(f"Well {w_id}", len(idx), color=colors[w_id])

plt.title("Sample Count per Well (Should be roughly equal)")
plt.ylabel("Number of 1.0s Windows")
plt.show()

# 选 Well 2 的前 100 个连续窗口
well_idx = np.where(meta[:, 0] == 2)[0][:100]
sample_vel = X[well_idx, 0, 0] # 取每个窗口的第一帧速度

plt.figure(figsize=(15, 4))
plt.plot(sample_vel)
plt.title("Velocity Trace (Well 2) - Check for sudden 'teleportation' spikes")
plt.show()