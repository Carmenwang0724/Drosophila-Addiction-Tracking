import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

# configuration
h5_path = "/Volumes/Quasar/Carmen/flies_data_2026/derivedCoor/0308_Group1_final.h5"
out_path = "x_tensor_v2.npz"

fps = 120
win = 120
stride = 30
conf_thr = 0.5
well_radius = 220

well_centers = np.array([
    [668, 463], [1165, 427], [1681, 405],
    [701, 951], [1185, 946], [1712, 914]
])

# load tracks and scores
print("loading data from h5...")
with h5py.File(h5_path, "r") as f:
    tracks = f["tracks"][:].astype(np.float32)
    scores = f["point_scores"][:].astype(np.float32)

n_ids, _, n_nodes, n_frames = tracks.shape
tx, ty = tracks[:, 0, 1, :], tracks[:, 1, 1, :]
ts = scores[:, 1, :]

# spatial re-indexing: map fragments to 6 wells
print("re-indexing fragments to wells...")
best_id_map = np.full((n_frames, 6), -1, dtype=np.int16)

for w_id in range(6):
    cx, cy = well_centers[w_id]
    dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)
    in_well = dist < well_radius
    
    # filter by confidence and spatial proximity
    masked_scores = np.where(in_well, ts, -1.0)
    ids = np.argmax(masked_scores, axis=0)
    ids[np.max(masked_scores, axis=0) == -1] = -1
    best_id_map[:, w_id] = ids

def extract_features(w_id):
    # build continuous trajectory for a single well
    t_idx = best_id_map[:, w_id]
    valid = t_idx >= 0
    pos = np.full((n_frames, 3, 2), np.nan)
    
    for f_idx in np.where(valid)[0]:
        pos[f_idx] = tracks[t_idx[f_idx], :, :, f_idx].T

    # linear interpolation for gaps
    f_axis = np.arange(n_frames)
    for n in range(3):
        for d in range(2):
            y = pos[:, n, d]
            mask = ~np.isnan(y)
            if mask.sum() > 10:
                pos[:, n, d] = interp1d(f_axis[mask], y[mask], kind='linear', fill_value="extrapolate")(f_axis)

    head, thx, abd = pos[:, 0], pos[:, 1], pos[:, 2]

    # f1: walking velocity
    vel = np.concatenate([[0], np.linalg.norm(np.diff(thx, axis=0), axis=1)])

    # f2: angular speed (unwrapped to prevent 2pi jumps)
    vec_h = head - thx
    angle_h = np.arctan2(vec_h[:, 1], vec_h[:, 0])
    ang_vel = np.concatenate([[0], np.diff(np.unwrap(angle_h))])

    # f3: spine angle (egocentric)
    cos_a, sin_a = np.cos(-angle_h), np.sin(-angle_h)
    rel_abd = abd - thx
    ax = rel_abd[:, 0] * cos_a - rel_abd[:, 1] * sin_a
    ay = rel_abd[:, 0] * sin_a + rel_abd[:, 1] * cos_a
    spine_ang = np.arctan2(ay, ax)

    # f4: thigmotaxis (normalized distance to center)
    dist_c = np.sqrt((thx[:, 0] - well_centers[w_id, 0])**2 + (thx[:, 1] - well_centers[w_id, 1])**2)
    thigmo = np.clip(dist_c / well_radius, 0, 1)

    # f5: jitter (local body length variance)
    blen = np.linalg.norm(head - thx, axis=1)
    ms = uniform_filter1d(blen**2, size=15)
    sm = uniform_filter1d(blen, size=15)**2
    jitter = np.sqrt(np.maximum(ms - sm, 1e-10))
    jitter[vel > np.percentile(vel[~np.isnan(vel)], 70)] = 0

    def smooth(v): return savgol_filter(np.nan_to_num(v), 7, 3)
    data = np.stack([smooth(vel), smooth(spine_ang), smooth(ang_vel), smooth(thigmo), smooth(jitter)])
    return data, valid

# processing and window slicing
all_x, all_meta = [], []
print("slicing windows...")

for w_id in range(6):
    feats, v_mask = extract_features(w_id)
    for start in range(0, n_frames - win, stride):
        # require 70% valid frames in window
        if np.mean(v_mask[start:start+win]) > 0.7:
            all_x.append(feats[:, start:start+win])
            all_meta.append([w_id, start])

x = np.array(all_x, dtype=np.float32)
meta = np.array(all_meta, dtype=np.int32)

np.savez_compressed(out_path, x=x, meta=meta)
print(f"process complete. saved {x.shape[0]} windows to {out_path}")