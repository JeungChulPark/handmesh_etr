"""Isolate why DexYCB depth_med tracks root_z poorly (r=0.63, resid 123mm).

Three hypotheses, measured on raw frames (no dataloader corruption unless asked):
  H1 object contamination -> compare hand-masked median vs full-bbox median.
  H2 train-time hole/dropout corruption -> compare clean vs corrupted median.
  H3 wrist-vs-surface offset only (benign) -> residual of clean masked median.
"""
import numpy as np
import cv2
from datasets.dexycb import DexYCB_RGBD, HAND_SEG_ID, JOINT_PERM, get_2D_annotation
from datasets.depth_synth import add_sensor_noise

HD = "/home/jucpark/DeepLearning/Datasets/Hand Dataset"
N = 1200

ds = DexYCB_RGBD(root=f"{HD}/DexYCB_full/data", mode="train", with_depth=True,
                 flip_left=True, return_abs_depth=True)
dc = ds.depth_cfg
rng = np.random.default_rng(0)

def sample_at(depth, uv, r=6):
    """robust depth at a 2D pixel: median of valid depth in a (2r+1) window."""
    H, W = depth.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < W and 0 <= v < H):
        return 0.0
    p = depth[max(0, v-r):v+r+1, max(0, u-r):u+r+1]
    pv = p[p > 0]
    return float(np.median(pv)) if pv.size else 0.0


# palm joints (wrist + MCPs) for a central, low-variance depth estimate
PALM = [0, 5, 9, 13, 17]
rootz, m_hand, m_box, m_corr, m_wrist, m_palm = [], [], [], [], [], []
valid_counts = []
empty_hand = 0
step = max(1, len(ds.samples) // N)
for s in ds.samples[::step][:N]:
    lab = np.load(s["label"])
    j3 = lab["joint_3d"][0].astype(np.float32)
    if np.any(j3 <= -1.0 + 1e-6) and (j3 < 0).all(axis=1).any():
        continue
    j3 = ds._to_metres(j3)[JOINT_PERM]
    K = s["K"].copy()
    d = cv2.imread(s["depth"], cv2.IMREAD_UNCHANGED)
    if d is None:
        continue
    depth = d.astype(np.float32) / 1000.0
    seg = lab["seg"]
    if s["side"] == "left":               # flip_left convention
        j3[:, 0] *= -1.0; K[0, 2] = depth.shape[1] - 1 - K[0, 2]
        depth = depth[:, ::-1].copy(); seg = seg[:, ::-1].copy()
    j2 = get_2D_annotation(j3, K)         # (21,2) px

    hand = (seg == HAND_SEG_ID) & (depth > 0)
    if hand.sum() < 50:
        empty_hand += 1
        continue
    # full-bbox median (object + arm + bg INCLUDED) over the hand bbox
    x0, y0 = j2.min(0); x1, y1 = j2.max(0)
    pad = 0.3 * max(x1 - x0, y1 - y0)
    xs, ys = slice(int(max(0, x0-pad)), int(x1+pad)), slice(int(max(0, y0-pad)), int(y1+pad))
    box = depth[ys, xs]; boxv = box[box > 0]
    # train-corrupted hand-masked depth (holes+dropout, like the loader)
    dmask = np.where(hand, depth, 0.0).astype(np.float32)
    dcorr = add_sensor_noise(dmask, sigma=dc["sigma"], dropout_p=dc["dropout_p"],
                             quant=dc["quant"], n_holes=dc["n_holes"],
                             hole_frac=dc["hole_frac"], rng=rng)
    cv = dcorr > 0

    rootz.append(float(j3[0, 2]))
    m_hand.append(float(np.median(depth[hand])))
    m_box.append(float(np.median(boxv)) if boxv.size else 0.0)
    m_corr.append(float(np.median(dcorr[cv])) if cv.any() else 0.0)
    valid_counts.append(int(hand.sum()))
    dmask_only = np.where(hand, depth, 0.0)
    m_wrist.append(sample_at(dmask_only, j2[0]))
    m_palm.append(float(np.median([v for v in (sample_at(dmask_only, j2[p]) for p in PALM) if v > 0]) if
                  any(sample_at(dmask_only, j2[p]) > 0 for p in PALM) else 0.0))

rootz = np.array(rootz); m_hand = np.array(m_hand)
m_box = np.array(m_box); m_corr = np.array(m_corr)
m_wrist = np.array(m_wrist); m_palm = np.array(m_palm)


def rel(med, z, tag):
    v = med > 0
    r = np.corrcoef(med[v], z[v])[0, 1]
    a, b = np.polyfit(med[v], z[v], 1)
    resid = (z[v] - (a*med[v]+b)).std()*1000
    off = (z[v]-med[v]).mean()*1000
    print(f"  {tag:26s} r={r:5.2f}  off(z-med)={off:+7.1f}mm  resid={resid:6.1f}mm  "
          f"a={a:4.2f}  med=0: {100*(~v).mean():4.1f}%")


print(f"DexYCB depth_med vs root_z  (N={len(rootz)}, empty-hand-mask skipped={empty_hand})")
rel(m_hand, rootz, "H3 clean hand-masked")
rel(m_box,  rootz, "H1 full-bbox (unmasked)")
rel(m_corr, rootz, "H2 train-corrupted masked")
rel(m_wrist, rootz, "H4 depth @ wrist px")
rel(m_palm,  rootz, "H4 depth @ palm(MCPs)")
print(f"  contamination |box-hand| median = {np.median(np.abs(m_box-m_hand))*1000:.1f} mm")
# robust: correlation on frames where depth@wrist is within a plausible band of root_z
v = m_wrist > 0
d = np.abs(m_wrist[v] - rootz[v])
for thr in (0.08, 0.05):
    keep = d < thr
    zz, ww = rootz[v][keep], m_wrist[v][keep]
    r = np.corrcoef(ww, zz)[0, 1]
    resid = (zz - np.polyval(np.polyfit(ww, zz, 1), ww)).std() * 1000
    print(f"  outlier-trimmed (|wrist-z|<{thr*1000:.0f}mm): kept {100*keep.mean():.0f}%  "
          f"r={r:.2f}  resid={resid:.1f}mm")
diff = (m_wrist - rootz)[m_wrist > 0] * 1000
print(f"  (depth@wrist - root_z) mm  p5/25/50/75/95 = "
      f"{np.percentile(diff,[5,25,50,75,95]).round(0)}")
print(f"  valid hand-depth px count  p5/50/95 = {np.percentile(valid_counts,[5,50,95]).astype(int)}")
