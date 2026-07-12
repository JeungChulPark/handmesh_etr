"""Diagnostic: per-dataset distributions of GT root (x,y,z) and depth_med, and how
well depth_med tracks root_z (the geometric-anchor assumption for metric MPJPE).

For each dataset we report root_z / depth_med ranges (units/convention sanity) and
the root_z<-depth_med relation: Pearson r, mean offset (root_z - depth_med), and
the residual std after a best-fit root_z = a*depth_med + b. A high r + small
residual means root_z can be *read* from depth_med (anchor will generalize); a low
r or huge per-dataset offset means the current free-MLP head has to memorize it.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

HD = "/home/jucpark/DeepLearning/Datasets/Hand Dataset"


def build(name, n):
    if name == "hanco":
        from datasets.hanco_ty import HanCo_ETRI_jitter
        ds = HanCo_ETRI_jitter(limit=5e6, with_depth=True, depth_source="render",
                               return_abs_depth=True)
    elif name == "dexycb":
        from datasets.dexycb import DexYCB_RGBD
        ds = DexYCB_RGBD(root=f"{HD}/DexYCB_full/data", mode="train", with_depth=True,
                         flip_left=True, return_abs_depth=True)
    elif name == "ho3d":
        from datasets.ho3d import HO3D_RGBD
        ds = HO3D_RGBD(root=f"{HD}/HO3D_full", mode="train", with_depth=True,
                       return_abs_depth=True)
    torch.manual_seed(0)
    n_train = int(len(ds) * 0.95)
    _, test = random_split(ds, [n_train, len(ds) - n_train])
    return DataLoader(test, batch_size=64, shuffle=False, num_workers=8), n


def collect(name, n):
    loader, _ = build(name, n)
    R, M = [], []
    seen = 0
    for item in loader:
        R.append(item["root"].numpy())
        M.append(item["depth_med"].numpy())
        seen += item["root"].shape[0]
        if seen >= n:
            break
    R = np.concatenate(R)[:n]            # [N,3] wrist root (m)
    M = np.concatenate(M)[:n].reshape(-1)  # [N] depth median (m)
    return R, M


def pct(a):
    p = np.percentile(a, [5, 50, 95])
    return f"[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--datasets", default="hanco,dexycb,ho3d")
    args = ap.parse_args()

    print(f"{'dataset':8s} {'N':>5s} | root_z 5/50/95   depth_med 5/50/95  | "
          f"r(z,med) off(z-med)mm  resid_std(mm)  scale_a  root_x_p5/95   root_y_p5/95")
    for name in args.datasets.split(","):
        R, M = collect(name, args.n)
        z = R[:, 2]
        valid = M > 0
        zz, mm = z[valid], M[valid]
        r = np.corrcoef(zz, mm)[0, 1]
        off = (zz - mm).mean() * 1000
        a, b = np.polyfit(mm, zz, 1)
        resid = (zz - (a * mm + b)).std() * 1000
        print(f"{name:8s} {len(z):5d} | {pct(z):16s} {pct(M):16s} | "
              f"{r:5.2f}  {off:+8.1f}      {resid:7.1f}     {a:5.2f}  "
              f"{np.percentile(R[:,0],[5,95])} {np.percentile(R[:,1],[5,95])}  "
              f"(med-invalid {100*(~valid).mean():.1f}%)")


if __name__ == "__main__":
    main()
