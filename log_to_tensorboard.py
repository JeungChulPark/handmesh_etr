# Copyright (c) 2026. Depth-Aware Hand Pose Initiative.
"""
Mirror the warm-start training's metrics into TensorBoard, WITHOUT wandb and
WITHOUT restarting the running job.

train_mobrecon_rgbd.py logs to a local wandb *offline* datastore
(wandb/offline-run-*/run-*.wandb). That file is just an on-disk record the
training already writes -- you never need the wandb UI or an account. This
script reads the scalars out of it and writes them to TensorBoard event files,
polling until training exits, so everything (loss + MPJPE curves) is viewable at

    tensorboard --logdir runs --port 6006     # http://localhost:6006

Scalars exposed (x-axis = training step):
    loss/total, loss/3d, loss/2d        every ~500 steps
    mpjpe/train_iter                    every ~500 steps
    mpjpe/train_epoch, mpjpe/test_epoch, loss/train_epoch   once per epoch
"""
import os, re, glob, json, time, argparse
from torch.utils.tensorboard import SummaryWriter

# wandb scalar key -> TensorBoard tag. Anything not listed (media metadata like
# depth.*, gt.*, _runtime, _timestamp) is ignored.
TAGS = {
    "loss": "loss/total",
    "loss_3d": "loss/3d",
    "loss_2d": "loss/2d",
    "mpjpe": "mpjpe/train_iter",
    "train_loss": "loss/train_epoch",
    "train_mpjpe": "mpjpe/train_epoch",
    "test_mpjpe": "mpjpe/test_epoch",
}


def find_wandb_file(explicit=None):
    if explicit and os.path.isfile(explicit):
        return explicit
    cands = glob.glob("wandb/*run-*/run-*.wandb")
    if not cands:
        return None
    # The actively-training run has the most recently modified .wandb file.
    return max(cands, key=os.path.getmtime)


def read_history(wandb_file):
    """Yield (step, {tag: value}) for every history record in the datastore."""
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2 as pb
    ds = DataStore()
    ds.open_for_scan(wandb_file)
    while True:
        try:
            raw = ds.scan_data()
        except Exception:
            break
        if raw is None:
            break
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") != "history":
            continue
        step, vals = None, {}
        for it in rec.history.item:
            key = it.key if it.key else ".".join(it.nested_key)
            try:
                v = json.loads(it.value_json)
            except Exception:
                continue
            if key == "_step":
                step = int(v)
            elif key in TAGS and isinstance(v, (int, float)):
                vals[TAGS[key]] = float(v)
        if step is not None and vals:
            yield step, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_file", default=None, help="explicit run-*.wandb (default: newest)")
    ap.add_argument("--logdir", default="runs/rgbd_warm30")
    ap.add_argument("--pid", default="logs/train_warm30.pid")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", default=30, type=int)
    args = ap.parse_args()

    writer = SummaryWriter(args.logdir)
    seen = set()  # (tag, step) already written

    def flush():
        wf = find_wandb_file(args.wandb_file)
        if not wf:
            return 0
        added = 0
        for step, vals in read_history(wf):
            for tag, v in vals.items():
                if (tag, step) not in seen:
                    writer.add_scalar(tag, v, step)
                    seen.add((tag, step))
                    added += 1
        writer.flush()
        return added

    def train_alive():
        try:
            os.kill(int(open(args.pid).read().strip()), 0)
            return True
        except Exception:
            return False

    n = flush()
    print(f"[tb] wrote {n} scalar points to {args.logdir}", flush=True)
    if args.once:
        writer.close()
        return

    while True:
        time.sleep(args.interval)
        added = flush()
        if added:
            print(f"[tb] +{added} points ({len(seen)} total) -> {args.logdir}", flush=True)
        if not train_alive():
            flush()
            print(f"[tb] training exited; final {len(seen)} points written. done.", flush=True)
            break
    writer.close()


if __name__ == "__main__":
    main()
