## Introduction

**Contributors:** Taeyun Woo ([taeyun.woo@kaist.ac.kr](mailto:taeyun.woo@kaist.ac.kr)), Minje Kim ([minjekim@kaist.ac.kr](mailto:minjekim@kaist.ac.kr))

This is hand pose estimation as a service work of ETRI (Electronics and Telecommunications Research Institute) service. 

<img src="./asset/18_2d.jpg" width="200px" height="200px" title="Result1"/> <img src="./asset/86_2d.jpg" width="200px" height="200px" title="Result2"/> <img src="./asset/result1.gif" width="200px" height="200px" title="Result3"/>


This work is largely based on [MobRecon](https://github.com/SeanChenxy/HandMesh) study.
## Repository Layout

Entry-point scripts live under `scripts/`, grouped by what they do. **Run them from
the repo root** — checkpoint, cache and capture paths (`mobrecon_ckpt/`, `pretrain/`,
`cache/`, `rgbd_captures*/`) are resolved relative to the current directory.

```
configs/        configs.yaml, configs_rgbd.yaml, configs_origin.yaml, hanco_configs.yaml
docs/           design notes, model cards, deployment guides
notebooks/      ONNX conversion notebooks
media/          rendered comparison videos / screenshots (gitignored)

scripts/
  train/        train_mobrecon_rgbd, train_hybrid, train_zlifter, train_fastvit_hand, ...
  infer/        infer_hybrid, infer_rgbd_zed, infer_ipad_stream, infer_rgbd_captures, ...
  eval/         eval_iphone, eval_rgbd, eval_pa_mpjpe, ...
  diag/         diag_* probes, ablation_depth*, cmp_zed_depth_modes
  export/       convert_* and export_*_onnx
  capture/      capture_zed_controlled, view_zed, cache_mediapipe, make_iphone_gt
  tools/        smoke_rgbd, unity_stream_hand, log_to_tensorboard, demo_hanco_video, test
  run/          *.sh launchers
  _bootstrap.py sys.path shim (see below)

utils.py loss.py runner.py     shared library, imported by name
models/ datasets/ conv/ depth_refine/   packages, imported by name
template/ asset/ ios/ unity/            assets and client-side projects
```

Example:

```bash
python scripts/train/train_hybrid.py --exp hybrid_B --mode locked
python scripts/infer/infer_hybrid.py --dir rgbd_captures --ckpt mobrecon_ckpt/hybrid_B/best.pt
bash   scripts/run/run_rgbd_train.sh
```

### `scripts/_bootstrap.py`

These scripts import repo-root modules (`from utils import ...`) and each other
(`from infer_rgbd_captures import load_capture`). Running `python scripts/infer/foo.py`
only puts `scripts/infer` on `sys.path`, so every entry point starts with:

```python
import _bootstrap  # noqa: F401
```

which puts the repo root and the other `scripts/<group>` directories back on the path.
Each group directory holds a symlink to `scripts/_bootstrap.py`. New scripts added under
`scripts/` should keep that first line.


## Data Preprocessing

We use dataset [Hanco](https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html) and [Obman](https://www.di.ens.fr/willow/research/obman/data/). We either use ground truth 3D hand joint or 2.5D hand joint predicted by [hand landmark detection of mediapipe](https://mediapipe-studio.webapps.google.com/studio/demo/hand_landmarker). 

**2.5D:** 

```bash
# Hanco dataset
$ mv ./datasets/Hanco/preprocess.py {path_to_Hanco} 
# Obman dataset
$ mv ./datasets/Obman/train/preprocess.py {path_to_Obman}/train  
$ mv ./datasets/Obman/val/preprocess.py {path_to_Obman}/val  
$ mv ./datasets/Obman/test/preprocess.py {path_to_Obman}/test  
```

After moving preprocess file to dataset, run the preprocess file.

```bash
# Hanco dataset
$ python preprocess.py

# Obman dataset
python val/preprocess.py
python train/[preprocess.py
python test/preprocess.py
```

Then, we can earn preprocessed file .json for each dataset. 

**3D:** 

For using 3D hand joint, we don’t need any preprocess task.

## Running

### Training

```bash
python main.py --cfg configs/configs.yaml --gpu 0 --dataset hanco
```

### Validation and Test

Include ’eval’ in PHASE in configs/configs.yaml, and for test include ‘test’. By running same command of training, the code will run validation and test. 

### Camera Test
We present two pretrained weight. You can download it on following [link](https://drive.google.com/drive/folders/1-Xdc5hVo0R7ajtS0e14x_id_wfXffZub?usp=drive_link). Put the weight files in pretrain folder.
The middle number corresponds to image size. (e.g ETRI_256_Large.pth --> means it gets 256 input image size)

```bash
python main.py --cfg configs/configs.yaml --gpu 0 --cam_test
```

## License

This work is based on MIT license.
