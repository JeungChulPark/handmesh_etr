# Reference: https://github.com/hassony2/obman_train

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
import json
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset
import argparse
from utils import *
import pickle

from sklearn.model_selection import train_test_split

class ObMan(Dataset):
    def __init__(self, config, set_type="train", device='cpu'):
        self.device = device
        # split = ['train', 'val', 'test']
        self.set_type = set_type
        if self.set_type == 'eval':
            self.set_type = 'val'
        # config.DATA_ROOT => Obman root
        self.obman_root = os.path.join(config.OBMAN_DATA_ROOT, self.set_type)
        self.image_dir = os.path.join(self.obman_root, "rgb") # rgb_hand
        self.meta_dir = os.path.join(self.obman_root, "meta")
        self.media_pipe = os.path.join(self.obman_root, 'mediapipe_pixel.json')
        self.img_size = config.IMG_SIZE
        # bring image name
        idxs = [
            int(imgname.split(".")[0])
            for imgname in sorted(os.listdir(self.meta_dir))
        ]
        prefix_template = "{:08d}"
        # self.prefixes = [prefix_template.format(idx) for idx in idxs]
        # Every image is captured in same camera settings
        self.K = np.array(
            [[480.0, 0.0, 128.0], 
             [0.0, 480.0, 128.0], 
             [0.0, 0.0, 1.0]
        ]
        ).astype(np.float32)

        with open(self.media_pipe, "r") as f:
            self.mp_K = json.load(f)
        self.image_name = list(self.mp_K.keys())
        self.img_resizer = transforms.Resize(self.img_size)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize([self.img_size, self.img_size]),
                transforms.ToTensor(),
            ]
        )
        self.image_raw_transform = transforms.ToTensor()

        self.cam_extr = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ]
        ).astype(np.float32)
    
        
    def __len__(self):
        return len(self.image_name)
         
    def __getitem__(self, idx):
        # return val: Image, 2d joints, 3d joints
        prefix = self.image_name[idx].split('.')[0]
        meta_path = os.path.join(
            self.meta_dir, "{}.pkl".format(prefix)
        )
        with open(meta_path, "rb") as meta_f:
            meta_info = pickle.load(meta_f)
        hand_side = meta_info['side']
        flip = False
        if hand_side == "right":
            flip=True
        image_path = os.path.join(self.image_dir, "{}.jpg".format(prefix))
        image_raw = Image.open(image_path)
        image_raw = self.image_raw_transform(image_raw)

        joints_2d = np.array(self.mp_K[prefix + '.jpg'])

        joints_3d = meta_info['coords_3d']
        joints_3d = self.cam_extr[:3, :3].dot(joints_3d.transpose()).transpose()

        bb, crop_joint= crop_bb(joints_2d, [self.img_size, self.img_size], margin=10)
        cropped_img = to_pil_image(image_raw[:, bb[0, 1]:bb[1, 1], bb[0, 0]:bb[1, 0]])

        resized_cropped_img = self.img_resizer(cropped_img)
        w, h = resized_cropped_img.size
        y_ratio, x_ratio = self.img_size/h, self.img_size/w
        
        crop_joint[:, 0] = crop_joint[:, 0] / y_ratio
        crop_joint[:, 1] = crop_joint[:, 1] / x_ratio

        image = self.image_transform(to_pil_image(image_raw))
        cropped_img = self.image_transform(cropped_img)
        
        return {
            "image": image,
            "cropped_image": cropped_img,
            "keypoints": crop_joint,
            "ori_keypoints": joints_2d,
            "keypoints3D": joints_3d,
            "K": self.K,
        }
    
# python .\datasets\freihand.py --cfg .\configs.yaml
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=os.path.join('misc', 'model', 'config.yaml'))
    parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
    
    opt = parser.parse_args()
    
    cfg = load_cfg(opt.cfg)
    device = setup_runtime(opt) 

    dataset = ObMan(cfg, 'train', device)
    print(len(dataset))
    import pdb; pdb.set_trace()
    for i in range(0, len(dataset), len(dataset)//10):
        print(i)
        data = dataset.__getitem__(i)
        img = draw_joint(data['image'], data['keypoints'], data['K'])
        cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)