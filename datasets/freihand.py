import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import argparse
from utils import *

from sklearn.model_selection import train_test_split
from datasets.augmentation import *


class FreiHAND_MediaPipe_only(Dataset):
    def __init__(self, config, set_type="train", device='cpu'):
        self.device = device
        self.image_dir = os.path.join(config.FREI_DATA_ROOT, "training/rgb")
        self.image_names = np.sort(os.listdir(self.image_dir))
        self.img_size = config.IMG_SIZE

        mp_anno = os.path.join(config.FREI_DATA_ROOT, "training/mediapipe_pixel.json")

        with open(mp_anno, "r") as f:
            self.mp_anno = json.load(f)
        
        self.image_names = list(self.mp_anno.keys())

        self.start_idx = 0
        self.end_idx = len(self.image_names) - 1

        if set_type == 'train':
            self.end_idx = int(0.9 * len(self.image_names))
        elif set_type == 'eval':
            self.start_idx = int(0.9 * len(self.image_names))
            self.end_idx = int(0.97 * len(self.image_names))
        elif set_type == 'test':
            self.start_idx = int(0.97 * len(self.image_names))

        self.image_names = self.image_names[self.start_idx:self.end_idx]
        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        
        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))
        image = self.image_transform(image_raw)
        image_raw = self.image_raw_transform(image_raw)
        
        keypoints = np.array(self.mp_anno[image_name])

        return {
            "image": image,
            "image_raw": image_raw,
            "keypoints": keypoints,
        }
    

class FreiHAND_MediaPipe(Dataset):
    def __init__(self, config, set_type="train", device='cpu'):
        self.device = device
        self.image_dir = os.path.join(config.FREI_DATA_ROOT, "training/rgb")
        self.image_names = np.sort(os.listdir(self.image_dir))
        self.img_size = config.IMG_SIZE

        mp_anno = os.path.join(config.FREI_DATA_ROOT, "mediapipe_xyz.json")
        mp_K = os.path.join(config.FREI_DATA_ROOT, "mediapipe_K.json")

        with open(mp_anno, "r") as f:
            self.mp_anno = json.load(f)
        with open(mp_K, "r") as f:
            self.mp_K = json.load(f)
        
        self.image_names = list(self.mp_anno.keys())

        self.start_idx = 0
        self.end_idx = len(self.image_names) - 1

        if set_type == 'train':
            self.end_idx = int(0.9 * len(self.image_names))
        elif set_type == 'eval':
            self.start_idx = int(0.9 * len(self.image_names))
            self.end_idx = int(0.97 * len(self.image_names))
        elif set_type == 'test':
            self.start_idx = int(0.97 * len(self.image_names))

        self.image_names = self.image_names[self.start_idx:self.end_idx]
        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        
        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))
        image = self.image_transform(image_raw)
        image_raw = self.image_raw_transform(image_raw)
        
        keypoints = np.array(self.mp_anno[image_name])
        _K =  np.array(self.mp_K[image_name])
        # import pdb; pdb.set_trace()
        return {
            "image": image,
            "image_raw": image_raw,
            "keypoints": keypoints,
            "K":_K,
        }
class FreiHAND_with_MediaPipe(Dataset):
    def __init__(self, config, set_type="train", device='cpu'):
        self.device = device
        self.image_dir = os.path.join(config.FREI_DATA_ROOT, "training/rgb")
        self.img_size = config.IMG_SIZE

        t_anno = os.path.join(config.FREI_DATA_ROOT, "training_xyz.json")
        t_K = os.path.join(config.FREI_DATA_ROOT, "training_K.json")

        mp_anno = os.path.join(config.FREI_DATA_ROOT, "mediapipe_xyz.json")

        self.image_names = np.sort(os.listdir(self.image_dir))
        with open(t_anno, "r") as f:
            self.t_anno = json.load(f)
        with open(t_K, "r") as f:
            self.t_K = json.load(f)
        with open(mp_anno, "r") as f:
            self.mp_anno = json.load(f)

        self.train_dict = {}

        for idx in range(len(self.t_anno)):
            self.train_dict[self.image_names[idx]] = {}
            self.train_dict[self.image_names[idx]]['anno'] = self.t_anno[idx]
            self.train_dict[self.image_names[idx]]['K'] = self.t_K[idx]




        self.start_idx = 0
        self.end_idx = len(self.t_anno) - 1

        if set_type == 'train':
            self.end_idx = int(0.9 * len(self.t_anno))
        elif set_type == 'eval':
            self.start_idx = int(0.9 * len(self.t_anno))
            self.end_idx = int(0.97 * len(self.t_anno))
        elif set_type == 'test':
            self.start_idx = int(0.97 * len(self.t_anno))

        self.image_names = self.image_names[self.start_idx:self.end_idx]
        self.mp_image_names = list(self.mp_anno.keys())

        self.image_names = list(set(self.image_names).intersection(self.mp_image_names))

        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        
        center_jittering = 0.2
        scale = 1 # or 2.2
        scale_jittering = 0.3
        max_rot = np.pi
        image_size = [240, 240]

        # color setting
        blur_radius = 0.5
        brightness = 0.5
        saturation = 0.5
        hue = 0.15
        contrast = 0.5


        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))

        keypoints = np.array(self.train_dict[image_name]['anno'])
        _K =  np.array(self.train_dict[image_name]['K'])
        keypoints_2d =  np.array(self.mp_anno[image_name])
        
        center = (self.img_size //2, self.img_size // 2)
        scale = 1
        
        rot = np.random.uniform (low = max_rot, high = max_rot)

        rot_mat = np.array(
            [[np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0, 0, -1]],
        ).astype(np.float32)

        # affine transform
        affinetrans, pst_rot_trans = get_affine_transform(center, scale, image_size, rot=rot)

        # joint 3d
        joints3d = rot_mat.dot(
            keypoints.transpose(1, 0)
        ).transpose()

        # color jittering
        blur_radius = random.random() * blur_radius
        image = image_raw.filter(ImageFilter.GaussianBlur(blur_radius))
        image = color_jitter(
            image,
            brightness= brightness,
            saturation=saturation,
            hue=hue,
            contrast=contrast,
        )

        image = transform_img(
            image, affinetrans, image_size
        )
        image = image.crop((0, 0, self.img_size, self.img_size))

        image = self.image_transform(image_raw)
        image_raw = self.image_raw_transform(image_raw)
        

        # import pdb; pdb.set_trace()
        return {
            "image": image,
            "image_raw": image_raw,
            "keypoints3D": joints3d,
            "keypoints": keypoints_2d,
            "K":_K,
        }


class FreiHAND(Dataset):
    def __init__(self, config, set_type="train", device='cpu'):
        self.device = device
        self.image_dir = os.path.join(config.FREI_DATA_ROOT, "training/rgb")
        self.img_size = config.IMG_SIZE

        t_anno = os.path.join(config.FREI_DATA_ROOT, "training_xyz.json")
        t_K = os.path.join(config.FREI_DATA_ROOT, "training_K.json")

        self.image_names = np.sort(os.listdir(self.image_dir))
        with open(t_anno, "r") as f:
            self.t_anno = json.load(f)
        with open(t_K, "r") as f:
            self.t_K = json.load(f)

        self.train_dict = {}

        for idx in range(len(self.t_anno)):
            self.train_dict[self.image_names[idx]] = {}
            self.train_dict[self.image_names[idx]]['anno'] = self.t_anno[idx]
            self.train_dict[self.image_names[idx]]['K'] = self.t_K[idx]

        self.start_idx = 0
        self.end_idx = len(self.t_anno) - 1

        if set_type == 'train':
            self.end_idx = int(0.9 * len(self.t_anno))
        elif set_type == 'eval':
            self.start_idx = int(0.9 * len(self.t_anno))
            self.end_idx = int(0.97 * len(self.t_anno))
        elif set_type == 'test':
            self.start_idx = int(0.97 * len(self.t_anno))

        self.image_names = self.image_names[self.start_idx:self.end_idx]
       
        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        
        center_jittering = 0.2
        scale = 1 # or 2.2
        scale_jittering = 0.3
        max_rot = np.pi
        image_size = [self.img_size, self.img_size]

        # color setting
        blur_radius = 0.5
        brightness = 0.5
        saturation = 0.5
        hue = 0.15
        contrast = 0.5


        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))

        keypoints = np.array(self.train_dict[image_name]['anno'])
        _K =  np.array(self.train_dict[image_name]['K'])

        center = (self.img_size //2, self.img_size // 2)
        scale = 1
        
        rot = np.random.uniform (low = max_rot, high = max_rot)

        rot_mat = np.array(
            [[np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0, 0, -1]],
        ).astype(np.float32)

        # affine transform
        affinetrans, pst_rot_trans = get_affine_transform(center, scale, image_size, rot=rot)

        # joint 3d
        joints3d = rot_mat.dot(
            keypoints.transpose(1, 0)
        ).transpose()

        # color jittering
        blur_radius = random.random() * blur_radius
        image = image_raw.filter(ImageFilter.GaussianBlur(blur_radius))
        image = color_jitter(
            image,
            brightness= brightness,
            saturation=saturation,
            hue=hue,
            contrast=contrast,
        )

        image = transform_img(
            image, affinetrans, image_size
        )
        image = image.crop((0, 0, self.img_size, self.img_size))

        image = self.image_transform(image_raw)
        image_raw = self.image_raw_transform(image_raw)
        

        # import pdb; pdb.set_trace()
        return {
            "image": image,
            "image_raw": image_raw,
            "keypoints3D": joints3d,
            "K":_K,
        }

# python .\datasets\freihand.py --cfg .\configs.yaml
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=os.path.join('configs.yaml'))
    parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
    
    opt = parser.parse_args()
    
    cfg = load_cfg(opt.cfg)
    device = setup_runtime(opt)
    dataset = FreiHAND_with_MediaPipe(cfg, 'eval', device)
    print(len(dataset))
    # import pdb; pdb.set_trace()
    for i in range(0, len(dataset), len(dataset)//50):
        import pdb; pdb.set_trace()
        data = dataset.__getitem__(i)
        # import pdb; pdb.set_trace()
        # img = draw_joint2D(data['image_raw'], data['keypoints'], data['K'])
        img = draw_joint2D(data['image'], data['keypoints'])
        cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)