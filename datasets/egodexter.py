import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
import json
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torch.utils.data import Dataset
import argparse
from utils import *

from sklearn.model_selection import train_test_split
from datasets.augmentation import *

def get_affine_transform(center, scale, res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [
        1,
    ])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)   
    t_mat[0, 2] = -res[1] / 2
    t_mat[1, 2] = -res[0] / 2
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [
        1,
    ])
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2],
                                                   scale, res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(
        np.float32)
def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[1]) / scale
    affinet[1, 1] = float(res[0]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    affinet[2, 2] = 1
    return affinet
# img: PIL
def transform_img(img, affine_trans, res):
    """
    Args:
    center (tuple): crop center coordinates
    scale (int): size in pixels of the final crop
    res (tuple): final image size
    """
    trans = np.linalg.inv(affine_trans)
    img = img.transform(
        tuple(res), Image.AFFINE, (trans[0, 0], trans[0, 1], trans[0, 2],
                                    trans[1, 0], trans[1, 1], trans[1, 2]))
    return img
def joint_transform(joint, center, affinetrans, res=[256, 256]):
    # joint: [21, 2] = [21, (x, y) in pixel]
    # affinetrans: [3, 3]
    img_center = np.array([res[0]/2, res[1]/2]) # [128, 128]
    tj = np.ones((21,3))
    tj[:,:2] = joint
    #transformed_joint = tj.dot(affinetrans) 
    transformed_joint = affinetrans.dot(tj.T)
    transformed_joint = transformed_joint.T
    # print(transformed_joint.shape)
    # transformed_joint[:,:2] += center #+= img_center # (21, 3)
    return transformed_joint[:, :2]
# res = resolution: [256, 256]



class EgoDexter_MediaPipe(Dataset):
    def __init__(self, config, set_type="train", device='cpu'):
        self.device = device

        self.image_dir = os.path.join(config.EGO_DATA_ROOT, "rgb")
        self.image_names = np.sort(os.listdir(self.image_dir))
        self.img_size = config.IMG_SIZE

        mp_anno = os.path.join(config.EGO_DATA_ROOT, "mediapipe_pixel.json")

        with open(mp_anno, "r") as f:
            self.mp_anno = json.load(f)
        
        self.image_names = list(self.mp_anno.keys())

        self.start_idx = 0
        self.end_idx = len(self.image_names) - 1

        if set_type == 'train':
            self.end_idx = int(0.85 * len(self.image_names))
        elif set_type == 'eval':
            self.start_idx = int(0.85 * len(self.image_names))
            self.end_idx = int(0.99 * len(self.image_names))
        elif set_type == 'test':
            self.start_idx = int(0.1 * len(self.image_names))

        random.shuffle(self.image_names)
        

        self.img_resizer = transforms.Resize(self.img_size)
        self.image_names = self.image_names[self.start_idx:self.end_idx]
        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize([self.img_size, self.img_size]),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        blur_radius = 0.5
        brightness = 0.5
        saturation = 0.5
        hue = 0.15
        contrast = 0.5

        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))
        
        image = image_raw.filter(ImageFilter.GaussianBlur(blur_radius))
        image = color_jitter(
            image,
            brightness= brightness,
            saturation=saturation,
            hue=hue,
            contrast=contrast,
        )   

        keypoints = np.array(self.mp_anno[image_name])

        center_jittering = 0.2
        scale_jittering = 0.5
        max_rot = np.pi
        center = [128, 128] # img size: [256, 256]
        scale = 256
        center_offsets = (center_jittering * np.random.uniform(low=-1, high=1, size=2))
        center = center + center_offsets.astype(int)
        # Scale jittering
        scale_jittering = scale_jittering * np.random.randn() + 0.8
        scale_jittering = np.clip(
            scale_jittering,
            1 - scale_jittering,
            1,
            )
        scale = scale * scale_jittering 
        rot = np.random.uniform(low=-max_rot, high=max_rot)
        affinetrans, post_rot_trans = get_affine_transform(
        center, scale, [256, 256], rot=rot
        )
        # image augmentation
        img = transform_img(self.img_resizer(image), affinetrans, [256, 256])
        # joint augmentation
        tj = joint_transform(keypoints[:, :2] * 256, center, affinetrans) / 256
        keypoints[:, :2] = tj

        # image = self.image_transform(img)
        image_raw = self.image_raw_transform(image_raw)
        
        bb, crop_joint= crop_bb(keypoints, [self.img_size, self.img_size], margin=10)
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
            "ori_keypoints": keypoints,
            "keypoints3D": np.zeros([21, 3]),
            "K": np.zeros([3, 3]),
        }
    


# python .\datasets\freihand.py --cfg .\configs.yaml
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=os.path.join('configs.yaml'))
    parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
    
    opt = parser.parse_args()
    
    cfg = load_cfg(opt.cfg)
    device = setup_runtime(opt) 
    dataset = EgoDexter_MediaPipe(cfg, 'eval', device)
    print(len(dataset))
    
    for i in range(0, len(dataset), len(dataset)//50):
        print(i)
        data = dataset.__getitem__(i)
        # import pdb; pdb.set_trace()
        img = draw_joint2D(data['image'], data['keypoints'])
        cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)