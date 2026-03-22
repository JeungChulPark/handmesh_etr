import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
from torchvision import transforms
import pickle 

class hamer_dataloader(Dataset):
    def __init__ (self, base_path='/root/data/ty_dataset/OXR/240908_ego/', set_type='train', load_verts=False, root_align=True):
        self.load_verts = load_verts
        self.root_align = root_align # joint[0] -> origin or not
        self.set_type = set_type
        self.img_root = base_path + 'crop_frames'
        self.img_names = os.listdir(self.img_root)
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256], antialias=True)
        ])
        annot_root = base_path + 'annotations/annot_with_verts.pkl' if load_verts \
                        else base_path + 'annotations/annot_without_verts.pkl'

        with open(annot_root, 'rb') as f:
            self.annotations = pickle.load(f)
        f.close()
        self.img_names_ori = self.img_names.copy()
        for img_n in self.img_names_ori:
            if img_n not in self.annotations.keys():
                self.img_names.remove(img_n)
                
        self.start_idx = 0
        self.end_idx = len(self.img_names)
        if set_type == 'train':
            self.end_idx = int(0.9 * len(self.img_names))
        elif set_type == 'eval':
            self.start_idx = int(0.9 * len(self.img_names))
            self.end_idx = int(0.97 * len(self.img_names))
        elif set_type == 'test':
            self.start_idx = int(0.97 * len(self.img_names))
        # annotations keys:
        # pred_cam / pred_cam_t / focal_length / pred_keypoints_3d / pred_keypoints_2d
        # pred_mano_params (global_orient [1, 3, 3] / hand_pose [15, 3, 3] / betas)
        # (optional, load_verts=True) pred_vertices

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__ (self, idx):
        img_name = self.img_names[idx]
        img = Image.open(os.path.join(self.img_root, img_name))
        
        annot = self.annotations[img_name]
        joints_3d = annot['pred_keypoints_3d']
        joints_2d = annot['pred_keypoints_2d'] + 0.5 # normalized in [0, 1]]
        # import pdb; pdb.set_trace()
        if self.root_align:
            root_pos = joints_3d[0].copy()
            joints_3d -= root_pos
        
        joints_25d = np.concatenate([joints_2d, joints_3d[:, -1:]], axis=-1)
        ret_dict = {
                'image': self.image_transforms(img),
                # 'cropped_image': self.image_transforms(img),
                'keypoints': joints_25d,
                'keypoints3D': joints_3d,
                
            }
            
        if self.load_verts:
            verts3d = annot['pred_vertices']
            if self.root_align:
                verts3d -= root_pose
            ret_dict['verts3d'] = verts3d

        return ret_dict