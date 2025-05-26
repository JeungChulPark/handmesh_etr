import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np
import os

from models.manolayer import ManoLayer
import argparse
from utils import *

from PIL import Image
import pickle


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1



class InterHand(Dataset):
    def __init__(self, cfg, set_type='train', device='cpu'):
        super(InterHand, self).__init__()
        self.img_size = cfg.IMG_SIZE
        self.device = device
        
        base_path = cfg.INTER_DATA_ROOT


        tmp = set_type
        if tmp == 'test':
            tmp = 'train'
        elif tmp == 'train':
            tmp = 'test'
        if tmp == 'eval':
            tmp = 'test'
        self.data_path = os.path.join(base_path, tmp)
        self.train_path = []

        for each_view in os.listdir(os.path.join(self.data_path, 'img')):
            view_path = os.path.join(self.data_path, 'img', each_view)
            for each_file in os.listdir(view_path):
                train_dict = {}
                each_file = each_file.split('.')[0]
                train_dict['img'] = os.path.join(view_path, each_file + '.jpg')
                train_dict['anno'] = os.path.join(self.data_path, 'anno', each_view, each_file + '.pkl')

                self.train_path.append(train_dict)

        if set_type == 'train':
            self.train_path = self.train_path[:int(0.9*len(self.train_path))]

        elif set_type == 'eval':
            self.train_path = self.train_path[int(0.9*len(self.train_path)):]

        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

        mano_base = 'misc'
        mano_path = {'left': os.path.join(mano_base, 'MANO_LEFT.pkl'), 'right': os.path.join(mano_base, 'MANO_RIGHT.pkl')}
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

    def load_hand(self, data):
        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']
        hand_dict = {}
        single_hand = False
        for idx, hand_type in enumerate(['left', 'right']):
            if data['mano_params'][hand_type] is None:
                h = ['left', 'right']
                single_hand = h[1 - idx]
                continue
            params = data['mano_params'][hand_type]
            handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(), torch.from_numpy(params['pose']).float(), torch.from_numpy(params['shape']).float(), trans=torch.from_numpy(params['trans']).float())
            
            handV = handV[0].numpy()
            handJ = handJ[0].numpy()
            handV = handV @ R.T + T
            handJ = handJ @ R.T + T

            handV2d = handV @ camera.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handJ2d = handJ @ camera.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]

            hand_dict[hand_type] = {
                                    'joints3d': handJ,
                                    'joints2d': handJ2d,
                                    'R': R @ params['R'][0],
                                    'camera': camera,
                                    'hand_type':hand_type
                                    }
        if single_hand:
            hand_dict = hand_dict[single_hand]
        # import pdb; pdb.set_trace()
            
        return hand_dict
    def __getitem__(self, idx):
        data_dict = {}
       
        train_i = self.train_path[idx]
        train_img_path = train_i['img']
        train_anno_path = train_i['anno']
       
        image_raw = Image.open(train_img_path)
        
        image = self.image_transform(image_raw)
        image_raw = self.image_raw_transform(image_raw)


        with open(train_anno_path, 'rb') as file:
            data = pickle.load(file)
        
        hand_dict = self.load_hand(data)

        
        data_dict['image'] = image
        data_dict['image_raw'] = image_raw
        data_dict['keypoints'] = hand_dict['joints3d'] 
        data_dict['keypoints2d'] = hand_dict['joints2d'] 
        data_dict['K'] = hand_dict['camera']

        return data_dict
    
    def __len__(self):
        return len(self.train_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=os.path.join('misc', 'model', 'config.yaml'))
    parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
    
    opt = parser.parse_args()
    
    cfg = load_cfg(opt.cfg)
    device = setup_runtime(opt) 
    dataset = InterHand(cfg, 'test', device)
    print(len(dataset))
    for i in range(0, len(dataset), len(dataset)//50):
        print(i)
        data = dataset.__getitem__(i)
        img = draw_joint(data['image_raw'], data['keypoints'], data['K'])
        cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)