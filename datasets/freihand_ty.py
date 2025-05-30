# freihand dataset

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
from torchvision import transforms
from pycocotools.coco import COCO
import pickle

from datasets.augmentation import *
from datasets.dataset_utils import *
import random

import mediapipe as mp


class Freihand(Dataset):
    def __init__(
        self,
        config=None,
        mode="train",
        img_size=256,
    ):
        self.mode = mode  # {"train", "eval"}
        self.use_pca = False
        self.center_idx = 0

        if mode == "train":
            img_path = "training/rgb"
            prefix = "training_{}.json"
            i2l_suffix = "freihand_train_{}.json"
        else:
            img_path = "evaluation/rgb"
            prefix = "evaluation_{}.json"
            i2l_suffix = "freihand_eval_{}.json"

        self.image_dir = os.path.join(r"../../Datasets/Hand Dataset/FreiHAND/FreiHAND_pub_v2", img_path)
        self.image_names = np.sort(os.listdir(self.image_dir))  # 130240
        self.image_size = [img_size, img_size]  # [224, 224] due to ViT
        self.i2l_annot_path = os.path.join(r"../../Datasets/Hand Dataset/FreiHAND/FreiHAND_pub_v2", "i2l")
        self.img_size = img_size

        dataset_path = r"../../Datasets/Hand Dataset/FreiHAND/FreiHAND_pub_v2"
        self.verts_path = os.path.join(dataset_path, prefix.format("verts"))
        self.mano_path = os.path.join(dataset_path, prefix.format("mano"))
        self.joint_path = os.path.join(dataset_path, prefix.format("xyz"))

        with open(self.verts_path, "r") as f:
            self.verts = json.load(f)
        f.close()
        if self.use_pca:
            self.mano = np.load(os.path.join("/root/colab_diff/dataset", mode + "_freihand_pca.npy"))
        else:
            with open(self.mano_path, "r") as f:
                self.mano = json.load(f)
            f.close()
        with open(self.joint_path, "r") as f:
            self.joint = json.load(f)
        f.close()

        self.K_path = os.path.join(dataset_path, prefix.format("K"))
        with open(self.K_path, "r") as f:
            self.K = json.load(f)
        f.close()

        i2l_annot = COCO(os.path.join(self.i2l_annot_path, i2l_suffix.format("coco")))
        self.prepare_bbox_from_coco(i2l_annot)

        # 32560
        assert len(self.verts) == len(self.mano), "the length of two annotation files are DIFFERENT!"
        assert len(self.verts) == len(self.joint), "the length of two annotation files are DIFFERENT!"
        assert len(self.verts) == len(self.K), "the length of two annotation files are DIFFERENT!"

        self.image_names = self.image_names[: len(self.verts)]

        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([img_size, img_size], antialias=None),
            ]
        )

        self.to_tensor_transform = transforms.ToTensor()

        print(f"{self.mode} dataset is loaded, length of annotations: {len(self.verts)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        ori_img = load_img(os.path.join(self.image_dir, image_name))

        aug_img, img2bb_trans, bb2img_trans, rot, _, cam, cam_nh, no_rot_trans = augmentation(
            ori_img,
            self.bbox[idx],
            self.mode,
            exclude_flip=True,
            rotation=True,
            cam_param=np.array(self.K[idx]),
            need_heatmap=False,
        )

        rot = rot * np.pi / 180.0
        rot_mat = np.array(
            [
                [np.cos(rot), -np.sin(rot), 0],
                [np.sin(rot), np.cos(rot), 0],
                [0, 0, 1],
            ]
        ).astype(np.float32)

        joint = np.array(self.joint[idx])
        # verts = np.array(self.verts[idx])

        xy = self.get_2D_annotation(joint, cam)  # cam is already rotated
        cam = np.array(self.K[idx])

        joint = joint.dot(rot_mat)
        # verts = verts.dot(rot_mat)

        root_xyz = joint[self.center_idx].copy()
        align_joint = joint - root_xyz  # wrist-oriented coord
        # align_verts = verts - root_xyz # wrist-oriented coord

        item = {
            "ori_image": self.to_tensor_transform(ori_img) / 255.0,
            "img": self.img_transform(aug_img.astype(np.float32)) / 255.0,
            "joint": joint,
            "align_joint": align_joint,
            "cam": cam,
            "root": root_xyz,
            "xy": xy,
        }
        return item

    def prepare_bbox_from_coco(self, coco_obj):
        bbox_list = []
        img_size = [256, 256]

        # for idx in coco_obj.anns.keys()[len(self.image_names)]:
        if self.mode == "train":
            for idx in range(0, len(self.image_names)):
                ann = coco_obj.anns[idx]
                bbox = process_bbox(np.array(ann["bbox"]), img_size[0], img_size[1])
                bbox_list.append(bbox)
        else:
            for idx in coco_obj.anns.keys():
                ann = coco_obj.anns[idx]
                bbox = process_bbox(np.array(ann["bbox"]), img_size[0], img_size[1])
                bbox_list.append(bbox)
        self.bbox = bbox_list

    def get_2D_annotation(self, xyz, K):
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.matmul(K, xyz.T).T
        return uv[:, :2] / uv[:, -1:]
