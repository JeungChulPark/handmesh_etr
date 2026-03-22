import sys
import os

import numpy as np
from PIL import Image
import json
import torch
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.utils.data import Dataset
import argparse
from tqdm import tqdm

from datasets.augmentation import *
from datasets.dataset_utils import *

# from augmentation import *
# from dataset_utils import *
import random

from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)


COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g", "color_val": [0, 255, 0]},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c", "color_val": [255, 255, 0]},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b", "color_val": [255, 0, 0]},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m", "color_val": [0, 255, 255]},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r", "color_val": [0, 0, 255]},
}


def cam2pixel(cam_coord, K):
    f = [K[0, 0], K[1, 1]]
    c = [K[0, 2], K[1, 2]]
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def tensor2img(input_tensor, batch_id=0, as_cv=False):  # input_tensor [B, 3(1), H, W]
    batch_id = batch_id if batch_id else 0
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    target_tensor = input_tensor[batch_id]

    if target_tensor.size(0) == 1:  # Mask img
        target_tensor = target_tensor.repeat(3, 1, 1)
    img = target_tensor.permute(1, 2, 0).detach().cpu().numpy()
    if not (target_tensor.max() > 1):
        img *= 255
    img = img.astype(np.uint8)

    return img[..., ::-1] if as_cv else img


def setup_runtime(args, seed=44, num_workers=4):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    # Setup CUDA
    cuda_device_id = args.gpu

    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = f"cuda" if torch.cuda.is_available() else "cpu"

    print(f"Environment: GPU {device} seed {seed} number of workers {num_workers}")

    return device


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    defaults_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), "defaults.yaml"))
    _C.merge_from_file(defaults_abspath)
    _C.set_new_allowed(False)
    return _C.clone()


def load_cfg(path=None):
    _C.set_new_allowed(True)
    _C.merge_from_file(path)
    _C.set_new_allowed(False)
    return _C.clone()


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    # def is_valid_normalized_value(value: float) -> bool:
    #   return (value > 0 or math.isclose(0, value)) and (value < 1 or
    #                                                      math.isclose(1, value))

    # if not (is_valid_normalized_value(normalized_x) and
    #         is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    #   return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_joint2D_numpy(img, keypoints, idx=None):
    if idx is not None:
        keypoints = keypoints[idx]

    # image_raw = tensor2img(img, batch_id=idx, as_cv=False).copy()
    img_size_x = img.shape[1]
    img_size_y = img.shape[0]

    px_landmark = []
    for each_land in keypoints:
        landmark_px = normalized_to_pixel_coordinates(each_land[0], each_land[1], img_size_x, img_size_y)
        px_landmark.append(list(landmark_px))

    px_landmark = np.array(px_landmark)

    for finger, params in COLORMAP.items():
        points = [px_landmark[params["ids"]]]
        cv2.polylines(img, points, False, params["color_val"], 2)

    return img


def draw_joint2D(img, keypoints, idx=None):
    if idx is not None:
        keypoints = keypoints[idx]

    image_raw = tensor2img(img, batch_id=idx, as_cv=False).copy()
    img_size_x = img.size(-1)
    img_size_y = img.size(-2)

    px_landmark = []
    for each_land in keypoints:
        landmark_px = normalized_to_pixel_coordinates(each_land[0], each_land[1], img_size_x, img_size_y)
        px_landmark.append(list(landmark_px))

    px_landmark = np.array(px_landmark)

    for finger, params in COLORMAP.items():
        points = [px_landmark[params["ids"]]]
        cv2.polylines(image_raw, points, False, params["color_val"])

    return image_raw


def get_2D_annotation(xyz, K):
    # xyz = np.array(xyz) # cuz already np array
    # K = np.array(K) # cuz already np array
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


def get_affine_transform(center, scale, res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1

    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(
        center.tolist()
        + [
            1,
        ]
    )[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -res[1] / 2
    t_mat[1, 2] = -res[0] / 2
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = (
        t_inv.dot(rot_mat)
        .dot(t_mat)
        .dot(
            center.tolist()
            + [
                1,
            ]
        )
    )
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2], scale, res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(np.float32)


def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[1]) / scale
    affinet[1, 1] = float(res[0]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + 0.5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + 0.5)
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
        tuple(res),
        Image.AFFINE,
        (trans[0, 0], trans[0, 1], trans[0, 2], trans[1, 0], trans[1, 1], trans[1, 2]),
    )
    return img


def joint_transform(joint, center, affinetrans, res=[256, 256]):
    # joint: [21, 2] = [21, (x, y) in pixel]
    # affinetrans: [3, 3]
    img_center = np.array([res[0] / 2, res[1] / 2])  # [128, 128]
    tj = np.ones((21, 3))
    tj[:, :2] = joint
    # transformed_joint = tj.dot(affinetrans)
    transformed_joint = affinetrans.dot(tj.T)
    transformed_joint = transformed_joint.T
    # print(transformed_joint.shape)
    # transformed_joint[:,:2] += center #+= img_center # (21, 3)
    return transformed_joint[:, :2]


class HanCo_ETRI(Dataset):
    def __init__(self, config=None, mode="train", img_size=128, limit=2e3):
        self.hanco_root = r"D:\Datasets\Hand Dataset\HanCo"
        self.image_dir = os.path.join(self.hanco_root, "rgb")
        self.img_size = img_size  # int
        self.mode = mode

        self.train_xyz = os.path.join(self.hanco_root, "xyz")
        self.train_cam = os.path.join(self.hanco_root, "calib")

        mp_anno = os.path.join(self.hanco_root, "mediapipe_pixel.json")
        with open(mp_anno, "r") as f:
            self.mp_anno = json.load(f)
        f.close()
        # self.image_ls = []
        # self.keypoints_2d = []
        # self.cam_anno = []
        # self.xyz_anno = []

        self.train_dict_list = []
        for scene_num, scene_dict in self.mp_anno.items():
            for cam_id, cam_dict in scene_dict.items():
                for img_name, img_dict in cam_dict.items():
                    img_id = img_name.split(".")[0]
                    if os.path.isfile(
                        os.path.join(self.train_cam, scene_num, img_id + ".json")
                    ) and os.path.isfile(os.path.join(self.train_xyz, scene_num, img_id + ".json")):
                        train_dict = {}
                        train_dict["image_path"] = os.path.join(scene_num, cam_id, img_id)
                        with open(os.path.join(self.train_cam, scene_num, img_id + ".json"), "r") as f:
                            cam_anno = json.load(f)
                            cam_num = int(cam_id[-1])
                            # self.cam_anno.append(cam_anno['M'][cam_num])
                            train_dict["cam_intr"] = cam_anno["K"][cam_num]
                            train_dict["cam_extr"] = cam_anno["M"][cam_num]

                        f.close()
                        with open(os.path.join(self.train_xyz, scene_num, img_id + ".json"), "r") as f:
                            xyz_anno = json.load(f)
                            # self.xyz_anno.append(xyz_anno)
                            train_dict["xyz_anno"] = xyz_anno
                        # train_dict['keypoints2d'] = img_dict
                        f.close()

                        self.train_dict_list.append(train_dict)

                if len(self.train_dict_list) > limit and limit > 0:
                    break
            if len(self.train_dict_list) > limit and limit > 0:
                break
        del self.mp_anno

        self.img_resizer = transforms.Resize([self.img_size, self.img_size])
        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([self.img_size, self.img_size]),
            ]
        )

    def __len__(self):
        return len(self.train_dict_list)

    def __getitem__(self, idx):
        # color augmentation options
        blur_radius = 0.5
        brightness = 0.5
        saturation = 0.5
        hue = 0.15
        contrast = 0.5

        # image load
        train_dict = self.train_dict_list[idx]
        image_name = train_dict["image_path"]
        image_raw = Image.open(os.path.join(self.image_dir, image_name + ".jpg"))

        # color augmentation
        blur_radius = random.random() * blur_radius
        image = image_raw.filter(ImageFilter.GaussianBlur(blur_radius))
        image = color_jitter(
            image,
            brightness=brightness,
            saturation=saturation,
            hue=hue,
            contrast=contrast,
        )

        # load joints3d, cam_intr, calculate joints2d
        joints3d = np.array(train_dict["xyz_anno"])

        # adjusting camera extrinsics
        cam_extr = np.array(train_dict["cam_extr"])
        joints3d = cam_extr @ np.concatenate((joints3d, np.ones((21, 1))), axis=1).T

        _K = np.array(train_dict["cam_intr"])
        joints2d = get_2D_annotation(joints3d, _K)  # cam projection

        # x_min, y_min, x_width, y_height
        # 10. -> margin for bbox
        margin = 15.0

        x_min = max(joints2d[:, 0].min() - margin, 0.0)
        y_min = max(joints2d[:, 1].min() - margin, 0.0)
        x_max = min(joints2d[:, 0].max() + margin, 256.0)
        y_max = min(joints2d[:, 1].max() + margin, 256.0)
        x_width = x_max - x_min
        y_height = y_max - y_min

        line_segment = max(x_width, y_height)  # for square bbox

        bbox = [x_min, y_min, line_segment, line_segment]
        image = np.array(image)
        aug_img, img2bb_trans, bb2img_trans, rot, _, _, _, _ = augmentation(
            image, bbox, self.mode, exclude_flip=True, rotation=True
        )  # FreiHAND dataset only contains right hands. do not perform flip aug.

        rot = rot * np.pi / 180.0
        rot_mat = np.array(
            [
                [np.cos(rot), -np.sin(rot), 0],
                [np.sin(rot), np.cos(rot), 0],
                [0, 0, 1],
            ]
        ).astype(np.float32)

        # Since mobrecon estimates root-aligned 3D verts, make root joint (i.e. joints3d[0]) at the origin
        rot_joints = joints3d.dot(rot_mat)
        root_xyz = rot_joints[0].copy()
        align_joints = rot_joints - root_xyz

        # affine transform to joints2d
        ori_2d_mat = np.concatenate((joints2d, np.ones((21, 1))), axis=1)
        kps = (img2bb_trans @ ori_2d_mat.T).T[:, :2]

        # image resizer
        return_img = self.image_transform(Image.fromarray(aug_img.astype(np.uint8)))
        return {
            "image": return_img,
            # "cropped_image": cropped_img,
            # "keypoints": crop_joint,
            "keypoints3D": align_joints,
            "keypoints2D": kps / 256.0,  # 256 normalization
            # "K":_K,
            # 'affine': img2bb_trans,
        }


def generate_fake_prevpose(joint_uvd, weight=1.0):
    # (21, 3), uv range : (0~256), d range : (-0.10 ~ 0.10)

    extra_uvd = np.copy(joint_uvd)
    extra_uvd = random_translate_pose(extra_uvd, weight=weight)

    noise_w = weight
    ref_value = 3.0
    extra_uvd[:, 0] += np.random.normal(-1 * ref_value * noise_w, ref_value * noise_w, 21)
    extra_uvd[:, 1] += np.random.normal(-1 * ref_value * noise_w, ref_value * noise_w, 21)
    extra_uvd[0:, 2] += np.random.normal(-0.003 * noise_w, 0.003 * noise_w, 21)

    return extra_uvd


def random_translate_pose(joint_uvd, weight=1.0):
    extra_uvd = np.copy(joint_uvd)
    ref_value = 7
    extra_uvd[:, 0] += np.random.normal(-1 * ref_value * weight, ref_value * weight, 1)
    extra_uvd[:, 1] += np.random.normal(-1 * ref_value * weight, ref_value * weight, 1)
    extra_uvd[0:, 2] += np.random.normal(-0.01 * weight, 0.01 * weight, 1)

    return extra_uvd


class HanCo_ETRI_jitter(Dataset):
    def __init__(self, config=None, mode="train", img_size=256, limit=2e3):
        # self.hanco_root -> hanco root
        self.hanco_root = r"D:\Datasets\Hand Dataset\HanCo"
        self.image_dir = os.path.join(self.hanco_root, "rgb")
        self.img_size = img_size  # int
        self.mode = mode

        self.train_xyz = os.path.join(self.hanco_root, "xyz")
        self.train_cam = os.path.join(self.hanco_root, "calib")

        mp_anno = os.path.join(self.hanco_root, "mediapipe_pixel.json")
        with open(mp_anno, "r") as f:
            self.mp_anno = json.load(f)
        f.close()

        self.train_dict_list = []
        for scene_num, scene_dict in tqdm(self.mp_anno.items()):
            for cam_id, cam_dict in scene_dict.items():
                for img_name, img_dict in cam_dict.items():
                    img_id = img_name.split(".")[0]
                    if os.path.isfile(
                        os.path.join(self.train_cam, scene_num, img_id + ".json")
                    ) and os.path.isfile(os.path.join(self.train_xyz, scene_num, img_id + ".json")):
                        train_dict = {}
                        train_dict["image_path"] = os.path.join(scene_num, cam_id, img_id)
                        with open(os.path.join(self.train_cam, scene_num, img_id + ".json"), "r") as f:
                            cam_anno = json.load(f)
                            cam_num = int(cam_id[-1])

                            train_dict["cam_intr"] = cam_anno["K"][cam_num]
                            train_dict["cam_extr"] = cam_anno["M"][cam_num]
                        f.close()
                        with open(os.path.join(self.train_xyz, scene_num, img_id + ".json"), "r") as f:
                            xyz_anno = json.load(f)
                            # self.xyz_anno.append(xyz_anno)
                            train_dict["xyz_anno"] = xyz_anno
                        # train_dict['keypoints2d'] = img_dict
                        f.close()

                        self.train_dict_list.append(train_dict)

                if len(self.train_dict_list) > limit and limit > 0:
                    break
            if len(self.train_dict_list) > limit and limit > 0:
                break
        del self.mp_anno

        self.img_resizer = transforms.Resize([self.img_size, self.img_size])
        self.image_raw_transform = transforms.ToTensor()
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Resize([self.img_size, self.img_size]),
            ]
        )

    def __len__(self):
        return len(self.train_dict_list)

    def __getitem__(self, idx):
        # color augmentation options
        blur_radius = 0.5
        brightness = 0.5
        saturation = 0.5
        hue = 0.15
        contrast = 0.5

        # image load
        train_dict = self.train_dict_list[idx]
        image_name = train_dict["image_path"]
        image_raw = Image.open(os.path.join(self.image_dir, image_name + ".jpg"))

        if self.mode == "train":
            # color augmentation
            blur_radius = random.random() * blur_radius
            image = image_raw.filter(ImageFilter.GaussianBlur(blur_radius))
            image = color_jitter(
                image,
                brightness=brightness,
                saturation=saturation,
                hue=hue,
                contrast=contrast,
            )
        else:
            image = image_raw

        # load joints3d, cam_intr, calculate joints2d
        joints3d = np.array(train_dict["xyz_anno"])

        # adjusting camera extrinsics
        cam_extr = np.array(train_dict["cam_extr"])

        joints3d = cam_extr @ np.concatenate((joints3d, np.ones((21, 1))), axis=1).T
        joints3d = joints3d.T[:, :3]
        intr = np.array(train_dict["cam_intr"])
        joints2d = get_2D_annotation(joints3d, intr)  # cam projection
        # cv2.imwrite('img.png', cv2.cvtColor(draw_joint2D(torch.from_numpy(image).permute(2, 0, 1), joints2d / 224, idx=None), cv2.COLOR_RGB2BGR))  --> True

        # x_min, y_min, x_width, y_height
        # 10. -> margin for bbox
        margin = 20.0
        random_margin_x = int(np.random.rand() * margin)
        random_margin_y = int(np.random.rand() * margin)

        x_min = max(joints2d[:, 0].min() - random_margin_x, 0.0)
        y_min = max(joints2d[:, 1].min() - random_margin_y, 0.0)
        x_max = min(joints2d[:, 0].max() + random_margin_y, 256.0)  # x->y
        y_max = min(joints2d[:, 1].max() + random_margin_x, 256.0)  # x->y
        x_width = x_max - x_min
        y_height = y_max - y_min

        line_segment = max(x_width, y_height)  # for square bbox
        bbox = [x_min, y_min, line_segment, line_segment]
        image = np.array(image)
        aug_img, img2bb_trans, bb2img_trans, rot, _, cam, cam_nh, _ = augmentation(
            image, bbox, self.mode, exclude_flip=True, rotation=True, cam_param=intr
        )  # FreiHAND dataset only contains right hands. do not perform flip aug.

        rot = -rot * np.pi / 180.0
        rot_mat = np.array(
            [
                [np.cos(rot), -np.sin(rot), 0],
                [np.sin(rot), np.cos(rot), 0],
                [0, 0, 1],
            ]
        ).astype(np.float32)

        # Since mobrecon estimates root-aligned 3D verts, make root joint (i.e. joints3d[0]) at the origin
        # rot_joints = rot_mat.dot(joints3d.T).T
        rot_joints = rot_mat.dot(joints3d.transpose(1, 0)).transpose(1, 0)
        root_xyz = rot_joints[0].copy()
        # import pdb; pdb.set_trace()

        align_joints = rot_joints - root_xyz

        # affine transform to joints2d
        ori_2d_mat = np.concatenate((joints2d, np.ones((21, 1))), axis=1)
        kps = (img2bb_trans @ ori_2d_mat.T).T[:, :2]

        kps25d = np.concatenate((kps, align_joints[:, 2:]), axis=1)
        # image resizer
        return_img = self.image_transform(Image.fromarray(aug_img.astype(np.uint8)))

        kps25d[:, :2] /= 256.0
        kps = kps25d[:, :2]  # (21, 2)

        new_cam = cam_nh
        new_cam[:, 2] = cam[:, 2]

        # cv2.imwrite('img.png', cv2.cvtColor(draw_joint2D(torch.from_numpy(image).permute(2, 0, 1), cam2pixel(rot_joints, new_cam)[:, :2] / 256, idx=None), cv2.COLOR_RGB2BGR))
        # cv2.imwrite('img2.png', cv2.cvtColor(draw_joint2D(torch.from_numpy(image).permute(2, 0, 1), cam2pixel(rot_joints, intr)[:, :2] / 256, idx=None), cv2.COLOR_RGB2BGR))

        return {
            # "image": aug_img,
            "image": return_img,
            "keypoints3D": align_joints,
            # "keypoints": kps25d,
            "keypoints2D": kps,  # 256 normalization
            # "add_noise": add_noise,
            # "prev_additional": prev
            "root": root_xyz,
            "cam": new_cam,
        }


# python .\datasets\freihand.py --cfg .\configs.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=os.path.join("configs.yaml"))
    parser.add_argument("--gpu", default=None, type=int, help="Specify a GPU device")

    opt = parser.parse_args()

    cfg = load_cfg(opt.cfg)
    device = setup_runtime(opt)
    dataset = HanCo_ETRI_jitter(cfg, mode="train", limit=1e2)
    print(len(dataset))
    # import pdb; pdb.set_trace()
    for i in range(0, len(dataset), len(dataset) // 50):
        print(i)
        data = dataset.__getitem__(i)
        # import pdb; pdb.set_trace()
        img = draw_joint2D(data["image"], data["keypoints2D"])
        # img = draw_joint(data['image'], data['keypoints2D'], data['K'])
        cv2.imwrite("img.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
