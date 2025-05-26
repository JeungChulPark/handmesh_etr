import torch
import torch.nn as nn
import random
import numpy as np
import os

from yacs.config import CfgNode as CN
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import math
from PIL import Image

N_KEYPOINTS = 21
N_IMG_CHANNELS = 3
RAW_IMG_SIZE = 224
MODEL_IMG_SIZE = 240

_C = CN(new_allowed=True)

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    defaults_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'defaults.yaml'))
    _C.merge_from_file(defaults_abspath)
    _C.set_new_allowed(False)
    return _C.clone()

def load_cfg(path=None):
    _C.set_new_allowed(True)
    _C.merge_from_file(path)
    _C.set_new_allowed(False)
    return _C.clone()


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

    device = f'cuda' if torch.cuda.is_available() else 'cpu'


    print(f"Environment: GPU {device} seed {seed} number of workers {num_workers}")

    return device

def tensor2img(input_tensor, batch_id=0, as_cv=False): # input_tensor [B, 3(1), H, W]
    batch_id = batch_id if batch_id else 0
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    target_tensor = input_tensor[batch_id]
    
    if target_tensor.size(0) == 1: # Mask img
        target_tensor = target_tensor.repeat(3, 1, 1)
    img = target_tensor.permute(1, 2, 0).detach().cpu().numpy()
    if not (target_tensor.max() > 1):
        img *= 255
    img = img.astype(np.uint8)
    
    return  img[..., ::-1] if as_cv else img


COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g", 'color_val': [0, 255, 0]},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c", 'color_val': [255, 255, 0]},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b", 'color_val': [255, 0, 0]},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m", 'color_val': [0, 255, 255]},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r", 'color_val': [0, 0, 255]},
}


def projectPoints(xyz, K):
    """
    Projects 3D coordinates into image space.
    Function taken from https://github.com/lmb-freiburg/freihand
    """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T

    
    return uv[:, :2] / uv[:, -1:]

def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t



def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2

def get_norm_params(dataloader):
    """
    Calculates image normalization parameters.
    Mean and Std are calculated for each channel separately.

    Borrowed from this StackOverflow discussion:
    https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch
    """
    mean = 0.0
    std = 0.0
    nb_samples = 0.0

    for i, sample in tqdm(enumerate(dataloader)):
        data = sample["image_raw"]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return {"mean": mean, "std": std}


def vector_to_heatmaps(keypoints):
    """
    Creates 2D heatmaps from keypoint locations for a single image
    Input: array of size N_KEYPOINTS x 2
    Output: array of size N_KEYPOINTS x MODEL_IMG_SIZE x MODEL_IMG_SIZE
    """
    heatmaps = np.zeros([N_KEYPOINTS, MODEL_IMG_SIZE, MODEL_IMG_SIZE])
    for k, (x, y) in enumerate(keypoints):
        x, y = int(x * MODEL_IMG_SIZE), int(y * MODEL_IMG_SIZE)
        if (0 <= x < MODEL_IMG_SIZE) and (0 <= y < MODEL_IMG_SIZE):
            heatmaps[k, int(y), int(x)] = 1

    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps


def blur_heatmaps(heatmaps):
    """Blurs heatmaps using GaussinaBlur of defined size"""
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
    return heatmaps_blurred



def heatmaps_to_coordinates(heatmaps):
    """
    Heatmaps is a numpy array
    Its size - (batch_size, n_keypoints, img_size, img_size)
    """
    batch_size = heatmaps.shape[0]
    sums = heatmaps.sum(axis=-1).sum(axis=-1)
    sums = np.expand_dims(sums, [2, 3])
    normalized = heatmaps / sums
    x_prob = normalized.sum(axis=2)
    y_prob = normalized.sum(axis=3)

    arr = np.tile(np.float32(np.arange(0, 128)), [batch_size, 21, 1])
    x = (arr * x_prob).sum(axis=2)
    y = (arr * y_prob).sum(axis=2)
    keypoints = np.stack([x, y], axis=-1)
    return keypoints / 128


def draw_joint(img, keypoints, K, idx=None):
    if idx is not None:
        keypoints, K= keypoints[idx], K[idx]
    ori_img_size = K[1, 2] * 2
    image_raw = tensor2img(img, batch_id=idx, as_cv=False).copy()
    keypoints_2d = projectPoints(keypoints, K) #  * img_size / ori_img_size

    px_landmark = np.array(keypoints_2d, dtype=np.int32)
    
    for finger, params in COLORMAP.items():
        points = [px_landmark[params['ids']]]
        cv2.polylines(image_raw, points, False, params['color_val'])
    
    return image_raw

def draw_joint2D(img, keypoints, idx=None):
    if idx is not None:
        keypoints= keypoints[idx]

    image_raw = tensor2img(img, batch_id=idx, as_cv=False).copy()
    img_size_x = img.size(-1)
    img_size_y = img.size(-2)
    
    px_landmark = []
    for each_land in keypoints: 
        landmark_px = normalized_to_pixel_coordinates(each_land[0], each_land[1], img_size_x, img_size_y)
        px_landmark.append(list(landmark_px))
    
    px_landmark = np.array(px_landmark)
    
    for finger, params in COLORMAP.items():
        points = [px_landmark[params['ids']]]
        cv2.polylines(image_raw, points, False, params['color_val'])

    return image_raw

def show_data(dataset, n_samples=12):
    """
    Function to visualize data
    Input: torch.utils.data.Dataset
    """
    n_cols = 4
    n_rows = int(np.ceil(n_samples / n_cols))
    plt.figure(figsize=[15, n_rows * 4])

    ids = np.random.choice(dataset.__len__(), n_samples, replace=False)
    for i, id_ in enumerate(ids, 1):
        sample = dataset.__getitem__(id_)

        image = sample["image_raw"].numpy()
        image = np.moveaxis(image, 0, -1)
        keypoints = sample["keypoints"].numpy()
        keypoints = keypoints * RAW_IMG_SIZE

        plt.subplot(n_rows, n_cols, i)
        plt.imshow(image)
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c="k", alpha=0.5)
        for finger, params in COLORMAP.items():
            plt.plot(
                keypoints[params["ids"], 0],
                keypoints[params["ids"], 1],
                params["color"],
            )
    plt.tight_layout()
    plt.show()


def show_batch_predictions(batch_data, model):
    """
    Visualizes image, image with actual keypoints and
    image with predicted keypoints.
    Finger colors are in COLORMAP.

    Inputs:
    - batch data is batch from dataloader
    - model is trained model
    """
    inputs = batch_data["image"]
    true_keypoints = batch_data["keypoints"].numpy()
    batch_size = true_keypoints.shape[0]
    pred_heatmaps = model(inputs)
    pred_heatmaps = pred_heatmaps.detach().numpy()
    pred_keypoints = heatmaps_to_coordinates(pred_heatmaps)
    import pdb; pdb.set_trace()
    images = batch_data["image_raw"].numpy()
    images = np.moveaxis(images, 1, -1)

    plt.figure(figsize=[12, 4 * batch_size])
    for i in range(batch_size):
        image = images[i]
        true_keypoints_img = true_keypoints[i] * RAW_IMG_SIZE
        pred_keypoints_img = pred_keypoints[i] * RAW_IMG_SIZE

        plt.subplot(batch_size, 3, i * 3 + 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(batch_size, 3, i * 3 + 2)
        plt.imshow(image)
        for finger, params in COLORMAP.items():
            plt.plot(
                true_keypoints_img[params["ids"], 0],
                true_keypoints_img[params["ids"], 1],
                params["color"],
            )
        plt.title("True Keypoints")
        plt.axis("off")

        plt.subplot(batch_size, 3, i * 3 + 3)
        plt.imshow(image)
        for finger, params in COLORMAP.items():
            plt.plot(
                pred_keypoints_img[params["ids"], 0],
                pred_keypoints_img[params["ids"], 1],
                params["color"],
            )
        plt.title("Pred Keypoints")
        plt.axis("off")
    plt.tight_layout()


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
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




def load_model(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    # delete unmatched total_ops total_params
    state_dict = []
    for n, p in checkpoint.items():
        if "total_ops" not in n and "total_params" not in n:
            state_dict.append((n, p))
    state_dict = dict(state_dict)
    
    model.load_state_dict(state_dict, strict=False)
    print('Loaded pre-trained weight: {}'.format(path))
    return model



def crop_bb(joint, img_size=[1920, 1080], margin=10):

    x_list = joint[:, 0]
    y_list = joint[:, 1]

    x_list = np.array(x_list)
    y_list = np.array(y_list)
        
    x_max = np.argmax(x_list)
    x_min = np.argmin(x_list)
    y_max = np.argmax(y_list)
    y_min = np.argmin(y_list)    
    center = np.array([(x_list[x_min] * img_size[0] + x_list[x_max] * img_size[0]) / 2, (y_list[y_min] * img_size[1] + y_list[y_max] * img_size[1]) / 2], np.int32)
    center = np.clip(center, 0, img_size[0])
    xx = x_list[x_max] * img_size[0] - center[0]
    yy = y_list[y_max] * img_size[1] - center[1]
    
    if xx > yy:
        line = xx + margin
    else:
        line = yy + margin
        
    bb = [[max(0, center[0] - line), max(0, center[1] - line)] , [min(img_size[0], center[0] + line), min(img_size[1], center[1] + line)]]
    joint_x = (x_list * img_size[0] - max(0, center[0] - line)) / (line * 2)
    joint_y = (y_list * img_size[1] - max(0, center[1] - line)) / (line * 2)
    return np.array(bb, np.int32), np.array([joint_x, joint_y, joint[:, -1]]).transpose([1, 0])


def crop_bb_mediapipe(hand_landmarks_list, img_size=[1920, 1080]):
    x_list = []
    y_list = []
    for i in range(21):
        x_list.append(hand_landmarks_list[i][0])
        y_list.append(hand_landmarks_list[i][1])
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    
    x_max = np.argmax(x_list)
    x_min = np.argmin(x_list)
    y_max = np.argmax(y_list)
    y_min = np.argmin(y_list)    
    center = [(x_list[x_min] * img_size[0] + x_list[x_max] * img_size[0]) / 2, (y_list[y_min] * img_size[1] + y_list[y_max] * img_size[1]) / 2]
    center = np.clip(center, 0, img_size[0])
    xx = x_list[x_max] * img_size[0] - center[0]
    yy = y_list[y_max] * img_size[1] - center[1]
    margin = 40
    if xx > yy:
        line = xx + margin
    else:
        line = yy + margin
    
    bb = [[max(0, center[0] - line), max(0, center[1] - line)] , [min(img_size[0]-2, center[0] + line), min(img_size[1]-2, center[1] + line)]]

    return np.array(bb, np.int32)