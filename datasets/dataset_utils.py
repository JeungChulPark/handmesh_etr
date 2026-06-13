# retreived from https://github.com/mks0601/I2L-MeshNet_RELEASE/blob/master/common/utils/preprocessing.py


import numpy as np
import cv2
import random
import math

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid):

    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1]
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img)

    x_center = (xmin+xmax)/2.; width = xmax-xmin
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, img_width, img_height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 224/224
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox

def get_aug_config(exclude_flip, rotation):
    scale_factor = 0.25
    rot_factor = 60
    color_factor =  0.2 # 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    # scale = 1.
    
    if rotation:
        rot = np.clip(np.random.randn(), -2.0,
                    2.0) * rot_factor if random.random() <= 0.6 else 0


    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    return scale, rot, color_scale, do_flip

def augmentation_encoder(img, bbox, data_split, exclude_flip=False, rotation=True, cam_param=None, need_heatmap=False):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config(exclude_flip, rotation)
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False

    # print("scale: ", scale, " rot: ", rot)

    img, trans, inv_trans, trans_nh = generate_patch_image(img, bbox, scale, rot, do_flip, [256, 256], need_heatmap=need_heatmap)
    img = np.clip(img * color_scale[None,None,:], 0, 255)

    if cam_param is not None:
        cam = trans @ cam_param
        cam = np.concatenate([cam, np.array([[0., 0., 1.]])], axis=0)

        _, no_rot_trans, _, _ = generate_patch_image(img, bbox, scale, 0., do_flip, [256, 256], need_heatmap=need_heatmap)


        if need_heatmap:
            cam_nh = trans_nh @ cam_param
            cam_nh = np.concatenate([cam_nh, np.array([[0., 0., 1.]])], axis=0)

            return img, trans, inv_trans, rot, do_flip, cam, cam_nh, None

        else:
            return img, trans, inv_trans, rot, do_flip, cam, None, no_rot_trans
    else:
        return img, trans, inv_trans, rot, do_flip, None, None, None


def augmentation_hamer(img, bbox, data_split, exclude_flip=False, rotation=True, cam_param=None, need_heatmap=False):
    scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False

    img, trans, inv_trans, trans_nh = generate_patch_image(img, bbox, scale, rot, do_flip, [224, 224], need_heatmap=need_heatmap)
    img = np.clip(img * color_scale[None,None,:], 0, 255)

    if cam_param is not None:
        cam = trans @ cam_param
        cam = np.concatenate([cam, np.array([[0., 0., 1.]])], axis=0)

        _, no_rot_trans, _, _ = generate_patch_image(img, bbox, scale, 0., do_flip, [224, 224], need_heatmap=need_heatmap)

        # maybe use trans_

        cam_nh = trans_nh @ cam_param
        cam_nh = np.concatenate([cam_nh, np.array([[0., 0., 1.]])], axis=0)

        return img, trans, inv_trans, rot, do_flip, cam, cam_nh, no_rot_trans
    else:
        return img, trans, inv_trans, rot, do_flip, None, None, None

def augmentation(img, bbox, data_split, exclude_flip=False, rotation=True, cam_param=None, need_heatmap=False):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config(exclude_flip, rotation)
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False

    img, trans, inv_trans, trans_nh = generate_patch_image(img, bbox, scale, rot, do_flip, [256, 256], need_heatmap=need_heatmap)
    img = np.clip(img * color_scale[None,None,:], 0, 255)

    if cam_param is not None:
        cam = trans @ cam_param
        cam = np.concatenate([cam, np.array([[0., 0., 1.]])], axis=0)

        _, no_rot_trans, _, _ = generate_patch_image(img, bbox, scale, 0., do_flip, [256, 256], need_heatmap=need_heatmap)

        # maybe use trans_

        cam_nh = trans_nh @ cam_param
        cam_nh = np.concatenate([cam_nh, np.array([[0., 0., 1.]])], axis=0)

        return img, trans, inv_trans, rot, do_flip, cam, cam_nh, no_rot_trans
    else:
        return img, trans, inv_trans, rot, do_flip, None, None, None


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape, need_heatmap=False):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
   
    if need_heatmap:
        bb_c_x_nh = float(112.)
        bb_c_y_nh = float(112.)
        bb_width_nh = float(224.)
        bb_height_nh = float(224.)
    
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    # #'''
    # bb_c_x = float(112.)
    # bb_c_y = float(112.)
    # bb_width = float(224.)
    # bb_height = float(224.)
    # #'''
    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    # if need_heatmap:
    #     heatmap_trans = gen_trans_from_patch_cv(bb_c_x_nh, bb_c_y_nh, bb_width_nh, bb_height_nh, out_shape[1], out_shape[0], scale, rot)
    no_rot_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, 0)

    return img_patch, trans, inv_trans, no_rot_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def generate_heatmap(joints, img_size=[224, 224], heatmap_size=[56, 56,], sigma=2):
    '''
    :param joints:  [num_joints, 2]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    
    target_weight = np.ones((21, 1), dtype=np.float32)
    # target_weight[:, 0] = 1. # 0 joint is always visible
    
    target = np.zeros((21,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3
    feat_stride = [img_size[0] / heatmap_size[0], img_size[1] / heatmap_size[1]]

    for joint_id in range(21):
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

        

        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        # print(f'mu_x {mu_x} / mu_y {mu_y} / ul 0 {ul[0]} / ul 1 {ul[1]} / br 0 {br[0]} / br 1 {br[1]}')


        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            print(f'ul 0 {ul[0]} / ul 1 {ul[1]} / br 0 {br[0]} / br 1 {br[1]}')

            if ul[0] >= heatmap_size[0]:
                mu_x = heatmap_size[0] + sigma

            if ul[1] >= heatmap_size[1]:
                mu_y = heatmap_size[1] + sigma

            if br[0] < 0:
                mu_x = 0 - sigma
                

            if br[1] < 0:
                mu_y = 0 - sigma

            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            '''
            return None
            target_weight[joint_id] = 0

            print(joint_id)
            print(joints[joint_id][0])
            print(joints[joint_id][1])

            print(mu_x)
            print(mu_y)


            print(ul[0])
            print(ul[1])
            print(br[0])
            print(br[1])
            '''
            # continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target