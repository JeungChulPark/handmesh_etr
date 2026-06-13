import mediapipe as mp
import os
import cv2
import math
import json
from tqdm import tqdm
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

IMG_SIZE = 240
categories = os.listdir('rgb/')


def cut_center(img):
    row, col, _ = img.shape
    min_size = min(row, col)
    
    image_cut = img[(row - min_size)//2:row - (row-min_size)//2, (col - min_size)//2:col - (col-min_size)//2]
                    
    return image_cut

res = {}
with mp_hands.Hands(    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    
    idx =  0
    for scene_num in categories:
        # import pdb; pdb.set_trace()
        data_path = os.path.join('rgb', scene_num)
        cam_views = os.listdir(data_path)
        cam_dict = {}
        for cams in cam_views: 
            data_path = os.path.join('rgb', scene_num, cams)
            imgs = os.listdir(data_path)
            img_dict = {}
            for img in imgs:
                image = cv2.imread(os.path.join(data_path, img), cv2.COLOR_BGR2RGB)
                row, col, _ = image.shape
                image = cut_center(image)
                results = hands.process(image)

                if results.multi_hand_landmarks:
                    hand_res = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for each_land in hand_landmarks.landmark:
                            hand_res.append([each_land.x, each_land.y, each_land.z])
                    # cv2.imwrite(os.path.join('rgb', f'{idx}.jpg'), image)
                    img_dict[img] = hand_res
                    idx += 1
            print(f'{scene_num} {cams} done')
            cam_dict[cams] = img_dict
        res[scene_num] = cam_dict
        
    with open(os.path.join('mediapipe_pixel.json'), 'w') as outfile:
        json.dump(res, outfile)