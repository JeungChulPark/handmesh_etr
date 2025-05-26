import mediapipe as mp
import os
import cv2
import math
import json
from tqdm import tqdm
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


files = os.listdir(os.path.join('train','rgb_hand'))

# with open("training_K.json", "r") as f:
#     K_matrix = json.load(f)


res = {}
# camera_res = {}
# import pdb; pdb.set_trace()
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
       
    for step, file_name in enumerate(files):
        file_name = file_name
        image = cv2.imread(os.path.join('train','rgb_hand', file_name), cv2.COLOR_BGR2RGB)
        row, col, _ = image.shape
        results = hands.process(image)

        if results.multi_hand_landmarks:
            hand_res = []
            for hand_landmarks in results.multi_hand_landmarks:
                for each_land in hand_landmarks.landmark:
                    hand_res.append([each_land.x, each_land.y, each_land.z])
                    
            res[file_name] = hand_res
            # camera_res[file_name] = each_K
           
        if step % 500 == 0 and step > 1:
            print(step)
        
    # import pdb; pdb.set_trace()



    with open(os.path.join('train', 'mediapipe_pixel.json'), 'w') as outfile:
        json.dump(res, outfile)
    # with open('mediapipe_K.json', 'w') as outfile:
    #     json.dump(camera_res, outfile)