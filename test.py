import cv2
import torch
from torchvision import transforms
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *

import argparse
from models.mobrecon_ds import LargeModel, SmallModel

from PIL import Image
from utils import COLORMAP
import time

capture = cv2.VideoCapture(0) #카메라 정보 받아옴
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #카메라 속성 설정
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # width:너비, height: 높이

img_size=240
inputScale=1.0/255


image_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
    )

def test_img():
    net = (cfg).to(device)
    net = load_model(net, os.path.join('pretrain', 'ETRI_224.pth'))
    net.eval()
    
    src = Image.open('1.jpg')
    src = image_transform(src)
    src = src.unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = net(src)
    
    print(out)
    res_img = draw_joint2D(src, out['keypoints'], idx=0)
    cv2.imshow("Output-Keypoints", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    
    ax = plt.axes(projection='3d')
    
    fig = plt.figure()
    xdata = np.array(out['keypoints'][0, :, 0])
    ydata = np.array(out['keypoints'][0, :, 1])
    zdata = np.array(out['keypoints'][0, :, 2])
    ax.scatter3D(xdata, ydata, zdata, c=zdata)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=os.path.join('configs.yaml'))
    parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
    parser.add_argument('--dataset', default='freihand', type=str, help='obman, freihand')
    
    opt = parser.parse_args()
    
    cfg = load_cfg(opt.cfg)
    device = setup_runtime(opt) 
    
    net = LargeModel(cfg).to(device) if cfg.MODEL.NAME == 'LargeModel' else SmallModel(cfg).to(device)
    net = load_model(net, cfg.MODEL.PRETRAIN)
    net.eval()


    # test_cam()
    test_img()
