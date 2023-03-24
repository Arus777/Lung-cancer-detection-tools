import argparse
import time

import os
import cv2

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import re
from model import Generator
import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import LUNA16DatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Test LUNA16 Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--model_name', default='netG_epoch_4_198.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))


"""
def getFileList(dir, Filelist, ext=None):
    
    #获取文件夹及其子文件夹中文件列表
    #输入 dir：文件夹根目录
    #输入 ext: 扩展名
    #返回： 文件路径列表
  
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


org_img_folder = 'F:/DeepLearning/yolov5_mask_recognition/yolov5-5.0/VOCdevkit/images/train'

# 检索文件
imglist = getFileList(org_img_folder, [], 'png')
print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')
imgname = os.path.splitext(os.path.basename(imgpath))[0]
    image = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    # 对每幅图像执行相关操作
"""
test_set = LUNA16DatasetFromFolder('F:/DeepLearning/yolov5_mask_recognition/yolov5-5.0/VOCdevkit/images/val/', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing LUNA16 datasets]')

for image_name, image in test_bar:
    image_name = image_name[0]
   # print(image_name)


    #IMAGE_NAME= imgname
    #image = Image.open(IMAGE_NAME)
    with torch.no_grad():
        image = Variable(image)
    if TEST_MODE:
        image = image.cuda()

    start = time.clock()
    with torch.no_grad():
        out = model(image)
    elapsed = (time.clock() - start)
    print('\ncost' + str(elapsed) + 's')
    #out_img = ToPILImage()(out[0].data.cpu())
    print(image_name)
    utils.save_image(out, 'data/4origin/val/' + image_name, padding=5)
    torch.cuda.empty_cache()
