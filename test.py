import os
import shutil
from time import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable as V

from utils.utils import get_patches, stitch_together
from model.EDB import Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--parameters_path', type=str, required=True, help='parameters_path')
parser.add_argument('--testimg_path', type=str, required=True, default='')
args = parser.parse_args()

BATCHSIZE_PER_CARD = 32

img_outdir = testimg_path + "_Binarized"

if torch.cuda.is_available():
    print("CUDA可用")
else:
    print("CUDA不可用，正在使用CPU")


class TTAFrame:
    def __init__(self, net):
        self.net = net().cuda(1)
        self.net.eval()

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        else:
            print("请将batch_size设为大于等于8")

    def test_one_img(self, img):
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float).unsqueeze(0).to("cuda:1")
        img = img.permute(0, 3, 1, 2)
        pred = self.net.forward(img)

        mask = pred.squeeze().cpu().data.numpy()

        return mask


    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))


solver = TTAFrame(Model)

if os.path.exists(img_outdir):
    shutil.rmtree(img_outdir)  # 清空输出文件夹
os.makedirs(img_outdir)

img_list = os.listdir(testimg_path)
img_list.sort()


start_time = time()
solver.load(model_path)

# 为当前 epoch 创建输出文件夹
mean_fm = 0
mean_psnr = 0
with torch.no_grad():
    for idx in range(len(img_list)):
        print("Processing %s: " % img_list[idx])
        if os.path.isdir(os.path.join(testimg_path, img_list[idx])):
            continue

        img_input = os.path.join(testimg_path, img_list[idx])

        img = cv2.imread(img_input)
        locations, patches, padding_info = get_patches(img, patch_h=256, patch_w=256)
        masks = []
        for idy in range(len(patches)):
            msk = solver.test_one_img(patches[idy])
            masks.append(msk)
        if padding_info:
            padding_h, padding_w = padding_info
            padded_shape = (img.shape[0] + padding_h, img.shape[1] + padding_w, img.shape[2])  # 填充后的形状
        else:
            padded_shape = img.shape  # 如果没有填充，使用原始形状
        prediction = stitch_together(locations, masks, padded_shape[0:2], 256, 256, padding_info)
        prediction[prediction >= 0.5] = 255
        prediction[prediction < 0.5] = 0

        # 保存结果到以 epoch 命名的文件夹
        fname, fext = os.path.splitext(img_list[idx])
        img_output = os.path.join(img_outdir, f"{fname}.tiff")
        cv2.imwrite(img_output, prediction.astype(np.uint8))
    print("Total running time: %f sec." % (time() - start_time))
    print("Finished!")
