

""" 
   Testing code for the paper "SalBiNet360: Saliency Prediction on 360 images with Local-Global Bifurcated Deep Network"
   which is submitted in IEEE VR2020
"""  


import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
from scipy import misc
from tqdm import tqdm
from SalBiNet360 import SalBiNet360
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=352, help='testing size')
parser.add_argument('--gpu', type=int, default=0, help='which GPU to run')
parser.add_argument('--type', type=str, default='global', choices=['global', 'local'], help='global or local')
parser.add_argument('--data', type=str, default='', help='path to testing data')
parser.add_argument('--pth', type=str, default='./pretrained/SalBiNet.pth', help='path to pretrained model')
parser.add_argument('--save_path', type=str, default='', help='path to save files')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

if opt.data == '':
    opt.data = './data/global' if opt.type == 'global' else './data/local'

if opt.save_path == '':
    opt.save_path = './output/global' if opt.type == 'global' else './output/local'

model = SalBiNet360()

print("====> loading model checkpoint")
model.load_state_dict(torch.load(opt.pth))

model.cuda()
model.eval()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

def trans_img(img, stride):
    tmp = np.zeros(img.shape, dtype=img.dtype)
    tmp[:, -stride:] = img[:, :stride]
    tmp[:, :img.shape[1]-stride] = img[:, stride:]
    return tmp

print("====> start testing")
test_loader = test_dataset(opt.data, opt.size, opt.type)
if opt.type == 'global':
    for i in tqdm(range(test_loader.size)):
        image, image_back, image_side1, image_side2, ori_size, name = test_loader.load_data()
        image = image.cuda()
        image_back = image.cuda()
        image_side1 = image_side1.cuda()
        image_side2 = image_side2.cuda()

        res, _ = model(image)
        res_back, _ = model(image_back)
        res_side1, _ = model(image_side1)
        res_side2, _ = model(image_side2)

        res = F.interpolate(res, size=ori_size, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        res_back = F.interpolate(res_back, size=ori_size, mode='bilinear', align_corners=False)
        res_back = res_back.sigmoid().data.cpu().numpy().squeeze()
        res_back = (res_back - res_back.min()) / (res_back.max() - res_back.min() + 1e-8)
        res_back = trans_img(res_back, ori_size[1] // 2)

        res_side1 = F.interpolate(res_side1, size=ori_size, mode='bilinear', align_corners=False)
        res_side1 = res_side1.sigmoid().data.cpu().numpy().squeeze()
        res_side1 = (res_side1 - res_side1.min()) / (res_side1.max() - res_side1.min() + 1e-8)
        res_side1 = trans_img(res_side1, (ori_size[1] // 4) * 3)

        res_side2 = F.interpolate(res_side2, size=ori_size, mode='bilinear', align_corners=False)
        res_side2 = res_side2.sigmoid().data.cpu().numpy().squeeze()
        res_side2 = (res_side2 - res_side2.min()) / (res_side2.max() - res_side2.min() + 1e-8)
        res_side2 = trans_img(res_side2, ori_size[1] // 4)

        res = 0.25*res + 0.25*res_back + 0.25*res_side1 + 0.25*res_side2

        misc.imsave(os.path.join(opt.save_path, name), res)
else:
    for i in tqdm(range(test_loader.size)):
        image, ori_size, name = test_loader.load_data()
        image = image.cuda()
        _, res = model(image)
        res = F.interpolate(res, size=ori_size, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        misc.imsave(os.path.join(opt.save_path, name), res)

print('Done')

