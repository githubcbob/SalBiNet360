

""" 
   Training code for the paper "SalBiNet360: Saliency Prediction on 360 images with Local-Global Bifurcated Deep Network"
   which is submitted in IEEE VR2020
"""   

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from SalBiNet360 import SalBiNet360
from data import get_loader, get_loader_val  
from utils import clip_gradient, adjust_lr

from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--gpu', type=int, default=0, help='which GPU to run')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)

### metrics constants
KLmiu = 2.4948
KLstd = 1.7421
CCmiu = 0.3932
CCstd = 0.2565
NSSmiu = 0.4539
NSSstd = 0.2631
bcemiu = 0.3194
bcestd = 0.1209
alpha = 0.05



### evaluation metrics 
def CC(output, target):
    output = (output - torch.mean(output)) / torch.std(output)
    target = (target - torch.mean(target)) / torch.std(target)
    num = (output - torch.mean(output)) * (target - torch.mean(target))
    out_square = (output - torch.mean(output)) * (output - torch.mean(output))
    tar_square = (target - torch.mean(target)) * (target - torch.mean(target))
    CC_score = torch.sum(num) / (torch.sqrt(torch.sum(out_square) * torch.sum(tar_square)))
    return CC_score

def NSS_global(output, fixationMap):
    output = (output-torch.mean(output))/torch.std(output)
    Sal = output*fixationMap
    NSS_score = torch.sum(Sal)/torch.sum(fixationMap)
#    print('NSS_global:', NSS_score)
    return NSS_score

def NSS_local(output, fixationMap):
    output = (output-torch.mean(output))/torch.std(output)
    Sal = output*fixationMap
    NSS_score = torch.sum(Sal)/torch.sum(fixationMap)
#    print('NSS_local:', NSS_score)
    return NSS_score


print('Learning Rate: {} Trainset: {}'.format(opt.lr, opt.trainsize))
model = SalBiNet360()
model.cuda()

print('====> loading pre-trained model checkpoint')
model_dict = model.state_dict()
pretrained_dict = torch.load('./pre-trained_on_SALICON.pth') #need to change!!!!!!!!!!!!!!!!!!!!!!!!!

pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)


image_root = './training_data/MCP_training/image256x192/'
gt_root = './training_data/MCP_training/salmap256x192/'
fixmap_root = './training_data/MCP_training/fixmap256x192/'
image_val_root = './training_data/MCP_training/image256x192/'
gt_val_root = './training_data/MCP_training/salmap256x192/'
fixmap_val_root = './training_data/MCP_training/fixmap256x192/'


train_loader = get_loader(image_root, gt_root, fixmap_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
val_loader = get_loader_val(image_val_root, gt_val_root, fixmap_val_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
total_val_step = len(val_loader)
counter = 0
counter_val = 0
print('total_step:', total_step)
print('total_val_step:', total_val_step)

CE = torch.nn.BCEWithLogitsLoss()

writer = SummaryWriter(comment='scalar')

def train(train_loader, model, optimizer, epoch, trainsize):
    model.train()

    global counter
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, fixmaps = pack
        images = Variable(images)
        gts = Variable(gts)
        fixmaps = Variable(fixmaps)
        images = images.cuda()
        gts = gts.cuda()
        fixmaps = fixmaps.cuda()

        global_map, local_map = model(images)

        loss1 = CE(global_map, gts) + alpha * (bcemiu + bcestd * ((1.)*((CC(global_map, gts) - CCmiu) / CCstd) - (1.)*((NSS_global(global_map, fixmaps) - NSSmiu) / NSSstd)))
        loss2 = CE(local_map, gts) + alpha * (bcemiu + bcestd * ((1.)*((CC(local_map, gts) - CCmiu) / CCstd) - (1.)*((NSS_local(local_map, fixmaps) - NSSmiu) / NSSstd)))
        loss = loss1 + loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        writer.add_scalar('Loss1', loss1.item(), counter)
        writer.add_scalar('Loss2', loss2.item(), counter)
        writer.add_scalar('Loss', loss.item(), counter)

        if i % 400 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))
        counter += 1


    save_path = './results/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'fine-tune.pth')  ## need to change!!!!!!!!!!!!!!!!!!!!!!!

def val(val_loader, model, epoch, trainsize):
    model.eval()
    global counter_val

    for i, pack in enumerate(val_loader, start=1):
        images_val, gts_val, fixmaps_val = pack
        
        images_val = Variable(images_val)
        gts_val = Variable(gts_val)
        fixmaps_val = Variable(fixmaps_val)
        images_val = images_val.cuda()
        gts_val = gts_val.cuda()
        fixmaps_val = fixmaps_val.cuda()

        global_map_val, local_map_val = model(images_val)

        loss1_val = CE(global_map_val, gts_val) + alpha * (bcemiu + bcestd * ((1.)*((CC(global_map_val, gts_val) - CCmiu) / CCstd) - (1.)*((NSS_global(global_map_val, fixmaps_val) - NSSmiu) / NSSstd)))
        loss2_val = CE(local_map_val, gts_val) + alpha * (bcemiu + bcestd * ((1.)*((CC(local_map_val, gts_val) - CCmiu) / CCstd) - (1.)*((NSS_local(local_map_val, fixmaps_val) - NSSmiu) / NSSstd)))
        loss_val = loss1_val + loss2_val


        writer.add_scalar('Loss1_val', loss1_val.item(), counter_val)
        writer.add_scalar('Loss2_val', loss2_val.item(), counter_val)
        writer.add_scalar('Loss_val', loss_val.item(), counter_val)

        if i % 400 == 0 or i == total_val_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1_val: {:.4f} Loss2_val: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_val_step, loss1_val.data, loss2_val.data))
        counter_val += 1

###validation_sample visualization
    transform = transforms.Compose([
#        transforms.Resize((trainsize, trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validation_sample = cv2.imread('path to validation image')     #need to change!!!!!!!!!!!!!!!!!!!!!!!
    validation_sample = cv2.resize(validation_sample, (352,352), interpolation=cv2.INTER_AREA)
    validation_sample = transform(validation_sample).unsqueeze(0)
    validation_sample = validation_sample.cuda()
    global_sample, local_sample = model(validation_sample)


    global_sample = F.upsample(global_sample, size=(352, 352), mode='bilinear', align_corners=False)
#    res = res.squeeze(0).squeeze(0)
#    res = res.sigmoid().data.cpu().numpy().squeeze()
    global_sample = global_sample.sigmoid()
    global_sample = (global_sample - global_sample.min()) / (global_sample.max() - global_sample.min() + 1e-8)

    global_sample = torch.cuda.FloatTensor(global_sample)
    global_sample = vutils.make_grid(global_sample, normalize=True, scale_each=True)

    local_sample = F.upsample(local_sample, size=(352, 352), mode='bilinear', align_corners=False)
#    att = att.squeeze(0).squeeze(0)
#    att = att.sigmoid().data.cpu().numpy().squeeze()
    local_sample = local_sample.sigmoid()
    local_sample = (local_sample - local_sample.min()) / (local_sample.max() - local_sample.min() + 1e-8)
#    att = att.cuda()
    
    local_sample = torch.cuda.FloatTensor(local_sample)
    local_sample = vutils.make_grid(local_sample, normalize=True, scale_each=True)
    writer.add_image('global_sample', global_sample, epoch)
    writer.add_image('local_sample', local_sample, epoch)


print("====> start training")

for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch, opt.trainsize)
    val(val_loader, model, epoch, opt.trainsize)

writer.close()
print('Done')
