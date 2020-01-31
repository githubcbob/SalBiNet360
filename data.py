import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch

Image.MAX_IMAGE_PIXELS = None


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, fixmap_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.startswith('train')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.startswith('train')]
        self.fixmaps = [fixmap_root + f for f in os.listdir(fixmap_root) if f.startswith('train')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.fixmaps = sorted(self.fixmaps)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
#            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
#            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.fixmap_transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        fixmap = self.binary_loader(self.fixmaps[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        fixmap = self.fixmap_transform(fixmap)
        return image, gt, fixmap

    def filter_files(self):
        assert len(self.images) == len(self.gts)
#        assert len(self.images) == len(self.fixmaps)
        images = []
        gts = []
        fixmaps = []
        for img_path, gt_path, fixmap_path in zip(self.images, self.gts, self.fixmaps):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            fixmap = Image.open(fixmap_path)
#            if img.size == gt.size and img.size == fixmap.size:
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                fixmaps.append(fixmap_path)
        self.images = images
        self.gts = gts
        self.fixmaps = fixmaps

    def rgb_loader(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.trainsize, self.trainsize), interpolation=cv2.INTER_AREA)
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (self.trainsize, self.trainsize), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=2)
        return img

    def resize(self, img, gt, fixmap):
        assert img.size == gt.size
#        assert img.size == fixmap.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

class SalObjDataset_val(data.Dataset):
    def __init__(self, image_val_root, gt_val_root, fixmap_val_root, trainsize):
        self.trainsize = trainsize
        self.images_val = [image_val_root + f for f in os.listdir(image_val_root) if f.startswith('val')]
        self.gts_val = [gt_val_root + f for f in os.listdir(gt_val_root) if f.startswith('val')]
        self.fixmaps_val = [fixmap_val_root + f for f in os.listdir(fixmap_val_root) if f.startswith('val')]


        self.images_val = sorted(self.images_val)
        self.gts_val = sorted(self.gts_val)
        self.fixmaps_val = sorted(self.fixmaps_val)
        self.filter_files()
        self.size = len(self.images_val)
        self.img_transform = transforms.Compose([
#            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
#            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.fixmap_transform = transforms.Compose([
#            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image_val = self.rgb_loader(self.images_val[index])

        gt_val = self.binary_loader(self.gts_val[index])
        fixmap_val = self.binary_loader(self.fixmaps_val[index])
        image_val = self.img_transform(image_val)
        gt_val = self.gt_transform(gt_val)
        fixmap_val = self.fixmap_transform(fixmap_val)
        return image_val, gt_val, fixmap_val

    def filter_files(self):
        assert len(self.images_val) == len(self.gts_val)
#        assert len(self.images_val) == len(self.fixmaps_val)
        images_val = []
        gts_val = []
        for img_val_path, gt_val_path in zip(self.images_val, self.gts_val):
            img_val = Image.open(img_val_path)
            gt_val = Image.open(gt_val_path)
            if img_val.size == gt_val.size:
                images_val.append(img_val_path)
                gts_val.append(gt_val_path)
        self.images_val = images_val
        self.gts_val = gts_val
#        self.fixmaps_val = fixmaps_val

    def rgb_loader(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.trainsize, self.trainsize), interpolation=cv2.INTER_AREA)
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (self.trainsize, self.trainsize), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=2)
        return img

    def resize(self, img_val, gt_val, fixmap_val):
        assert img_val.size == gt_val.size
#        assert img_val.size == fixmap_val.size
        w, h = img_val.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img_val.resize((w, h), Image.BILINEAR), gt_val.resize((w, h), Image.NEAREST)
        else:
            return img_val, gt_val

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, fixmap_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, fixmap_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_loader_val(image_val_root, gt_val_root, fixmap_val_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset_val(image_val_root, gt_val_root, fixmap_val_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader




class test_dataset:
    def __init__(self, image_root, testsize, data_type='global'):
        self.testsize = testsize
        self.data_type = data_type

        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.size = len(self.images)
        self.index = 0

    def trans_img(self, img, stride):
        res = np.zeros(img.shape, dtype=img.dtype)
        res[:, :stride, :] = img[:, -stride:, :]
        res[:, stride:, :] = img[:, :img.shape[1]-stride, :]
        res = Image.fromarray(res)
        res = self.transform(res).unsqueeze(0)
        return res

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.jpg'
        ori_size = list(image.size)[::-1]

        if self.data_type == 'global':
            image_array = np.array(image)
            image_back = self.trans_img(image_array, image_array.shape[1] // 2)
            image_side1 = self.trans_img(image_array, (image_array.shape[1] // 4) * 3)
            image_side2 = self.trans_img(image_array, image_array.shape[1] // 4)

            image = self.transform(image).unsqueeze(0)
            self.index += 1
            return image, image_back, image_side1, image_side2, ori_size, name
        else:
            image = self.transform(image).unsqueeze(0)
            self.index += 1
            return image, ori_size, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
