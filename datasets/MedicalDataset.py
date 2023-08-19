import random
import cv2
import torch
from torchvision import transforms as TR
from torchvision.transforms import functional
import os
from PIL import Image
import numpy as np
from torch.utils import data
import nibabel as nib
from scipy import ndimage

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):

        opt.load_size = 256 if for_metrics else 256
        opt.crop_size = 256 if for_metrics else 256
        opt.label_nc = 37
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels = self.list_images()

    def __len__(self, ):
        return len(self.images)

    def __getitem__(self, idx):
        image = nib.load(self.images[idx])
        image = image.get_fdata()
        label = nib.load(self.labels[idx])
        label = label.get_fdata()
        # numpy array
        image, label = self.transforms(image, label)

        return {"image": image, "label": label}

    def list_images(self):
        mode = "val2" if self.opt.phase == "test2" else "train2"
        images = []
        path_img = os.path.join(self.opt.dataroot, mode, "CT")
        for item in sorted(os.listdir(path_img)):
            images.append(os.path.join(path_img, item))
        labels = []
        path_lab = os.path.join(self.opt.dataroot, mode, "SEG")
        for item in sorted(os.listdir(path_lab)):
            labels.append(os.path.join(path_lab, item))
        assert len(images) == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))

        return images, labels

    def transforms(self, image, label):
        assert image.size == label.size
        # normalize
        label = label.astype(np.int)
        image = (image - image.min()) / (image.max() - image.min())
        # flip
        #if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            #if random.random() < 0.5:
                #image = np.fliplr(image)
                #label = np.fliplr(label)
            #elif random.random() < 0.5:
                #image = ndimage.rotate(image, 90)
                #label = ndimage.rotate(label, 90)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)



        return image, label