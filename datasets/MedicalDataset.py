import random
import cv2
import torch
from torchvision import transforms as TR
from torchvision.transforms import functional
import os
from PIL import Image
import numpy as np
from torch.utils import data


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
        image = Image.open(self.images[idx]) #self.mixed_index[idx]
        image = image.convert('RGB')
        label = Image.open(self.labels[idx])
        image, label = self.transforms(image, label)
        label = label * 255

        return {"image": image, "label": label}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" else "train"
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
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.crop(image, 72, 65, new_width, new_height)
        label = TR.functional.crop(label, 72, 65, new_width, new_height)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = np.asarray(image)
        label = np.asarray(label)
        label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label