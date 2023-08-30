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
        self.num_patches_per_image = 15
        self.opt = opt
        self.patch_size = 128
        self.stride = 64

        self.for_metrics = for_metrics
        self.images, self.labels = self.list_images()


    def __len__(self, ):
        return len(self.images)* self.num_patches_per_image

    def __getitem__(self, idx):
        image_idx = idx // self.num_patches_per_image
        patch_idx = idx % self.num_patches_per_image
        image = nib.load(self.images[image_idx])
        image = image.get_fdata()
        label = nib.load(self.labels[image_idx])
        label = label.get_fdata()
        if patch_idx < 5:
            patch_image, patch_label = self.get_patch(patch_idx, image, label)
        elif patch_idx > 11:
            patch_image, patch_label = self.get_patch(patch_idx-7, image, label)
        else:
            patch_image, patch_label = self.get_patch(5, image, label)

        image_patch, label_patch = self.transforms(patch_image, patch_label)
        return {"image": image_patch, "label": label_patch}

    def get_patch(self, patch_idx, image, label):

        x_start = (patch_idx % ((256 - self.patch_size) // self.stride + 1)) * self.stride
        y_start = (patch_idx // ((256 - self.patch_size) // self.stride + 1)) * self.stride

        patch_image = image[x_start:x_start + self.patch_size, y_start:y_start + self.patch_size, :]
        patch_label = label[x_start:x_start + self.patch_size, y_start:y_start + self.patch_size, :]

        return patch_image, patch_label


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
        assert image.shape == label.shape
        # normalize
        label = label.astype(np.int)
        max_value = np.max(image)
        min_value = np.min(image)

        if max_value - min_value == 0:
            image = np.zeros_like(image, dtype=np.uint8)
        else:
            image = (image - min_value) / (max_value - min_value)

        image = Image.fromarray(image)
        # Apply data augmentation
        image = self.augmentation(image)
        label = self.augmentation(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)


        return image, label

    def augmentation(self, image):
        transform = TR.Compose([
            TR.RandomHorizontalFlip(),
            TR.RandomVerticalFlip(),

        ])
        return transform(image)
