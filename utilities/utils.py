import torch
import numpy as np
import random
import time
import os
import matplotlib

from datasets.utils import preprocess_input

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch = (start_iter + 1) // dataset_size
    start_iter = (start_iter + 1) % dataset_size
    return start_epoch, start_iter


def labelcolormap(N):
    if N == 37:
        cmap = np.array([
            [0, 0, 0],  # 0 - Background
            [255, 0, 0],  # 1 - Class 1 (Red)
            [0, 255, 0],  # 2 - Class 2 (Green)
            [0, 0, 255],  # 3 - Class 3 (Blue)
            [255, 255, 0],  # 4 - Class 4 (Yellow)
            [0, 255, 255],  # 5 - Class 5 (Cyan)
            [255, 0, 255],  # 6 - Class 6 (Magenta)
            [255, 128, 0],  # 7 - Class 7 (Orange)
            [128, 0, 255],  # 8 - Class 8 (Purple)
            [0, 255, 128],  # 9 - Class 9 (Lime)
            [128, 255, 0],  # 10 - Class 10 (Chartreuse)
            [0, 128, 255],  # 11 - Class 11 (Sky Blue)
            [255, 0, 128],  # 12 - Class 12 (Rose)
            [128, 255, 255],  # 13 - Class 13 (Aquamarine)
            [255, 128, 255],  # 14 - Class 14 (Violet)
            [255, 255, 128],  # 15 - Class 15 (Light Yellow)
            [128, 128, 128],  # 16 - Class 16 (Gray)
            [192, 192, 192],  # 17 - Class 17 (Silver)
            [255, 128, 128],  # 18 - Class 18 (Light Red)
            [128, 255, 128],  # 19 - Class 19 (Light Green)
            [128, 128, 255],  # 20 - Class 20 (Light Blue)
            [255, 255, 0],  # 21 - Class 21 (Light Yellow)
            [0, 255, 255],  # 22 - Class 22 (Light Cyan)
            [255, 0, 255],  # 23 - Class 23 (Light Magenta)
            [255, 128, 0],  # 24 - Class 24 (Light Orange)
            [128, 0, 255],  # 25 - Class 25 (Light Purple)
            [0, 255, 128],  # 26 - Class 26 (Light Lime)
            [128, 255, 0],  # 27 - Class 27 (Light Chartreuse)
            [0, 128, 255],  # 28 - Class 28 (Light Sky Blue)
            [255, 0, 128],  # 29 - Class 29 (Light Rose)
            [128, 255, 255],  # 30 - Class 30 (Light Aquamarine)
            [255, 128, 255],  # 31 - Class 31 (Light Violet)
            [255, 255, 128],  # 32 - Class 32 (Pale Yellow)
            [192, 192, 192],  # 33 - Class 33 (Pale Silver)
            [255, 128, 128],  # 34 - Class 34 (Pale Red)
            [128, 255, 128],  # 35 - Class 35 (Pale Green)
            [128, 128, 255],  # 36 - Class 36 (Pale Blue)
        ], dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


class results_saver():
    def __init__(self, opt):
        path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)
        self.path_label = os.path.join(path, "label")
        self.path_image = os.path.join(path, "image")
        self.path_to_save = {"label": self.path_label, "image": self.path_image}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_image, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            im = tens_to_lab(label[i], self.num_cl)
            self.save_im(im, "label", name[i])
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name[i])

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        # print(name.split("/")[-1])
        im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))


class results_saver_mid_training():
    def __init__(self, opt, current_iteration):
        path = os.path.join(opt.results_dir, opt.name, current_iteration)
        self.path_label = os.path.join(path, "label")
        self.path_image = os.path.join(path, "image")
        self.path_to_save = {"label": self.path_label, "image": self.path_image}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_image, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            im = tens_to_lab(label[i], self.num_cl)
            self.save_im(im, "label", name[i])
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name[i])

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))


class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.name, "progress.txt")

    def __call__(self, epoch, cur_iter):
        if cur_iter != 0:
            avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = cur_iter

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d - iter %d], time:%.3f \n' % (epoch, self.num_epochs, cur_iter, avg))
        print('[epoch %d/%d - iter %d], time:%.3f' % (epoch, self.num_epochs, cur_iter, avg))
        return avg


class LossRecorder:
    def __init__(self, opt):
        self.losses = []
        self.opt = opt

    def append(self, loss):
        self.losses.append(loss)

    def plot(self):
        plt.figure()
        plt.plot(self.losses, label='Training Loss')
        plt.xlabel('Iteration / 100')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'loss'), dpi=600)
        plt.close()


class losses_saver():
    def __init__(self, opt):
        self.name_list = ["Generator", "Vgg", "GAN", "edge", 'featMatch', "D_fake", "D_real", "LabelMix", "Du_fake",
                          "Du_real", "Du_regularize"]
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss
        self.loss = []
        self.cur_estimates = np.zeros(len(self.name_list))
        print(len(self.name_list))
        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)

    def __call__(self, epoch, loss):
        self.loss.append(loss)
        self.plot_losses()

    def plot_losses(self):
        fig, ax = plt.subplots(1)
        n = np.array(range(len(self.loss))) * self.opt.freq_smooth_loss
        plt.plot(n[1:], self.loss[1:])
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '.png'), dpi=600)
        plt.close(fig)


class image_saver():
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = os.path.join(opt.checkpoints_dir, opt.name, "images") + "/"
        self.opt = opt
        self.num_cl = 37
        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, model, image, label, cur_iter):
        self.save_images(label, "groundtruth", cur_iter, is_label=True)
        self.save_images(image, "image", cur_iter, is_image=True)
        with torch.no_grad():
            model.eval()
            pred = model(image)
            self.save_images(pred, "segmentation", cur_iter, is_label=True)

    def save_images(self, batch, name, cur_iter, is_label=False, is_image=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab_color(batch[i], self.num_cl)
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i + 1)
            plt.axis("off")
            if is_image:
                plt.imshow(im, cmap='gray')
            else:
                plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path + str(cur_iter) + "_" + name)
        plt.close()


def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))


def tens_to_lab(tens, num_cl):
    label_tensor = GreyScale(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy


def tens_to_lab_color(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy


###############################################################################
# Code below from
# https://github.com/visinf/1-stage-wseg/blob/38130fee2102d3a140f74a45eec46063fcbeaaf8/datasets/utils.py
# Modified so it complies with the Cityscapes label map colors (fct labelcolormap)
###############################################################################

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def GreyScale(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = label
        color_image[1][mask] = label
        color_image[2][mask] = label
    return color_image


def save_accuracy_to_txt(class_accuracy, filepath):
    with open(filepath, 'w') as f:
        for class_idx, accuracy in enumerate(class_accuracy):
            f.write(f"Class {class_idx}: {accuracy:.2%}\n")


def save_class_counts_to_txt(pre_counts, label_counts, filepath):
    with open(filepath, 'w') as f:
        for class_idx, (counts, count) in enumerate(zip(pre_counts,label_counts) ):
            f.write(f"Class {class_idx}: {counts}  {count} \n")


def check_accuracy(loader, model, opt, device="cuda"):
    model.eval()
    class_correct = torch.zeros(opt.num_classes, dtype=torch.float).to(device)
    class_total = torch.zeros(opt.num_classes, dtype=torch.float).to(device)
    class_pred = torch.zeros(opt.num_classes, dtype=torch.float).to(device)
    with torch.no_grad():
        for index, data in enumerate(loader):
            image, label = preprocess_input(opt, data)
            label = torch.argmax(label, dim=1, keepdim=True)
            preds = torch.argmax(model(image), dim=1, keepdim=True)
            flattened_labels = label.view(-1)
            flattened_preds = preds.view(-1)
            class_counts_preds = torch.bincount(flattened_preds, minlength=opt.num_classes)
            class_counts_labels = torch.bincount(flattened_labels, minlength=opt.num_classes)
            correct = preds.eq(label.view_as(preds))
            for i in range(opt.num_classes):
                class_pred[i] += preds[label == i].sum().item()
                class_correct[i] += correct[label == i].sum().item()
                class_total[i] += (label == i).sum().item()
            if index == 500:
                break
    class_accuracy = class_correct / class_total
    save_accuracy_to_txt(class_accuracy, os.path.join(opt.checkpoints_dir, 'class_accuracy.txt'))
    save_class_counts_to_txt(class_counts_preds, class_counts_labels, os.path.join(opt.checkpoints_dir, 'class_counts'
                                                                                                        '.txt'))
    model.train()