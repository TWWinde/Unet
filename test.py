import torch
from tqdm import tqdm

from checkpoints_manager import CheckpointsManager
from datasets.utils import preprocess_input
from utilities import utils
from datasets import dataloaders
from networks.UNET import UNet
from loss_functions.dice_loss import SoftDiceLoss
import config
from utilities.utils import LossRecorder, check_accuracy
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support



def calculate_iou(pred, target):
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    iou = intersection / union
    return iou


import numpy as np


def calculate_iou1(prediction, target, num_classes): #37
    iou_list = []

    for class_id in range(0, num_classes - 1):  # 类别从1到num_classes

        pred_mask = (prediction == class_id).astype(np.uint8)

        target_mask = (target == class_id).astype(np.uint8)


        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        # 计算IoU
        iou = intersection / (union + 1e-8)
        iou_list.append(iou)

    mean_iou = sum(iou_list) / len(iou_list)

    return mean_iou

#--- read options ---#
opt = config.read_arguments(train=False)
print(opt.phase)

#--- create dataloader ---#
_,_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = UNet(opt, num_classes=39, in_channels=1)
saver = CheckpointsManager(model, opt.checkpoints_dir)
model.eval()

for i, data_i in tqdm(enumerate(dataloader_val)):
    image, label = preprocess_input(opt, data_i)
    pred = model(image).cpu().detach()
    label = torch.argmax(label, dim=1, keepdim=True)
    pred = torch.argmax(pred, dim=1, keepdim=True)
    miou = calculate_iou1(pred, label, 37)
    conf_matrix = confusion_matrix(label.ravel(), pred.ravel())

    precision, recall, f1_score, support = precision_recall_fscore_support(label.ravel(), pred.ravel(),
                                                                           labels=range(37))
