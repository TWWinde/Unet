import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from checkpoints_manager import CheckpointsManager
from datasets.utils import preprocess_input
from utilities import utils
from datasets import dataloaders
from networks.UNET import UNet
from loss_functions.dice_loss import SoftDiceLoss
import config
from utilities.utils import LossRecorder, check_accuracy

opt = config.read_arguments(train=True)
print("nb of gpus: ", torch.cuda.device_count())
cur_step = 0
already_started = True
im_saver = utils.image_saver(opt)
visualizer_losses = utils.losses_saver(opt)
loss_recorder = LossRecorder(opt)


def loss_calculation(pred, label):
    dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax for DICE Loss!
    ce_loss = torch.nn.CrossEntropyLoss()  # No softmax for CE Loss -> is implemented in torch!
    pred_softmax = F.softmax(pred, dim=1)
    loss = dice_loss(pred_softmax, label) + ce_loss(pred, label)

    return loss


def train_fn(loader, model, optimizer, opt, cur_step):
    for batch_idx, data in enumerate(loader):
        cur_step +=1
        image, label = preprocess_input(opt, data)  # [16, 1, 256, 256], [16, 39, 256, 256]
        target_element = 1
        bool_tensor = label == target_element
        # 计算 True 值的数量
        count = bool_tensor.sum().item()
        print(f"Number of {target_element}s in the tensor: {count}")
        model.zero_grad()
        optimizer.zero_grad()

        # forward
        predictions = model(image)  # [16, 39, 256, 256]
        loss = loss_calculation(predictions, label)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if cur_step % 2000 == 0:
            im_saver.visualize_batch(model, image, label, cur_step)
            loss_recorder.append(loss.item())




print(':::::::::::::::::::start training:::::::::::::::::::::::::::::')
# load model
model = UNet(opt, num_classes=39, in_channels=1)
model.to('cuda')
model.train()


# load data
dataloader, dataloader_val = dataloaders.get_dataloaders(opt)

# load checkpoint
saver = CheckpointsManager(model, opt.checkpoints_dir)
cur_step = saver.load_last_checkpoint()
start_epoch = int(cur_step / (len(dataloader.dataset) / opt.batch_size))
num_training_steps = int(opt.num_epochs * len(dataloader.dataset) / opt.batch_size)
print('total training steps:', num_training_steps)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min')

for epoch in range(start_epoch, opt.num_epochs):
    train_fn(dataloader, model, optimizer, opt, cur_step)
    cur_step += (len(dataloader.dataset) // opt.batch_size)
    saver.save_checkpoint(cur_step)
    check_accuracy(dataloader_val, model, opt, device='cuda')
    loss_recorder.plot()


torch.save(model.state_dict(), os.path.join(opt.checkpoints_dir, "Unet_model_final.tar"))
print("The training has successfully finished")
