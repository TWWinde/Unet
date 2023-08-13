import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from datasets.utils import preprocess_input
from utilities import utils
from datasets import dataloaders
from networks.UNET import UNet
from loss_functions.dice_loss import SoftDiceLoss
import config

opt = config.read_arguments(train=True)
print("nb of gpus: ", torch.cuda.device_count())
model = UNet(opt, num_classes=37, in_channels=1)
model.to('cuda')
dataloader, dataloader_val = dataloaders.get_dataloaders(opt)
im_saver = utils.image_saver(opt)

dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax for DICE Loss!
ce_loss = torch.nn.CrossEntropyLoss()  # No softmax for CE Loss -> is implemented in torch!

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min')
visualizer_losses = utils.losses_saver(opt)

print('===============================TRAIN===============================')
model.train()


def loopy_iter(dataset):
    while True:
        for item in dataset:
            yield item


cur_iter = 0

already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))


def validate(model, dataloader, Epoch):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    num_subset_samples = int(len(dataloader.dataset) * 0.1)
    subset_indices = random.sample(range(len(dataloader.dataset)), num_subset_samples)

    with torch.no_grad():
        for data in subset_indices:
            images, labels = preprocess_input(opt, data)

            pred = model(images)
            loss = dice_loss(pred_softmax, labels.squeeze()) + ce_loss(pred, label.squeeze())

            total_loss += loss.item()
            num_samples += images.size(0)

        average_loss = total_loss / num_samples
        print('Epoch: {0} Training Loss: {1:.4f}'.format(Epoch, average_loss))


# --- the training loop ---#


for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(dataloader):

        cur_iter = epoch * len(dataloader) + i

        image, label = preprocess_input(opt, data_i)
        model.zero_grad()
        optimizer.zero_grad()

        # Shape of data_batch = [1, b, c, w, h]
        # Desired shape = [b, c, w, h]
        # Move data and target to the GPU

        pred = model(image)
        pred_softmax = F.softmax(pred,dim=1)
        # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
        loss = dice_loss(pred_softmax, label.squeeze()) + ce_loss(pred, label.squeeze())
        loss.backward()
        optimizer.step()

        # Some logging and plotting
        if (i % opt.freq_plot_loss) == 0:
            print('Steps: {0} Training Loss: {1:.4f}'.format(i, loss))

        if cur_iter % opt.freq_save_latest == 0:
            torch.save(model.state_dict(), os.path.join(opt.checkpoints_dir, "Unet_model.tar"))
        #visualizer_losses(cur_iter, loss)

        if cur_iter % opt.freq_print == 0:
            pass
           # im_saver.visualize_batch(model, image, label, cur_iter)

        validate(model, dataloader_val, epoch)


torch.save(model.state_dict(), os.path.join(opt.checkpoints_dir, "Unet_model_final.tar"))

print("The training has successfully finished")





