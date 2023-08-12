import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from utilities import utils

from datasets import dataloaders
from networks.UNET import UNet
from loss_functions.dice_loss import SoftDiceLoss
import config

opt = config.read_arguments(train=True)
print("nb of gpus: ", torch.cuda.device_count())
model = UNet(opt,num_classes=37, in_channels=1)
model.to('cuda')
dataloader,dataloader_supervised, dataloader_val = dataloaders.get_dataloaders(opt)
im_saver = utils.image_saver(opt)

dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax for DICE Loss!
ce_loss = torch.nn.CrossEntropyLoss()  # No softmax for CE Loss -> is implemented in torch!

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min')
visualizer_losses = utils.losses_saver(opt)
# If directory for checkpoint is provided, load it.


print('=====TRAIN=====')
model.train()


def loopy_iter(dataset):
    while True :
        for item in dataset :
            yield item
cur_iter = 0
#--- the training loop ---#
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(dataloader):

        cur_iter = epoch*len(dataloader) + i

        image, label = data_i
        model.zero_grad()
        optimizer.zero_grad()

        # Shape of data_batch = [1, b, c, w, h]
        # Desired shape = [b, c, w, h]
        # Move data and target to the GPU

        pred = model(image)
        pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

        loss = dice_loss(pred_softmax, label.squeeze()) + ce_loss(pred, label.squeeze())
        # loss = self.ce_loss(pred, target.squeeze())

        loss.backward()
        optimizer.step()

        # Some logging and plotting
        if (i % opt.freq_plot_loss) == 0:
            print('Epoch: {0} Loss: {1:.4f}'.format(epoch, loss))
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        visualizer_losses(cur_iter, loss)

        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model, image, label, cur_iter)




    def validate(self, epoch):
        self.elog.print('VALIDATE')
        self.model.eval()

        data = None
        loss_list = []

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch['seg'][0].long().to(self.device)

                pred = model(data)
                pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

                loss = dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())


        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, float(np.mean(loss_list))))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)

        self.clog.show_image_grid(data.float().cpu(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        self.clog.show_image_grid(target.float().cpu(), name="mask_val", title="Mask", n_iter=epoch)
        self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_val", normalize=True, scale_each=True, n_iter=epoch)

    def test(self):
        from evaluation.evaluator import aggregate_scores, Evaluator
        from collections import defaultdict

        self.elog.print('=====TEST=====')
        self.model.eval()

        pred_dict = defaultdict(list)
        gt_dict = defaultdict(list)

        batch_counter = 0
        with torch.no_grad():
            for data_batch in self.test_data_loader:
                print('testing...', batch_counter)
                batch_counter += 1

                # Get data_batches
                mr_data = data_batch['data'][0].float().to(self.device)
                mr_target = data_batch['seg'][0].float().to(self.device)

                pred = self.model(mr_data)
                pred_argmax = torch.argmax(pred.data.cpu(), dim=1, keepdim=True)

                fnames = data_batch['fnames']
                for i, fname in enumerate(fnames):
                    pred_dict[fname[0]].append(pred_argmax[i].detach().cpu().numpy())
                    gt_dict[fname[0]].append(mr_target[i].detach().cpu().numpy())

        test_ref_list = []
        for key in pred_dict.keys():
            test_ref_list.append((np.stack(pred_dict[key]), np.stack(gt_dict[key])))

        scores = aggregate_scores(test_ref_list, evaluator=Evaluator, json_author=self.config.author, json_task=self.config.name, json_name=self.config.name,
                                  json_output_file=self.elog.work_dir + "/{}_".format(self.config.author) + self.config.name + '.json')

        print("Scores:\n", scores)

    def segment_single_image(self, data):
        self.model = UNet(num_classes=self.config.num_classes, in_channels=self.config.in_channels)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # a model must be present and loaded in here
        if self.config.model_dir == '':
            print('model_dir is empty, please provide directory to load checkpoint.')
        else:
            self.load_checkpoint(name=self.config.model_dir, save_types=("model",))

        self.elog.print("=====SEGMENT_SINGLE_IMAGE=====")
        self.model.eval()
        self.model.to(self.device)

        # Desired shape = [b, c, w, h]
        # split into even chunks (lets use size)
        with torch.no_grad():

            ######
            # When working entirely on CPU and in memory, the following lines replace the split/concat method
            # mr_data = data.float().to(self.device)
            # pred = self.model(mr_data)
            # pred_argmax = torch.argmax(pred.data.cpu(), dim=1, keepdim=True)
            ######

            ######
            # for CUDA (also works on CPU) split into batches
            blocksize = self.config.batch_size

            # number_of_elements = round(data.shape[0]/blocksize+0.5)     # make blocks large enough to not lose any slices
            chunks = [data[i:i+blocksize, ::, ::, ::] for i in range(0, data.shape[0], blocksize)]
            pred_list = []
            for data_batch in chunks:
                mr_data = data_batch.float().to(self.device)
                pred_dict = self.model(mr_data)
                pred_list.append(pred_dict.cpu())

            pred = torch.Tensor(np.concatenate(pred_list))
            pred_argmax = torch.argmax(pred, dim=1, keepdim=True)

        # detach result and put it back to cpu so that we can work with, create a numpy array
        result = pred_argmax.short().detach().cpu().numpy()

        return result

