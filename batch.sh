#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=medical
#SBATCH --output=medical%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --nodes=1


# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
pyenv activate venv
module load cuda/11.3

# Run your python code
# For single GPU use this
# CUDA_VISIBLE_DEVICES=0 python /no_backups/s1449/OASIS/dataloaders/get_2D_images.py
#--name USIS_cityscapes --dataset_mode cityscapes --gpu_ids 0 \
#--dataroot /data/public/cityscapes  \
#--batch_size 2 --model_supervision 0  \
#--Du_patch_size 64 --netDu wavelet  \
#--netG 0 --channels_G 64 \
#--num_epochs 500

CUDA_VISIBLE_DEVICES=0 python train.py --name medicals --dataset_mode medicals --gpu_ids 1 \
--dataroot /misc/data/private/autoPET
