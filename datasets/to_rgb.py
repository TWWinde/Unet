import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import nibabel as nib

def set_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path, ignore_errors=True)

os.makedirs('/misc/data/private/autoPET/data_nnunet/train/images', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/data_nnunet/train/labels', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/data_nnunet/test/images', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/data_nnunet/test/labels', exist_ok=True)



n=0
for item in sorted(os.listdir('/misc/data/private/autoPET/train3/CT')):
    image = Image.open(os.path.join('/misc/data/private/autoPET/train3/CT', item))
    image = image.convert('RGB')
    image.save(os.path.join(f'/misc/data/private/autoPET/data_nnunet/train/images/ct_slice_{n}'))
    n+=1

n=0
for item in sorted(os.listdir('/misc/data/private/autoPET/test3/CT')):
    image = Image.open(os.path.join('/misc/data/private/autoPET/test3/CT', item))
    image = image.convert('RGB')
    image.save(os.path.join(f'/misc/data/private/autoPET/data_nnunet/test/images/ct_slice_{n}'))
    n+=1

n = 0
for item in sorted(os.listdir('/misc/data/private/autoPET/train3/SEG')):
    original_file_path = os.path.join('/misc/data/private/autoPET/train3/SEG', item)
    new_file_path = f'/misc/data/private/autoPET/data_nnunet/train/labels/ct_slice_{n}'
    shutil.copyfile(original_file_path, new_file_path)
    n += 1

n = 0
for item in sorted(os.listdir('/misc/data/private/autoPET/test3/SEG')):
    original_file_path = os.path.join('/misc/data/private/autoPET/train3/SEG', item)
    new_file_path = f'/misc/data/private/autoPET/data_nnunet/test/labels/ct_slice_{n}'
    shutil.copyfile(original_file_path, new_file_path)
    n += 1
