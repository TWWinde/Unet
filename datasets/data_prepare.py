import os
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def get_2d_images(ct_path, label_path):
    n = 0

    for i in range(int(len(ct_path) * 0.9)):
        nifti_img = nib.load(ct_path[i])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[i])
        seg_3d = nifti_seg.get_fdata()

        for z in range(seg_3d.shape[2]):
            seg_slice = seg_3d[:, :, z]
            seg_slice = seg_slice[72:328, 65:321]
            img_slice = img_3d[:, :, z]
            img_slice = img_slice[72:328, 65:321]
            new_affine1 = nifti_img.affine.copy()
            sliced_nifti_img = nib.Nifti1Image(img_slice, new_affine1)
            nib.save(sliced_nifti_img, f'/misc/data/private/autoPET/train2/CT/CT_slice_{n}.nii.gz')
            new_affine2 = nifti_seg.affine.copy()
            sliced_nifti_seg = nib.Nifti1Image(seg_slice, new_affine2)
            nib.save(sliced_nifti_seg, f'/misc/data/private/autoPET/train2/SEG/sliced_image_{n}.nii.gz')
            n += 1

    print("finished train data set")
    n = 0
    for j in range(int(len(ct_path) * 0.9), int(len(ct_path) * 0.95)):
        nifti_img = nib.load(ct_path[j])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[j])
        seg_3d = nifti_seg.get_fdata()

        for z in range(seg_3d.shape[2]):
            seg_slice = seg_3d[:, :, z]
            seg_slice = seg_slice[72:328, 65:321]
            img_slice = img_3d[:, :, z]
            img_slice = img_slice[72:328, 65:321]
            new_affine1 = nifti_img.affine.copy()
            sliced_nifti_img = nib.Nifti1Image(img_slice, new_affine1)
            nib.save(sliced_nifti_img, f'/misc/data/private/autoPET/test2/CT/CT_slice_{n}.nii.gz')
            new_affine2 = nifti_seg.affine.copy()
            sliced_nifti_seg = nib.Nifti1Image(seg_slice, new_affine2)
            nib.save(sliced_nifti_seg, f'/misc/data/private/autoPET/test2/SEG/sliced_image_{n}.nii.gz')
            n += 1

    print("finished test data set")
    n = 0
    for k in range(int(len(ct_path) * 0.95), len(ct_path)):
        nifti_img = nib.load(ct_path[k])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[k])
        seg_3d = nifti_seg.get_fdata()

        for z in range(seg_3d.shape[2]):
            seg_slice = seg_3d[:, :, z]
            seg_slice = seg_slice[72:328, 65:321]
            img_slice = img_3d[:, :, z]
            img_slice = img_slice[72:328, 65:321]
            new_affine1 = nifti_img.affine.copy()
            sliced_nifti_img = nib.Nifti1Image(img_slice, new_affine1)
            nib.save(sliced_nifti_img, f'/misc/data/private/autoPET/val2/CT/CT_slice_{n}.nii.gz')
            new_affine2 = nifti_seg.affine.copy()
            sliced_nifti_seg = nib.Nifti1Image(seg_slice, new_affine2)
            nib.save(sliced_nifti_seg, f'/misc/data/private/autoPET/val2/SEG/sliced_image_{n}.nii.gz')
            n += 1

    print("finished validation data set")





def list_images(path):
    ct_path = []
    label_path = []
    # read autoPET files names
    names = os.listdir(path)
    ct_names = list(filter(lambda x: x.endswith('0001.nii.gz'), names))

    for i in range(len(ct_names)):
        ct_path.append(os.path.join(path, ct_names[i]))
        label_path.append(os.path.join(path, ct_names[i].replace('0001.nii.gz', '0002.nii.gz')))

    return ct_path, label_path


os.makedirs('/misc/data/private/autoPET/train2/CT', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/train2/SEG', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/test2/CT', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/test2/SEG', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/val2/CT', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/val2/SEG', exist_ok=True)
path_imagesTr = "/misc/data/private/autoPET/imagesTr"
root_dir = "/misc/data/private/autoPET/"
test_path = '/misc/data/private/autoPET/train/SEG'

ct_paths, label_paths = list_images(path_imagesTr)

get_2d_images(ct_paths, label_paths)

#unique_value = set()
# for item in label_paths:
#     nifti_seg = nib.load(item)
#     seg_3d = nifti_seg.get_fdata()
#     flat_data = seg_3d.flatten().tolist()
#     for value in flat_data:
#         unique_value.add(value)

#for item in sorted(os.listdir(test_path)):
    #image_path = os.path.join(test_path, item)
    #image = cv2.imread(image_path)
    #flat_data = image.flatten().tolist()
    #for value in flat_data:
        #unique_value.add(value)

#print(unique_value)
#print('number of classes:',len(unique_value))
def count_pixels(dir):
    for item in os.listdir(dir):
        image = Image.open(os.path.join(dir,item))
        image_array = np.array(image)
        unique_pixel_values, pixel_counts = np.unique(image_array, return_counts=True)
        class_pixel_counts = np.zeros(37, dtype=int)
        for value, count in zip(unique_pixel_values, pixel_counts):
            class_pixel_counts[value] = count

    return class_pixel_counts

#total_pixel_count_train = count_pixels('/misc/data/private/autoPET/train1/SEG')
#total_pixel_count_test = count_pixels('/misc/data/private/autoPET/test1/SEG')
#total_pixel_count_val = count_pixels('/misc/data/private/autoPET/test1/SEG')

#pixel_count = total_pixel_count_val + total_pixel_count_test + total_pixel_count_train
def percentage(vector):
    pixel_number = vector.sum()
    return vector.astype(float) / pixel_number

#total_persentage = percentage(pixel_count)
#train_persentage = percentage(total_pixel_count_train)
#test_persentage = percentage(total_pixel_count_test)
#val_persentage = percentage(total_pixel_count_val)

#with open('/no_backups/s1449/Unet/class_statistics', 'w') as f:
    #for class_idx, (p1, p2, p3, p4) in enumerate(zip(total_persentage, train_persentage, test_persentage, val_persentage )):
        #f.write(f'Class {class_idx}:  {p1}         {p2}        {p3}        {p4}  \n  ')


total_pixel_counts = [0] * 50
def count_pixel_value(root_path):
    for item in os.listdir(os.path.join(root_path,'train2','SEG')):
        nifti_seg = nib.load(os.path.join(root_path,'train2','SEG', item))
        seg_3d = nifti_seg.get_fdata()
        unique_values, counts = np.unique(seg_3d, return_counts=True)
        for value, count in zip(unique_values, counts):
            total_pixel_counts[int(value)] += count

    for item in os.listdir(os.path.join(root_path,'test2','SEG')):
        nifti_seg = nib.load(os.path.join(root_path,'test2','SEG', item))
        seg_3d = nifti_seg.get_fdata()
        unique_values, counts = np.unique(seg_3d, return_counts=True)
        for value, count in zip(unique_values, counts):
            total_pixel_counts[int(value)] += count

    for item in os.listdir(os.path.join(root_path, 'val2', 'SEG')):
        nifti_seg = nib.load(os.path.join(root_path, 'val2', 'SEG', item))
        seg_3d = nifti_seg.get_fdata()
        unique_values, counts = np.unique(seg_3d, return_counts=True)
        for value, count in zip(unique_values, counts):
            total_pixel_counts[int(value)] += count

    with open('/no_backups/s1449/Unet/pixel_counts_nib', 'w') as f:
        for value, count in enumerate(total_pixel_counts):
            f.write(f'Pixel value {value}:   Count = {count} \n ')


count_pixel_value(root_dir)


print('finished image')

