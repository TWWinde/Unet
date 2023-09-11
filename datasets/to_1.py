import os
import cv2
import nibabel as nib
import numpy as np


def rename_copy(root_dir):

    name = ['train3','test3','val3']
    n = 1
    #for i in range(3):
        #for item in sorted(os.listdir(os.path.join(root_dir, name[i], 'SEG'))):
            #original_file_path = os.path.join(root_dir, name[i], 'SEG', item)
            #new_filename = f'body_{n:06}.nii.gz'
            #destination_folder = '/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_AutoPET/labelsTr'
            #new_file_path = os.path.join(destination_folder, new_filename)
            #shutil.copyfile(original_file_path, new_file_path)
            #n+=1


    print('label finished')
    n = 1
    for i in range(3):
        for item in sorted(os.listdir(os.path.join(root_dir, name[i], 'CT1'))):
            original_file_path = os.path.join(root_dir, name[i], 'CT1', item)
            new_filename = f'body_{n:06}_0001.nii.gz'
            destination_folder = '/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_raw/Dataset521_AutoPET/imagesTr'
            new_file_path = os.path.join(destination_folder, new_filename)
            gray_image = cv2.imread(original_file_path, cv2.IMREAD_GRAYSCALE)
            # 调整图像维度，使其成为 (1, 256, 256)
            gray_image = np.expand_dims(gray_image, axis=0)
            #gray_image = np.transpose(gray_image, (2, 0, 1))
            nifti_image = nib.Nifti1Image(gray_image, affine=np.eye(4))  # 这里使用单位矩阵作为仿射矩阵
            nib.save(nifti_image, new_file_path)
            n+=1
    print('images finished')


root_dir = "/misc/data/private/autoPET/"
rename_copy(root_dir)