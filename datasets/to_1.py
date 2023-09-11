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
        for item in sorted(os.listdir(os.path.join(root_dir, name[i], 'CT'))):
            original_file_path = os.path.join(root_dir, name[i], 'CT', item)
            destination_folder = '/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_raw/Dataset521_AutoPET/imagesTr'
            image = cv2.imread(original_file_path)
            b, g, r = cv2.split(image)
            blue_channel = np.expand_dims(b, axis=0)
            green_channel = np.expand_dims(g, axis=0)
            red_channel = np.expand_dims(r, axis=0)
            new_filename0 = f'body_{n:06}_0000.nii.gz'
            new_filename1 = f'body_{n:06}_0001.nii.gz'
            new_filename2 = f'body_{n:06}_0002.nii.gz'
            new_file_path0 = os.path.join(destination_folder, new_filename0)
            new_file_path1 = os.path.join(destination_folder, new_filename1)
            new_file_path2 = os.path.join(destination_folder, new_filename2)
            nifti_image0 = nib.Nifti1Image(red_channel, affine=np.eye(4))  # 这里使用单位矩阵作为仿射矩阵
            nib.save(nifti_image0, new_file_path0)
            nifti_image1 = nib.Nifti1Image(green_channel, affine=np.eye(4))  # 这里使用单位矩阵作为仿射矩阵
            nib.save(nifti_image1, new_file_path1)
            nifti_image2 = nib.Nifti1Image(blue_channel, affine=np.eye(4))  # 这里使用单位矩阵作为仿射矩阵
            nib.save(nifti_image2, new_file_path2)
            n+=1
    print('images finished')


root_dir = "/misc/data/private/autoPET/"
rename_copy(root_dir)