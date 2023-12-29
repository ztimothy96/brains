import glob
import numpy as np
import nibabel as nib
import os
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# shrink function copy-pasted from elsewhere...
def shrink(data, rows, cols, depths):
    scale1, scale2, scale3 =2, 2, 2
    while scale1*rows<=data.shape[0]:
        scale1 *= 2
    while scale2*cols<=data.shape[1]:
        scale2 *= 2
    while scale3*depths<=data.shape[2]:
        scale3 *= 2

    scale1, scale2, scale3 = scale1/2, scale2/2, scale3/2
    scale1 = min(scale1, scale2)
    scale2 = min(scale1, scale2)

    min1 = int((data.shape[0]-scale1*rows)/2)
    min2 = int((data.shape[1]-scale2*cols)/2)
    min3 =  int((data.shape[2]-scale3*depths)/2)
    max1 = int((data.shape[0]+scale1*rows)/2)
    max2 = int((data.shape[1]+scale2*cols)/2)
    max3 =  int((data.shape[2]+scale3*depths)/2)
    # print(data.shape, scale1, scale2, scale3, min1, min2, min3, max1, max2, max3)
    
    data2 = data[min1:max1, min2:max2, min3:max3]

    # print(data2.shape, rows, cols, depths)
    newdata = data2.reshape(rows, int(data2.shape[0]/rows), cols, int(data2.shape[1]/cols), depths, int(data2.shape[2]/depths)).sum(axis=1).sum(axis=2).sum(axis=3)
    return scaler.fit_transform(newdata.reshape(-1, newdata.shape[-1])).reshape(newdata.shape)


def preprocess_img(data):
    data = (data - data.min()) / data.max() # normalize grayscale from 0 to 1
    return data

def preprocess_mask(data):
    data[data!=0] = 1.0 # 1.0 is cancer, 0.0 is not cancer
    return data

def resize_and_save(save_name, data):
    if data.shape[0] >= 256 and data.shape[1] >= 256:
        try:
            data=shrink(data, 256, 256, 64)
        except Exception:
            return False
    np.save(save_name, data)
    return True


'''
    input: assume that data is downloaded as .nii files in NIFTIs_Images, NIFTIs_Masks folders
    output: preprocess data and save as np arrays
        - normalize image values to range [0.0, 1.0]
        - (**DONE**) reshape all data to 128x128x128
        - (**DONE**) combine non-cancer masks
        - upsample areas with more cancer?
'''
MSK_FOLDER = 'NIFTIs_Masks'
IMG_FOLDER = 'NIFTIs_Images'
DATA_FOLDER = 'data'

# remove all previously saved npy arrays
def remove_all_npy_in_dir(dir):
    fnames = sorted(glob.glob("{}/*.npy".format(dir)))
    for fn in fnames:
        os.remove(fn)
    return

remove_all_npy_in_dir(MSK_FOLDER)
remove_all_npy_in_dir(IMG_FOLDER)


img_list = sorted(glob.glob('{}/*img.nii'.format(IMG_FOLDER)))
name_list = [fn.replace('_img.nii', '').replace(IMG_FOLDER+'/', '') for fn in img_list]

for fn in name_list:
    img_name = '{}/{}_img.nii'.format(IMG_FOLDER, fn)
    msk_name = '{}/{}_msk.nii'.format(MSK_FOLDER, fn)
    img_data=nib.load(img_name).get_fdata()
    msk_data=nib.load(msk_name).get_fdata()
    preprocess_img(img_data)
    preprocess_mask(msk_data)
    resize_and_save('{}/{}_img.npy'.format(DATA_FOLDER, fn), img_data)
    resize_and_save('{}/{}_msk.npy'.format(DATA_FOLDER, fn), img_data)
