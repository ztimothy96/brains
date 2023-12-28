import nibabel as nib
from nilearn import plotting

import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

scaler = MinMaxScaler()
encoder = OneHotEncoder(categories='auto', sparse=False)

msk_list=sorted(glob.glob("NIFTIs_Masks/*msk.nii"))
img_list=sorted(glob.glob("NIFTIs_Images/*img.nii"))
saved_values = []
max_nb_values = 0

def show_img(vol):
    plotting.plot_img(vol)
    plt.show()

def save_img(msk_data):
    msk_data[msk_data%10==1] = 1
    msk_data[msk_data%10==2] = 2
    msk_data[msk_data%10==0] = 0
    #msk_data=shrink(msk_data, 128, 128, 128)
    msk_data = np.eye(3)[msk_data]
    save_name = msk_name.replace("_msk.nii", ".npy")
    print(f"save to {save_name}")
    np.save(save_name, msk_data)
    
for fname in (img_list[:3]):
    print('HI')
    print(fname)
    vol=nib.load(fname)
    show_img(vol)
    data=vol.get_fdata()
    # save_img(data)
