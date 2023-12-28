import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

scaler = MinMaxScaler()
encoder = OneHotEncoder(categories='auto', sparse=False)

msk_list=sorted(glob.glob("msk/*msk.nii"))
saved_values = []
max_nb_values = 0


for i in range(len(msk_list)):
    msk_name =  msk_list[i]
    msk_data=nib.load(msk_name).get_fdata()
    msk_data[msk_data%10==1] = 1
    msk_data[msk_data%10==2] = 2
    msk_data[msk_data%10==0] = 0
    #msk_data=shrink(msk_data, 128, 128, 128)
    msk_data = np.eye(3)[msk_data]
    save_name = msk_name.replace("_msk.nii", ".npy")
    print(f"save to {save_name}")
    np.save(save_name, msk_data)
    
    #plt.figure(figsize=(12, 8))

    #plt.subplot(221)
    #plt.imshow(msk_data[:,:,64], cmap='gray')
    #plt.title('Image flair')
    #plt.show()