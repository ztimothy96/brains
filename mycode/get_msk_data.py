import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

scaler = MinMaxScaler()
encoder = OneHotEncoder(categories='auto', sparse=False)

img_list=sorted(glob.glob("*/*img.nii"))
msk_list=sorted(glob.glob("msk/*msk.nii"))
saved_values = []
max_nb_values = 0


for i in range(len(msk_list)):
    msk_name = msk_list[i]
    msk_data=nib.load(msk_name).get_fdata().astype(int)
    #msk_data=scaler.fit_transform(msk_data.reshape(-1, msk_data.shape[-1])).reshape(msk_data.shape)

    save_name = msk_name.replace("_msk.nii", ".npy")
    print(f"save to {save_name}")

    msk_data[msk_data%10==1] = 1
    msk_data[msk_data%10==2] = 2

    msk_data =  np.eye(3)[msk_data]
    np.save(save_name, msk_data)

    #print(msk_data.shape)

    plt.figure(figsize=(24, 24))

    plt.subplot(221)
    plt.imshow(msk_data[:,:,32], cmap='gray')
    plt.title('Image flair')
    plt.show()


    unique_values = np.unique(msk_data)
    nb_unique_values = len(np.unique(msk_data))
    
    if nb_unique_values > max_nb_values:
        max_nb_values = nb_unique_values
        saved_values = unique_values

print(f"Maximum number of values in all segmentation images: {max_nb_values}")
print(f"Values: {saved_values}")
