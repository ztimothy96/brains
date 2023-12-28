import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

scaler = MinMaxScaler()
encoder = OneHotEncoder(categories='auto', sparse=False)

img_list=sorted(glob.glob("*/*.npy"))
msk_list=sorted(glob.glob("masks_scaled/*/*.npy"))
saved_values = []
max_nb_values = 0

def shrink(data, rows, cols, depths):
    scale1, scale2, scale3 =2, 2, 2
    while scale1*rows<=data.shape[0]:
        scale1 *= 2
    while scale2*cols<=data.shape[1]:
        scale2 *= 2
    while scale3*depths<=data.shape[2]:
        scale3 *= 2

    scale1, scale2, scale3 = scale1/2, scale2/2, scale3/2

    min1 = int((data.shape[0]-scale1*rows)/2)
    min2 = int((data.shape[1]-scale2*cols)/2)
    min3 =  int((data.shape[2]-scale3*depths)/2)
    max1 = int((data.shape[0]+scale1*rows)/2)
    max2 = int((data.shape[1]+scale2*cols)/2)
    max3 =  int((data.shape[2]+scale3*depths)/2)
    
    data2 = data[min1:max1, min2:max2, min3:max3]

    newdata = data2.reshape(rows, int(data2.shape[0]/rows), cols, int(data2.shape[1]/cols), depths, int(data2.shape[2]/depths), 3).sum(axis=1).sum(axis=2).sum(axis=3)
    newdata = np.eye(3)[np.argmax(newdata, axis=3)]
    print(newdata.shape, newdata[...,0].sum(), newdata[...,1].sum()+newdata[...,2].sum())
    return newdata

for i in range(len(msk_list)):
    msk_name = msk_list[i]
    msk_data=np.load(msk_name)
    print(msk_data.shape)
    if msk_data.shape[0]>=256 and msk_data.shape[1]>=256 and msk_data.shape[2]>=64:
        try:
            #msk_data=scaler.fit_transform(msk_data.reshape(-1, msk_data.shape[-1])).reshape(msk_data.shape)
            msk_data=shrink(msk_data, 256, 256, 64)
            #print(msk_data[32][32][32])

            save_name = msk_name.replace(".npy", "_scaled.npy")
            print(f"{i} save to {save_name}")
            np.save(save_name, msk_data)
            
            #plt.figure(figsize=(24, 24))

            #plt.subplot(221)
            #plt.imshow(msk_data[:,:,32], cmap='gray')
            #plt.title('Image flair')
            #plt.show()
        except Exception:
            pass
