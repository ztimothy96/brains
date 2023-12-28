import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

scaler = MinMaxScaler()
encoder = OneHotEncoder(categories='auto', sparse=False)

img_list=sorted(glob.glob("img/*.npy"))
msk_list=sorted(glob.glob("msk/*msk.nii"))
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
    scale1 = min(scale1, scale2)
    scale2 = min(scale1, scale2)

    min1 = int((data.shape[0]-scale1*rows)/2)
    min2 = int((data.shape[1]-scale2*cols)/2)
    min3 =  int((data.shape[2]-scale3*depths)/2)
    max1 = int((data.shape[0]+scale1*rows)/2)
    max2 = int((data.shape[1]+scale2*cols)/2)
    max3 =  int((data.shape[2]+scale3*depths)/2)
    print(data.shape, scale1, scale2, scale3, min1, min2, min3, max1, max2, max3)
    
    data2 = data[min1:max1, min2:max2, min3:max3]

    print(data2.shape, rows, cols, depths)
    newdata = data2.reshape(rows, int(data2.shape[0]/rows), cols, int(data2.shape[1]/cols), depths, int(data2.shape[2]/depths)).sum(axis=1).sum(axis=2).sum(axis=3)
    return scaler.fit_transform(newdata.reshape(-1, newdata.shape[-1])).reshape(newdata.shape)

for i in range(len(img_list)):
    img_name = img_list[i]
    img_data=np.load(img_name)
    if img_data.shape[0]>=256 and img_data.shape[1]>=256:
        #img_data=scaler.fit_transform(img_data.reshape(-1, img_data.shape[-1])).reshape(img_data.shape)
        try:
            img_data=shrink(img_data, 256, 256, 64)
            print(img_data[32][32][32])

            save_name = img_name.replace(".npy", "_scaled.npy")
            print(f"save to {save_name}")
            #np.save(save_name, img_data)
            
            plt.figure(figsize=(24, 24))

            plt.subplot(221)
            plt.imshow(img_data[:,:,32], cmap='gray')
            plt.title('Image flair')
            plt.show()

        except Exception:
            pass