import nibabel as nib
import nilearn as nil
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import torch

brain_vol = nib.load('NIFTIs_Images/10005_19010618_eTRA-3D-T1Gd_img.nii')

# What is the type of this object?
print(type(brain_vol))

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)