import glob
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


root = 'data'
img_suffix = '_img.npy'
msk_suffix = '_msk.npy'

class MRIDataset(Dataset):
    def __init__(self, names, transform = None):
        super(MRIDataset, self).__init__()
        self.names = names
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        fn = self.names[index]
        img_name = '{}{}'.format(fn, img_suffix)
        msk_name = '{}{}'.format(fn, msk_suffix)
        img = np.load(img_name)
        mask = np.load(msk_name)
        return img, mask
    
def get_MRI_train_test_datasets():
    img_list = glob.glob('{}/*{}'.format(root, img_suffix))
    names = [fn.replace(img_suffix, '') for fn in img_list]
    names_train, names_test = train_test_split(names, test_size=0.25)
    datasets = {
        'train': MRIDataset(names_train),
        'val': MRIDataset(names_test)
    }
    return datasets

if __name__ == '__main__':
    datasets = get_MRI_train_test_datasets()
    for x in ['train', 'val']:
        print('length of dataset: {}'.format(len(datasets[x])))