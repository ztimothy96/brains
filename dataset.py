import glob
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


root = 'data'
img_suffix = '_img.npy'
msk_suffix = '_msk.npy'

class MRIDataset(Dataset):
    def __init__(self, names, transform=None):
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
        if self.transform:
            img = self.transform(img)
        return img, mask
    
def get_MRI_train_test_datasets(train_transform=None, test_transform=None):
    img_list = glob.glob('{}/*{}'.format(root, img_suffix))
    names = [fn.replace(img_suffix, '') for fn in img_list]
    train_names, test_names = train_test_split(names, test_size=0.25)
    train_dataset = MRIDataset(train_names, transform=train_transform)
    test_dataset = MRIDataset(test_names, transform=test_transform)
    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = get_MRI_train_test_datasets()
    for dataset in [train_dataset, test_dataset]:
        print('length of dataset: {}'.format(len(dataset)))

    X, y = train_dataset[0]
    print(X.shape)
    print(y.shape)