from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.utils import shuffle as shf
import pandas as pd

def process_data(filename='./driver_imgs_list.csv'):
    data = pd.read_csv(filename, encoding='utf-8')
    return data


def train_test_split(data_df,test_size=0.2,shuffle=True,random_state=None):
    if shuffle:
        data_df = shf(data_df,random_state = random_state)
    train = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    test = data_df[:int(len(data_df) * test_size)].reset_index(drop=True)
    return train,test


class MyDataSet(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_label = torch.tensor(int(self.data.loc[item,'classname'][1:]),dtype=torch.long)
        img_name = self.data.loc[item, 'img']

        img_path = "./imgs/train/c{}/{}".format(img_label,img_name)
        img = Image.open(img_path)
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        if self.transform is not None:
            img = self.transform(img)

        return img, img_label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels