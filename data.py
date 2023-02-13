import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random


class CelebDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.root_dir) if f.endswith('.jpg') or f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        index1 = random.randint(0, len(self.image_files) - 1)
        index2 = random.randint(0, len(self.image_files) - 1)
        while index2 == index1:
            index2 = random.randint(0, len(self.image_files) - 1)

        image_path = os.path.join(self.root_dir, self.image_files[index1])
        image1 = np.array(Image.open(image_path).convert('RGB'))

        image_path2 = os.path.join(self.root_dir, self.image_files[index2])
        image2 = np.array(Image.open(image_path2).convert('RGB'))

        if self.transform is not None:
            aug = self.transform(image=image1, mask=image2)
            image1 = aug['image']
            image2 = aug['mask']
        # print(image2.shape)
        return image1, image2


class RandomImagePairSampler(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for i in range(0, len(indices), 2):
            yield indices[i], indices[i+1]


def random_sample(dataset):
    index1 = random.randint(0, len(dataset) - 1)
    index2 = random.randint(0, len(dataset) - 1)
    while index2 == index1:
        index2 = random.randint(0, len(dataset) - 1)
    return dataset[index1], dataset[index2]


def collate_fn(batch):
    return random.sample(batch, 2)