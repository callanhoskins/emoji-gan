import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def dataset_split(dataset, val_split=0.2, test_split=0.2):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {'train': Subset(dataset, train_idx),
                'val': Subset(dataset, val_idx),
                'test': Subset(dataset, test_idx)}
    return datasets

class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, num_workers=4, shuffle=True):
        self.dataset = dataset
        self.image_path = image_path
        self.image_size = image_size
        self.batch = batch_size
        self.shuffle = shuffle
        self.train = train
        self.num_workers = num_workers

    def loader(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size, self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        dataset = ImageFolder(self.image_path, transform=transforms)
        datasets = dataset_split(dataset)
        dataloaders = {x : DataLoader(datasets[x], batch_size=self.batch, shuffle=self.shuffle, num_workers=self.num_workers) for x in ['train', 'val', 'test']}
        return dataloaders
