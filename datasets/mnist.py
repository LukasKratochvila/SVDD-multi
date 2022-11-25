from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms

import numpy as np
import torch

class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0, one_class: bool = False):
        super().__init__(root)
        
        self.n_classes = 10  # 0: normal, 1: outlier
        self.normal_classes = normal_class#tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        #self.outlier_classes.remove(normal_class)
        self.name = "MNIST_Dataset"
        
        self.norm = normal_class[0] if isinstance(normal_class, tuple) else normal_class

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[self.norm][0]],
                                                             [min_max[self.norm][1] - min_max[self.norm][0]])])

        target_transform = None #transforms.Lambda(lambda x: x-1)# None # transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyMNIST(n_classes=self.n_classes, sample=True,
                            root=self.root, train=True, download=True,
                            transform=transform, target_transform=target_transform)
        
        test_set = MyMNIST(n_classes=self.n_classes, validation=False,
                                root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)
        
        validation_set = MyMNIST(n_classes=self.n_classes, validation=True,
                                root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)
        
        # Subsets to normal class
        if one_class:
            self.n_classes = len(self.normal_classes)
            train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
            self.train_set = Subset(train_set, train_idx_normal)
            test_idx_normal = get_target_label_idx(test_set.test_labels.clone().data.cpu().numpy(), self.normal_classes)
            self.test_set = Subset(test_set, test_idx_normal)
            val_idx_normal = get_target_label_idx(validation_set.test_labels.clone().data.cpu().numpy(), self.normal_classes)
            self.validation_set = Subset(validation_set, val_idx_normal)
        else:
            self.train_set = train_set
            self.test_set = test_set
            self.validation_set = validation_set


class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, n_classes: int, validation=False, sample=False, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)
        if validation:
            data,labels=[],[]
            for i in range(n_classes):
                data.append(self.test_data[self.test_labels == i][:100])
                labels.append(self.test_labels[self.test_labels == i][:100])
            self.test_data = torch.Tensor(np.concatenate(data,axis=0))
            self.test_labels = torch.Tensor(np.concatenate(labels,axis=0)).type(dtype=torch.LongTensor)
        if sample:
            data,labels=[],[]
            for i in range(n_classes):
                data.append(self.train_data[self.train_labels == i][:10])
                labels.append(self.train_labels[self.train_labels == i][:10])
            self.train_data = torch.Tensor(np.concatenate(data,axis=0))
            self.train_labels = torch.Tensor(np.concatenate(labels,axis=0)).type(dtype=torch.LongTensor)
            
    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target.numpy(), index  # only line changed
