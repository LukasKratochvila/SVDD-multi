#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:04:55 2021

@author: kratochvila
"""
from PIL import Image
import os
import os.path
import json
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from torchvision.datasets.utils import check_integrity

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import torchvision.transforms as transforms
from torch.utils.data import Subset

import numpy as np
import torch

class JOYBUY_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int=0, Res: int = 32,
                 validation: bool = True, precompute: bool = False):
        super().__init__(root)

        self.n_classes = 24 # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, self.n_classes))
        self.outlier_classes.remove(normal_class)
        self.name = "JOYBUY_Dataset"

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.12278110533952713, 0.48555073142051697),
                   (-0.25080981850624084, 0.660620927810669),
                   (-0.2279401421546936, 0.4902990162372589),
                   (-0.19261640310287476, 0.47622451186180115),
                   (-0.3263070285320282,0.40692266821861267),
                   (-0.24773985147476196,0.555292010307312),
                   (-0.31460490822792053,0.4956718683242798),
                   (-0.32529082894325256,0.4599974453449249),
                   (-0.1555253267288208,0.4479055404663086),
                   (-0.26625385880470276,0.5181570053100586),
                   (-0.31362852454185486,0.46142637729644775),
                   (-0.26174306869506836,0.49237361550331116),
                   (-0.5593557357788086,0.5188949704170227),
                   (-0.12982499599456787,0.7611070275306702),
                   (-0.3553692698478699,0.43311455845832825),
                   (-0.35010001063346863,0.4435591697692871),
                   (-0.22321926057338715,0.43905070424079895),
                   (-0.2981839179992676,0.4398650825023651),
                   (-0.3913074731826782,0.4234130382537842),
                   (-0.3016900420188904,0.48525261878967285),
                   (-0.26137515902519226,0.42268654704093933),
                   (-0.36716705560684204,0.6510715484619141),
                   (-0.29929134249687195,0.4885651767253876),
                   (-0.21230000257492065,0.44281092286109924)]        

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        trans=[transforms.Resize((Res,Res)),#160,120
               transforms.ToTensor(),
               transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
               transforms.Normalize([min_max[normal_class][0]] * 3,
                                    [min_max[normal_class][1] - min_max[normal_class][0]] * 3)]
        transform = transforms.Compose(trans[1:] if Res == 0 else trans)

        target_transform = None #transforms.Lambda(lambda x: torch.Tensor(x).type(dtype=torch.LongTensor)) # transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = JOYBUY(root=self.root, n_classes=self.n_classes, train=True,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        #train_idx_normal = get_target_label_idx(train_set.train_labels, self.normal_classes)
        #self.train_set = Subset(train_set, train_idx_normal)
        self.train_set = train_set

        self.test_set = JOYBUY(root=self.root, n_classes=self.n_classes, train=False,
                                  transform=transform, target_transform=target_transform)
        if validation:
            self.validation_set = JOYBUY(root=self.root, n_classes=self.n_classes, train=False,
                                         transform=transform, target_transform=target_transform,validation=True)
        # Precompute min max
        if precompute:
            print("Compute min max for all classes")
            target_transform=transforms.Lambda(lambda x: x)
            train_set = JOYBUY(root=self.root, n_classes=self.n_classes, train=True,
                              transform=transform, target_transform=target_transform)
            test_set = JOYBUY(root=self.root, n_classes=self.n_classes, train=False,
                                  transform=transform, target_transform=target_transform)
            test=list(test_set)
            train=list(train_set)
            for clas in range(self.n_classes):
                s=[x[0] for x in train if x[1]==clas]
                n=[s.append(x[0]) for x in test if x[1]==clas]
                print("Class {} : min: {}, max: {}".format(clas,min([torch.min(x) for x in s]),max([torch.max(x) for x in s])))
        
class JOYBUY(data.Dataset):
    """`MyData Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``mydata-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        version (int, optional): Version of dataset, Possible ``100`` or ``300``, default 100.

    """
    base_folder_list = ['joy-batches-py']
    base_folder = None
    train_list = None
    test_list = None

    def __init__(self, root, n_classes: int, train=True,
                 transform=None, target_transform=None, validation=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        
        self._load_hashes()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            # self.train_data = []
            # self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                # self.train_data.append(entry['data'])
                # if 'labels' in entry:
                #     self.train_labels.append(entry['labels'])
                # else:
                #     self.train_labels.append(entry['fine_labels'])
                fo.close()
                self.train_data = entry['data']
                self.train_labels = entry['labels']
            # self.train_data = np.concatenate(self.train_data)
            # self.train_data = self.train_data.reshape((100, 640, 480, 3))
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            if validation:
                data,labels=[],[]
                for i in range(n_classes):
                    data.append(self.test_data[np.array(self.test_labels) == i][:100])
                    labels.append(np.array(self.test_labels)[np.array(self.test_labels) == i][:100])
                self.test_data = np.concatenate(data,axis=0)
                self.test_labels = np.concatenate(labels,axis=0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def _load_hashes(self):
        self.base_folder = self.base_folder_list[0]
        root = self.root
        if self.train_list == None:
            hashes_file=os.path.join(root, self.base_folder, 'hashes.json')
            assert os.path.exists(hashes_file), 'The hashes.json doesnt exist'
            with open(hashes_file) as hashes_buffer:    
                hashes = json.loads(hashes_buffer.read())
            keys = list(hashes.keys())
            self.train_list = list()
            self.test_list = list()
            for key in keys:
                if 'train' in key:
                    self.train_list.append(list((key,hashes[key])))
                elif 'test' in key:
                    self.test_list.append(list((key,hashes[key])))