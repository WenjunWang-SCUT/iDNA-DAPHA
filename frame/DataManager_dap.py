import pickle
import torch
import torch.utils.data as Data

import numpy as np
from util import util_file


class DataManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.config = learner.config

        self.mode = self.config.mode

        # label
        self.train_label = []
        self.test_label = None
        # raw_data
        self.train_dataset = []
        self.test_dataset = None
        # iterator
        self.train_dataloader = None
        self.test_dataloader = None

    def load_data(self):
        train_type = []
        index = 0
        
        for key,file_train_data in self.config.path_train_data.items():
            item_train_dataset, item_train_label = util_file.load_tsv_format_data(file_train_data)
            self.train_dataset.extend(item_train_dataset)
            self.train_label.extend(item_train_label)
            train_type.extend([index]*len(item_train_dataset))
            index += 1
        self.train_dataloader = self.construct_dataset_type(self.train_dataset, self.train_label, self.config.cuda,
                                                       self.config.batch_size, train_type)
      
    def construct_dataset_type(self, sequences, labels, cuda, batch_size, type):
        if cuda:
            labels = torch.cuda.LongTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        dataset = MyDataSet_type(sequences, labels, type)
        data_loader = Data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      drop_last=False,
                                      shuffle=True)
        print('len(data_loader)', len(data_loader))
        return data_loader

    def construct_dataset(self, sequences, labels, cuda, batch_size):
        if cuda:
            labels = torch.cuda.LongTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        dataset = MyDataSet(sequences, labels)
        data_loader = Data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      drop_last=False,
                                      shuffle=True)
        print('len(data_loader)', len(data_loader))
        return data_loader

    def get_dataloder(self, name):
        return_data = None
        if name == 'train_set':
            return_data = self.train_dataloader
        elif name == 'test_set':
            return_data = self.test_dataloader

        return return_data


class MyDataSet_type(Data.Dataset):
    def __init__(self, data, label, type):
        self.data = data
        self.label = label
        self.type = type

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.type[idx]

class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
