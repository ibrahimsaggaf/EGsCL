import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import torch


def data_loader(X, y, batch_size):
    data = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    return loader
        

class AgeingData:
    def __init__(self, train_size, cv, seed, file_X, file_y, delimiter, header, file_X2=None, 
                 delimiter2=None, header2=None):
        self.file_X = file_X
        self.file_y = file_y
        self.delimiter = delimiter
        self.header = header
        self.file_X2 = file_X2
        self.delimiter2 = delimiter2
        self.header2 = header2
        self.train_size = train_size
        self.seed = seed
        self.folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)


    def csv_reader(self, data, delimiter=',', labels=False, header=False):
        rows = []
        with open(data, 'r') as file:
            if labels:
                for row in file:
                    rows.append(row.split('%')[1])

                return np.array(rows, dtype=np.int32)

            else:
                reader = csv.reader(file, delimiter=delimiter)
                if header:
                    _ = reader.__next__()

                for row in reader:
                    if header:
                        row.pop(0)
                    if row[-1] == '':
                        row.pop(-1)
                    
                    rows.append(row)

                return np.array(rows, dtype=np.float32)


    def split(self):
        X = self.csv_reader(self.file_X, delimiter=self.delimiter, header=self.header)
        if self.file_X2 is not None:
            X2 = self.csv_reader(self.file_X2, delimiter=self.delimiter2, header=self.header2)
            X = np.hstack([X, X2])
            
        self.n = X.shape[0]
        self.dim = X.shape[1]
        y = self.csv_reader(self.file_y, labels=True)
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(X, y, 
                                                                              train_size=self.train_size, 
                                                                              random_state=self.seed, 
                                                                              shuffle=True, 
                                                                              stratify=y)

