import torch
import torch.nn as nn
import torch.optim as opt

from pathlib import Path
import numpy as np

from data import data_loader
from utils import get_SVM_performance
from losses import SupConLoss, SimCLRLoss
from networks import Encoder

class BaseModel:
    def __init__(self, data, val_metric, loss, epochs, batch_size, step, temperature, lr, wd, device, res_path):
        self.data = data
        self.val_metric = val_metric
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.step = step
        self.temperature = temperature
        self.lr = lr
        self.wd = wd
        self.device = device
        self.res_path = res_path

    def _reset_fold_results(self):
        self.fold_res = {
            'loss': [],
            f'train_{self.val_metric}': [],
            f'val_{self.val_metric}': [],
            f'best_val_{self.val_metric}': -np.inf,
            f'best_val_{self.val_metric}_epoch': None, 
            'test_mcc': None,
            'test_f1': None,
            'test_ap': None,
            'SVM_best_params': None,
            'test_y': None,
            'probs': None
        }


    def _weights_init(self, net):
        if type(net) == nn.Linear:
            net.weight.data.normal_(0.0, 0.02)
            net.bias.data.fill_(0)
        

    def _get_new_model(self):
        self.enc = Encoder(**self.enc_kwargs).to(self.device)
        self.enc.apply(self._weights_init)
        self.opt = opt.Adam(self.enc.parameters(), lr=self.lr, weight_decay=self.wd)


    def _load_best_model(self, fold):
        self.best_enc = Encoder(**self.enc_kwargs).to(self.device)
        checkpoint = torch.load(
            Path(
                self.res_path, f'EGsCL--{self.loss}--{fold}--{self.val_metric}--{self.beta}_checkpoint.pt'
            )
        )
        self.best_enc.load_state_dict(checkpoint)
        self.best_enc.eval()
    
    
    def _model_eval(self, fold, epoch, train_h, train_y, val_h, val_y):
        res = get_SVM_performance(self.val_metric,
            train_h.detach().cpu().numpy(), train_y.numpy().flatten(), 
            val_h.detach().cpu().numpy(), val_y.numpy().flatten(), 
            self.data.folds
        )
        self.fold_res[f'val_{self.val_metric}'].append(res[f'test_{self.val_metric}'])
        self.fold_res[f'train_{self.val_metric}'].append(res[f'train_{self.val_metric}'])

        if res[f'test_{self.val_metric}'] >= self.fold_res[f'best_val_{self.val_metric}']:
            torch.save(
                self.enc.state_dict(), Path(
                    self.res_path, f'EGsCL--{self.loss}--{fold}--{self.val_metric}--{self.beta}_checkpoint.pt'
                )
            )
            self.fold_res[f'best_val_{self.val_metric}'] = res[f'test_{self.val_metric}']
            self.fold_res[f'best_val_{self.val_metric}_epoch'] = epoch


    def _get_test_performance(self, train_X, train_y, test_X, test_y):
        with torch.no_grad():
            train_h, _ = self.best_enc(train_X.to(self.device))
            test_h, _ = self.best_enc(test_X.to(self.device))

        res = get_SVM_performance(self.val_metric,
            train_h.detach().cpu().numpy(), train_y.numpy().flatten(), 
            test_h.detach().cpu().numpy(), test_y.numpy().flatten(), 
            self.data.folds
        )

        self.fold_res['test_mcc'] = res['test_mcc']
        self.fold_res['test_f1'] = res['test_f1']
        self.fold_res['test_ap'] = res['test_ap']
        self.fold_res['SVM_best_params'] = res['best_params']
        self.fold_res['test_y'] = res['test_y']
        self.fold_res['probs'] = res['probs']        


    def _fit(self, *args):

        raise NotImplementedError('The _fit method is not implemeneted')
    
    
    def fit_cv(self):
        self.cv_res = {}
        val_X = torch.tensor(self.data.val_X, dtype=torch.float32)
        val_y = torch.tensor(self.data.val_y, dtype=torch.long).view(-1, 1)

        for fold, (train_fold, test_fold) in enumerate(self.data.folds.split(self.data.train_X, 
                                                                             self.data.train_y)):
            train_X = torch.tensor(self.data.train_X[train_fold], dtype=torch.float32)
            train_y = torch.tensor(self.data.train_y[train_fold], dtype=torch.long).view(-1, 1)
            test_X = torch.tensor(self.data.train_X[test_fold], dtype=torch.float32)
            test_y = torch.tensor(self.data.train_y[test_fold], dtype=torch.long).view(-1, 1)

            self._get_new_model()
            self._reset_fold_results()
            self._fit(fold, train_X, train_y, val_X, val_y)

            self._load_best_model(fold)
            self._get_test_performance(train_X, train_y, test_X, test_y)

            self.cv_res[fold] = self.fold_res


class EGsCL(BaseModel):
    def __init__(self, data, val_metric, loss, epochs, batch_size, step, temperature, lr, wd, device, res_path, 
                 enc_kwargs, beta):
        super().__init__(
            data = data,
            val_metric = val_metric,
            loss = loss,
            epochs = epochs,
            batch_size = batch_size,
            step = step,
            temperature = temperature,
            lr = lr,
            wd = wd,
            device = device,
            res_path = res_path
        )
        self.enc_kwargs = enc_kwargs
        self.beta = beta


    def _fit(self, fold, train_X, train_y, val_X, val_y):
        batch_size = train_X.size(0) if self.batch_size == -1 else self.batch_size
        data_loader_ = data_loader(train_X, train_y, batch_size)

        if self.loss == 'SupCon':
            criterion = SupConLoss(self.device, self.temperature)
        elif self.loss == 'SimCLR':
            criterion = SimCLRLoss(self.device, self.temperature)
        else:
            raise NotImplementedError(
                f'{self.loss} is not implemeneted. Please enter one of the following losses: '\
                'SupCon or SimCLR'
            )

        # Model training
        for epoch in range(self.epochs + 1):
            self.enc.train()

            for x, y in data_loader_:
                x = x.to(self.device)
                y = y.to(self.device)

                self.enc.zero_grad()

                if self.beta == 'fixed':
                    x_i = x + torch.normal(mean=0, std=1, size=x.size(), device=self.device)
                    x_j = x + torch.normal(mean=0, std=1, size=x.size(), device=self.device)

                else:
                    x_i = x + torch.normal(mean=train_X.mean() + self.beta, std=train_X.std(), size=x.size(), device=self.device)
                    x_j = x + torch.normal(mean=train_X.mean() - self.beta, std=train_X.std(), size=x.size(), device=self.device)

                _, proj_i = self.enc(x_i)
                _, proj_j = self.enc(x_j)

                if self.loss == 'SupCon':
                    loss = criterion(proj_i, proj_j, y)
                else:
                    loss = criterion(proj_i, proj_j)

                loss.backward()
                self.opt.step()

                self.fold_res['loss'].append(loss.item())

            # Model evaluation
            if epoch % self.step == 0:
                self.enc.eval()
                with torch.no_grad():
                    train_h, _ = self.enc(train_X.to(self.device))
                    val_h, _ = self.enc(val_X.to(self.device))

                self._model_eval(fold, epoch, train_h, train_y, val_h, val_y)
