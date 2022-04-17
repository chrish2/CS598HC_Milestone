
import os
import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import Dataset
from model import LSTM



def train(model, train_loader, val_loader, n_epochs, optimizer, criterion):
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        print(len(train_loader.dataset))
        for x, y in train_loader:
            optimizer.zero_grad()
            # print(x)

            y_hat = model(x)
            #
            # print(type(y))
            # print(type(y_hat))

            y_hat = y_hat.to(torch.float32)
            y = y.to(torch.float32)

            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch + 1, train_loss))
        p, r, f, roc_auc = eval(model, val_loader)
        print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'.format(epoch + 1, p, r, f,
                                                                                               roc_auc))
    return round(roc_auc, 2)


from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def eval(model, val_loader):

    model.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    model.eval()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_logit = model(x, masks, rev_x, rev_masks)
        y_hat = y_logit >= 5

        y_score = torch.cat((y_score, y_logit.detach().to('cpu')), dim=0)
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc



if __name__ == '__main__':
    batch_size = 20
    train_dataset = Dataset(data_path='./data/cleaned2.txt', is_val=False)
    print(len(train_dataset.data))
    val_dataset = Dataset(data_path='./data/cleaned2.txt', is_val=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # instantiate the model
    model = LSTM(2, 128, 2, 200)

    # load the loss function
    criterion = nn.BCELoss()
    # load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 5
    train(model, train_loader, val_loader, n_epochs, optimizer, criterion)
