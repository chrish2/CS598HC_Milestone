
import os
import time
import numpy as np
# from pyrsistent import T

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import Dataset

from LSTM import LSTM
from BiLSTM_Attention import BiLSTM_Attention
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score



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

            y_hat = y_hat.to(torch.float32).squeeze()
            y = y.to(torch.float32)

            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch + 1, train_loss))
    
    #     print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch + 1, train_loss))
    # eval(model, val_loader)
        # p, r, f, roc_auc = eval(model, val_loader)
        # print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'.format(epoch + 1, p, r, f,
        #                                                                                        roc_auc))
    # return round(roc_auc, 2)


def eval(model, val_loader):
    val_loss = 0
    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    model.eval()

    ass = []
    # print(len(val_loader))
    count = 0
    for x, y in val_loader:
        # print(count)
        # count += 1
        y_hat = model(x)
        y_hat = y_hat.to(torch.float32).squeeze()
        y = y.to(torch.float32)
        # print(y_hat, y_hat.size())
        y_hat_= [ yy.detach().numpy() for yy in y_hat]
        threshold = np.percentile(y_hat_, 75)
        y_hat2 = (y_hat > threshold).float() * 1
        y_hat2 = y_hat2
        ass.append(y_hat)
        y_pred = torch.cat((y_pred,  y_hat2.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)

        loss = criterion(y_hat, y)

        loss.backward()
        val_loss += loss.item()
    
    # total_count, correct = len(y_true), 0
    # for y, y_hat in zip(y_true, y_pred):
    #     if y[0] == y_hat[0]:
    #         correct += 1
    # incorrect = total_count - correct
    # print( correct / total_count)

    with open("BiLSTM_Attention.txt", 'w') as f:
        for i in range(len(y_pred)):
                f.write(str(float(y_pred[i])) +"|"+ str(float(y_true[i])) + "\n")


    # print(y_true)
    # print(y_pred)


    val_loss = val_loss / len(val_loader)
    print(val_loss)

    # p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    # print('{:.2f}, r:{:.2f}, f: {:.2f}',p,r,f)


if __name__ == '__main__':
    batch_size = 20
    train_dataset = Dataset(data_path='./data/cleaned.txt', is_val=False)
    print(len(train_dataset.data))
    val_dataset = Dataset(data_path='./data/cleaned.txt', is_val=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # y_hat = torch.tensor([1.0, 2.0,3.0, 4, 5])
    # y_hat_ = torch.tensor([1.0, 2.0,3.0, 4, 5])
    # tt = np.percentile(y_hat, 75)
    # print(np.percentile(y_hat, 75))
    # print( (y_hat > tt).float() * 1)
    # instantiate the model

    # model = BiLSTM_Attention(2, 128, 1, 200)
    model = BiLSTM(2, 128, 2, 200)

    # load the loss function
    criterion = nn.BCELoss()
    # load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_epochs = 5
    train(model, train_loader, val_loader, n_epochs, optimizer, criterion)
    eval(model, val_loader)
