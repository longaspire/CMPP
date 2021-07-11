import torch
import torch.nn as nn
import math
import time
from torch import optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ..utils.utils import read_data,norm_Adj,evaluation,evaluation_,integrated_loss,GruMulti
import numpy as np
import os
import psutil
# np.random.seed(2021)
scaler_mean = MinMaxScaler()
scaler_variance = MinMaxScaler()


seq_len = 10
pre_len = 1
adj, features, labels, idx_train, idx_validation, idx_test, columns = read_data(seq_len=seq_len,pre_len=pre_len,train_rate=0.625,validation_rate=0.125)
norm_adj = norm_Adj(adj)
features = torch.FloatTensor(features)
features = torch.transpose(features,1,2)
labels = torch.FloatTensor(labels)
labels = torch.transpose(labels,1,2)

norm_adj = torch.FloatTensor(norm_adj)
in_channels = 2
out_channels = 16
hidden_size = 16
output_size = 1
LR = 0.01

model = GruMulti(in_channels,hidden_size,output_size)
optimizer = optim.Adam(model.parameters(),lr=LR)
batch_size = 64
batches = int(len(idx_train)/batch_size)
time_interval = 'ten_secs'#1min, 5mins


def label_transform(idx_train):
    label1, label2 = labels[idx_train][:, :, :, 0], labels[idx_train][:, :, :, 1]
    label1 = torch.reshape(label1, (-1, output_size))
    label2 = torch.reshape(label2, (-1, output_size))
    return label1, label2

# Model_save_path = './se_model.pkl'
def train(epoch):
    t = time.time()
    model.train()
    for batch in range(batches):
        optimizer.zero_grad()
        batch_idx = idx_train[batch_size*batch:batch_size*(batch+1)]
        out1, out2 = model(features[batch_idx])
        label1, label2 = label_transform(batch_idx)
        loss = integrated_loss(out1,out2,label1,label2)
        loss.backward()
        optimizer.step()
    # torch.save(model, Model_save_path)

    model.eval()
    vl = len(idx_validation)
    vst = time.time()
    out_val1, out_val2 = model(features[idx_validation])
    vet = time.time()
    label_val1, label_val2 = label_transform(idx_validation)
    loss_val = integrated_loss(out_val1, out_val2, label_val1, label_val2)
    out_val = torch.cat([out_val1,out_val2],dim=0)
    label_val = torch.cat([label_val1,label_val2],dim=0)
    rmse1, mae1, acc1 = evaluation_(label_val1.detach().numpy(),out_val1.detach().numpy())
    acc1_mape = evaluation(label_val1.detach().numpy(),out_val1.detach().numpy())[-1]
    rmse2, mae2, acc2 = evaluation(label_val2.detach().numpy(),out_val2.detach().numpy())
    acc2_norm = evaluation_(label_val2.detach().numpy(),out_val2.detach().numpy())[-1]
    # rmse, mae, acc = evaluation(label_val.detach().numpy(),out_val.detach().numpy())

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val_MEAN(mape): {:.4f}'.format(acc1_mape),
          'acc_val_MEAN(norm): {:.4f}'.format(acc1),
          'mae_val_MEAN{:.4f}'.format(mae1),
          'rmse_val_MEAN:{:.4f}'.format(rmse1),
          'acc_val_VAR(mape): {:.4f}'.format(acc2),
          'acc_val_VAR(norm): {:.4f}'.format(acc2_norm),
          'mae_val_VAR:{:.4f}'.format(mae2),
          'rmse_val_VAR:{:.4f}'.format(rmse2),
          'time: {:.4f}s'.format(time.time() - t),
          'validation dataset len:{vl}'.format(vl = vl),
          'validation spent time: {tl}'.format(tl=vet-vst),
          'validation time per sample:{pl}'.format(pl=round((vet-vst)/vl, 3)))

    return loss.item(),loss_val.item(),acc1,acc1_mape,mae1,rmse1,acc2_norm,acc2,mae2,rmse2,(time.time() - t), round((vet-vst)/vl, 5)

train_epochs = 100
fine_train_epochs = 10
t_total = time.time()
epoch_results = []
time_eva = []

for epoch in range(train_epochs):
    loss_train, loss_val, acc_val_1, acc_val_1_mape, mae_val_1, rmse_val_1,acc_val_2,acc_val_2_mape,mae_val_2,rmse_val_2, elasped_time, etps = train(epoch)
    time_eva.append(etps)
    epoch_results.append([loss_train, loss_val, acc_val_1, acc_val_1_mape, mae_val_1, rmse_val_1,acc_val_2,acc_val_2_mape,mae_val_2,rmse_val_2, elasped_time])
    print('average epts among epoch:', np.average(time_eva))



# def test(target = None):
#     t = time.time()
#     estimators.eval()
#     if target == 'mean':
#         ix = 0
#     else:
#         ix = 1
#     out_test = estimators(features[idx_test][:,:,:,[ix]])
#     label_test1, label_test2 = label_transform(idx_test)
#     if target == 'mean':
#         loss_test = F.mse_loss(out_test, label_test1)
#         rmse1, mae1, acc1 = evaluation_(label_test1.detach().numpy(), out_test.detach().numpy())
#         acc1_mape = evaluation(label_test1.detach().numpy(), out_test.detach().numpy())[-1]
#     else:
#         loss_test = F.mse_loss(out_test, label_test2)
#         rmse1, mae1, acc1 = evaluation_(label_test2.detach().numpy(), out_test.detach().numpy())
#         acc1_mape = evaluation(label_test2.detach().numpy(), out_test.detach().numpy())[-1]
#     print("Test set results:",
#           'Target:{target}'.format(target=target),
#           "loss= {:.4f}".format(loss_test.item()),
#           'acc_val_norm: {:.4f}'.format(acc1),
#           'acc_val_mape: {:.4f}'.format(acc1_mape),
#           'mae_val:{:.4f}'.format(mae1),
#           'rmse_val:{:.4f}'.format(rmse1),
#           'time: {:.4f}s'.format(time.time() - t))
#
# test(target=target)




