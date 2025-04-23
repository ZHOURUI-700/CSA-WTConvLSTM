import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.Dataloader import TEC_Dataset
from Models.CSA_WTConvLSTM_Model import CSA_WTConvLSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

zr_id = random.randint(0, 1000000000)
seed = zr_id
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
time_step = 12
predict_step = 12
batch_size = 20
input_dim = 1
CSA_hidden_dim = [4]
CSA_num_layers = 1
WTConvLSTM_hidden_dim = [4]
WT_num_layers = 1
height = 71
width = 73
learning_rate = 0.001
epochs = 3

train_data = TEC_Dataset(path='./Dataset/', time_step=time_step, predict_step=predict_step, train=True)
validation_data = TEC_Dataset(path='./Dataset/', time_step=time_step, predict_step=predict_step, validation=True)

train_dataloader = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
validation_dataloader = DataLoader(validation_data, batch_size, shuffle=True, pin_memory=True)

CSA_WTConvLSTM_Execute = CSA_WTConvLSTM(input_dim=input_dim, CSA_hidden_dim=CSA_hidden_dim,
                                        CSA_num_layers=CSA_num_layers,
                                        WTConvLSTM_hidden_dim=WTConvLSTM_hidden_dim,
                                        WT_num_layers=WT_num_layers, height=height, width=width, kernel_size=3,
                                        predict_step=predict_step).cuda()

loss_function = nn.MSELoss()
optimizer = optim.Adam(CSA_WTConvLSTM_Execute.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True, threshold=0.00001,
                              threshold_mode='abs', cooldown=10, min_lr=0, eps=1e-08)

print(CSA_WTConvLSTM_Execute)

best_val_loss = 1

for epoch in range(epochs):
    loss_train_total = 0
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for idx, (inputs, targets) in loop:
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs = inputs.cuda()

        targets = targets.type(torch.cuda.FloatTensor)
        targets = targets.cuda()

        outputs = CSA_WTConvLSTM_Execute(inputs)

        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        loss_train_total += loss.item()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch}/{epochs}]')
        loop.set_postfix_str({'Loss': loss.item()})
    print("\nepoch:", epoch, "loss_train_total / len(train_dataloader):", loss_train_total / len(train_dataloader))

    CSA_WTConvLSTM_Execute.eval()
    loss_test_total = 0
    rmse1 = 0
    rmse2 = 0
    loop1 = tqdm(enumerate(validation_dataloader), total=len(validation_dataloader))
    with torch.no_grad():
        for idx, (inputs, targets) in loop1:
            inputs = inputs.type(torch.cuda.FloatTensor)
            inputs = inputs.cuda()

            targets = targets.type(torch.cuda.FloatTensor)
            targets = targets.cuda()

            outputs = CSA_WTConvLSTM_Execute(inputs)

            loss = loss_function(outputs, targets)
            loss_test_total += loss.item()
            rmse1 = torch.sqrt(torch.mean((targets - outputs) ** 2))
            rmse1 += rmse1.item()

            TEC_MAX = 147.1
            TEC_MIN = 0.0
            result = (TEC_MAX - TEC_MIN) * outputs + TEC_MIN
            true = (TEC_MAX - TEC_MIN) * targets + TEC_MIN
            rmse2 = torch.sqrt(torch.mean((true - result) ** 2))
            rmse2 += rmse2.item()

            loop1.set_description(f'Epoch [{epoch}/{epochs}]')
            loop1.set_postfix_str({'Loss': loss.item()})

        print("epoch:", epoch, "loss_test_total / len(test_dataloader):", loss_test_total / len(validation_dataloader))
        print("epoch:", epoch, "rmse1:", rmse1)
        print("epoch:", epoch, "rmse2:", rmse2)

        average_epoch_loss = loss_test_total / len(validation_dataloader)

    if best_val_loss > average_epoch_loss:
        if epoch == 0:
            torch.save(CSA_WTConvLSTM_Execute.state_dict(),
                       f'./id_{zr_id}_CSA-WTConvLSTM.pth')
            best_val_loss = average_epoch_loss
            print("best_loss" + 'update' + ', pth saved!')
            temp = epoch
        else:
            os.remove(
                f'./id_{zr_id}_CSA-WTConvLSTM.pth')
            torch.save(CSA_WTConvLSTM_Execute.state_dict(),
                       f'./id_{zr_id}_CSA-WTConvLSTM.pth')
            best_val_loss = average_epoch_loss
            print("best_loss" + 'update' + ', pth saved!')
            temp = epoch
    print('-' * 30)
    scheduler.step(average_epoch_loss)

test_data = TEC_Dataset(path='./Dataset/', time_step=time_step, predict_step=predict_step, test=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)
CSA_WTConvLSTM_Execute.eval()
out_list = []
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_dataloader):
        if idx == 0:
            true = targets.cuda()
        else:
            true = torch.cat((true, targets.cuda()), 0)
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs = inputs.cuda()
        out = CSA_WTConvLSTM_Execute(inputs)
        out_list.append(out)
    print(len(out_list))
outputs = torch.cat(out_list, 0)
outputs = outputs.view(-1, 1, 71, 73)
true = true.view(-1, 1, 71, 73)

true = true.cpu().numpy()
result = outputs.cpu().numpy()

TEC_MAX = 147.1
TEC_MIN = 0.0
result = (TEC_MAX - TEC_MIN) * result + TEC_MIN
true = (TEC_MAX - TEC_MIN) * true + TEC_MIN
np.save("./CSA_WTConvLSTM.npy", result)
