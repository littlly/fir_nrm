import os, time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import averager
from seetings import settings

from trendLayer import TrendNormalize

s = settings()

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.cnn1 = nn.Conv3d(4,36,kernel_size=11,padding=2)
        self.cnn2 = nn.Conv3d(36,8,kernel_size=8, padding=6)
        self.avg3d1 = nn.AvgPool3d(kernel_size=5, stride=2)
        self.avg3d2 = nn.AvgPool3d(kernel_size=4, stride=2)
        self.act = nn.ReLU()
        self.fla = nn.Flatten()
        self.lin1 = nn.Linear(15552,2048)
        self.lin2 = nn.Linear(2048,256)
        self.lin3 = nn.Linear(256,2*s.output_horizon)
        self.tre = TrendNormalize()

    def forward(self, x):

        x = self.cnn1(x)
        x = self.act(x)
        x = self.avg3d1(x)
        x = self.cnn2(x)
        x = self.act(x)
        x = self.avg3d2(x)
        x = self.fla(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.act(x)
        x = self.lin3(x)
        x = torch.reshape(x, (-1,2,s.output_horizon))
        x = self.tre(x)

        return x

    def init_model(self):
        for m in self.children():
            if isinstance(m,(nn.Linear,nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
        print('init mode successful!')


    def train_model(self, train_iter, val_iter, criterion = nn.MSELoss()):
        print('train on gpu')
        self.init_model()
        self.cuda()
        optimizer = optim.Adam(self.parameters(), lr = s.lr)
        loss = nn.MSELoss

        model_save_path = os.path.join(s.log_path)

        try:
            os.mkdir(model_save_path)
        except:
            pass

        model_path = os.path.join(model_save_path, 'model_best.pth')
        model_state_file = open(os.path.join(model_save_path, 'model.txt'), 'w')
        training_log_file = open(os.path.join(model_save_path, 'training.log'), 'w')
        validation_log_file = open(os.path.join(model_save_path, 'validation.log'), 'w')

        training_log_file.write('Epoch,Loss\n')
        validation_log_file.write('Epoch,Loss\n')


        for epoch in range(s.num_epoch):
            self.train()
            epoch_start = time.time()
            train_loss_aver = []
            for i, data in enumerate(train_iter, 0 ):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                output = self(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss_aver.append(loss)

            val_loss = self.__val(val_iter,criterion)

            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            train_loss = sum(train_loss_aver)/len(train_loss_aver)

            print('[{:.4f}s] [{}]/[{}] Loss: {:.4f} VALLoss: {:.4f}'.format(epoch_duration,epoch+1,s.num_epoch,train_loss,val_loss))

            training_log_file.write('{},{}\n'.format(epoch, train_loss))
            validation_log_file.write('{},{}\n'.format(epoch, val_loss))
            training_log_file.flush()
            validation_log_file.flush()

        model_state_file.write("Model's state_dict:\n")
        for param_tensor in self.state_dict():
            model_state_file.write('{},{}\n'.format(param_tensor, self.state_dict()[param_tensor].size()))
        model_state_file.write("Model's state_dict:\n")
        for var_name in optimizer.state_dict():
            model_state_file.write('{},\n{}\n'.format(var_name, optimizer.state_dict()[var_name]))

        training_log_file.close()
        torch.save(self.state_dict(),s.log_path + '/model_best.pth')
        print("save successful!")


    def __val(self, test_iter, criterion):
        self.eval()
        val_loss_aver = []
        n = len(test_iter)
        with torch.no_grad():

            for i, data in enumerate(test_iter, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                output = self(inputs)
                loss = criterion(output, labels)
                val_loss_aver.append(loss.item())

        val_loss = sum(val_loss_aver)/len(val_loss_aver)
        return val_loss

    def test_model(self, path_model, test_iter):
        self.load_state_dict(torch.load(path_model))
        self.cuda()
        print("load successful!")
        val_loss = self.__val(test_iter, nn.MSELoss())
        output = []
        for i, data in enumerate(test_iter, 0 ):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            output.append(self(inputs).cpu().detach().numpy())
        return val_loss, output




        













