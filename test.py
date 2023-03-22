import torch
import random, datetime, argparse, os
import numpy as np
from lib import Data, load_array, model, draw_graph_station
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



from seetings import settings

s = settings()
data_use = Data()
model_use = model()
[X_train, y_train], [X_val, y_val], [X_test, y_test] = data_use.load_data_cnn()

test_iter = load_array((torch.from_numpy(X_test),torch.from_numpy(y_test)), 1, shuffle = False)
path_model =  s.log_path + '/model_best.pth'

val_loss, predict = model_use.test_model(path_model,test_iter)
i = random.randint(1,len(predict))

draw_graph_station(data_use,y_test,np.array(predict),drawu=True)





