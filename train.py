import torch
import random, datetime, argparse, os
import numpy as np
from lib import Data, load_array, model
import torch.nn as nn
import torch.optim as optim

from seetings import settings

s = settings()
data_use = Data()
model_use = model()
[X_train, y_train], [X_val, y_val], [X_test, y_test] = data_use.load_data_cnn()

train_iter = load_array((torch.from_numpy(X_train),torch.from_numpy(y_train)), s.bitch_size, shuffle = False)
val_iter = load_array((torch.from_numpy(X_val),torch.from_numpy(y_val)), 1, shuffle = True)

model_use.train_model(train_iter, val_iter)


