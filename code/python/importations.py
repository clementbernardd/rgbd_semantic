import os
os.system('pip3 install pickle5')
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import torch
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import pickle5 as pickle
from torch.autograd import Variable
import math
from copy import deepcopy
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import models


N = 21

''' Mapping betweem VGG-16 net parameters and the Encoder parameters '''
dict_conversion_vgg_name = {
    'encoder.cbr1.cbr.0.weight' : '0.weight',
    'encoder.cbr1.cbr.0.bias' : '0.bias',
    'encoder.cbr1.cbr.1.weight' : '1.weight',
    'encoder.cbr1.cbr.1.bias' : '1.bias',
    'encoder.cbr1.cbr.1.running_mean' : '1.running_mean',
    'encoder.cbr1.cbr.1.running_var' : '1.running_var',
    'encoder.cbr1.cbr.1.num_batches_tracked' : '1.num_batches_tracked',
    'encoder.cbr2.cbr.0.weight' : '3.weight',
    'encoder.cbr2.cbr.0.bias' : '3.bias' ,
    'encoder.cbr2.cbr.1.weight' : '4.weight',
    'encoder.cbr2.cbr.1.bias' : '4.bias',
    'encoder.cbr2.cbr.1.running_mean' : '4.running_mean',
    'encoder.cbr2.cbr.1.running_var' : '4.running_var',
    'encoder.cbr2.cbr.1.num_batches_tracked' : '4.num_batches_tracked',
    'encoder.cbr3.cbr.0.weight' : '7.weight',
    'encoder.cbr3.cbr.0.bias' : '7.bias' ,
    'encoder.cbr3.cbr.1.weight' : '8.weight',
    'encoder.cbr3.cbr.1.bias' : '8.bias',
    'encoder.cbr3.cbr.1.running_mean' : '8.running_mean',
    'encoder.cbr3.cbr.1.running_var' : '8.running_var',
    'encoder.cbr3.cbr.1.num_batches_tracked' : '8.num_batches_tracked',
    'encoder.cbr4.cbr.0.weight' : '10.weight',
    'encoder.cbr4.cbr.0.bias' : '10.bias' ,
    'encoder.cbr4.cbr.1.weight' : '11.weight',
    'encoder.cbr4.cbr.1.bias' : '11.bias',
    'encoder.cbr4.cbr.1.running_mean' : '11.running_mean',
    'encoder.cbr4.cbr.1.running_var' : '11.running_var',
    'encoder.cbr4.cbr.1.num_batches_tracked' : '11.num_batches_tracked',
    'encoder.cbr5.cbr.0.weight' : '14.weight',
    'encoder.cbr5.cbr.0.bias' : '14.bias' ,
    'encoder.cbr5.cbr.1.weight' : '15.weight',
    'encoder.cbr5.cbr.1.bias' : '15.bias',
    'encoder.cbr5.cbr.1.running_mean' : '15.running_mean',
    'encoder.cbr5.cbr.1.running_var' : '15.running_var',
    'encoder.cbr5.cbr.1.num_batches_tracked' : '15.num_batches_tracked',
    'encoder.cbr6.cbr.0.weight' : '17.weight',
    'encoder.cbr6.cbr.0.bias' : '17.bias' ,
    'encoder.cbr6.cbr.1.weight' : '18.weight',
    'encoder.cbr6.cbr.1.bias' : '18.bias',
    'encoder.cbr6.cbr.1.running_mean' : '18.running_mean',
    'encoder.cbr6.cbr.1.running_var' : '18.running_var',
    'encoder.cbr6.cbr.1.num_batches_tracked' : '18.num_batches_tracked',
    'encoder.cbr7.cbr.0.weight' : '20.weight',
    'encoder.cbr7.cbr.0.bias' : '20.bias' ,
    'encoder.cbr7.cbr.1.weight' : '21.weight',
    'encoder.cbr7.cbr.1.bias' : '21.bias',
    'encoder.cbr7.cbr.1.running_mean' : '21.running_mean',
    'encoder.cbr7.cbr.1.running_var' : '21.running_var',
    'encoder.cbr7.cbr.1.num_batches_tracked' : '21.num_batches_tracked',
    'encoder.cbr8.cbr.0.weight' : '24.weight',
    'encoder.cbr8.cbr.0.bias' : '24.bias' ,
    'encoder.cbr8.cbr.1.weight' : '25.weight',
    'encoder.cbr8.cbr.1.bias' : '25.bias',
    'encoder.cbr8.cbr.1.running_mean' : '25.running_mean',
    'encoder.cbr8.cbr.1.running_var' : '25.running_var',
    'encoder.cbr8.cbr.1.num_batches_tracked' : '25.num_batches_tracked',
    'encoder.cbr9.cbr.0.weight' : '27.weight',
    'encoder.cbr9.cbr.0.bias' : '27.bias' ,
    'encoder.cbr9.cbr.1.weight' : '28.weight',
    'encoder.cbr9.cbr.1.bias' : '28.bias',
    'encoder.cbr9.cbr.1.running_mean' : '28.running_mean',
    'encoder.cbr9.cbr.1.running_var' : '28.running_var',
    'encoder.cbr9.cbr.1.num_batches_tracked' : '28.num_batches_tracked',
    'encoder.cbr10.cbr.0.weight' : '30.weight',
    'encoder.cbr10.cbr.0.bias' : '30.bias' ,
    'encoder.cbr10.cbr.1.weight' : '31.weight',
    'encoder.cbr10.cbr.1.bias' : '31.bias',
    'encoder.cbr10.cbr.1.running_mean' : '31.running_mean',
    'encoder.cbr10.cbr.1.running_var' : '31.running_var',
    'encoder.cbr10.cbr.1.num_batches_tracked' : '31.num_batches_tracked',
    'encoder.cbr11.cbr.0.weight' : '34.weight',
    'encoder.cbr11.cbr.0.bias' : '34.bias' ,
    'encoder.cbr11.cbr.1.weight' : '35.weight',
    'encoder.cbr11.cbr.1.bias' : '35.bias',
    'encoder.cbr11.cbr.1.running_mean' : '35.running_mean',
    'encoder.cbr11.cbr.1.running_var' : '35.running_var',
    'encoder.cbr11.cbr.1.num_batches_tracked' : '35.num_batches_tracked',
    'encoder.cbr12.cbr.0.weight' : '37.weight',
    'encoder.cbr12.cbr.0.bias' : '37.bias' ,
    'encoder.cbr12.cbr.1.weight' : '38.weight',
    'encoder.cbr12.cbr.1.bias' : '38.bias',
    'encoder.cbr12.cbr.1.running_mean' : '38.running_mean',
    'encoder.cbr12.cbr.1.running_var' : '38.running_var',
    'encoder.cbr12.cbr.1.num_batches_tracked' : '38.num_batches_tracked',
    'encoder.cbr13.cbr.0.weight' : '40.weight',
    'encoder.cbr13.cbr.0.bias' : '40.bias' ,
    'encoder.cbr13.cbr.1.weight' : '41.weight',
    'encoder.cbr13.cbr.1.bias' : '41.bias',
    'encoder.cbr13.cbr.1.running_mean' : '41.running_mean',
    'encoder.cbr13.cbr.1.running_var' : '41.running_var',
    'encoder.cbr13.cbr.1.num_batches_tracked' : '41.num_batches_tracked'
}

dict_conversion_vgg_name_gcn = {  key.replace('encoder','gcn_encoder') : dict_conversion_vgg_name[key]  for key in dict_conversion_vgg_name   }
