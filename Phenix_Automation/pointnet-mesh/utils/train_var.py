from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('..')
# sys.path.append(os.path.join(BASE_DIR, 'pointnet'))
# sys.path.append(os.path.join(BASE_DIR, 'utils'))
from pointnet.basic_model import basic
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np




if __name__ == "__main__":
    dt1 = torch.empty(1, 3, 1020)
    torch.nn.init.uniform_(dt1)
    dt2 = torch.empty(1,3, 1200)
    torch.nn.init.uniform_(dt2)
    dt3 = torch.empty(1, 3, 1500)
    torch.nn.init.uniform_(dt3)
    mdl = basic()
    dt1_out = mdl.forward(dt1)
    print(dt1_out.shape)
    dt2_out = mdl.forward(dt2)
    print(dt2_out.shape)
    dt3_out = mdl.forward(dt3)
    print(dt3_out.shape)