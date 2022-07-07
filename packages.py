#operating system operations
import os

#pytorch
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor') #put everything on GPU
import torch.nn as nn #neural network tools
import torch.nn.functional as F #some helpful functions live here 
from torch.optim import Adam #this is my optimizer (fancy SGD)
from torch.utils.data import DataLoader, Dataset #batching tools

#pytorch lightning: automated training wrapper 
import pytorch_lightning as pl

#arrays etc
import numpy as np

#normal distribution and solver, respectively
from scipy.stats import norm
from scipy.optimize import fsolve

#floor and ceiling functions
from math import floor, ceil

#progress bar 
from tqdm import tqdm

#hide some messages I don't want to see
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

#plotting
import matplotlib.pyplot as plt