#operating system operations
import os

#pytorch
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor') #put everything on GPU
import torch.nn as nn #neural network tools
from torch.optim import Adam, LBFGS #this is my optimizer (fancy SGD)
from torch.utils.data import DataLoader, Dataset #batching tools
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#pytorch lightning: automated training wrapper 
#import pytorch_lightning as pl

import torch_optimizer as optim

#arrays etc
import numpy as np

#normal distribution and solver, respectively
from scipy.stats import norm
from scipy.optimize import fsolve, minimize, root_scalar

#floor and ceiling functions
from math import floor, ceil

#progress bar 
from tqdm import tqdm

#hide some messages I don't want to see
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

#plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines
from matplotlib.ticker import StrMethodFormatter

import argparse