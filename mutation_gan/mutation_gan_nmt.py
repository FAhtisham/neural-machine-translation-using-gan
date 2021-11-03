
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm 

import os
from timeit import default_timer as timer

from Dataset import *
from models import *

DEVICE = torch.device("cuda:3" if torch.cuda.is_available else "cpu")



