#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import os
import shutil
import pickle
from datetime import datetime
import multiprocessing as mp

from tub_nmd_VIII_model import * 


res_folder = "./res/"+str(datetime.now()).split('.')[0]+'/'