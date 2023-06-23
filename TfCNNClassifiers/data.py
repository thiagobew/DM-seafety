import sys
sys.dont_write_bytecode = True

import os
from PIL import Image, ImageFilter
from skimage import io, filters
import os, os.path
import sys
import time
import pickle
import random
import pandas as pd
import numpy as np
import skimage
from skimage import data, io
import cv2
from sklearn.utils import shuffle
import shutil
import matplotlib.pyplot as plt
from typing import Callable

IMG_HEIGHT = 480
IMG_WIDTH = 640
IMG_CHANNELS = 3
random.seed(6)

cur_dir = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(cur_dir)

def apply_sobel(dataset_path: str):
    dataSetPath = os.path.join(IMG_DIR, dataset_path)
    folders = ['down', 'carry', 'start_comm', 'up', 'five', 'four', 'num_delimiter',
            'one', 'two', 'end_comm', 'here', 'mosaic', 'backwards', 'three', 'boat', 'photo']

    count = []
    total = 0
    dirs = os.listdir(dataSetPath)
    for dir in dirs:
        count.append((len(os.listdir(os.path.join(dataSetPath, dir))), dir))
        total += count[-1][0]

    for folder in folders:
        folderPath = os.path.join(dataSetPath, folder)
        files = os.listdir(folderPath)
        if (len(files) < total / len(folders)): 
            for filename in files:

                if filename.endswith('.jpg') or filename.endswith('.png'):

                    img = Image.open(os.path.join(folderPath, filename))

                    img_inverted = img.transpose(Image.FLIP_LEFT_RIGHT)

                    img_inverted.save(os.path.join(folderPath, filename))
            
    count = []       
    for dir in dirs:
        count.append((len(os.listdir(os.path.join(dataSetPath, dir))), dir))
        total += count[-1][0]

    for folder in folders:
        folderPath = os.path.join(dataSetPath, folder)
        files = os.listdir(folderPath)
        if (len(files) < total / len(folders)): 
            for filename in files:

                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image = io.imread(os.path.join(folderPath, filename), as_gray=True)

                    # Aplica o filtro sobel para detectar as bordas
                    edge_sobel = filters.sobel(image)
                    path_to_save = os.path.join(folderPath, filename)
                    
                    io.imsave(path_to_save, edge_sobel, plugin='pil')
