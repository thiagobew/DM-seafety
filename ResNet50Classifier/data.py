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
IMG_DIR = os.path.join(cur_dir, 'images')
CLASSES = ['none', 'start', 'end', 'up', 'down', 'photo', 'backward', 'carry', 'boat', 'here', 'mosaic', 'delimiter', 'one', 'two', 'three', 'four', 'five']
NUM_CLASSES = 17

def get_image_count(IMG_DIR, CLASSES, NUM_CLASSES):
    data_summary = pd.DataFrame(CLASSES)
    data_summary.columns = ['label']
    data_summary['label_id'] = range(-1,NUM_CLASSES-1)
    
    for split in ['train', 'test']:
        split_count = []
        for j in range(NUM_CLASSES):
            DIR = IMG_DIR + "/" + split + "/" + CLASSES[j]
            class_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
            split_count.append(class_count)
        data_summary[split] = split_count

    data_summary['total'] = data_summary['train'] + data_summary['test']
    print(sum(data_summary['test']))

    return data_summary

def flip(img):
    # use case: for left-handed people
    new_img = np.fliplr(img)
    return new_img

def rotate(img, angle):
    new_img = skimage.transform.rotate(img, angle=angle, mode='reflect')
    return new_img

def scale(img, zoom_factor):
    height, width = img.shape[:2] # This is the final desired shape as well
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def translate(img, hpixels, vpixels):
    #img = cv2.imread('images/input.jpg')
    num_rows, num_cols = img.shape[:2]

    translation_matrix = np.float32([ [1,0,hpixels], [0,1,vpixels] ])
    new_img = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    return new_img

def data_augmentation(IMG_DIR, CLASSES, NUM_CLASSES):
    target_count = 1500
    split = 'train'

    for j in range(1, NUM_CLASSES):
        print(j, CLASSES[j])
        DIR = os.path.join(IMG_DIR, split, CLASSES[j])
        file_lst = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        random.shuffle(file_lst)

        itr = 0
        ctr = 0
        while len(file_lst) + ctr < target_count:
            file_dir = DIR + "/" + file_lst[itr]
            filename = file_dir.rsplit('/')[-1]
            filename = filename.replace('.jpg', '')
            #print(file_dir, filename)
            img = cv2.imread(file_dir)

            # rotation
            if random.random() > 0.6:
                angle = random.randint(-10, 10)
                rot_img = rotate(img, angle)
                rotate_token = True
            else:
                rot_img = img
                rotate_token = False

            # translation
            if random.random() > 0.6:
                htranslate = random.randint(-10, 10)
                vtranslate = random.randint(-10, 10)
                trans_img = translate(rot_img, htranslate, vtranslate)
            else:
                trans_img = rot_img

            NEW_IMG_DIR = DIR + "/" + filename + "_" + str(ctr) + ".jpg"

            if rotate_token:
                plt.imsave(NEW_IMG_DIR, trans_img[:,:,::-1])
            else:
                cv2.imwrite(NEW_IMG_DIR, trans_img)

            ctr += 1
            itr += 1

            if itr == len(file_lst):
                itr = 0
    return


def split_train_val():
    #split train and val
    CLASSES.remove('none')
    for j in range(NUM_CLASSES-1):
        DIR = os.path.join(IMG_DIR,  "train", CLASSES[j])
        file_lst = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        random.shuffle(file_lst)

        train_lst = file_lst[:int(len(file_lst)*0.8)]
        val_lst = file_lst[int(len(file_lst)*0.8):]

        dest_dir = os.path.join(IMG_DIR, "val", CLASSES[j])
        os.makedirs(dest_dir, exist_ok=True)
        
        for file in val_lst:
            shutil.move(os.path.join(DIR, file), os.path.join(IMG_DIR, "val", CLASSES[j], file))
    return


def apply_sobel(dataset_path:str):
    dataSetPath = os.path.join(IMG_DIR, dataset_path)
    folders = ['down', 'carry', 'start', 'up', 'five', 'four', 'delimiter',
            'one', 'two', 'end', 'here', 'mosaic', 'backward', 'three', 'boat', 'photo']

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

def resizeAndGrayscale(folder: str):
    folders = ['down', 'carry', 'start', 'up', 'five', 'four', 'delimiter',
            'one', 'two', 'end', 'here', 'mosaic', 'backward', 'three', 'boat', 'photo']
    for a in folders:
        curr_img_dir = os.path.join(IMG_DIR, folder)

        curr_out_dir = os.path.join(IMG_DIR, folder)

        if not os.path.exists(curr_out_dir):
            os.makedirs(curr_out_dir)

        for img in os.listdir(os.path.join(curr_img_dir,a)):
            curr_img = Image.open(os.path.join(curr_img_dir,a, img))

            # Convert the image to grayscale
            curr_img = curr_img.convert('L')

            curr_img.save(os.path.join(curr_img_dir, a, img))

def preprocess_images(method: Callable[[str], None]):
    for set in ["train", "val", "test"]:
        method(set)
        
        
if __name__ == '__main__':
    data_summary = get_image_count(IMG_DIR, CLASSES, NUM_CLASSES)
    print('Total number of images: ' + str(data_summary['total'].sum()))

    data_augmentation(IMG_DIR, CLASSES, NUM_CLASSES)
    split_train_val()
    preprocess_images(resizeAndGrayscale)

