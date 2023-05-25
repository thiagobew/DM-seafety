import os
from PIL import Image

dataSetPath = os.path.join(os.getcwd(), "dataset")
folders = ['down', 'carry', 'start_comm', 'up', 'five', 'four', 'num_delimiter',
           'one', 'two', 'end_comm', 'here', 'mosaic', 'backwards', 'three', 'boat', 'photo']

for folder in folders:
    folderPath = os.path.join(dataSetPath, folder)
    files = os.listdir(folderPath)
    if (len(files) < 1300): 
        for filename in files:

            if filename.endswith('.jpg') or filename.endswith('.png'):

                img = Image.open(os.path.join(folderPath, filename))

                img_inverted = img.transpose(Image.FLIP_LEFT_RIGHT)

                img_inverted.save(os.path.join(folderPath, 'inverted_' + filename))
