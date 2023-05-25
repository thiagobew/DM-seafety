import os
from PIL import Image, ImageFilter
from skimage import io, filters

dataSetPath = os.path.join(os.getcwd(), "dataset")
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

                img_inverted.save(os.path.join(folderPath, 'inverted_' + filename))
         
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
                path_to_save = os.path.join(folderPath, 'sobel_' + filename)
                path_to_save.replace('.jpg', '.png')
                
                io.imsave(path_to_save, edge_sobel, plugin='pil')