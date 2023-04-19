from PIL import Image
import os
import numpy as np
from multiprocessing import Process


# Plot simples para ver médias das imagens em um folder
# (em linux, adaptar path para outros SOs)

path = "/dataset/caddy-gestures-complete-v2-release-all-scenarios-fast.ai/"


def mean(folder: str):
    fullpath = os.getcwd() + path + folder + "/"
    files = os.listdir(fullpath)
    images = [file for file in files]
    w, h = Image.open(fullpath + images[0]).size
    N = len(images)

    arr = np.zeros((h, w, 3), float)
    std_arr = []

    # Build up average pixel intensities, casting each image as an array of floats
    for im in images:
        curr_path = fullpath + im
        imarr = np.array(Image.open(curr_path), dtype=float)
        #std_arr.append(imarr)
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    #stdev = np.std(np.array(std_arr, dtype=np.uint8), axis=0)
    #Image.fromarray(stdev).save(folder + '_stdev.png')

    # Generate, save and preview final image
    out = Image.fromarray(arr, mode="RGB")
    out.save("meanImgs/" + folder + "_average.png")

# Não está funcionando
def meanAndStd(folder):
    imgs_path = fullpath = os.getcwd() + path + folder + "/"
    imgs_path = os.listdir(fullpath)
    
    images = []
    for img in imgs_path:
        im = Image.open(fullpath + img)
        images.append(np.array(im))

    means = np.mean(images, axis=0)
    stdev = np.std(images, axis=0)
    
    Image.fromarray(stdev.astype(np.uint8)).save("stdevImgs/" + folder + '_stdev.png')
    Image.fromarray(means.astype(np.uint8)).save("meanImgs/" + folder + "_average.png")

def getAllMeansStds():
    folders = os.listdir(os.getcwd() + path)
    print(folders)
    joined = []
    for folder in folders:
        if False:
            x = Process(target=meanAndStd, args=(folder,))
            joined.append(x)
            x.start()
        else:
            meanAndStd(folder)
        
    for x in joined:
        x.join()


if __name__ == '__main__':
    getAllMeansStds()
