from PIL import Image
import os
import numpy as np
from scipy import ndimage

# Plot simples para ver médias das imagens em um folder 
# (em linux, adaptar path para outros SOs)

path = "/dataset/caddy-gestures-complete-v2-release-all-scenarios-fast.ai/"

def mean(folder: str):
    fullpath = os.getcwd() + path + folder + "/"
    files = os.listdir(fullpath)
    images = [file for file in files]
    w,h = Image.open(fullpath + images[0]).size
    N = len(images)
    
    arr = np.zeros((h,w,3), float)
    
    # Build up average pixel intensities, casting each image as an array of floats
    for im in images:
        curr_path = fullpath + im
        imarr = np.array(Image.open(curr_path), dtype=float)
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)

    # Generate, save and preview final image
    out = Image.fromarray(arr, mode="RGB")
    out.save("meanImgs/" + folder + "_average.png")
    out.show()

# Não está funcionando
def STDnMean(folder):
    fullpath = ""
    files = os.listdir(fullpath)
    arr = np.array([Image.open(fullpath + img) for img in files])
    rgb_values = np.concatenate(
        arr, 
        axis=0
    ) / 255.

    # rgb_values.shape == (n, 3), 
    # where n is the total number of pixels in all images, 
    # and 3 are the 3 channels: R, G, B.

    # Each value is in the interval [0; 1]

    mu_rgb = np.mean(rgb_values, axis=0)  # mu_rgb.shape == (3,)
    std_rgb = np.std(rgb_values, axis=0)  # std_rgb.shape == (3,)
    
    scipyMean = ndimage.mean(arr);
    
    # Generate final image
    outMean = Image.fromarray(scipyMean, mode="RGB")
    outMean.save(folder + "_mean.png")
    
    outSTD = Image.fromarray(std_rgb, mode="RGB")
    outSTD.save(folder + "_std.png")

def getAllMeans():
    folders = os.listdir(os.getcwd() + path)
    for folder in folders:
        mean(folder)

if __name__ == '__main__':
    getAllMeans()
    