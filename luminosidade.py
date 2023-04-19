
import os 
from skimage import io
from skimage.color import rgb2gray

images_path = []
for x in (os.listdir("dataset")):
    for image in os.listdir("dataset/"+x):
        images_path.append(x+"/"+image )


image_paths = images_path
images = [io.imread("dataset/"+path) for path in image_paths]

# Converte as imagens para escala de cinza
gray_images = [rgb2gray(image) for image in images]

# Calcula a média da luminosidade de cada imagem
mean_luminosity = [gray_image.mean() for gray_image in gray_images]

# Exibe as médias de luminosidade
for i, path in enumerate(image_paths):
    print(f"A média de luminosidade da imagem {i+1} ({path}) é {mean_luminosity[i]:.2f}")
