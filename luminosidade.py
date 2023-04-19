
import os 
from skimage import io
from skimage.color import rgb2gray

images_path = []

image_paths = os.listdir("meanImgs")
images = [io.imread("meanImgs/"+path) for path in image_paths]

# Converte as imagens para escala de cinza
gray_images = [rgb2gray(image) for image in images]

# Calcula a média da luminosidade de cada imagem
mean_luminosity = [gray_image.mean() for gray_image in gray_images]

# Exibe as médias de luminosidade
for i, path in enumerate(image_paths):
    print(f"A média de luminosidade da imagem {i+1} ({path}) é {mean_luminosity[i]:.2f}")

soma = 0
for luminosidade in mean_luminosity:
    soma+=luminosidade

media = soma/len(mean_luminosity)
print(f"A média da luminosidade de todas as imagens é:{media:.2f}")