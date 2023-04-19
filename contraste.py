from skimage import io, color, exposure
import matplotlib.pyplot as plt

# Carrega a imagem
image = io.imread('dataset/backwards/backwards-0004.jpg')

# Converte para escala de cinza
gray_image = color.rgb2gray(image)

# Equaliza o histograma para aumentar o contraste
equalized_image = exposure.equalize_hist(gray_image)

# Exibe as imagens
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Imagem em escala de cinza')

ax[1].imshow(equalized_image, cmap='gray')
ax[1].set_title('Imagem com contraste ajustado')

plt.show()
