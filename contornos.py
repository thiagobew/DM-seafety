from skimage import io, color, filters, feature
from skimage.measure import find_contours
import matplotlib.pyplot as plt

# Carrega a imagem
image = io.imread('dataset/backwards/backwards-0004.jpg')

# Converte para escala de cinza
gray_image = color.rgb2gray(image)

# Encontra os contornos
contours = find_contours(gray_image)

# Cria uma figura com a imagem original e os contornos
fig, ax = plt.subplots(figsize=(8, 8))

# Plota a imagem original
ax.imshow(image)

# Plota os contornos em vermelho
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

# Remove os eixos
ax.axis('off')

plt.show()
