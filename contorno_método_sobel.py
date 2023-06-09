from skimage import io, filters
import matplotlib.pyplot as plt

# Carrega a imagem
path = 'dataset/mosaic/mosaic-0002.jpg'
image = io.imread(path, as_gray=True)

# Aplica o filtro sobel para detectar as bordas
edge_sobel = filters.sobel(image)

# Cria uma figura com a imagem original e as bordas
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

# Plota a imagem original
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Imagem Original')

# Plota as bordas detectadas
axes[1].imshow(edge_sobel, cmap='gray')
axes[1].set_title('Bordas (filtro Sobel)')

# Remove os eixos
for ax in axes:
    ax.axis('off')

plt.show()
