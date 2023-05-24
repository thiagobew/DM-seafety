import matplotlib.pyplot as plt
import os
def main():
    count = []
    folder = []

    total = 0
    
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    dirs = os.listdir(dataset_path)
    for dir in dirs:
        folder.append(dir)
        count.append(len(os.listdir(os.path.join(dataset_path, dir))))
        total += count[-1]

    # Creating horizontal bar plot
    plt.barh(folder, [100 * x / total for x in count], color='green')

    # Setting labels and title
    plt.xlabel('Número de imagens (%)')
    plt.ylabel('Categorias')
    plt.title(f'Imagens do dataset (total de {total}) por categoria')

    # Displaying the plot
    plt.show()


if __name__ == '__main__':
    main()
