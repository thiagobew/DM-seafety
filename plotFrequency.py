import matplotlib.pyplot as plt


def main():
    count = []
    folder = []

    total = 0
    
    with open("fileCountEachFolder.txt", "r") as f:
        data = f.readlines()
        for line in data:
            line = line.split()
            folder.append(line[0])
            count.append(int(line[1]))
            total += int(line[1])

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
