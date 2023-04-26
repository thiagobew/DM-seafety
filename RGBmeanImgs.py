import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

path = os.path.join(os.getcwd(), "meanImgs")


def main():
    imageNames = os.listdir(path)

    fig, axs = plt.subplots(len(imageNames) // 4, 4,
                            figsize=(3, len(imageNames) // 2), sharex=True, sharey=True)
    fig.suptitle('RGB Histograms of Mean Images', fontsize=16)
    plt.tight_layout()

    namesAxes = zip(imageNames, axs.flatten())

    for tup in namesAxes:
        imgName = tup[0]
        ax = tup[1]

        img = Image.open(os.path.join(path, imgName))

        img_array = np.array(img)

        for j, color in enumerate(['red', 'green', 'blue']):
            ax.hist(img_array[:, :, j].ravel(), bins=256,
                    range=(0, 256), color=color, alpha=0.7, label=color.capitalize(), histtype='step')
        ax.set_xlim([0, 256])
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')
        ax.set_title(imgName.removesuffix('_average.png'))
        ax.legend()

    plt.subplots_adjust(left=0.1, right=0.9,
                        bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

    plt.show()


if __name__ == '__main__':
    main()
