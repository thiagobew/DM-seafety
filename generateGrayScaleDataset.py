from PIL import Image
from multiprocessing import Process
import os

folders = ['down', 'carry', 'start_comm', 'up', 'five', 'four', 'num_delimiter',
           'one', 'two', 'end_comm', 'here', 'mosaic', 'backwards', 'three', 'boat', 'photo']

# Set the directory where the images are located
img_dir = os.path.join(os.getcwd(), "dataset",
                       "caddy-gestures-complete-v2-release-all-scenarios-fast.ai")

# Set the directory where you want to save the resized and grayscale images
out_dir = os.path.join(os.getcwd(), "dataset", "grayScale")

# Set the desired size of the resized images
size = (500, 500)


def resizeAndGrayscale(folder: str):
    curr_img_dir = os.path.join(img_dir, folder)

    curr_out_dir = os.path.join(out_dir, folder)

    if not os.path.exists(curr_out_dir):
        os.makedirs(curr_out_dir)

    for img in os.listdir(curr_img_dir):
        curr_img = Image.open(os.path.join(curr_img_dir, img))

        curr_img = curr_img.resize(size)

        # Convert the image to grayscale
        curr_img = curr_img.convert('L')

        curr_img.save(os.path.join(curr_img_dir, img))


def main(paralize=False):
    joined = []
    for folder in folders:
        if paralize:
            x = Process(target=resizeAndGrayscale, args=(folder,))
            joined.append(x)
            x.start()
        else:
            resizeAndGrayscale(folder)

    for x in joined:
        x.join()


if __name__ == '__main__':
    main(True)
