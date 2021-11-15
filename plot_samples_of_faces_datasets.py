"""Plot samples from the Deepfakes and Synthetic Faces datasets."""
import os
import random
import matplotlib.pyplot as plt

from common import FIGURES_DIR
from utils import load_dataset

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def normalize(image):
    """Normalize an image pixel values to [0, ..., 1]."""
    return (image - image.min()) / (image.max() - image.min())


def main():
    """Load the Deepfakes and Synthetic Faces datasets, sample real and fake
    images from them and plot them in a single image."""
    # create deepfakes dataset
    fakes_dataset_train = load_dataset('fakes_dataset', 'train')
    # sample indices of real and fake images
    image_idx_1 = random.choice(range(int(len(fakes_dataset_train) / 2)))
    image_idx_2 = random.choice(range(int(len(fakes_dataset_train) / 2),
                                         len(fakes_dataset_train)))
    images_samples = plt.figure()
    plt.subplot(2, 2, 1)
    im, label = fakes_dataset_train[image_idx_1]
    if label == 0:
        im_type = 'real'
    else:
        im_type = 'fake'
    plt.imshow(normalize(im).permute(1, 2, 0))
    plt.title('Deepfakes dataset ' + im_type +' image')
    plt.subplot(2, 2, 2)
    im, label = fakes_dataset_train[image_idx_2]
    if label == 0:
        im_type = 'real'
    else:
        im_type = 'fake'
    plt.imshow(normalize(im).permute(1, 2, 0))
    plt.title('Deepfakes dataset ' + im_type +' image')

    # create synthetic faces dataset
    synthetic_dataset_train = load_dataset('synthetic_dataset', 'train')

    image_idx_1 = random.choice(range(int(len(synthetic_dataset_train) / 2)))
    image_idx_2 = random.choice(range(int(len(synthetic_dataset_train) / 2),
                                         len(synthetic_dataset_train)))
    plt.subplot(2, 2, 3)
    im, label = synthetic_dataset_train[image_idx_1]
    if label == 0:
        im_type = 'real'
    else:
        im_type = 'fake'
    plt.imshow(normalize(im).permute(1, 2, 0))
    plt.title('Deepfakes dataset ' + im_type +' image')
    plt.subplot(2, 2, 4)
    im, label = synthetic_dataset_train[image_idx_2]
    if label == 0:
        im_type = 'real'
    else:
        im_type = 'fake'
    plt.imshow(normalize(im).permute(1, 2, 0))
    plt.title('Deepfakes dataset ' + im_type +' image')
    images_samples.set_size_inches((8, 8))
    images_samples.savefig(os.path.join(FIGURES_DIR, 'datasets_samples.png'))


if __name__ == "__main__":
    main()
