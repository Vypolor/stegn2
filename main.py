import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow, show, imread, imsave
from lab2.task import embedding_watermark, detecting_p
from utils.distortion import smooth_bulk, scalerest_bulk, salt_pepper_bulk, cyclic_shift

PATH_TO_SOURCE_IMAGE = 'bridge.tif'

if __name__ == '__main__':
    # read source image
    source_image = imread(PATH_TO_SOURCE_IMAGE)
    ll_place, watermark_vector, invert_dwt_image = embedding_watermark(source_image)
    p = detecting_p(ll_place, watermark_vector, invert_dwt_image)
    #1
    cyclic_shift_images = cyclic_shift(invert_dwt_image, 0.1, 0.9, 0.1)
    cyclic_shift_rhos = []
    for i in range(0, cyclic_shift_images.shape[0]):
        cyclic_shift_rhos.append(detecting_p(ll_place, watermark_vector, cyclic_shift_images[i]))

    plt.title('Rhos (cyclic_shift)')
    x = np.arange(0.1, 1, 0.1)
    plt.plot(x, cyclic_shift_rhos)
    plt.show()


    #2
    scalerest_images = scalerest_bulk(invert_dwt_image, 0.55, 1.45, 0.15)
    scalerest_rhos = []
    for i in range(0, scalerest_images.shape[0]):
        scalerest_rhos.append(detecting_p(ll_place, watermark_vector, scalerest_images[i]))

    plt.title('Rhos (scale_rest)')
    x = np.arange(0.55, 1.50, 0.15)
    plt.plot(x, scalerest_rhos)
    plt.show()

    #3
    smooth_images = smooth_bulk(invert_dwt_image, 3, 15, 2)

    smooth_rhos = []
    for i in range(0, smooth_images.shape[0]):
        smooth_rhos.append(detecting_p(ll_place, watermark_vector, smooth_images[i]))

    plt.title('Rhos (smooth)')
    x = np.arange(3, 17, 2)
    plt.plot(x, smooth_rhos)
    plt.show()

    #4
    salt_pepper_images = salt_pepper_bulk(invert_dwt_image, 0.05, 0.5, 0.05)
    salt_pepper_rhos = []
    for i in range(0, salt_pepper_images.shape[0]):
        salt_pepper_rhos.append(detecting_p(ll_place, watermark_vector, salt_pepper_images[i]))

    plt.title('Rhos (salt_pepper)')
    x = np.arange(0.05, 0.55, 0.05)
    plt.plot(x, salt_pepper_rhos)
    plt.show()
