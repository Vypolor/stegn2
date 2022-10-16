import numpy as np
from skimage.io import imshow, show, imread, imsave
import pywt

PATH_TO_SOURCE_IMAGE = 'bridge.tif'
ALPHA = 0.7


def create_watermark(size: int) -> np.ndarray:
    vector = np.linspace(1, 50, size)
    mean = np.mean(vector)
    sd = np.std(vector)
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((vector - mean) / sd) ** 2)
    return prob_density


def insert_watermark(watermark_values: np.ndarray, place: np.ndarray) -> np.ndarray:
    return place + watermark_values * ALPHA


if __name__ == '__main__':
    # 1 read source image
    source_image = imread(PATH_TO_SOURCE_IMAGE)

    # 2 dwt source image
    dwt_image = pywt.wavedec2(source_image, wavelet='haar', level=2)

    # 3 get LL place from dwt image with level 2
    ll_place = dwt_image[0]

    # 4 get watermark
    watermark_length = int(ll_place.shape[0] * ll_place.shape[1] * 0.5)
    zeros = np.zeros((int(ll_place.shape[0] / 2), ll_place.shape[1]))
    # watermark = np.vstack((create_watermark(watermark_length).reshape(int(ll_place.shape[0] / 2),
    #                                                                       ll_place.shape[1]), zeros))

    watermark = np.vstack((zeros, create_watermark(watermark_length).reshape(int(ll_place.shape[0] / 2),
                                                                             ll_place.shape[1])))

    # 5 insert watermark
    ll_place_with_watermark = insert_watermark(watermark, ll_place)
    dwt_image[0] = ll_place_with_watermark

    # 6 save image
    invert_dwt_image = pywt.waverec2(dwt_image, wavelet='haar').astype(np.uint8)
    imsave('result.tif', invert_dwt_image)
