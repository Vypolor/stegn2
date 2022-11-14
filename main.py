import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow, show, imread, imsave
import pywt
import cv2
import random

PATH_TO_SOURCE_IMAGE = 'bridge.tif'
ALPHA = 0.7
STEP = 0.02
MATH_EXPECTATION = 160
SIGMA = 80


def create_watermark(length, math_expectation, sigma):
    result = np.zeros(length)
    for i in range(0, length, 1):
        result[i] = random.gauss(math_expectation, sigma)
    return result


def insert_watermark(watermark_values: np.ndarray, place: np.ndarray, alpha: float) -> np.ndarray:
    return place + watermark_values * alpha


def detecting(source_watermark_vector, result_watermark_vector):
    numerator = np.sum(source_watermark_vector * result_watermark_vector)
    denominator = np.sqrt(np.sum(source_watermark_vector ** 2)) * np.sqrt(np.sum(result_watermark_vector ** 2))
    return numerator / denominator


def auto_alpha_selection(source_image):
    result = {}
    alpha = 0.1
    while alpha < 1:
        alpha += STEP
        # 5 insert watermark
        ll_place_with_watermark = insert_watermark(watermark, ll_place, alpha)
        dwt_image[0] = ll_place_with_watermark

        # 6 save image
        invert_dwt_image = pywt.waverec2(dwt_image, wavelet='haar').astype(np.uint8)
        imsave('result.tif', invert_dwt_image)

        # 7 (task 5 - read result image)
        watermarked_image = imread('result.tif')
        watermarked_dwt_image = pywt.wavedec2(watermarked_image, wavelet='haar', level=2)

        # 8 getting watermark
        ll_place_with_watermark = watermarked_dwt_image[0]
        extracted_watermark = (ll_place_with_watermark - ll_place) / alpha

        extracted_watermark_vector_length = extracted_watermark.shape[0] * extracted_watermark.shape[1]
        extracted_watermark_vector = extracted_watermark.reshape(extracted_watermark_vector_length)
        extracted_watermark_vector = extracted_watermark_vector[int(extracted_watermark_vector_length / 2):]
        # extracted_watermark_vector = extracted_watermark_vector[0:int(extracted_watermark_vector_length / 2)]

        p = detecting(watermark_vector, extracted_watermark_vector)
        psnr = cv2.PSNR(source_image, watermarked_image)
        if p > 0.9:
            result[psnr] = alpha
        print(f'p: {p}, alpha: {alpha}, PSNR: {psnr}')
    min_psnr = max(result.keys())
    max_alpha = result[min_psnr]
    print(f'Result: alpha: {max_alpha}, Max PSNR: {min_psnr}')
    return max_alpha


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
    watermark_vector = create_watermark(watermark_length, MATH_EXPECTATION, SIGMA)
    watermark = np.vstack((zeros, watermark_vector.reshape(int(ll_place.shape[0] / 2),
                                                           ll_place.shape[1])))
    print(watermark_vector)

    # 5 insert watermark
    ll_place_with_watermark = insert_watermark(watermark, ll_place, ALPHA)
    dwt_image[0] = ll_place_with_watermark

    # 6 save image
    invert_dwt_image = pywt.waverec2(dwt_image, wavelet='haar').astype(np.uint8)
    imsave('result.tif', invert_dwt_image)

    # 7 (task 5 - read result image)
    watermarked_image = imread('result.tif')
    watermarked_dwt_image = pywt.wavedec2(watermarked_image, wavelet='haar', level=2)

    # 8 (task 6) getting watermark
    ll_place_with_watermark = watermarked_dwt_image[0]
    extracted_watermark = (ll_place_with_watermark - ll_place) / ALPHA
    extracted_watermark[extracted_watermark < 0] = 0
    extracted_watermark[extracted_watermark > 255] = 255

    extracted_watermark_vector_length = extracted_watermark.shape[0] * extracted_watermark.shape[1]
    extracted_watermark_vector = extracted_watermark.reshape(extracted_watermark_vector_length)
    extracted_watermark_vector = extracted_watermark_vector[int(extracted_watermark_vector_length / 2):]
    # extracted_watermark_vector = extracted_watermark_vector[0:int(extracted_watermark_vector_length / 2)]

    first_p = detecting(watermark_vector, extracted_watermark_vector)
    print(first_p)
    # task 7
    print(auto_alpha_selection(source_image))
    p_arr = []
    p_arr.append(first_p)
    for i in range(100):
        random_noise_image_length = 512 * 512
        random_noise_image = create_watermark(random_noise_image_length, MATH_EXPECTATION, SIGMA)
        random_noise_image = random_noise_image.reshape(512, 512)
        dwt_image = pywt.wavedec2(random_noise_image, wavelet='haar', level=2)
        watermark_test = dwt_image[0]
        watermark_test = watermark_test.reshape(watermark_test.shape[0] * (watermark_test.shape[1]))
        watermark_length = watermark_test.shape[0] / 2
        watermark_test = watermark_test[int(watermark_length):]
        p = detecting(watermark_vector, watermark_test)

        p_arr.append(p)
    print(p_arr)
    distance = np.arange(0, 100)
    plt.plot(distance, p_arr)
    plt.show()
    # print(p)
