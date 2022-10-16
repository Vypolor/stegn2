import numpy as np
from skimage.io import imshow, show, imread, imsave
import pywt

PATH_TO_SOURCE_IMAGE = 'bridge.tif'
ALPHA = 0.7
STEP = 0.05


def create_watermark(size: int) -> np.ndarray:
    vector = np.linspace(1, 50, size)
    mean = np.mean(vector)
    sd = np.std(vector)
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((vector - mean) / sd) ** 2)
    return prob_density


def insert_watermark(watermark_values: np.ndarray, place: np.ndarray, alpha: float) -> np.ndarray:
    return place + watermark_values * alpha


def detecting(source_watermark_vector, result_watermark_vector):
    numerator = np.sum(source_watermark_vector * result_watermark_vector)
    denominator = np.sqrt(np.sum(source_watermark_vector ** 2)) * np.sqrt(np.sum(result_watermark_vector ** 2))
    return numerator / denominator


def auto_alpha_selection():
    result_p = 0.0
    alpha = 0.0
    alpha_value = alpha
    while alpha < 100:
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
        if p > result_p:
            result_p = p
            alpha_value = alpha
    return result_p, alpha_value


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

    watermark_vector = create_watermark(watermark_length)
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

    # 8 getting watermark
    ll_place_with_watermark = watermarked_dwt_image[0]
    extracted_watermark = (ll_place_with_watermark - ll_place) / ALPHA

    extracted_watermark_vector_length = extracted_watermark.shape[0] * extracted_watermark.shape[1]
    extracted_watermark_vector = extracted_watermark.reshape(extracted_watermark_vector_length)
    extracted_watermark_vector = extracted_watermark_vector[int(extracted_watermark_vector_length / 2):]
    # extracted_watermark_vector = extracted_watermark_vector[0:int(extracted_watermark_vector_length / 2)]

    p = detecting(watermark_vector, extracted_watermark_vector)

    print(auto_alpha_selection())

    print(p)
