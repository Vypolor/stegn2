import pywt
import numpy as np
import random

MATH_EXPECTATION = 160
ALPHA = 0.7
SIGMA = 80


def create_watermark(length, math_expectation, sigma):
    result = np.zeros(length)
    for i in range(0, length, 1):
        result[i] = random.gauss(math_expectation, sigma)
    return result


def insert_watermark(watermark_values: np.ndarray, place: np.ndarray, alpha: float) -> np.ndarray:
    return place + watermark_values * alpha


def embedding_watermark(source_image):

    dwt_image = pywt.wavedec2(source_image, wavelet='haar', level=2)
    ll_place = dwt_image[0]
    watermark_length = int(ll_place.shape[0] * ll_place.shape[1] * 0.5)
    zeros = np.zeros((int(ll_place.shape[0] / 2), ll_place.shape[1]))

    # Create watermark
    watermark_vector = create_watermark(watermark_length, MATH_EXPECTATION, SIGMA)
    watermark = np.vstack((zeros, watermark_vector.reshape(int(ll_place.shape[0] / 2),
                                                           ll_place.shape[1])))
    # Insert watermark
    ll_place_with_watermark = insert_watermark(watermark, ll_place, ALPHA)
    dwt_image[0] = ll_place_with_watermark

    # Save image
    invert_dwt_image = pywt.waverec2(dwt_image, wavelet='haar').astype(np.uint8)
    return ll_place, watermark_vector, invert_dwt_image


def detecting_p(ll_place, source_watermark_vector, image_with_watermark):
    watermarked_dwt_image = pywt.wavedec2(image_with_watermark, wavelet='haar', level=2)
    ll_place_with_watermark = watermarked_dwt_image[0]
    extracted_watermark = (ll_place_with_watermark - ll_place) / ALPHA
    extracted_watermark_vector_length = extracted_watermark.shape[0] * extracted_watermark.shape[1]
    extracted_watermark_vector = extracted_watermark.reshape(extracted_watermark_vector_length)
    extracted_watermark_vector = extracted_watermark_vector[int(extracted_watermark_vector_length / 2):]
    numerator = np.sum(source_watermark_vector * extracted_watermark_vector)
    denominator = np.sqrt(np.sum(source_watermark_vector ** 2)) * np.sqrt(np.sum(extracted_watermark_vector ** 2))
    return numerator / denominator
