import PIL.Image
import cv2
import numpy as np
import numpy.random
import pandas as pd
from lab2.task import detecting_p


def smooth(image, M=3):
    return cv2.blur(image, (M, M))


def smooth_bulk(image, p_min, p_max, p_delta):
    smooth_images = []
    items_count = int(np.round((p_max - p_min) / p_delta)) + 1

    p_current = p_min
    for i in range(0, items_count):
        smooth_images.append(smooth(image, p_current))
        p_current += p_delta

    return np.array(smooth_images)


def scalerest_bulk(image, p_min, p_max, p_delta):
    scalerest_images = []
    items_count = int(np.round((p_max - p_min) / p_delta)) + 1

    p_current = p_min
    temp_image = image.copy()
    original_size = image.shape[0]
    for i in range(0, items_count):
        new_shape = int(original_size * p_current)
        img = PIL.Image.fromarray(temp_image)
        resized_image = img.resize((new_shape, new_shape), PIL.Image.ANTIALIAS)
        rest_image = resized_image.resize((original_size, original_size), PIL.Image.ANTIALIAS)
        scalerest_images.append(np.asarray(rest_image))
        p_current += p_delta
        temp_image = np.asarray(rest_image)

    return np.array(scalerest_images)


def convert_value(q, p, value=255):
    return value if q > p else 0


def salt_pepper_bulk(image, p_min, p_max, p_delta):
    salt_pepper_images = []
    items_count = int(np.round((p_max - p_min) / p_delta)) + 1

    p_current = p_min
    size = image.shape[0]
    for i in range(0, items_count):
        q = p_current / 2
        zeros_arr = np.zeros((size, size))
        zeros_arr[numpy.random.rand(*zeros_arr.shape) < q] = 255
        salt_pepper = zeros_arr
        noise_image = image + salt_pepper
        noise_image[noise_image > 255] = 255
        salt_pepper_images.append(noise_image)
        p_current += p_delta

    return np.array(salt_pepper_images)


def cyclic_shift(image, p_min, p_max, p_delta):
    cyclic_shift_images = []
    items_count = int(np.round((p_max - p_min) / p_delta)) + 1
    p_current = p_min
    N1 = image.shape[0]
    N2 = image.shape[1]
    for i in range(0, items_count):
        rN1 = N1 * p_current
        rN2 = N2 * p_current
        temp_image = image.copy()
        for j in range(0, N1):
            for k in range(0, N2):
                temp_image[j][k] = temp_image[int((j + rN1) % N1)][int((k + rN2) % N2)]

        cyclic_shift_images.append(temp_image)
        p_current += p_delta

    return np.array(cyclic_shift_images)


def scalerest(image, p):
    original_size = image.shape[0]
    new_shape = int(original_size * p)
    img = PIL.Image.fromarray(image)
    resized_image = img.resize((new_shape, new_shape), PIL.Image.ANTIALIAS)
    rest_image = resized_image.resize((original_size, original_size), PIL.Image.ANTIALIAS)

    return np.array(rest_image)


def salt_pepper(image, p):
    size = image.shape[0]
    q = p / 2
    zeros_arr = np.zeros((size, size))
    zeros_arr[numpy.random.rand(*zeros_arr.shape) < q] = 255
    salt_pepper = zeros_arr
    noise_image = image + salt_pepper
    noise_image[noise_image > 255] = 255
    return noise_image


def scalerest_salt_pepper_bulk(watermark_vector, ll_place, image, p1_min, p1_max, p1_delta, p2_min, p2_max, p2_delta):
    scalerest_saltpepper = []
    items_count_scalerest = int(np.round((p1_max - p1_min) / p1_delta)) + 1
    items_count_salt_pepper = int(np.round((p2_max - p2_min) / p2_delta)) + 1

    p1_array = [0.55, 0.7, 0.85, 1, 1.15, 1.3, 1.45]
    p2_array = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    p1_current = p1_min
    p2_current = p2_min
    rho = np.zeros((items_count_scalerest, items_count_salt_pepper))
    for i in range(0, items_count_scalerest):
        for j in range(0, items_count_salt_pepper):
            scalerest_image = scalerest(image, p1_current)
            scalerest_and_salt_pepper = salt_pepper(scalerest_image, p2_current)
            scalerest_saltpepper.append(scalerest_and_salt_pepper)
            rho[i][j] = detecting_p(ll_place, watermark_vector, scalerest_and_salt_pepper)
            p1_current += p1_delta
        p2_current += p2_delta
    df = pd.DataFrame(rho, columns=p2_array, index=p1_array)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    return np.array(scalerest_saltpepper)
