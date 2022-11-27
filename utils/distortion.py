import PIL.Image
import cv2
import numpy as np
import numpy.random


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