from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import cv2


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def binar(img: np.ndarray, down_threshold: int) -> np.ndarray:
    img_bin = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_bin[i][j] < down_threshold:
                img_bin[i][j] = 0
            else:
                img_bin[i][j] = 255
    return img_bin


def freq_hist(image: np.ndarray) -> None:
    for i in range(np.shape(image)[2]):
        flattened = image[:, :, i].flatten()
        plt.figure(figsize=(20, 8))
        plt.title(label='Гистограмма освещенности')
        plt.hist(flattened, bins=255)
        plt.show()


def download_data(path: str):
    data = []
    for path_image in sorted(os.listdir(path=path)):
        # Opens image
        image = Image.open(path + path_image)
        image = ImageOps.exif_transpose(image)
        # Appends pixels
        data.append(np.array(image))
    return data


def linear_correction(image: np.ndarray) -> np.ndarray:
    temp_image = np.zeros_like(image)
    for i in range(np.shape(image)[2]):
        flattened = image[:, :, i].flatten()
        max_arg = np.argmax(flattened)
        max_num = flattened[max_arg]
        min_arg = np.argmin(flattened)
        min_num = flattened[min_arg]
        flattened = (flattened - min_num) * (255 / (max_num - min_num))
        flattened[max_arg] = 255
        flattened[min_arg] = 0
        temp_image[:, :, i] = np.reshape(flattened, np.shape(image)[:2])
    return temp_image


def nonlinear_gamma_corr(image: np.ndarray, c=1, gamma=1) -> np.ndarray:
    c = c
    gamma = gamma
    temp = np.array(copy.copy(image), dtype=float)
    for i in range(np.shape(image)[2]):
        temp[:, :, i] = temp[:, :, i] / 255
    return np.array(c * np.power(temp, gamma) * 255, dtype=int)


def nonlinear_log_corr(image: np.ndarray, c=1) -> np.ndarray:
    temp = np.array(copy.copy(image), dtype=float)
    for i in range(np.shape(image)[2]):
        temp[:, :, i] = temp[:, :, i] / 255
    return np.array(c * np.log10(1 + temp) * 255, dtype=int)


def median_filtration(image: np.ndarray, filter_size: int) -> np.ndarray:
    image_with_new_borders = make_borders(image, filter_size)
    corrected = copy.copy(image_with_new_borders)
    temp = np.zeros((filter_size, filter_size))
    for i in range(np.shape(image_with_new_borders)[2]):
        # Разворачиваем весь массив канала
        flattened = image_with_new_borders[:, :, i].flatten()
        # Считаем количество вертикальных перемещений
        vertical_strides_number = np.shape(image_with_new_borders[:, :, i])[0] - (filter_size - 1)
        for v in range(vertical_strides_number):
            # Считаем количество горизонтальных перемещений
            horizontal_strides_number = np.shape(image_with_new_borders[:, :, i])[1] - (filter_size - 1)
            for h in range(horizontal_strides_number):
                # Заполняем временный массив, размерами фильтра, значениями из основного массива (изображения)
                for m in range(filter_size):
                    temp[m][0:filter_size] = flattened[np.shape(image_with_new_borders[:, :, i])[1] * (m + v) + h:
                                                       filter_size + h + np.shape(image_with_new_borders[:, :, i])[1]
                                                       * (m + v)]
                corrected[v - 1 + filter_size - (filter_size // 2)][h - 1 + filter_size - (filter_size // 2)][i] = \
                    np.sort(temp.flatten())[int(len(temp.flatten()) / 2)]
    return delete_borders(corrected, filter_size)


def mean_filtration(image: np.ndarray, filter_size: int) -> np.ndarray:
    image_with_new_borders = make_borders(image, filter_size)
    corrected = copy.copy(image_with_new_borders)
    temp = np.zeros((filter_size, filter_size))
    for i in range(np.shape(image_with_new_borders)[2]):
        # Разворачиваем весь массив канала
        flattened = image_with_new_borders[:, :, i].flatten()
        # Считаем количество вертикальных перемещений
        vertical_strides_number = np.shape(image_with_new_borders[:, :, i])[0] - (
                filter_size - 1)
        for v in range(vertical_strides_number):
            # Считаем количество горизонтальных перемещений
            horizontal_strides_number = np.shape(image_with_new_borders[:, :, i])[1] - (
                    filter_size - 1)
            for h in range(horizontal_strides_number):
                # Заполняем временный массив, размерами фильтра, значениями из основного массива (изображения)
                for m in range(filter_size):
                    temp[m][0:filter_size] = flattened[np.shape(image_with_new_borders[:, :, i])[1] * (
                            m + v) + h: filter_size + h + np.shape(image_with_new_borders[:, :, i])[1] * (m + v)]
                # Перенос в новое изображение
                corrected[v - 1 + filter_size - (filter_size // 2)][h - 1 + filter_size - (filter_size // 2)][i] = int(
                    np.mean(temp.flatten()))
    return delete_borders(corrected, filter_size)


def gaussian_filtration(image: np.ndarray, filter_size: int, sigma: float) -> np.ndarray:
    image_with_new_borders = make_borders(image, filter_size)
    corrected = copy.copy(image_with_new_borders)
    temp = np.zeros((filter_size, filter_size))
    filter_matrix = form_gaussian_filter(filter_size, sigma)
    for i in range(np.shape(image_with_new_borders)[2]):
        # Разворачиваем весь массив канала
        flattened = image_with_new_borders[:, :, i].flatten()
        # Считаем количество вертикальных перемещений
        vertical_strides_number = np.shape(image_with_new_borders[:, :, i])[0] - (
                filter_size - 1)
        for v in range(vertical_strides_number):
            horizontal_strides_number = np.shape(image_with_new_borders[:, :, i])[1] - (
                    filter_size - 1)  # Считаем количество горизонтальных перемещений
            for h in range(horizontal_strides_number):
                # Заполняем временный массив, размерами фильтра, значениями из основного массива (изображения)
                for m in range(filter_size):
                    temp[m][0:filter_size] = flattened[np.shape(image_with_new_borders[:, :, i])[1] * (
                            m + v) + h: filter_size + h + np.shape(image_with_new_borders[:, :, i])[1] * (m + v)]
                # Перенос в новое изображение
                corrected[v:filter_size + v, h:filter_size + h, i] = \
                    corrected[v:filter_size + v, h:filter_size + h, i] + temp * filter_matrix
    return delete_borders(corrected, filter_size)


def form_gaussian_filter(filter_size: int, sigma: float):
    filter_matrix = np.zeros(shape=(filter_size, filter_size))
    current_coordinate = [-int(filter_size / 2), int(filter_size / 2)]
    for row_n in range(len(filter_matrix)):
        for col_n in range(len(filter_matrix)):
            filter_matrix[row_n][col_n] = 1 / (2 * np.pi * np.square(sigma)) * np.exp(
                -(np.square(current_coordinate[0]) + np.square(current_coordinate[1])) / (2 * np.square(sigma)))
            current_coordinate[0] = current_coordinate[0] + 1
        current_coordinate[0] = -int(filter_size / 2)
        current_coordinate[1] = current_coordinate[1] - 1
    return filter_matrix / np.sum(filter_matrix)


def make_borders(image: np.ndarray, filter_size: int) -> np.ndarray:
    corrected = np.zeros(
        (np.shape(image)[0] + 2 * int(filter_size / 2), np.shape(image)[1] + 2 * int(filter_size / 2), 3), dtype=int)
    for i in range(np.shape(image)[2]):
        temp = copy.copy(image[:, :, i])
        temp = np.insert(temp, 0, temp[:int(filter_size / 2)][::-1], axis=0)
        temp = np.insert(temp, np.shape(temp)[0], temp[::-1][:int(filter_size / 2)], axis=0)
        temp = np.insert(temp, [0], temp[:, :int(filter_size / 2)][:, ::-1], axis=1)
        temp = np.insert(temp, [np.shape(temp)[1]], temp[:, ::-1][:, :int(filter_size / 2)], axis=1)
        corrected[:, :, i] = temp
    return corrected


def delete_borders(image: np.ndarray, filter_size: int) -> np.ndarray:
    return image[
           int(filter_size / 2): np.shape(image[:, :, 0])[0] - int(filter_size / 2),
           int(filter_size / 2):np.shape(image[:, :, 0])[1] - int(filter_size / 2), :
           ]


def expanding(k_size: int, image: np.ndarray) -> np.ndarray:
    temp = copy.copy(image)
    height, width = np.shape(temp)
    max_num = image.max()
    for i in range(height):
        for j in range(width):
            window = image[
                     window_calc(i, k_size, height): window_calc(i, k_size, height, left=False),
                     window_calc(j, k_size, width): window_calc(j, k_size, width, left=False)
                     ]
            if max_num in window:
                temp[i][j] = max_num
    return temp


def narrowing(k_size: int, image: np.ndarray) -> np.ndarray:
    temp = copy.copy(image)
    height, width = np.shape(image)
    for i in range(height):
        for j in range(width):
            window = image[
                     window_calc(i, k_size, height): window_calc(i, k_size, height, left=False),
                     window_calc(j, k_size, width): window_calc(j, k_size, width, left=False)
                     ]
            if (window == 0).any():
                temp[i][j] = 0
    return temp


def window_calc(index: int, k_size: int, size: int, left=True):
    if left:
        temp = index - int(k_size / 2)
        if temp < 0:
            return 0
        else:
            return temp
    else:
        temp = index + int(k_size / 2) + 1
        if temp > size:
            return size
        else:
            return temp
