from functions import *

# r - отключение экранирования
photos = download_data("G:/Photos/")

I = 1  # номер изображения для вывода
photo = photos[I]
plt.imshow(photos[I])  # вывод изображения на экран

freq_hist(photo)

photo = cv2.resize(photo, (int(np.shape(photo)[1] / 4), int(np.shape(photo)[0] / 4)))
plt.imshow(photo)
plt.show()

corrected = nonlinear_log_corr(photo, c=1)
plt.imshow(corrected)
plt.show()

plt.plot(corrected[700])
plt.show()

a = copy.copy(corrected)

a[:, :, 0] = binar(a[:, :, 0], 90)
a[:, :, 1] = binar(a[:, :, 1], 60)
a[:, :, 2] = binar(a[:, :, 2], 60)
# img_gray = np.mean(a, axis=-1)
a[0:600, :] = 0
a = rgb2gray(a)
img_gray = binar(a, 150)
plt.imshow(img_gray, cmap='gray')
plt.show()

temp = expanding(30, img_gray)
temp = narrowing(21, temp)

plt.imshow(temp, cmap='gray')
plt.show()


sobel_matrix_1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
sobel_matrix_2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 4

sob_image = np.zeros(temp.shape)
for i in range(1, len(sob_image) - 1):
    for j in range(1, len(sob_image[0]) - 1):
        sob_image[i, j] = (np.abs(np.sum(temp[i - 1:i + 2, j - 1:j + 2] * sobel_matrix_1)) +
                           np.abs(np.sum(temp[i - 1:i + 2, j - 1:j + 2] * sobel_matrix_2))) / 2

plt.imshow(sob_image, cmap='gray')
plt.show()
