from functions import *

photos = download_data("G:/Photos/")
image_num = 2  # номер изображения для вывода
photo = photos[image_num]
plt.imshow(photos[image_num])  # вывод изображения на экран
plt.show()
photo = cv2.resize(photo, (int(np.shape(photo)[1] / 4), int(np.shape(photo)[0] / 4)))

if image_num == 0:
    corrected = nonlinear_log_corr(photo, c=0.6)
    plt.imshow(corrected)
    plt.show()

if image_num == 2:
    corrected = nonlinear_log_corr(photo, c=0.7)
    plt.imshow(corrected)
    plt.show()
