import os
import numpy as np
import cv2
from datetime import datetime
import shutil

# Vấn đề: now.strftime("%d/%m/%Y %H:%M:%S") tạo tên tệp có ký tự /, : không hợp lệ trên Windows.

path_root = 'image_data'

# img = cv2.imread(r"test.png")

class CheckAndSaveImg():
    def check_exists(self, label):
        return os.path.exists(path_root + '/' + label)
    def save_image(self, label, image_array: np.array):
        if self.check_exists(label= label) == False:
            now = datetime.now()
            name = now.strftime("%H-%M-%S %d-%m-%Y")
            path_dir = os.path.join(path_root, label)
            os.makedirs(path_dir, exist_ok= True)
            cv2.imwrite(os.path.join(path_dir, name + '.jpg'), image_array)
        else:
            print('Đã lưu trước đó, yêu cầu xóa ...')
    def delete_image(self, label):
        path_dir = path_root + '/' + label
        try:
            shutil.rmtree(path_dir)  # Xóa cả thư mục và file bên trong
        except FileNotFoundError:
            print('Không tồn tại label')
    def get_data(self, label):
        if self.check_exists(label= label):
            path = path_root + '/' + label
            img_name = str(os.listdir(path= path)[0])
            img_array = cv2.imread(path + '/' + img_name)

            date_time_img = img_name[:-4].replace('-', ':', 2)

            return date_time_img, img_array
        else:
            # print('Xe chưa vào bãi')
            return 'Xe chưa vào bãi'

# OJ= CheckAndSaveImg()
# # OJ.save_image(label= '36X1-4359', image_array= img)
# a = OJ.get_data(label= '36X1-4359')

# print(a[1])