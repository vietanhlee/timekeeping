from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
from MainUi import Ui_MainWindow
from ultralytics import YOLO
from ImageDetect import ImageDetect
from PyQt5.QtCore import QTimer

# load model phát hiện gương mặt
model = YOLO(r'model/yolov11n-face.pt')

class HandlePageGetData(Ui_MainWindow):
    '''class cho lớp lấy dữ liệu'''
    def __init__(self, MainWindow):
        super().__init__(MainWindow)
        # self.stackedWidget.setCurrentWidget(self.page_get_data)
        self.count = 0 # Biến đếm số ảnh đã lưu lại
        self.mode_cam = 'off' # 0: tắt camera, 1: bật camera hiện ảnh bình thường, 2: detect ảnh ...

        self.timer = QTimer()
        self.timer.start(30)  # Update every 30ms (approx 33 FPS)

        # Khi bấm các nút này chương trình sẽ chạy hàm tương ứng
        self.push_get_data_face.clicked.connect(lambda : self.timer.timeout.connect(self.start_detect))
        self.push_stop_get_data.clicked.connect(lambda : self.timer.timeout.connect(self.update_frame))
        self.number_face = None # Biến số mặt người cần lấy tối đa
        # self.cap = cv2.VideoCapture(0)

    def start_detect(self):
        # Đặt lại biến từ dữ liệu người dùng nhập 
        self.number_face = int(self.get_number_face.toPlainText())
        
        # Chỉnh sửa biến flag cho chế độ hiển thị hiện tại
        if(self.mode_cam == 'update_frame'):
            self.timer.timeout.disconnect(self.update_frame)
        
        # Nếu số người đã đủ
        if(self.count >= self.number_face):
            self.timer.timeout.connect(self.update_frame)
            self.count = 0
        # Đặt lại chế độ của camera
        self.mode_cam = 'start_detect'
        
        # Đọc ảnh
        check_done, frame = self.cap.read()
        if(check_done == False):
            print('Đọc ảnh không thành công')
        # Đảo ảnh
        frame = cv2.flip(frame, 1)
        
        # Lấy tên người được đọc ảnh
        name_lalble = self.get_name_face.toPlainText()
        ID = ImageDetect(image_input= frame, index= self.count, name_lable= name_lalble)
        
        # Hiển thị lên màn chính ảnh đầu ra
        img_out = ID.image_output
        img_out = self.convert_qimg(img_out)
        self.cam_view_main.setPixmap(QPixmap.fromImage(img_out).scaled(self.cam_view_main.size()))

        # Hiển thị gương mặt được cắt lên khung hình nhỏ bên cạnh
        img_face = ID.img_face
        img_face = self.convert_qimg(img_face)

        self.view_face.setPixmap(QPixmap.fromImage(img_face).scaled(self.view_face.size()))
        
        # Tăng biến đếm lên 1
        self.count += 1
    
    # Hàm hiển thị camera bình thường
    def update_frame(self):
        if(self.mode_cam == 'start_detect'):
            self.timer.timeout.disconnect(self.start_detect)
        self.mode_cam = 'update_frame'

        # Đọc một khung hình từ camera
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            # Chuyển đổi khung hình từ BGR (OpenCV) sang RGB (Qt)
            frame = self.convert_qimg(frame)
            # Hiển thị khung hình trên QLabel
            self.cam_view_main.setPixmap(QPixmap.fromImage(frame).scaled(self.cam_view_main.size()))

    def closeEvent(self, event):
        # Release the camera and stop the timer when the application closes
        if self.cap != None and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        event.accept()

    # Hàm chuyển đổi từ image dạng arr sang qimage để phù hợp hiển thị
    def convert_qimg(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        res = QImage(image.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        return res
    
if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = HandlePageGetData(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())