from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2
import cvzone
from MainUi import Ui_MainWindow
from ultralytics import YOLO
from keras.api.models import load_model
import pickle
from check_and_save_img import CheckAndSaveImg

face_model = YOLO(r'model/yolov11n-face.pt')


class HandlePageRun(Ui_MainWindow):
    def __init__(self, MainWindow):
        super().__init__(MainWindow)
        # self.stackedWidget.setCurrentWidget(self.page_run)
        self.mode_cam_run = 'off'
        self.timerr = QTimer()
        # Start the camera feed
        self.timerr.start(30)  # Update every 30ms (approx 33 FPS)

        self.OJ = CheckAndSaveImg()
        self.image_input= np.array([])
        self.name = None
        # Thiết lập kết nối các nút khi nhấn vào
        self.push_run.clicked.connect(lambda : self.timerr.timeout.connect(self.start_predict))
        self.push_stop.clicked.connect(lambda : self.timerr.timeout.connect(self.update_frame_run))
        self.push_check_in.clicked.connect(self.check_in)
        self.push_check_out.clicked.connect(self.check_out)
        self.push_export.clicked.connect(self.export_data)
        self.lb = []
        self.model_cnn = None
        # self.cap = cv2.VideoCapture(0)
    def export_data(self):
        pass

    def check_in(self):
        if self.name != None and self.image_input.shape != (0, ):
            if not self.OJ.check_exists(label= self.name):
                self.OJ.save_image(image_array= self.image_input, label= self.name)
        
    def check_out(self):
        if self.name != None and self.OJ.check_exists(label= self.name):
            self.OJ.delete_image(label= self.name)

    # Phần này tương tự handel_page_run
    # Hàm dự đoán gương mặt
    def start_predict(self):
        if len(self.lb) == 0:
            # Lấy label của data là file chứa categories của OneHotEnCoder()
            with open('model/categories.pkl', 'rb') as f:
                cat = pickle.load(f)
            self.lb = np.array(cat[0]) # cat là mảng 2 chiều vd: [['label']], chuyển về numpy để thao tác tiện luôn
        
        if self.model_cnn == None:
            self.model_cnn = load_model(r'model/model_cnn.h5')
            
        if(self.mode_cam_run == 'update_frame_run'):
            self.timerr.timeout.disconnect(self.update_frame_run)
        self.mode_cam_run = 'start_predict'

        # Đọc một khung hình từ camera
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        frame_copy = frame.copy()

        face_result = face_model.predict(frame, conf = 0.6, verbose = False)

        if(ret == 0):
            print('Không thể đọc ảnh')
        else:
            boxes_xyxy = face_result[0].boxes.xyxy.tolist()
            # Nếu không thấy người nào
            if(len(boxes_xyxy) == 0):
                self.name = None
                self.image_input = np.array([])

                self.label_set_name.setText('Không tìm thấy gương mặt')
                self.labe_set_time.setText('Không tìm thấy gương mặt')
                self.cam_view_in.setText('Không tìm thấy gương mặt')
            # Trường hợp len > 0
            for box in boxes_xyxy:
                txt = None
                x, y, x2, y2 = map(int, box)
                w = x2 - x # width: chiều rộng
                h = y2 - y # height: chiều cao
                # Vẽ bounding box
                cvzone.cornerRect(frame, [x, y, w, h], rt = 0)
                # Cắt và chuẩn hóa dữ liệu
                img_cut = frame[y:y + h, x:x + w] 
                img_cut = cv2.resize(img_cut, (128, 128)) 
                img_cut = img_cut.astype('float32') / 255 
                # yêu cầu về đầu vào của input_shape của model (thêm batch size = 1)
                img_cut_expanded_0 = np.expand_dims(img_cut, axis = 0) 
                # Gán nhãn
                arr_predict = self.model_cnn.predict(img_cut_expanded_0, verbose = 0)
                # Lấy index có tỉ lệ cao nhất, axis = 1 vì đây là mảng 2 chiều [[data]] 
                predicted_label_index = np.argmax(arr_predict, axis = 1)
                # Lấy tỉ lệ phần trăm tương ứng, các mảng ở đây đều là 2D nên cần trỏ truy cập vào phần tử đầu 
                accuracy = arr_predict[0][predicted_label_index[0]] 
                
                # Lấy tên người đó
                name =  self.lb[predicted_label_index[0]]

                # Chấp nhận gương mặt
                if accuracy >= 0.8:
                    self.image_input = frame_copy
                    self.name = name    
                    
                    txt = name + ' ' + str(round(accuracy * 100, 2)) + ' %'
                
                    # Nếu đã check in
                    if self.OJ.check_exists(label= name):
                        # Lấy time và image lúc check in
                        time, img_in = self.OJ.get_data(label= name)
                        img_in = self.convert_qimg(image= img_in)
                        
                        self.cam_view_in.setPixmap(QPixmap.fromImage(img_in).scaled(self.cam_view_in.size()))
                        self.label_set_name.setText(name)
                        self.labe_set_time.setText(time)
                    else:
                        self.cam_view_in.setText('Chưa check in')
                        self.label_set_name.setText(name)
                        self.labe_set_time.setText('Chưa check in')
                
                # Trường hợp độ chính xác thấp
                else:
                    self.image_input = np.array([])
                    self.name = None
                    
                    txt = 'unknow'
                    
                    self.cam_view_in.setText('gương mặt không tồn tại')
                    self.label_set_name.setText('gương mặt không tồn tại')
                    self.labe_set_time.setText('gương mặt không tồn tại')
    
                cv2.putText(img = frame, org= (x + w // 2 - 90, y - 25), text = txt, 
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0,0), thickness= 3)
                print(arr_predict[0], self.lb[predicted_label_index[0]], str(round(accuracy * 100, 2)) + ' %') # Hiển thị data lên màn console

            # Chuyển đổi khung hình từ BGR (OpenCV) sang RGB (Qt)
            frame = self.convert_qimg(frame)
            # Hiển thị khung hình trên QLabel
            self.cam_view_main_2.setPixmap(QPixmap.fromImage(frame).scaled(self.cam_view_main_2.size()))
    
    # Hàm hiển thị bình thường
    def update_frame_run(self):
        if(self.mode_cam_run == 'start_predict'):
            self.timerr.timeout.disconnect(self.start_predict)
        self.mode_cam_run = 'update_frame_run'

        self.image_input = np.array([])
        self.name = None
        
        self.cam_view_in.setText('bấm nhận diện để chạy')
        self.label_set_name.setText('bấm nhận diện để chạy')
        self.labe_set_time.setText('bấm nhận diện để chạy')

        # Đọc một khung hình từ camera
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            # Chuyển đổi khung hình từ BGR (OpenCV) sang RGB (Qt)
            frame = self.convert_qimg(frame)
            # Hiển thị khung hình trên QLabel
            self.cam_view_main_2.setPixmap(QPixmap.fromImage(frame).scaled(self.cam_view_main_2.size()))
    
    def closeEvent(self, event):
        # Release the camera and stop the timer when the application closes
        if self.cap != None and self.cap.isOpened():
            self.cap.release()
        self.timerr.stop()
        event.accept()

    def convert_qimg(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        res = QImage(image.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        return res
    
# if __name__ == '__main__':
#     import sys

#     app = QApplication(sys.argv)
#     MainWindow = QMainWindow()
#     ui = HandlePageRun(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())