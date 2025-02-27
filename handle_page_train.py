from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
from MainUi import Ui_MainWindow
import pickle
import numpy as np
import cv2
import os
import io
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Flatten # type: ignore
from tensorflow.keras.models import Sequential# type: ignore
from tensorflow.keras.optimizers import Adam# type: ignore
from tensorflow.keras import Input# type: ignore
from tensorflow.keras.callbacks import Callback # type: ignore

# Callback để ghi log từng epoch
class TrainLogger(Callback):
    def __init__(self, log_signal, log_percent):
        super().__init__()
        self.log_signal = log_signal
        self.log_percent = log_percent
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        message = f"Epoch {epoch+1}: " + " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.log_signal.emit(message)
        self.log_percent.emit(int(65 + 35 * (epoch + 1) / 20))
# Cần phải tạo 1 luồng phụ để thực hiện training để tránh bị delay giao diện UI
class HandelPageTrain(Ui_MainWindow, QThread):
    update_log_signal = pyqtSignal(str)  # Khai báo signal để đưa thông tin về luồng chính để hiển thị lên giao diện UI
    update_log_percent = pyqtSignal(int)

    def __init__(self, MainWindow):
        QThread.__init__(self)
        Ui_MainWindow.__init__(self, MainWindow)
        self.data_processed = None
        self.label_processed = None
        self.stackedWidget.setCurrentWidget(self.page_train)
        self.push_training.clicked.connect(self.start_training)
        self.note_out = ''
        self.update_log_signal.connect(self.update_log) # Khi có str mới cho vào thì nó sẽ kết nối với hàm update_log
        self.update_log_percent.connect(self.update_percent)
        self.cat = None # categrories của khi encoder
    def start_training(self):
        self.note_out = ''
        self.log_res.setText('- Bắt đầu xử lý...')
        self.start() # Khi luồng phụ chạy lệnh này thì nó sẽ chạy qua hàm run()
    
    def run(self):
        self.process_img_to_numpy()
        self.train()
    
    # Hàm xử lý ảnh thô
    def process_img_to_numpy(self):
        list_label = os.listdir('data_image_raw')
        data_img = []
        label = []

        for item in list_label:
            path_label = os.path.join('data_image_raw', item)
            list_image = os.listdir(path_label)
            
            for image in list_image:
                path_image = os.path.join(path_label, image)
                matrix = cv2.imread(path_image)
                data_img.append(matrix)
                label.append(item)
            
            self.update_log_signal.emit(f'- Đã xử lý xong ảnh của: {item} với số ảnh: {len(list_image)}')
        self.update_log_percent.emit(20)
        data_img = np.array(data_img) 
        cat_label = set(label.copy())
        label = np.array(label).reshape(-1, 1)

        self.update_log_signal.emit(f'- Shape của data: {data_img.shape}\nVới các label {cat_label}')
        self.update_log_percent.emit(40)
        encoder = OneHotEncoder(sparse_output=False)
        self.label_processed = encoder.fit_transform(label)
        self.data_processed = data_img.astype('float32') / 255
        
        self.cat = encoder.categories_
        self.update_log_percent.emit(50)

    def train(self):
        lb = np.array(self.cat[0])
        num_class = lb.size
        self.update_log_percent.emit(55)
        xtrain, xtest, ytrain, ytest = train_test_split(self.data_processed, self.label_processed, test_size=0.2)
        self.update_log_percent.emit(60)

        model_cnn = Sequential([
            Input(shape=(128, 128, 3)),
            
            Conv2D(16, (3, 3), activation='relu'),
            MaxPool2D((2, 2), padding='same'),

            Flatten(),
            
            Dense(32, activation='relu'),
            
            Dropout(0.5),
            
            Dense(num_class, activation='softmax')
        ])
        
        model_cnn.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc', 'precision'])
        
        # Ghi summary model
        model_summary = self.get_model_summary(model_cnn)
        self.update_log_signal.emit(model_summary)
        self.update_log_percent.emit(65)
        self.update_log_signal.emit('- Bắt đầu train ...')

        # Sử dụng callback để log từng epoch
        train_logger = TrainLogger(self.update_log_signal, self.update_log_percent)

        model_cnn.fit(xtrain, ytrain, epochs= 20, validation_data=(xtest, ytest), batch_size=32, callbacks=[train_logger])
        
        self.update_log_signal.emit('- Đã train xong')

        with open('model/categories.pkl', 'wb') as f:
            pickle.dump(self.cat, f)        
        self.update_log_signal.emit('- Đã lưu categories encoder labels')
        model_cnn.save('model/model_cnn.h5', include_optimizer= True)
        self.update_log_signal.emit('- Hoàn tất, có thể trải nghiệm ngay!')
        
    def get_model_summary(self, model):
        """Ghi model summary vào string và trả về."""
        stream = io.StringIO()
        sys.stdout = stream
        model.summary()
        sys.stdout = sys.__stdout__
        return stream.getvalue()

    # Hàm giúp hiển thị trạng thái/câu dẫn lên luồng chính cho UI 
    def update_log(self, message):
        self.note_out += f'\n\n{message}'
        self.log_res.setText(self.note_out)
        self.log_res.adjustSize()
        self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())

    def update_percent(self, value):
        self.progressBar.setValue(value)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main = QMainWindow()
    
    ui = HandelPageTrain(MainWindow=main)
    main.show()
    sys.exit(app.exec_())
