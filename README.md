# Phần mềm nhận diện gương mặt phục vụ cho điểm danh hoặc chấm công
## Chi tiết về code chạy console và lý thuyết các model AI
- Phần project này chủ yếu là phát triển giao diện cho code nhận diện gương mặt chạy console. Hay nói cách khác thì chỉ là phần front end dựa trên framework Pyqt5 và thêm một số tính năng khác

- Chi tiết bản code chạy trên console ở [đây](https://github.com/vietanhlee/face-recognition)

## Cơ sở lý thuyết  
### Phát hiện gương mặt
- Các gương mặt trên được cắt trực tiếp nhờ model phát hiện gương mặt người được train trên mạng YOLO dựa theo dataset về gương mặt con người đã được gán nhãn theo format của YOLO v11

  > YOLO là một mô hình mạng CNN cho việc phát hiện, nhận dạng, phân loại đối tượng. Yolo được tạo ra từ việc kết hợp giữa các convolutional layers và connected layers.Trong đóp các convolutional layers sẽ trích xuất ra các feature của ảnh, còn full-connected layers sẽ dự đoán ra xác suất đó và tọa độ của đối tượng. Thông tin về YOLO có thể tìm thấy ở [đây](https://docs.ultralytics.com/#what-are-the-licensing-options-available-for-ultralytics-yolo)


   ![](https://oditeksolutions.com/wp-content/uploads/2025/01/Fashionable-Blog-Banner.webp)
  <p align = 'center'> Phát hiện gương mặt với YOLO </p>

### Nhận diện các gương mặt
- Ứng dụng mô hình CNN để train dữ liệu nhận diện các gương mặt. Các gương mặt được thu thập, gán nhãn, sau đó tiền xử lý dữ liệu và đưa vào CNN để train tạo ra model có thể phân biệt gương mặt mỗi người

  > Layer cuối cùng là lớp Dense có số unit bằng số người cần phân biệt và activation là hàm softmax. Từ đó có thể phát triển code nhận diện gương mặt (nếu độ tin cậy < 90% thì gán nhãn là 'unknow' ngược lại thì ghhi nhãn của nó lên màn hình)

  ![](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/10/59954intro-to-CNN.webp)
    <p align = 'center'> Minh họa cấu trúc CNN </p>
## Cách chạy phần mềm

- **B1**: clone project trên về và chạy lệnh sau ở thư mục vừa clone đó trong terminal:
    ``` bash
    pip install -r 'requirements.txt'
    ```
- **B2**: chạy file `handle_main.py` để trải nghiệm

## Demo phần mềm
### Khi mở phần mềm
- Trang hiển thị hướng dẫn dùng sẽ hiện đầu tiên
![](https://raw.githubusercontent.com/vietanhlee/face-recognition-Qt5/refs/heads/main/display_github/Screenshot%202025-02-10%20200413.png)

### Chức năng `thu thập dữ liệu` 
- Khi điền tên và số gương mặt cần cắt phần mềm sẽ tự động lấy đầy đủ số ảnh cắt gương mựt đó và lưu vào thư mục riêng

- Nhìn thẳng vào camerea rồi bấm `run` để chương trình tự động lấy đủ số lượng gương mặt cần lấy. Nên quay nhiều hướng khác nhau để data đa dạng, tránh overfitting. Code tự điều chỉnh ánh sáng và tương phản nên không nhất thiết cần thu thập gương mặt mọi người ở cùng vị trí (nhưng có vẫn là hơn)

- Làm tương tự cho những người còn lại đến khi hết.

    ![](https://raw.githubusercontent.com/vietanhlee/face-recognition-Qt5/refs/heads/main/display_github/Screenshot%202025-02-10%20200519.png)

- Khi nhấn tạm dừng thì phần mềm sẽ dừng việc thu thập dữ liệu gương mặt
    ![](https://raw.githubusercontent.com/vietanhlee/face-recognition-Qt5/refs/heads/main/display_github/Screenshot%202025-02-10%20200610.png)

### Chức năng `xử lý và train`
- Chọn số người cần nhận diện để phần mềm chọn cấu hình model phù hợp
- Thanh tiến trình bên dưới hiển thị bên dưới
![](https://raw.githubusercontent.com/vietanhlee/face-recognition-Qt5/refs/heads/main/display_github/Screenshot%202025-02-10%20200633.png)

### Chức năng `chạy thử`
- Khi nhấn chạy thử phần mềm sẽ thực hiện việc dự đoán và gán nhãn cho từng khuôn mặt được phát hiện
![](https://raw.githubusercontent.com/vietanhlee/face-recognition-Qt5/refs/heads/main/display_github/Screenshot%202025-02-10%20201258.png)


# Hiện đang phát triển thêm chức năng phục vụ cho việc chấm công