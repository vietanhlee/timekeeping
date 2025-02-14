# Phần mềm nhận diện gương mặt phục vụ cho điểm danh hoặc chấm công
## Chi tiết về code chạy console và lý thuyết các model AI
- Phần project này chủ yếu là phát triển giao diện cho code nhận diện gương mặt chạy console. Hay nói cách khác thì chỉ là phần front end dựa trên framework Pyqt5 và thêm một số tính năng khác

- Chi tiết bản code nhận diện gương mặt chạy trên console và lí thuyết áp dụng ở [đây](https://github.com/vietanhlee/face-recognition)


## Cách chạy phần mềm

- **B1**: clone project trên về và chạy lệnh sau ở thư mục vừa clone đó trong terminal:
    ``` bash
    pip install -r 'requirements.txt'
    ```
- **B2**: chạy file `handle_main.py` để trải nghiệm

## Demo phần mềm
### Khi mở phần mềm
- Trang hiển thị hướng dẫn dùng sẽ hiện đầu tiên
![](https://raw.githubusercontent.com/vietanhlee/timekeeping/refs/heads/main/display_github/Screenshot%202025-02-13%20132040.png)

### Chức năng `thu thập dữ liệu` 
- Khi điền tên và số gương mặt cần cắt phần mềm sẽ tự động lấy đầy đủ số ảnh cắt gương mặt đó và lưu vào thư mục riêng

- Nhìn thẳng vào camerea rồi bấm `run` để chương trình tự động lấy đủ số lượng gương mặt cần lấy. Nên quay nhiều hướng khác nhau để data đa dạng, tránh overfitting. Code tự điều chỉnh ánh sáng và tương phản nên không nhất thiết cần thu thập gương mặt mọi người ở cùng vị trí (nhưng có vẫn là hơn)

- Làm tương tự cho những người còn lại đến khi hết.

    ![](https://raw.githubusercontent.com/vietanhlee/timekeeping/refs/heads/main/display_github/Screenshot%202025-02-13%20132224.png)

- Khi nhấn tạm dừng thì phần mềm sẽ dừng việc thu thập dữ liệu gương mặt
    ![](https://raw.githubusercontent.com/vietanhlee/timekeeping/refs/heads/main/display_github/Screenshot%202025-02-13%20132251.png)

### Chức năng `xử lý và train`
- Chọn số người cần nhận diện để phần mềm chọn cấu hình model phù hợp
- Thanh tiến trình bên dưới hiển thị bên dưới

![](https://raw.githubusercontent.com/vietanhlee/timekeeping/refs/heads/main/display_github/Screenshot%202025-02-13%20132140.png)

### Chức năng `chấm công`
- Khi không nhận diện được gương mặt

![](https://raw.githubusercontent.com/vietanhlee/timekeeping/refs/heads/main/display_github/Screenshot%202025-02-13%20132318.png)

- Khi nhận diện được gương mặt và đã `Check in` trước đó. Giao diện sẽ hiển thị lại thời gian check in và hình ảnh lúc check in

![](https://raw.githubusercontent.com/vietanhlee/timekeeping/refs/heads/main/display_github/Screenshot%202025-02-13%20132336.png)

- Khi bấm `Check out` các thông số được thiết đặt lại từ đầu là chưa check in

![](https://raw.githubusercontent.com/vietanhlee/timekeeping/refs/heads/main/display_github/Screenshot%202025-02-13%20132401.png)

- Khi bấm `Check in` một lần nữa. Chương trình sẽ lưu data lại, bao gồm ảnh lúc check in và thời gian lúc check in. Hiển thị thông số mỗi khi chính gương mặt đó lọt vào ống kính

![](https://raw.githubusercontent.com/vietanhlee/timekeeping/refs/heads/main/display_github/Screenshot%202025-02-13%20132414.png)

- Khi bấm `Xuất EXCEL` chương trình sẽ lưu lại tất cả thông tin check in và check out của ngày hiện tại vào file `.csv`