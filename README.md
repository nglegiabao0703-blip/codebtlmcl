# Dự đoán Khả năng Hủy Đặt phòng Khách sạn (Hotel Booking Cancellation Prediction)

Bài tập lớn môn **Nhập môn Học máy** – Xây dựng quy trình học máy hoàn chỉnh trên dữ liệu dạng bảng (Tabular Data) nhằm dự đoán khả năng khách hàng hủy đặt phòng khách sạn (`is_canceled`).

## 1. Giới thiệu bài toán & Dữ liệu
* **Loại bài toán:** Phân loại nhị phân (Binary Classification) với biến mục tiêu `is_canceled` (`0`: Không hủy, `1`: Hủy phòng).
* **Nguồn dữ liệu:** Bộ dữ liệu `hotel_bookings.csv` (Hotel Booking Demand Dataset – Kaggle / Antonio, Almeida và Nunes, 2019), gồm **119.390 bản ghi** và **32 thuộc tính** ban đầu.
* **Đặc điểm sau làm sạch:** Loại bỏ **31.994 bản ghi trùng lặp**, giữ lại **87.396 bản ghi** hợp lệ. Loại bỏ các cột gây rò rỉ dữ liệu (`reservation_status`, `reservation_status_date`, `assigned_room_type`).

> **Lưu ý về dữ liệu:** Do dung lượng tệp lớn, bộ dữ liệu `hotel_bookings.csv` có thể được tải trực tiếp từ [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) và đặt tại thư mục gốc của dự án trước khi chạy chương trình.

## 2. Cấu trúc thư mục dự án

```text
Hotel_Cancellation_Prediction/
├── hotel_bookings.csv           # Dữ liệu đầu vào (đặt tại thư mục gốc)
├── 01_EDA.py                    # Khám phá dữ liệu, kiểm tra thiếu/trùng lặp và thống kê mô tả
├── 02_Preprocessing.py          # Tiền xử lý dữ liệu theo Pipeline A (cơ bản) và chia tập Train/Test
├── 03_Training.py               # Huấn luyện và đánh giá 3 thuật toán: KNN, Decision Tree, Naive Bayes
├── 04_Pipeline_Comparison.py    # So sánh hiệu năng giữa Pipeline A (Cơ bản) và Pipeline B (Cải tiến)
├── requirements.txt             # Danh sách các thư viện Python cần thiết
├── README.md                    # Tài liệu hướng dẫn dự án
└── AI_USAGE.md                  # Khai báo minh bạch việc sử dụng công cụ AI