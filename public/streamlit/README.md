# Facebook Sentiment Analysis Dashboard (Streamlit)

Ứng dụng Dashboard tương tác phân tích cảm xúc bài đăng Facebook xây dựng bằng **Python**, **Streamlit**, **Pandas**, **Plotly** và **Scikit-learn**.

## Tính năng chính

1. **Thống kê tổng quan (KPI Cards)**: Tổng số bài đăng, số lượng và tỷ lệ phần trăm bài đăng Positive, Neutral, Negative (tự động cập nhật theo bộ lọc).
2. **Biểu đồ phân bố cảm xúc (Plotly)**: Biểu đồ cột (Bar Chart) hiển thị số lượng và biểu đồ tròn (Donut Chart) hiển thị tỷ trọng phần trăm với quy ước màu nhất quán (Positive: Xanh lá, Neutral: Cam/Vàng, Negative: Đỏ).
3. **Phân tích tương tác theo cảm xúc**: So sánh `Likes`, `Comments`, `Shares` theo 2 chế độ (**Trung bình trên mỗi bài đăng** và **Tổng lượng tương tác**) qua Grouped Bar Chart và bảng thống kê chi tiết.
4. **Bộ lọc dữ liệu đồng bộ (Sidebar)**: Lọc đa giá trị theo Danh mục (`category`), Nhóm cảm xúc (`sentiment`), Ngôn ngữ (`language`), hỗ trợ "Chọn tất cả" và tùy chọn loại bỏ bài đăng spam (`is_spam == 1`).
5. **Danh sách bài đăng & Kết quả dự đoán**: Tìm kiếm từ khóa, lọc chỉ các bài dự đoán sai, sắp xếp theo thời gian, phân biệt trực quan nhãn thực tế vs nhãn dự đoán, và xuất dữ liệu đã lọc ra file CSV.
6. **Dự đoán cảm xúc trực tiếp (Machine Learning)**: Sử dụng **TF-IDF** kết hợp với **Naive Bayes (MultinomialNB)** và **Decision Tree**, huấn luyện & đánh giá chuẩn trên tập **Train/Test Split (80/20)** (không rò rỉ dữ liệu), hiển thị nhãn dự đoán và xác suất từng lớp.

---

## Cấu trúc thư mục

```text
.
├── app.py                  # Mã nguồn chính của ứng dụng Streamlit
├── Facebook_data_txt.csv   # Dữ liệu bài đăng Facebook
├── requirements.txt        # Danh sách thư viện Python cần thiết
└── README.md               # Hướng dẫn cài đặt và triển khai
```

---

## Hướng dẫn chạy trên máy cá nhân (Local)

### Bước 1: Tạo và kích hoạt môi trường ảo (khuyến nghị)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 3: Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Mở trình duyệt tại địa chỉ `http://localhost:8501`.

---

## Hướng dẫn đưa lên GitHub và triển khai lên Render

### 1. Đưa mã nguồn lên GitHub

```bash
git init
git add app.py requirements.txt Facebook_data_txt.csv README.md
git commit -m "Initial commit: Facebook Sentiment Analysis Dashboard"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

### 2. Triển khai (Deploy) lên Render

1. Đăng nhập vào [Render](https://render.com) và chọn **New + → Web Service**.
2. Kết nối với repository GitHub vừa tạo.
3. Cấu hình các thông số triển khai:
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
     ```
4. Nhấn **Create Web Service** và đợi Render hoàn tất build.
