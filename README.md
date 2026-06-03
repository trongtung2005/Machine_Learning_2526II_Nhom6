# PHÂN LOẠI ĐA NHÃN CẢM XÚC VĂN BẢN TRÊN TẬP DỮ LIỆU GOEMOTIONS

Đề tài tập trung nghiên cứu, triển khai thực nghiệm và đánh giá đối chứng hiệu năng của 05 mô hình Học máy truyền thống và Học sâu trong bài toán phân loại đa nhãn (Multi-label Classification) trên tập dữ liệu biểu đạt 28 sắc thái cảm xúc GoEmotions của Google.

Slide canva nhóm 6: https://canva.link/w24w1erhkvcqcl5

Dataset: https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset/data

##  Tổng Quan Hệ Thống

Hệ thống xử lý và nhận diện cảm xúc văn bản được thiết kế theo một đường ống dữ liệu (Pipeline) khép kín:
1. Tiền xử lý dữ liệu (NLP Pipeline): Chuẩn hóa văn bản thô (lowercase), loại bỏ các ký tự đặc biệt, dấu câu, phân tách từ (Tokenization) và chuẩn hóa gốc từ (Lemmatization).
2. Biểu diễn từ vựng (Word Embedding): Chuyển đổi chuỗi văn bản đã làm sạch sang không gian vector bằng phương pháp TF-IDF (cho các mô hình truyền thống) và kiến trúc nhúng từ tự huấn luyện Word2Vec (cho các mạng học sâu).
3. Huấn luyện mô hình: Thực nghiệm song song 05 thuật toán phân loại đa nhãn:
   * Học máy truyền thống: Hồi quy Logistic (OVR), LinearSVC (OVR), Random Forest.
   * Học sâu (Deep Learning): Mạng tích chập 1D CNN, Mạng bộ nhớ dài-ngắn hạn hai chiều (Bi-LSTM).
4. Chiến lược tối ưu ranh giới: Thay vì áp dụng một ngưỡng cố định 0.5 thông thường, hệ thống tích hợp thuật toán tinh chỉnh ngưỡng cắt độc lập (Threshold Tuning) cho từng lớp nhãn dựa trên điểm số cực đại của F1-Score trên tập Validation.

##  Phân Tích Kết Quả Trên tập Test

### 1. Đánh giá định lượng trên Ma trận nhầm lẫn (Confusion Matrix)
* Hiện tượng mất cân bằng dữ liệu (Class Imbalance): Các ô True Negative (Mô hình đoán đúng nhãn 0) chiếm mật độ khổng lồ. Điều này hoàn toàn đồng nhất với đặc tính thưa thớt của dữ liệu đa nhãn thực tế khi một câu bình luận thường chỉ chứa từ 1 đến 2 sắc thái cảm xúc trong tổng số 28 lớp nhãn.
* Các nhãn hiệu năng cao: Hệ thống bắt trúng tối ưu (True Positive cao, False Negative thấp) ở các nhãn có tính biểu đạt từ vựng mạnh như Amusement, Gratitude, Love và Neutral.
* Hạn chế tồn tại: Hiện tượng thiếu hụt dữ liệu huấn luyện nghiêm trọng (Data Sparsity) khiến các nhãn thiểu số như Grief, Pride đạt hiệu năng thấp. Các nhãn có ranh giới ngữ nghĩa mập mờ (Annoyance, Realization) ghi nhận tỷ lệ nhận diện sai (False Positive) tương đối đáng kể.

### 2. So sánh đối chứng định tính giữa 05 Thuật toán
* Hồi quy Logistic & LinearSVC (OVR): Thể hiện ưu thế phân tách ranh giới tuyến tính dứt khoát trên các câu tường minh (như nhãn Gratitude đạt điểm số độ tự tin 83.58% và khoảng cách hình học Z = 1.3457). Tuy nhiên, độ tự tin bị sụt giảm lớn khi đối mặt với các từ ngữ mang tính ẩn dụ phức tạp.
* Random Forest: Cho khả năng bao quát dữ liệu cực tốt nhờ cấu trúc tập hợp cây quyết định độc lập. Giữ sự cân bằng và độ tự tin cao nhất khi bóc tách chính xác đồng thời cả hai nhãn đối lập Amusement (74.53%) và Sadness (66.89%) ở câu chứa ngữ cảnh tương phản (Kịch bản 2).
* Mạng 1D CNN: Sở hữu năng lực trích xuất các đặc trưng cục bộ (n-gram mang tín hiệu cảm xúc) rất mạnh thông qua cơ chế trượt của bộ lọc và lớp Max Pooling. Đạt độ tự tin tối ưu nhất ở câu ngắn tường minh (Gratitude vọt lên mức 98.84%), nhưng có xu hướng lược bỏ các sắc thái phụ ở câu phức dài.
* Mạng Bi-LSTM: Chứng minh ưu thế tuyệt đối của cơ chế các cổng nhớ trong việc duy trì mạch ngữ nghĩa xuyên suốt chuỗi tuyến tính, giúp mô hình phân phối xác suất mềm mại và giữ lại trọn vẹn các sắc thái cảm xúc hỗn hợp đan xen tốt hơn cấu trúc CNN.

##  Kịch Bản Thực Nghiệm Định Tính (Live Test)

Để kiểm tra trực quan năng lực suy luận thực tế và khả năng phản ứng của bộ lọc ranh giới quyết định, hệ thống được chạy thử nghiệm đồng thời qua 04 kịch bản văn bản đầu vào (chưa từng xuất hiện trong tập huấn luyện):

* Kịch bản 1 (Câu cổ vũ, đa sắc thái): "Good luck with your final exam tomorrow, you'll do great!"
* Kịch bản 2 (Câu chứa từ ngữ tương phản, ẩn dụ): "I laughed so hard at this that my ribs actually hurt."
* Kịch bản 3 (Câu tường minh, đặc trưng từ vựng mạnh): "Thank you so much for helping me out with this project!"
* Kịch bản 4 (Câu thể hiện trạng thái tâm lý hỗn hợp): "I'm really nervous about the interview, but also quite excited."

##  Hướng Dẫn Khởi Chạy Trên Google Colab

Dự án được cấu trúc dưới dạng các tệp Notebook độc lập, tối ưu hóa để thực thi trực tiếp trên nền tảng đám mây Google Colab. Người dùng chỉ cần tải các tệp mã nguồn về, mở trên Colab và chọn chế độ "Run all" để hệ thống tự động thiết lập môi trường, cài đặt các thư viện bổ sung, tiến hành huấn luyện mô hình và kích hoạt cổng thực nghiệm Live Test.
