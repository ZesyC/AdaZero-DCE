# Lưu ý khi implement AdaZero-DCE

## 1. Những điểm dễ đi nhầm

- Không train lại DCE-Net. Baseline giữ nguyên trọng số Epoch99.pth, chỉ train
  head. Train lại DCE-Net vừa tốn công vừa làm mất tính so sánh với baseline.
- Không tính nhãn oracle-n trên tập test. Nhãn chỉ tính trên tập train
  (our485). Lẫn test vào train là data leakage, giảng viên hỏi là chết.
- Không bỏ ngưỡng ảnh rất tối (mean < 0.10 -> n = 8). Thử nghiệm cho thấy bỏ
  ngưỡng này làm PSNR trên LOL gốc tụt 0.19 dB so với baseline.
- Không dùng classifier thay regressor. Bản classifier phân loại n thành 8 lớp
  cho kết quả thất thường khi gặp mức sáng chưa thấy lúc train; bản hồi quy
  khái quát tốt hơn.
- Đặc trưng chỉ cần mean, std, histogram 8 bin. Đừng thêm đặc trưng phức tạp
  khi chưa có bằng chứng nó giúp, dễ overfit trên 1940 mẫu.
- Đánh giá phải giữ đúng 2 tập: eval15 gốc (chứng minh không tụt) và tập giả
  lập vừa tối (chứng minh cải tiến). Thiếu một trong hai là bảng số không đủ
  sức thuyết phục.

## 2. Phân biệt với VAR-LIDE (arXiv:2511.18591)

VAR-LIDE đã làm adaptive iteration cho Zero-DCE, nên trong báo cáo bắt buộc
trích dẫn. Giữ khác biệt ở đúng những điểm sau, đừng implement theo hướng
của họ:

- Họ dùng điểm visibility từ VLM (nặng, cần pipeline GPP-LLIE riêng). Mình
  dùng đặc trưng độ sáng rẻ tiền, không phụ thuộc mô hình ngoài.
- Họ train unsupervised để làm conditioning cho VAR deblurring. Mình train có
  giám sát trên nhãn oracle-n cho bài toán enhancement độc lập.
- Họ không có cơ chế bảo vệ ảnh rất tối. Mình có ngưỡng mean < 0.10 giữ n = 8.
- Họ không đo định lượng hiệu quả của riêng phần adaptive iteration. Mình có
  bảng PSNR/SSIM so trực tiếp với fixed-8, đó là đóng góp thực nghiệm.

Đừng đi lệch sang hướng của họ: không đưa VLM vào, không gắn thêm deblurring,
không biến head thành module conditioning cho mô hình sinh. Giữ bài toán gọn
là enhancement độc lập.

## 3. Ưu điểm của phương pháp mình

- Nhẹ: head ~10K tham số, train vài phút trên CPU, inference thêm không đáng
  kể so với 1 lần forward DCE-Net.
- Không cần dữ liệu hay mô hình phụ trợ ngoài cặp low/high có sẵn của LOL.
- Ổn định: kết quả lặp lại qua 3 seed, không tụt trên tập LOL gốc.
- Có số đo định lượng trực tiếp cho adaptive iteration, thứ VAR-LIDE chưa làm.
- Dễ trình bày: ý tưởng trực quan (ảnh càng sáng càng cần ít vòng lặp), hình minh
  họa đường cong PSNR theo n cho từng ảnh là bằng chứng trực quan mạnh.
