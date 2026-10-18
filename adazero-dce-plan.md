# Kế hoạch dự án: AdaZero-DCE (Zero-DCE với số vòng lặp thích ứng)

## 1. Mục tiêu

Cải tiến Zero-DCE (CVPR 2020): thay số vòng lặp đường cong cố định (8) bằng số
vòng lặp thích ứng theo từng ảnh, do một head học được dự đoán. Giữ nguyên
DCE-Net, không train lại baseline.

## 2. Baseline và dữ liệu

- Repo gốc: https://github.com/Li-Chongyi/Zero-DCE
- Trọng số dùng sẵn: `Epoch99.pth` (không train lại DCE-Net).
- Train head: our485 (485 cặp low/high của LOLv1).
- Test: eval15 (15 cặp), đo PSNR/SSIM, so với fixed-8.
- Baseline đã đo: PSNR 14.86, SSIM 0.5927 trên eval15.

## 3. Các bước thực hiện

### Bước 1: Tính nhãn oracle-n (offline, 1 lần)

Với mỗi cặp (low, GT) trong tập train:
1. Chạy 1 lần DCE-Net trên ảnh low, lấy 8 alpha map.
2. Áp dụng lần lượt n = 1..8 vòng đường cong, được 8 ảnh output.
3. Tính PSNR từng output với GT, lấy n cho PSNR cao nhất làm nhãn oracle-n.

Mở rộng tập train: ngoài ảnh low gốc, tạo thêm ảnh giả lập bằng cách nhân
GT với hệ số 0.3, 0.45, 0.6 rồi tính oracle-n cho chúng. Tổng 1940 mẫu.
Tuyệt đối không tính oracle-n trên ảnh test.

### Bước 2: Trích đặc trưng

Mỗi ảnh đầu vào -> vector 10 chiều: mean, std và histogram 8 bin của ảnh xám.
Chuẩn hóa bằng mean/std tính trên tập train.

### Bước 3: Train head hồi quy

- Kiến trúc: MLP 10-128-64-1, ReLU, Dropout 0.1 (khoảng 10K tham số).
- Loss: SmoothL1. Optimizer: Adam lr 5e-3. Scheduler: CosineAnnealing.
- 1200 epoch, full batch. Target: oracle-n dạng số thực.
- Train vài phút trên CPU. Lưu state dict, mean/std chuẩn hóa.

### Bước 4: Inference

Với mỗi ảnh đầu vào:
1. Nếu mean độ sáng < 0.10: n = 8 (vùng này oracle gần như luôn bằng 8).
2. Còn lại: n = làm tròn output của head, kẹp trong [1, 8].
3. Chạy DCE-Net 1 lần lấy alpha map, áp dụng đúng n vòng đường cong.

### Bước 5: Đánh giá

Đo PSNR/SSIM trên eval15 gốc và 3 bản giả lập (GTx0.3, GTx0.45, GTx0.6),
so sánh với fixed-8. Kết quả kỳ vọng:
- eval15 gốc: 14.86 / 0.5927 (không tụt so với baseline).
- GTx0.45: khoảng 18.18 / 0.8862 (tăng ~0.97 dB).
- GTx0.6: khoảng 19.55 / 0.9031 (tăng ~3.13 dB).
- Ổn định qua nhiều seed.

## 4. Lưu ý kỹ thuật

- File trọng số head lưu state dict dưới key 'reg'; lúc load phải đọc đúng key.
- Nhãn oracle-n chỉ tính trên tập train.
- Code tham khảo đã kiểm chứng: `zero_dce_adaptive_n.py`, trọng số mẫu:
  `head_adaptive_n.pth` (cùng thư mục).

## 5. Caveat cho báo cáo

- Tập vừa tối là ảnh giả lập (nhân GT với hệ số), không phải ảnh thiếu sáng
  thật; con số cải tiến đo trên giả lập.
- Trên phân bố LOL thật (toàn ảnh rất tối), adaptive tương đương fixed-8 vì
  8 vòng vốn đã gần tối ưu ở đó (oracle chỉ hơn 0.18 dB).
- Paper liên quan cần trích dẫn: VAR-LIDE (arXiv:2511.18591) cũng chỉ ra
  điểm yếu số vòng lặp cố định và cũng dùng MLP dự đoán số vòng lặp tối ưu
  (họ dùng điểm visibility từ VLM). Điểm khác biệt: mình dùng đặc trưng độ
  sáng rẻ tiền không cần VLM, head ~10K tham số, có ngưỡng bảo vệ ảnh rất tối,
  và làm enhancement độc lập thay vì conditioning cho pipeline deblurring.
