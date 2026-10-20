# AdaZero-DCE: số vòng đường cong thích ứng

Bài tập lớn môn Mạng nơron và học sâu, Trường Công nghệ thông tin Phenikaa.
Sinh viên: Trần Minh Sang, Ngô Quang Thiện. Giảng viên: TS. Đặng Thị Thúy An.

AdaZero-DCE giữ DCE-Net `Epoch99.pth` đóng băng và train head hồi quy
10→128→64→1 (9.729 tham số) để chọn 1–8 bước đầu tiên của chuỗi đường cong.
Mean RGB < 0.10 giữ n=8; các ảnh còn lại dùng dự đoán head được round/clamp.
Head có giám sát từ oracle-n trên train; không gọi toàn bộ phương pháp là zero-reference.

## Notebook và báo cáo

- [Notebook có outputs thực thi trên Kaggle](notebook/AdaZero_DCE_Complete_Evaluation_Executed.ipynb)
- [Notebook nguồn](notebook/AdaZero_DCE_Complete_Evaluation.ipynb)
- [Báo cáo PDF](report/AdaZero-DCE_Bao-cao.pdf) · [LaTeX](report/main.tex)
- [Hướng dẫn notebook](notebook/README.md) · [Hướng dẫn Kaggle](kaggle/README.md)
- [Full LOLv1](reports/full-report.md) · [LOLv2-real](reports/lolv2-real-report.md)

Notebook báo cáo chạy mới inference/metric với ba head đã train; không train lại.
Biểu đồ loss và oracle được đọc từ artifact train full, có ghi rõ nguồn lịch sử.
Các bảng mới nằm trong `notebook/execution/adazero-notebook-results/`.

## Kết quả đã đo

Trung bình ba seed 0/1/2; delta = adaptive − fixed-8. Cùng một implementation
PSNR/SSIM cho hai nhánh. Không đối chiếu SSIM tuyệt đối với số kỳ vọng trong plan cũ.

| Dữ liệu | PSNR fixed-8 | PSNR adaptive | ΔPSNR | SSIM fixed-8 | SSIM adaptive | ΔSSIM |
|---|---:|---:|---:|---:|---:|---:|
| eval15 gốc | 14.8611 | 14.8611 | 0.0000 | 0.562653 | 0.562653 | 0.000000 |
| GT×0.3 | 18.3276 | 18.2442 | -0.0834 | 0.891626 | 0.897264 | +0.005638 |
| GT×0.45 | 17.2052 | 18.2826 | +1.0774 | 0.867250 | 0.900064 | +0.032814 |
| GT×0.6 | 16.4163 | 19.6761 | +3.2598 | 0.853672 | 0.914620 | +0.060947 |
| LOLv2-real Test | 18.0598 | 17.8576 | -0.2022 | 0.579800 | 0.581354 | +0.001555 |

15/15 ảnh eval15 gốc và 87/100 ảnh LOLv2 được guard giữ n=8. Kết quả trên
GT nhân hệ số là giả lập sRGB; chưa chứng minh cải thiện khái quát trên ảnh thật.
Trên LOLv2, cả ba seed đều giảm PSNR, dù SSIM tăng nhẹ.

## Ưu điểm và giới hạn

- Số vòng thích ứng giúp hạn chế tăng sáng quá mức trên ảnh vừa tối; đo được
  +1.08 và +3.26 dB ở GT×0.45/0.6 so với Zero-DCE baseline (fixed-8).
- Head khoảng 10K tham số, không cần VLM hoặc mô hình phụ trợ ngoài DCE-Net;
  train được trên CPU với chi phí thấp. Thực tế khoảng 6.49–6.76 giây/head khi
  đã có cache; tạo oracle là bước riêng, đã dùng GPU trong lần chạy này.
- Ngưỡng bảo vệ ảnh rất tối giữ kết quả không tụt trên LOLv1 eval15 gốc.
- DCE-Net đóng băng hoàn toàn, gắn thêm head mà không train lại baseline.

Mức tăng trên chỉ được chứng minh ở ảnh giả lập bằng cách nhân GT, chưa được
chứng minh trên ảnh vừa tối thật. Ảnh thật bằng baseline trên LOLv1 hoặc hơi
kém theo PSNR trung bình toàn tập trên LOLv2-real (khoảng -0.20 dB); một số ảnh
riêng lẻ có thể giảm nhiều hơn. Head hiện học theo phân bố LOLv1 và chưa khái
quát tốt sang LOLv2-real. Ý tưởng thích ứng đã có trong VAR-LIDE; đóng góp của
bài tập là cách làm nhẹ và bảng định lượng tự thực hiện so với fixed-8.

## Sử dụng

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 AdaZero-DCE_code/infer.py input.png output.png --head runs/kaggle-full/full/head_seed0.pth
```

Train lại full nếu cần tái lập từ đầu (notebook báo cáo không thực hiện bước này):

```sh
python3 AdaZero-DCE_code/experiment.py --data /path/to/lol_dataset --output runs/full-new --train-pairs 485 --epochs 1200 --seeds 0 1 2
```

Dữ liệu cần `our485/{low,high}/*.png` và `eval15/{low,high}/*.png`, tên cặp khớp.
485 cặp train được mở rộng thành 1940 đầu vào: low gốc và GT×0.3/0.45/0.6.
`factor=1.0` nghĩa là **low gốc**, không phải GT×1. Oracle chỉ tính trên our485.
Chuẩn hóa feature chỉ dùng train. Head: ReLU, dropout 0.1, SmoothL1, Adam 5e-3,
CosineAnnealingLR, full batch 1200 epoch. Không chọn threshold/seed/epoch bằng test.

PSNR RGB float [0,1]; SSIM RGB uniform 7×7, sample covariance, data_range=1.
Giữ nguyên độ phân giải, không crop, resize, lượng tử hóa hoặc GT mean correction.
Metric tính từng ảnh rồi lấy trung bình. LOLv2 chỉ tải 100 cặp Test, không dùng Train.

## Cấu trúc và nguồn gốc

- `zero_dce_adaptive_n.py`: feature, head, oracle, suy luận.
- `AdaZero-DCE_code/experiment.py`: train head và đánh giá LOLv1; mặc định CLI là pilot 96 cặp, nên **truyền `--train-pairs 485` cho full**.
- `AdaZero-DCE_code/evaluate_lolv2.py`: đánh giá test bằng checkpoint đóng băng.
- `AdaZero-DCE_code/`: backbone và tài nguyên kế thừa Zero-DCE. `lowlight_train.py`
  là trainer backbone tham chiếu, không phải pipeline train head hiện tại.
  Xem [phân loại mã kế thừa](AdaZero-DCE_code/README.md) trước khi sử dụng.
- Tài liệu baseline: dùng liên kết nguồn tác giả ở phần trích dẫn; bản lưu
  website và tài nguyên dư thừa đã được loại bỏ.
- `reports/`, `runs/`, `deliverables/`: kết quả và snapshot các lần chạy trước.
- `notebook/`: notebook trình bày cùng kết quả thực thi.
- `report/`: PDF báo cáo, nguồn LaTeX, hình và bảng trong cùng một thư mục.

Tên Zero-DCE trong công trình, mã nền và checkpoint được giữ đúng nguồn gốc.
Đợt đồng bộ tài liệu không đổi tên hoặc sửa code/trọng số/notebook cũ.

## Trích dẫn và giấy phép

Nền tảng: Guo et al., **Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement**, CVPR 2020, [repo tác giả](https://github.com/Li-Chongyi/Zero-DCE).
Mã nền được công bố cho nghiên cứu học thuật theo Attribution-NonCommercial 4.0;
giữ nguyên thông tin bản quyền và điều kiện sử dụng của upstream.

Công trình liên quan: Dong et al., [VAR-LIDE, arXiv:2511.18591](https://arxiv.org/abs/2511.18591),
sử dụng visibility từ VLM cho bài toán joint enhancement/deblurring. Dự án không
nhận là công trình đầu tiên dùng adaptive iterations và chưa so định lượng với VAR-LIDE.
