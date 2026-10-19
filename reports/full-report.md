# AdaZero-DCE — báo cáo full our485, 19/10/2026

Đã train đủ **485 cặp our485 → 1.940 mẫu** và đánh giá trên eval15 gốc cùng
ba mức giả lập. Không dùng subset 96 cặp. Chạy đủ seed **0, 1, 2**, mỗi seed
1.200 epoch; không chọn seed tốt nhất để đại diện kết quả.

[Kaggle full, version 1](https://www.kaggle.com/code/sangtranminh211/adazero-dce-full-20261019/versions/1)
đã COMPLETE. Notebook private, internet off, Tesla T4. Head train trên CPU;
GPU dùng cho DCE-Net tính alpha maps/oracle và đánh giá.

## Cấu hình giữ nguyên

- DCE-Net đóng băng từ `Epoch99.pth`; không train lại backbone.
- 485 input low gốc và 3×485 input GT×0.3/0.45/0.6. `factor=1.0` là low gốc, không phải GT×1.
- Tính oracle-n mới cho toàn bộ 1.940 mẫu train, tối ưu PSNR trên n=1..8; một forward DCE-Net/input. Không tính oracle trên eval15.
- Feature 10 chiều: mean/std ảnh xám trung bình RGB và histogram 8 bin. Chuẩn hóa chỉ từ đủ 1.940 mẫu train.
- MLP 10→128→64→1, ReLU, dropout 0.1, 9.729 tham số; hồi quy n.
- SmoothL1, Adam lr 5e-3, CosineAnnealing T_max=1200, full batch, 1200 epoch, seed 0/1/2.
- Mean <0.10 giữ n=8; còn lại round/clamp về [1,8].
- Toàn bộ 6 file nguồn/test/trọng số đóng gói có hash giống pilot. Không thay kiến trúc, loss, ngưỡng hay cách đo sau khi xem eval15.

## Giao thức đánh giá

**Chỉ so fixed-8 và adaptive trong cùng implementation** `experiment.metrics`.
Không đối chiếu số SSIM tuyệt đối với plan, không đổi metric để khớp con số nào.
Mỗi điều kiện có 15 ảnh; giữ độ phân giải gốc, RGB float32 [0,1], không
resize/crop/lượng tử hóa trước khi đo. PSNR dùng `skimage.metrics.peak_signal_noise_ratio`
với data_range=1. SSIM dùng `structural_similarity`, channel_axis=2,
data_range=1, cửa sổ đều 7×7, sample covariance; trung bình qua kênh RGB.

Bảng lấy trung bình metric từng ảnh, rồi tổng hợp qua 3 seed. **± là population
std giữa 3 seed**, không phải khoảng tin cậy theo ảnh. Δ = adaptive − fixed-8.
Fixed-8 không phụ thuộc seed và được dùng cùng một đầu ra baseline cho cả ba head.

## Kết quả tổng hợp

| Điều kiện eval15 | Fixed-8 PSNR | Adaptive PSNR, mean ± std | Δ PSNR, dB |
|---|---:|---:|---:|
| LOL low gốc | 14.8611 | 14.8611 ± 0.0000 | +0.0000 ± 0.0000 |
| GT×0.3 | 18.3276 | 18.2442 ± 0.0514 | -0.0834 ± 0.0514 |
| GT×0.45 | 17.2052 | 18.2826 ± 0.1615 | +1.0774 ± 0.1615 |
| GT×0.6 | 16.4163 | 19.6761 ± 0.0924 | +3.2598 ± 0.0924 |

| Điều kiện eval15 | Fixed-8 SSIM | Adaptive SSIM, mean ± std | Δ SSIM |
|---|---:|---:|---:|
| LOL low gốc | 0.562653 | 0.562653 ± 0.000000 | +0.000000 ± 0.000000 |
| GT×0.3 | 0.891626 | 0.897264 ± 0.000630 | +0.005638 ± 0.000630 |
| GT×0.45 | 0.867250 | 0.900064 ± 0.001480 | +0.032814 ± 0.001480 |
| GT×0.6 | 0.853672 | 0.914620 ± 0.001649 | +0.060947 ± 0.001649 |

## Kết quả từng seed

| Điều kiện | Seed | Adaptive PSNR | Δ PSNR | Adaptive SSIM | Δ SSIM | n TB |
|---|---:|---:|---:|---:|---:|---:|
| LOL low gốc | 0 | 14.8611 | +0.0000 | 0.562653 | +0.000000 | 8.000 |
| LOL low gốc | 1 | 14.8611 | +0.0000 | 0.562653 | +0.000000 | 8.000 |
| LOL low gốc | 2 | 14.8611 | +0.0000 | 0.562653 | +0.000000 | 8.000 |
| GT×0.3 | 0 | 18.3133 | -0.0143 | 0.897251 | +0.005625 | 7.400 |
| GT×0.3 | 1 | 18.1901 | -0.1374 | 0.898042 | +0.006416 | 7.267 |
| GT×0.3 | 2 | 18.2291 | -0.0984 | 0.896499 | +0.004873 | 7.333 |
| GT×0.45 | 0 | 18.1534 | +0.9482 | 0.899351 | +0.032101 | 6.600 |
| GT×0.45 | 1 | 18.5104 | +1.3052 | 0.902125 | +0.034875 | 6.733 |
| GT×0.45 | 2 | 18.1841 | +0.9789 | 0.898717 | +0.031467 | 6.667 |
| GT×0.6 | 0 | 19.7406 | +3.3244 | 0.914977 | +0.061305 | 6.133 |
| GT×0.6 | 1 | 19.7421 | +3.3259 | 0.916436 | +0.062764 | 6.067 |
| GT×0.6 | 2 | 19.5455 | +3.1292 | 0.912446 | +0.058773 | 6.133 |

## Diễn giải

- **LOL gốc:** giữ nguyên cả PSNR và SSIM ở cả 3 seed. Cả 15 ảnh đều mean <0.10 nên được bảo vệ bằng n=8; đây không phải bằng chứng head cải thiện ảnh LOL thật.
- **GT×0.45:** tăng PSNR 0.9482–1.3052 dB qua 3 seed, SSIM tăng 0.031467–0.034875. Dấu cải thiện nhất quán, mức tăng có dao động theo seed.
- **GT×0.6:** tăng PSNR 3.1292–3.3259 dB, SSIM tăng 0.058773–0.062764 qua 3 seed. Cả 15 ảnh tăng PSNR khi lấy trung bình qua seed.
- **GT×0.3:** PSNR còn giảm 0.0143–0.1374 dB qua 3 seed, dù SSIM tăng 0.004873–0.006416. Vì vậy chưa thể kết luận adaptive tốt hơn đồng thời mọi điều kiện/metric.

Đếm ảnh theo ΔPSNR trung bình qua 3 seed:

| Điều kiện | Tăng | Bằng | Giảm | n TB |
|---|---:|---:|---:|---:|
| LOL low gốc | 0 | 15 | 0 | 8.000 |
| GT×0.3 | 3 | 6 | 6 | 7.333 |
| GT×0.45 | 9 | 0 | 6 | 6.667 |
| GT×0.6 | 15 | 0 | 0 | 6.111 |

## Kiểm chứng và chi phí

Đã kiểm tra artifact tải từ Kaggle:

- Manifest có đủ 485 tên train duy nhất; oracle CSV có 1.940 cặp (file, factor) duy nhất, đúng 485 mẫu/mức sáng.
- Cache có shape (1940,10), target shape (1940,); target khớp từng dòng oracle CSV. Histogram nhãn có tổng 1.940.
- Mean/std trong cả 3 checkpoint khớp chính xác thống kê của toàn bộ cache train, không lấy thống kê pilot hoặc test.
- Mỗi head có đủ lịch sử 1.200 loss; checkpoint load được với key `reg` và `weights_only=True`.
- 180 dòng đánh giá = 15 ảnh × 4 điều kiện × 3 seed; 12 dòng summary được tính lại và khớp CSV từng ảnh.
- Các ảnh dưới ngưỡng đều n=8. Hash source/test/weights từ output khớp bundle đã gửi và pilot; hash Epoch99 giữ nguyên.
- 5 unit tests pass local và trên Kaggle. Xem [verification.json](full/verification.json) để kiểm tra số lượng và checksum checkpoint.

Thời gian experiment thực tế: **206.55 giây** (~3.44 phút), trong đó đọc/kiểm tra dữ liệu, khởi tạo và tính oracle **173.74 giây**. Train head theo seed 0/1/2: 6.76 / 6.49 / 6.51 giây. Không gồm thời gian chờ cấp máy, khởi động Kaggle và tải artifact.

Train MAE theo seed 0/1/2: 0.6205 / 0.6017 / 0.6131 vòng; đây là số đo trên train, không phải validation.

## File kết quả và cách chạy lại

- [Bảng tổng hợp 3 seed](full/aggregate.csv), [bảng từng seed](full/summary.csv), [số đo từng ảnh](full/per_image.csv).
- [Manifest](full/manifest.json), [nhãn oracle train](full/train_oracles.csv), [results JSON](full/results.json), [source hashes](full/source_hashes.json), [metadata Kaggle](full/kernel-metadata.json).
- Checkpoint local: `runs/kaggle-full/full/head_seed0.pth`, `head_seed1.pth`, `head_seed2.pth`. Giữ cả ba, không chọn theo điểm test.
- Cache/loss/ảnh minh họa: `runs/kaggle-full/full/`. Ảnh ghép: input | fixed-8 | adaptive seed 0 | GT; dùng ảnh eval đầu tiên theo tên cố định.
- Log: `runs/kaggle-full/adazero-dce-full-20261019.log`.

```sh
python3 kaggle/build_notebook.py --full
kaggle kernels push -p kaggle/generated-full --accelerator NvidiaTeslaT4 --timeout 3600
kaggle kernels output sangtranminh211/adazero-dce-full-20261019 -p runs/kaggle-full
```

## Giới hạn

Cải tiến định lượng chủ yếu trên GT nhân hệ số, chưa chứng minh cải tiến trên
ảnh thiếu sáng thật. Eval15 chỉ có 15 ảnh; std 3 seed không thay thế kiểm tra
trên dữ liệu độc lập. Chưa benchmark latency end-to-end và chưa thử dữ liệu
ngoài LOLv1. Không tăng epoch, thay threshold hay tune theo kết quả test lần này.

Phải trích dẫn [Zero-DCE](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html)
và [VAR-LIDE](https://arxiv.org/abs/2511.18591) khi viết báo cáo nghiên cứu;
không tuyên bố lần đầu đề xuất adaptive iterations.
