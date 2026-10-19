# AdaZero-DCE — khái quát trên LOLv2-real test, 19/10/2026

Đã đánh giá đủ **100 cặp LOLv2-real test**, dùng nguyên **3 head seed 0/1/2
đã train trên toàn bộ LOLv1 our485**, không train/fine-tune lại. Kết quả:
**adaptive giảm PSNR ở cả 3 seed, SSIM tăng rất nhẹ**. Chưa có bằng chứng
phương pháp cải thiện chất lượng đồng thời cả hai metric khi chuyển sang LOLv2-real.

## Nguồn dữ liệu và phạm vi tải

Trang [BMVC2018 do người dùng cung cấp](https://daooshee.github.io/BMVC2018website/)
là trang LOL ban đầu. [Repo của nhóm tác giả](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)
có liên kết LOL-v2 (extension, Sparse Gradient Regularized Deep Retinex Network,
TIP 2021), gồm [bản Google Drive](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view).
Lần chạy dùng [mirror Kaggle tanhyml/lol-v2-dataset](https://www.kaggle.com/datasets/tanhyml/lol-v2-dataset),
tải riêng file test qua API; không tải toàn bộ archive hay gắn toàn bộ dataset.
Đây là mirror, không phải tải trực tiếp từ Google Drive của tác giả; chưa đối
chiếu từng checksum với archive trên Drive.

Allowlist gồm đúng 200 file `LOL-v2/Real_captured/Test/Low/low00690.png` …
`low00789.png` và `Test/Normal/normal00690.png` … `normal00789.png`. Ghép cặp
theo ID 00690–00789, xác nhận đúng 100 low và 100 normal. Tổng 75.970.040 byte.
Không có URL `/Train/` trong danh sách tải; không đọc ảnh LOLv2 train. Mỗi file
được kiểm tra size, đọc xác thực PNG, lưu SHA256. Xem [source_files.csv](lolv2-real/source_files.csv)
và [downloads.json](lolv2-real/downloads.json). Ảnh test tạm đã xóa khỏi output
notebook sau đánh giá, giữ lại checksum và metric để truy vết.

## Thực thi và cấu hình

[Kaggle evaluation, version 1](https://www.kaggle.com/code/sangtranminh211/adazero-dce-lolv2-real-test-20261019/versions/1)
đã COMPLETE, private, Tesla T4; internet bật chỉ để tải các file test.
Nguồn checkpoint/code là output notebook full LOLv1 trước đó, qua kernel source.

- DCE-Net Epoch99 và cả ba head đều eval mode, requires_grad=False, inference_mode; không gọi trainer/optimizer, không tính oracle.
- Kiểm tra checksum head khớp bản LOLv1 full, checksum backbone/code khớp bản trước; hash checkpoint trước/sau inference không đổi.
- Giữ nguyên normalization mu/sd từ LOLv1 full, không tính lại theo LOLv2.
- Mean <0.10 → n=8; còn lại round/clamp head về [1,8], giống eval15. Không đổi threshold theo dữ liệu test.
- Chạy trực tiếp low gốc, không tạo các mức sáng giả lập LOLv2.
- Một forward DCE-Net/input lấy alpha maps; fixed-8 và adaptive lấy đúng prefix tương ứng. Dự đoán n không sử dụng GT.
- GT chỉ dùng tính metric. Giữ nguyên kích thước RGB 600×400, float32 [0,1], không resize/crop/GT mean correction hay lượng tử hóa trước khi đo.

## Kết quả trung bình 100 ảnh

Dùng chính `experiment.metrics` của LOLv1: PSNR RGB data_range=1; SSIM
skimage channel_axis=2, cửa sổ đều 7×7, sample covariance, data_range=1.
Hai phương pháp dùng cùng implementation. Trung bình metric từng ảnh;
Δ = adaptive − fixed-8, không đối chiếu SSIM tuyệt đối với plan.

| Phương pháp | PSNR | Δ PSNR, dB | SSIM | Δ SSIM |
|---|---:|---:|---:|---:|
| Fixed-8 | 18.0598 | — | 0.579800 | — |
| Adaptive seed 0 | 17.8628 | -0.1969 | 0.581476 | +0.001676 |
| Adaptive seed 1 | 17.8759 | -0.1839 | 0.581646 | +0.001846 |
| Adaptive seed 2 | 17.8340 | -0.2258 | 0.580941 | +0.001141 |
| Adaptive trung bình 3 seed | 17.8576 | **-0.2022** | 0.581354 | **+0.001555** |

Độ lệch chuẩn qua seed: PSNR **0.0175 dB**, SSIM **0.000300** (population std của 3 seed, không phải khoảng tin cậy trên 100 ảnh).

## Ngưỡng tối và phân bố n

**87/100 ảnh** có mean <0.10, nên cả ba seed dùng n=8 và khớp fixed-8 chính xác
trên những ảnh đó. Chỉ 13 ảnh còn lại sử dụng quyết định của head khác fixed-8.
Với mỗi seed: **2 ảnh tăng PSNR, 87 ảnh bằng, 11 ảnh giảm**.

| Seed | n=4 | n=5 | n=6 | n=7 | n=8 | n TB |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 2 | 4 | 7 | 87 | 7.79 |
| 1 | 2 | 0 | 0 | 11 | 87 | 7.81 |
| 2 | 2 | 0 | 2 | 9 | 87 | 7.79 |

Riêng 13 ảnh không bị guard, ΔPSNR trung bình theo seed 0/1/2 là
−1.5150 / −1.4144 / −1.7372 dB. Mức giảm PSNR trên toàn tập được giảm nhẹ
nhờ 87 ảnh giữ nguyên baseline, không nên diễn giải delta toàn tập nhỏ là head
khái quát tốt trên các ảnh nó thực sự thay đổi n.

Dữ liệu này cho thấy cải tiến trên các mức GT nhân hệ số của LOLv1 chưa chuyển
thành cải thiện PSNR trên ảnh thật LOLv2-real. SSIM tăng nhẹ không bù cho kết
luận riêng về PSNR; không tuyên bố tốt hơn fixed-8 nói chung. Chưa có thí nghiệm
phân lập nguyên nhân, nên không quy hoàn toàn thất bại này cho một đặc trưng hay
một tham số cụ thể.

## Kiểm chứng và artifact

- Đúng 100 cặp, 300 dòng adaptive (100×3 seed), không thiếu/trùng ID.
- 200 URL tải đều thuộc Test; size và checksum trong download manifest khớp file inference.
- Checkpoint 3 seed khớp [checksum LOLv1 full](full/verification.json); backbone và head giữ nguyên hash trước/sau.
- Tính lại toàn bộ summary từ CSV từng ảnh khớp số đã báo; mọi ảnh dưới ngưỡng đều n=8 và delta bằng 0.
- Hai kiểm thử mới pass local và trên Kaggle: ghép cặp/thiếu cặp và tính nhất quán của metric. Không chạy test train-head trong notebook evaluation.
- Python compile và `git diff --check` pass; đã xem ảnh minh họa input | fixed-8 | adaptive seed 0 | GT, ảnh đầu theo ID, không chọn theo điểm.

Thời gian inference/đo metric: **28.76 giây**, không gồm thời gian cấp máy, tải ảnh hay xuất/tải notebook.

File kết quả:

- [Từng seed](lolv2-real/summary.csv), [từng ảnh](lolv2-real/per_image.csv), [aggregate](lolv2-real/aggregate.json).
- [Manifest/checksum](lolv2-real/manifest.json), [verification](lolv2-real/verification.json), [results JSON](lolv2-real/results.json).
- [Download manifest](lolv2-real/downloads.json), [allowlist](lolv2-real/source_files.csv), [metadata mirror](lolv2-real/source_metadata.json).
- [Hash code evaluation](lolv2-real/source_hashes.json), [metadata notebook](lolv2-real/kernel-metadata.json).
- Log local: `runs/kaggle-lolv2/adazero-dce-lolv2-real-test-20261019.log`.
- Ảnh minh họa local: `runs/kaggle-lolv2/lolv2-results/example.png`.

```sh
python3 kaggle/build_lolv2_notebook.py
kaggle kernels push -p kaggle/generated-lolv2 --accelerator NvidiaTeslaT4 --timeout 1800
kaggle kernels output sangtranminh211/adazero-dce-lolv2-real-test-20261019 -p runs/kaggle-lolv2
```

Chưa thử dữ liệu ngoài LOLv1/LOLv2-real, chưa benchmark latency end-to-end;
không dùng kết quả test này để chỉnh head hay threshold trong lần chạy hiện tại.
