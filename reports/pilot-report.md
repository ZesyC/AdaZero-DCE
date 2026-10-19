# Báo cáo pilot AdaZero-DCE — 19/10/2026

Đã triển khai và chạy thử trên Kaggle. **Chưa train toàn bộ 485 cặp.** Pipeline
đạt mục tiêu tăng PSNR trên GT×0.45/0.6 và giữ nguyên LOL gốc, nhưng có hồi quy
PSNR ở GT×0.3. Không đủ bằng chứng để khẳng định tốt hơn trên ảnh thiếu sáng thật.

## Thí nghiệm thực tế

- [Kaggle pilot, version 1](https://www.kaggle.com/code/sangtranminh211/adazero-dce-pilot-20261019/versions/1): COMPLETE; notebook private, internet off, Tesla T4.
- [LOL dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset): kiểm tra 485 cặp our485 và 15 cặp eval15, tên low/high khớp; kiểm tra hash nội dung không trùng giữa train/test.
- Lấy 96/485 cặp ngẫu nhiên không hoàn lại (seed chọn mẫu 42), sinh 384 mẫu gồm low gốc và GT×0.3/0.45/0.6. Giữ độ phân giải gốc.
- DCE-Net Epoch99 đóng băng, SHA256 `a4395acb874f320375d9704997cef874eaaaaa26a1777ceb29a92b70f74c3612`; không cập nhật trọng số.
- Oracle n=1..8 tính **chỉ từ 96 cặp train**, một forward DCE-Net/input; cache dùng chung cho 3 seed. Không sinh oracle trên eval15.
- MLP 10→128→64→1, 9,729 tham số, ReLU, Dropout 0.1; SmoothL1, Adam 0.005, CosineAnnealing, full batch 1200 epoch; seed 0/1/2.
- Chuẩn hóa mean/std từ 384 mẫu train. Input mean <0.10 luôn n=8; các ảnh còn lại round/clamp đầu ra về [1,8].
- Không dùng eval15 để chọn head, epoch, threshold hay seed. Báo cả ba seed; không chọn seed tốt nhất.

## Kết quả

PSNR tính RGB float [0,1], không resize/crop hay lượng tử hóa ảnh trước khi đo.
SSIM chính: skimage, RGB channel average, cửa sổ đều 7×7, sample covariance.
Trung bình metric từng ảnh; adaptive là **mean ± population std qua 3 seed**.
Độ lệch chuẩn này chỉ đo độ nhạy với seed train, không phải khoảng tin cậy trên ảnh.

| Input eval15 | Fixed-8 PSNR | Adaptive PSNR | Δ dB | Fixed-8 SSIM | Adaptive SSIM | n TB |
|---|---:|---:|---:|---:|---:|---:|
| LOL low gốc | 14.8611 | 14.8611 ± 0.0000 | 0.0000 | 0.5627 | 0.5627 | 8.00 |
| GT×0.3 | 18.3276 | 18.0065 ± 0.0397 | **−0.3210** | 0.8916 | 0.8962 | 7.18 |
| GT×0.45 | 17.2052 | 18.1086 ± 0.0386 | **+0.9034** | 0.8672 | 0.8985 | 6.62 |
| GT×0.6 | 16.4163 | 19.4426 ± 0.0646 | **+3.0264** | 0.8537 | 0.9113 | 6.18 |

Ở GT×0.3, các seed giảm 0.2930–0.3771 dB. Ở GT×0.45 tăng
0.8762–0.9579 dB; GT×0.6 tăng 2.9703–3.1169 dB.

Đếm ảnh theo ΔPSNR trung bình qua seed:

| Điều kiện | Tăng | Bằng | Giảm |
|---|---:|---:|---:|
| GT×0.3 | 3 | 5 | 7 |
| GT×0.45 | 7 | 2 | 6 |
| GT×0.6 | 15 | 0 | 0 |

15/15 ảnh LOL gốc đều dưới ngưỡng tối, nên giữ nguyên fixed-8 bằng cơ chế bảo vệ.
Điều này xác nhận cơ chế hoạt động, **không chứng minh head học tốt hơn trên ảnh
LOL thật**. Với GT×0.3, ảnh 79.png giảm trung bình khoảng 3.07 dB, 55.png giảm
2.47 dB. Không nên chỉ báo trung bình của hai điều kiện thuận lợi.

## SSIM chưa tái lập được số trong plan

PSNR baseline tái lập đúng 14.86 dB, nhưng SSIM 0.5927 trong plan **chưa tái
lập được**. [Audit Kaggle độc lập](https://www.kaggle.com/code/sangtranminh211/adazero-dce-ssim-audit-20261019)
đã COMPLETE, chỉ đánh giá lại checkpoint, không train hoặc tune head. Bốn
quy ước phổ biến cho fixed-8 trên LOL gốc cho:

| Quy ước | SSIM |
|---|---:|
| RGB uniform 7×7 (giao thức chính) | 0.562653 |
| RGB Gaussian 11×11, sigma 1.5, population covariance | 0.559149 |
| Grayscale BT.709 uniform 7×7 | 0.672043 |
| Grayscale BT.709 Gaussian 11×11 | 0.670413 |

Chưa có mã đo gốc của con số 0.5927, nên chưa xác định được nguồn chênh lệch.
Không khẳng định chỉ do cửa sổ hoặc chuyển grayscale: cả bốn cách trên đều
không khớp. Giữ nguyên giao thức chính đã định trước và báo rõ khác biệt; không
điều chỉnh metric để khớp số kỳ vọng. Với giao thức chính, audit khớp cả 12
kết quả adaptive của run đầu (sai số <1e-7); GT×0.45/0.6 đều tăng SSIM qua
cả bốn quy ước đã kiểm tra. Các bảng audit nằm trong `reports/pilot/ssim_*`.

## Chi phí và kiểm chứng

- Tính oracle, đọc/kiểm tra dữ liệu và khởi tạo: 46.46 giây.
- Train head từng seed: 2.83 / 2.81 / 2.86 giây, trên CPU của Kaggle.
- Tổng chạy experiment: 68.26 giây, không gồm thời gian chờ cấp máy, khởi động Kaggle và tải artifact.
- Train MAE seed 0/1/2: 0.576 / 0.581 / 0.553 vòng. Đây là train MAE, không phải validation MAE.
- 5 unit tests pass local và trên Kaggle: feature/validation, threshold/round/clamp/checkpoint, curve prefixes/oracle/inference, seed reproducibility, pairing/leakage guard.
- Đầu ra refactor model khớp từng giá trị với code gốc trên tensor thử: output vòng 4, vòng 8 và alpha maps. `git diff --check` và Python compile pass.
- Tải đủ 3 checkpoint, load an toàn với `weights_only=True`, smoke test inference ảnh 600×400 local thành công.
- Hash cả 6 file code/test/trọng số từ output Kaggle khớp bundle gửi lên.

Số vòng giảm không đồng nghĩa tăng tốc toàn mạng theo cùng tỷ lệ: DCE-Net vẫn
forward một lần. Chưa benchmark latency/VRAM end-to-end, không tuyên bố speedup.

## Quyết định train đủ dữ liệu

**Đề xuất train đủ 485 cặp nếu mục tiêu là hoàn tất thực nghiệm học thuật**:
pilot cho hiệu quả nhất quán ở GT×0.45/0.6 và chi phí head rất nhỏ. Đó là một
thí nghiệm tiếp theo để đo tác động của cỡ tập train; không bảo đảm khắc phục
GT×0.3 hay tăng điểm thêm. Giữ nguyên cấu hình và báo cả 3 seed, cả bốn điều kiện.

Nếu mục tiêu là cải thiện ảnh thiếu sáng thật, pilot hiện chưa cung cấp lý do
mạnh để train đủ: LOL gốc chỉ bằng baseline. Cần thêm dữ liệu thật có phân bố
độ sáng đa dạng để kiểm tra khả năng khái quát.

Chỉ có một subset seed, 15 ảnh test và chưa có tập validation riêng; chưa đo
độ nhạy theo cách lấy subset. Không điều chỉnh mô hình theo các kết quả eval15
này rồi coi eval15 là kiểm định độc lập mới. Nếu muốn tune cấu hình tiếp, tách
validation theo **cặp ảnh gốc** từ our485 trước khi augmentation.

Lệnh full đã hỗ trợ nhưng **chưa chạy**:

```sh
python3 AdaZero-DCE_code/experiment.py --data /path/to/lol_dataset --output runs/full --train-pairs 485 --epochs 1200 --seeds 0 1 2
```

## Artifact

- `reports/pilot/`: manifest, results.json, bảng tổng hợp, metric từng ảnh, nhãn oracle train, hash mã nguồn.
- `runs/kaggle-pilot/pilot/`: 3 checkpoint `head_seed{0,1,2}.pth` (key `reg`), cache feature/target, loss history, ảnh minh họa và bản metric gốc tải từ Kaggle.
- `runs/kaggle-pilot/adazero-dce-pilot-20261019.log`: log thực thi Kaggle.
- Ảnh minh họa: `runs/kaggle-pilot/pilot/example_factor{1.0,0.3,0.45,0.6}.png`; từ trái sang phải: input | fixed-8 | adaptive seed 0 | GT. Chọn ảnh đầu theo tên cố định, không chọn theo điểm tốt nhất.

## Bối cảnh nghiên cứu

Phải trích dẫn [Guo et al., Zero-DCE, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html)
và [Dong et al., VAR-LIDE, arXiv:2511.18591](https://arxiv.org/abs/2511.18591).
VAR-LIDE đã có adaptive curve estimation dùng visibility từ VLM trong hệ thống
joint enhancement/deblurring. Khác biệt triển khai ở đây là feature độ sáng
rẻ, head hồi quy có nhãn oracle, bảo vệ ảnh rất tối và enhancement độc lập.
Không tuyên bố tính mới là lần đầu dùng số vòng thích ứng; không suy diễn việc
một thành phần không xuất hiện từ riêng abstract của paper.
