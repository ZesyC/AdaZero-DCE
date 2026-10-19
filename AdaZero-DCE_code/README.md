# Thành phần nền dùng trong AdaZero-DCE

Thư mục giữ cấu trúc mã kế thừa để không phá import, checkpoint và các notebook
Kaggle đã chạy. Các entry point của AdaZero-DCE đã được gom vào đây; mã baseline vẫn giữ nguyên.

| File | Vai trò trong dự án hiện tại |
|---|---|
| `experiment.py`, `infer.py`, `evaluate_lolv2.py` | Train head, inference và đánh giá LOLv2 của AdaZero-DCE. |
| `model.py` | Backbone DCE-Net đang được pipeline AdaZero-DCE gọi để sinh tám bản đồ đường cong. |
| `snapshots/Epoch99.pth` | Checkpoint backbone đóng băng, đang sử dụng trực tiếp. |
| `lowlight_train.py` | Trainer backbone kế thừa; không dùng để train head AdaZero-DCE. |
| `lowlight_test.py` | Script test baseline kế thừa; không phải entry point inference thích ứng. |
| `dataloader.py`, `Myloss.py` | Các thành phần hỗ trợ trainer baseline, giữ nguyên để bảo toàn mã kế thừa. |

## Entry point của AdaZero-DCE

Chạy từ thư mục gốc repository:

- Train head: `AdaZero-DCE_code/experiment.py --train-pairs 485 --epochs 1200 --seeds 0 1 2`
  cùng các tham số đường dẫn trong [README chính](../README.md).
- Inference: `AdaZero-DCE_code/infer.py` với head đã train.
- Đánh giá LOLv2: `AdaZero-DCE_code/evaluate_lolv2.py`.
- Đọc toàn bộ thí nghiệm: [notebook có outputs](../notebook/AdaZero_DCE_Complete_Evaluation_Executed.ipynb).

Không chạy `lowlight_train.py` để tái lập kết quả báo cáo: script này train
backbone, còn thí nghiệm hiện tại chỉ train head và giữ Epoch99 đóng băng.
Các script đã chuyển thư mục và cập nhật import; thuật toán và checkpoint giữ nguyên.

Nguồn nền: Guo et al., Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement, CVPR 2020; https://github.com/Li-Chongyi/Zero-DCE.
Giữ nguyên thông tin tác giả và điều kiện Attribution-NonCommercial 4.0 của mã nền.
