# Notebook trình bày AdaZero-DCE

- `AdaZero_DCE_Complete_Evaluation.ipynb`: nguồn, 19 cell (10 code).
- `AdaZero_DCE_Complete_Evaluation_Executed.ipynb`: bản chạy thực tế trên Kaggle,
  đủ execution_count, 36 output blocks, 8 hình, không có cell lỗi.
- `execution/`: giữ CSV, hình, manifest và log tải từ Kaggle. Bản notebook
  trùng lặp và HTML xuất lại đã được dọn; notebook chính ở ngay thư mục này
  vẫn giữ nguyên outputs thực tế, không điền thủ công.

[Kaggle run](https://www.kaggle.com/code/sangtranminh211/adazero-dce-complete-evaluation-report)
riêng tư, GPU T4, hoàn tất ngày 19/10/2026. Giữ nguyên ba head full LOLv1.
Không train lại. Biểu đồ train đọc từ artifact trước; các metric test được tính mới.
Cấu trúc trình bày theo notebook mẫu GTSRB: mục đánh số, giải thích trước code,
bảng và hình ngay sau từng bước; nội dung chuyển sang bài toán AdaZero-DCE.

## Kết quả và kiểm tra

- 180 bản ghi LOLv1: 15 ảnh × 4 điều kiện × 3 seed.
- 300 bản ghi LOLv2-real: 100 ảnh × 3 seed.
- Đúng 200 file LOLv2 Test, không tải Train.
- Cell cuối PASS: khớp kết quả trước trong atol=1e-6, trọng số không thay đổi.
- `execution/adazero-notebook-results/`: bảng mean/std/per-image và hình cho báo cáo.

## Chạy lại

```sh
kaggle kernels push -p notebook/kaggle --accelerator NvidiaTeslaT4 --timeout 1800
kaggle kernels status sangtranminh211/adazero-dce-complete-evaluation-report
kaggle kernels output sangtranminh211/adazero-dce-complete-evaluation-report -p notebook/execution-new
```

`kaggle/runner.ipynb` chứa notebook nguồn và thực thi bằng nbclient trên Kaggle,
rồi xuất bản có outputs thành file `.ipynb` riêng. Cách này tránh nhầm bản source
lấy qua `kernels pull` với notebook đã thực thi. Hai `kernel_sources` cung cấp
code/checkpoint full và module đánh giá LOLv2; dataset LOLv1 được gắn theo metadata.
Internet bật chỉ để tải từng ảnh LOLv2 Test theo allowlist.
