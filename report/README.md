# Báo cáo bài tập lớn AdaZero-DCE

Bìa và cách tổ chức theo report_mau.pdf: Phenikaa, mục lục, các mục đánh số,
đoạn phân tích và nhận xét sau bảng/hình. Sinh viên Trần Minh Sang, Ngô Quang Thiện;
giảng viên TS. Đặng Thị Thúy An; lớp Mạng nơron và học sâu-1-1-26(N02).

Số liệu: `../notebook/execution/adazero-notebook-results/`, chạy mới trên Kaggle.
Các bảng LaTeX được xuất từ CSV; hình lấy từ kết quả chương trình thực nghiệm. Loss là lịch sử
train, không phải một lần train mới. Logo trích từ PDF mẫu do người dùng cung cấp.

Mở `main.tex` hoặc thư mục này trong trình biên tập LaTeX. Biên dịch bằng XeLaTeX
hai lần từ thư mục `report/`. Nguồn sử dụng font hỗ trợ tiếng Việt.
Bản PDF cuối: [AdaZero-DCE_Bao-cao.pdf](AdaZero-DCE_Bao-cao.pdf), nằm cùng
nguồn LaTeX trong thư mục `report/`.

Để biên dịch trực tiếp ra đúng tên PDF (chạy hai lần):

```sh
xelatex -interaction=nonstopmode -halt-on-error -jobname=AdaZero-DCE_Bao-cao main.tex
```
