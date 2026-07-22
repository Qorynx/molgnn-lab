# molgnn-lab

`molgnn-lab` là framework dùng chung để huấn luyện và so sánh các mô hình graph neural
network trên đồ thị phân tử 2D. Framework thống nhất quá trình đọc dữ liệu, tạo đặc
trưng nguyên tử/liên kết, chia dữ liệu, huấn luyện, đánh giá và lưu kết quả; kiến trúc
mô hình được chọn bằng file YAML.

## Yêu cầu

- Python 3.11
- Windows hoặc Linux
- GPU là tùy chọn; framework có thể chạy trên CPU

## Cài đặt

Từ thư mục gốc của source code:

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Hoặc trên Linux:

```bash
source .venv/bin/activate
```

Cài framework và toàn bộ dependency runtime:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
molgnn --version
```

## Chuẩn bị dữ liệu

Dataset không được phân phối kèm repository. Người dùng cung cấp file CSV riêng với
ít nhất một cột SMILES và một cột target. Ví dụ cấu trúc cho regression:

```text
smiles,target
<molecular SMILES>,<numeric value>
```

Với binary classification, target phải là `0` hoặc `1`. Có thể khai báo nhiều cột
target trong `data.target_columns`. Tạo thư mục `data/` ở root và đặt dataset tại
`data/dataset.csv`, hoặc sửa `data.path` trong config để trỏ tới vị trí khác.

## Chạy thử

Repository cung cấp một config độc lập tại `configs/example.yaml`. Config này dùng
GCN baseline cho bài toán regression và random split `80/10/10`.

Kiểm tra config:

```bash
molgnn validate-config --config configs/example.yaml
```

Huấn luyện:

```bash
molgnn train --config configs/example.yaml
```

Kết quả được ghi vào `runs/example_gcn_regression/`. Thư mục output, checkpoint và
dataset local đều được Git bỏ qua.

## Các mô hình có sẵn

- `gcn_baseline`
- `attentivefp`
- `dmpnn`
- `hignn`
- `molecular_graph_embedding`
- `trimnet_2020`

Tên mô hình được đặt tại `model.name`; tham số kiến trúc được truyền qua
`model.parameters`.

## Chia dữ liệu

`data.split` hỗ trợ:

- `random`: random split tái lập theo seed.
- `scaffold`: Bemis–Murcko scaffold split tương thích Chemprop/astartes; các phân tử
  cùng scaffold luôn nằm trong cùng partition.
- `predefined`: đọc nhãn `train`, `validation` hoặc `test` từ cột được chỉ định bởi
  `data.split_column`.

Tỷ lệ được khai báo theo thứ tự train/validation/test trong `data.split_ratios`.

## Task và metric

Regression dùng `loss: mse` và hỗ trợ `rmse`, `mae`, `r2`. Binary classification dùng
`loss: bce_with_logits` và hỗ trợ `roc_auc`, `prc_auc`, `accuracy`,
`balanced_accuracy`. Target scaling chỉ áp dụng cho regression và được fit trên tập
train.

## Output

Mỗi experiment lưu config đã resolve, split assignment và metric tổng hợp. Mỗi seed
có thư mục riêng chứa checkpoint tốt nhất/gần nhất, lịch sử huấn luyện và dự đoán trên
tập test. Có thể khai báo nhiều seed bằng `experiment.seeds`.
