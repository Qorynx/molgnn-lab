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

Framework không khóa một CUDA toolkit cụ thể. Nếu cần bản PyTorch dành riêng cho CPU
hoặc một phiên bản CUDA nhất định, hãy cài bản PyTorch phù hợp vào environment trước,
sau đó chạy lệnh cài đặt trên; `pip` sẽ giữ lại bản đã cài nếu thỏa khoảng phiên bản
khai báo trong `pyproject.toml`.

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

Chạy benchmark nhiều model:

```bash
molgnn benchmark --config benchmark.yaml
```

Lệnh `benchmark` dùng danh sách `models` trong YAML và chạy từng model trên mọi
seed đã khai báo. Nếu bỏ `models`, nó chạy toàn bộ built-in model có benchmark
được bật. `molgnn train` vẫn giữ luồng single-model tương thích ngược và hỗ trợ
hai runtime hook cục bộ.

Kết quả được ghi vào `runs/example_gcn_regression/`. Thư mục output, checkpoint và
dataset local đều được Git bỏ qua.

## Các mô hình có sẵn

- `gcn_baseline`
- `attentivefp`
- `dmpnn`
- `hignn`
- `himnet`
- `molecular_graph_embedding`
- `trimnet_2020`

Tên mô hình được đặt tại `model.name`; tham số kiến trúc được truyền qua
`model.parameters`.

## Kiểm tra input contract

Lệnh `describe-model` chỉ in ra contract runtime công khai của một model đã
đăng ký: tensor bắt buộc, helper transform tùy chọn, prediction reducer và
benchmark metadata. Nó không chứa tài liệu nội bộ hay metadata provenance;
do đó an toàn để dùng trong package/app phát hành. Lệnh chỉ đọc metadata,
không khởi tạo training run và không thay đổi model hay featurizer.

```bash
molgnn describe-model --model hignn
molgnn describe-model --model dmpnn --format json
```

Output `text` phù hợp để đọc nhanh trong terminal. Output `json` phù hợp để
tích hợp vào tooling hoặc kiểm tra programmatically.

Các `graph_transform_name` dưới đây là helper có sẵn của project, không phải
feature pipeline bắt buộc. Featurizer bên ngoài có thể tạo tensor trực tiếp và
bỏ qua helper, miễn batch cuối cùng đáp ứng đúng required fields và ngữ nghĩa
graph của model.

| Nhóm model | Required batch fields | Helper bundled (tùy chọn) |
| --- | --- | --- |
| `gcn_baseline` | `x`, `edge_index`, `batch` | Không có |
| `attentivefp`, `trimnet_2020` | `x`, `edge_index`, `edge_attr`, `batch` | Không có |
| `dmpnn` | `x`, `edge_index`, `edge_attr`, `reverse_edge_index`, `batch` | `directed_edges` thêm reverse-edge map |
| `hignn` | `x`, `edge_index`, `edge_attr`, `brics_edge_index`, `brics_edge_attr`, `atom_to_fragment`, `batch` | `brics_fragments` thêm BRICS fragment view |
| `himnet` | `himnet_x`, `himnet_edge_index`, `himnet_edge_attr`, `himnet_reverse_edge_index`, `himnet_node_batch`, `himnet_node_type`, `himnet_fp` | `himnet_inputs` thêm unified hierarchy và fingerprint views |
| `molecular_graph_embedding` | `mge_x`, `edge_index`, `mge_edge_attr`, `batch` | `coley_2017_features` thêm feature tensors cho MGE |

Điểm chung là `edge_index` và `batch` luôn mô tả graph batch thực tế; tên và
số chiều feature phụ thuộc từng core architecture. Khi dùng custom featurizer,
hãy xem output của `describe-model` thay vì giả định canonical feature schema
của project là bắt buộc.

- Với D-MPNN, `reverse_edge_index` phải map mỗi directed edge sang đúng cạnh
  đảo chiều và phải là involution: `reverse_edge_index[reverse_edge_index]`
  trả về từng edge ban đầu.
- Với HiGNN, `brics_edge_index`/`brics_edge_attr` là graph giữ lại sau khi cắt
  BRICS; `atom_to_fragment` gán mỗi atom vào connected-component fragment của
  chính molecule đó.
- Với MGE, helper bundled tạo `mge_x`/`mge_edge_attr` theo default 32/8. Custom
  feature schema vẫn hợp lệ nếu đồng thời đặt `input_atom_dim`/`input_bond_dim`
  của model khớp với tensor cung cấp.

## Hook tùy biến cho `molgnn train`

CLI có hai điểm nối Python cục bộ để dùng feature pipeline hoặc cách train riêng
mà không thêm plugin schema vào YAML:

```powershell
molgnn train --config experiment.yaml `
  --featurizer .\my_featurizer.py:featurize `
  --training-strategy .\my_training_strategy.py:fit
```

Mỗi selector có dạng `path.py:top_level_callable` (hoặc
`dotted.module:top_level_callable`). Không truyền hai option này giữ nguyên hoàn
toàn luồng mặc định: canonical RDKit featurizer, helper transform tương ứng model,
AdamW và fit loop có sẵn. `validate-config` không import hay thực thi hook.

Hook được thực thi như Python cục bộ đáng tin cậy; chỉ dùng file do bạn kiểm soát.
Selector đã dùng cũng được ghi vào `runtime_hooks` trong config artifact của run.

Featurizer nhận một SMILES hợp lệ cùng label đã parse và phải trả về
`molgnn.data.MolecularData`:

```python
from molgnn.featurizer import featurize_smiles


def featurize(smiles, *, targets, target_mask, sample_id):
    data = featurize_smiles(
        smiles,
        targets=targets,
        target_mask=target_mask,
        sample_id=sample_id,
    )
    # Thay data.x/data.edge_attr hoặc thêm các fields riêng của model tại đây.
    return data
```

Để giữ CSV parsing, split, target scaling và PyG batching thống nhất, output phải
thỏa shared sample contract: `x`, `edge_index`, `edge_attr`, `y`, `y_mask`,
`sample_id`, với feature width ổn định cho toàn dataset. Runner tự gắn lại
`smiles`. Với D-MPNN, HiGNN hoặc MGE, hook hoặc cung cấp **đầy đủ** các derived
fields bắt buộc của model, hoặc không cung cấp field nào để helper bundled của
project tạo chúng; cung cấp dở dang sẽ lỗi sớm.

Training strategy thay optimizer và fit loop, nhưng runner vẫn quản lý seed,
DataLoader, checkpoint, đánh giá test và artifact. Cách ít rủi ro nhất là tái sử
dụng fit loop chuẩn rồi chỉ đổi optimizer/scheduler:

```python
import torch

from molgnn.trainer import StrategyResult, fit as default_fit


def fit(model, loaders, task_adapter, training, *, device, target_names, on_epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    result = default_fit(
        model,
        loaders,
        optimizer,
        task_adapter,
        epochs=training.epochs,
        patience=training.patience,
        monitor=training.monitor,
        monitor_mode=training.monitor_mode,
        device=device,
        target_names=target_names,
        callbacks=(on_epoch,),
    )
    return StrategyResult(result, optimizer.state_dict())
```

Một fit loop tự viết cũng hợp lệ nếu trả về `StrategyResult` chứa `FitResult` hợp
lệ và gọi `on_epoch` cho mỗi epoch để giữ loss/metrics history đầy đủ.

## Chia dữ liệu

`data.split` hỗ trợ:

- `random`: random split tái lập theo seed.
- `scaffold`: Bemis–Murcko scaffold split tương thích Chemprop/astartes; các phân tử
  cùng scaffold luôn nằm trong cùng partition.
- `predefined`: đọc nhãn `train`, `validation` hoặc `test` từ cột được chỉ định bởi
  `data.split_column`.

Tỷ lệ được khai báo theo thứ tự train/validation/test trong `data.split_ratios`.

`data.split_seed_mode` quyết định cách tái sử dụng split khi khai báo nhiều
`experiment.seeds`:

- `first_experiment_seed` (mặc định): tạo một split từ seed đầu tiên và dùng
  chung cho mọi seed.
- `per_experiment_seed`: tạo lại split và target scaler theo từng seed; các
  model tại cùng một seed vẫn dùng chung split đó.

Với `predefined`, partition lấy từ CSV nên policy này không thay đổi split.

## Task và metric

Regression dùng `loss: mse` và hỗ trợ `rmse`, `mae`, `r2`. Binary classification dùng
`loss: bce_with_logits` và hỗ trợ `roc_auc`, `prc_auc`, `accuracy`,
`balanced_accuracy`. Target scaling chỉ áp dụng cho regression và được fit trên tập
train.

## Output

Mỗi experiment lưu config đã resolve, split assignment và metric tổng hợp. Mỗi seed
có thư mục riêng chứa `split.csv`, checkpoint tốt nhất/gần nhất, lịch sử huấn luyện
và dự đoán trên tập test. Có thể khai báo nhiều seed bằng `experiment.seeds`.

Các artifact chính của mỗi seed:

- `loss_history.csv`: optimization loss, train evaluation loss và validation loss theo
  từng epoch.
- `metrics_history.csv`: toàn bộ train/validation metrics theo epoch, learning rate và
  thời gian chạy.
- `test_history.csv`: test loss và toàn bộ test metrics của best checkpoint.
- `test_predictions.csv`: đúng ba cột `smiles`, `target`, `prediction`; multitask dùng
  JSON array theo thứ tự `data.target_columns`.

Ở cấp experiment, `summary.csv` lưu kết quả test của từng seed và
`aggregate_metrics.json` lưu mean/std/min/max cùng danh sách giá trị hữu hạn của từng
metric qua các seed.

## Phạm vi runtime

Package runtime chỉ chứa implementation, input contract và validation để người
dùng có thể thay featurizer hoặc training strategy mà không phải mang theo tài
liệu nội bộ. Nó không tuyên bố tái lập feature schema, training protocol hay
kết quả benchmark cụ thể.

Canonical runner biểu diễn mỗi liên kết phân tử bằng hai cạnh ngược chiều và
không đưa self-loop vào input. Mọi model chặn cạnh nối giữa các sample trong
cùng batch; các model có phương trình nguồn tự cộng trạng thái node cũng chặn
self-loop, còn `dmpnn` kiểm tra thêm reverse-edge map. Nếu dùng featurizer riêng,
hãy giữ các quy ước này khi muốn dùng đúng input contract; `gcn_baseline` vẫn giữ
hành vi PyG hợp lệ với đồ thị directed/self-loop.
