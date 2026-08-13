# molgnn-lab

`molgnn-lab` là framework dùng chung để huấn luyện và so sánh các mô hình graph neural
network trên đồ thị phân tử 2D và complex ligand–pocket 3D. Framework thống nhất quá trình đọc dữ liệu, tạo đặc
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

`PotentialNet` chạy trực tiếp với `data.source: csv_smiles` như các model 2D khác:
khi sample không có `pos`, model chỉ chạy bond stage rồi readout trên toàn bộ atom.
Nó cũng hỗ trợ `data.source: pdbbind_complex`; khi sample có tọa độ, model chạy thêm
spatial stage trên complex. Với source này, `data.path` là CSV manifest cục bộ có
đường dẫn ligand (`.sdf`, `.mol` hoặc `.mol2`) và pocket/protein (`.pdb`), các
target, và tùy chọn `complex_id`/split. Đường dẫn tương đối được tính từ thư mục của
manifest. Ví dụ phần `data` cho complex:

```yaml
data:
  source: pdbbind_complex
  path: ../data/pdbbind/manifest.csv
  ligand_path_column: ligand_file
  protein_path_column: pocket_file
  id_column: complex_id
  target_columns: [affinity]
  split: predefined
  split_column: split
  strip_hydrogens: true
```

Không có cơ chế tự tải PDBBind. Manifest làm cho nguồn cấu trúc, split và atom order
được khai báo rõ ràng và tái lập được.

## Chạy thử

Repository cung cấp một config độc lập tại `configs/example.yaml`. Config này dùng
GCN baseline cho bài toán regression và random split `80/10/10`.

Kiểm tra config:

```bash
molgnn validate-config --config configs/example.yaml
```

Hai smoke config nhẹ cho các kiến trúc attention/message-passing mới là
`configs/ampnn_smoke.yaml` và `configs/emnn_smoke.yaml`; chúng dùng fixture
regression nhỏ của repository. Kiểm tra bằng lệnh trên hoặc chạy bằng
`molgnn train --config <smoke-config>`.

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
- `ampnn`
- `dimenet`
- `dmpnn`
- `emnn`
- `gpspp`
- `hignn`
- `himnet`
- `molecular_graph_embedding`
- `mpnn`
- `mpnn_3d_distance_bins`
- `potentialnet`
- `resgat`
- `fragnet`
- `trimnet_2020`
- `weave`

Tên mô hình được đặt tại `model.name`; tham số kiến trúc được truyền qua
`model.parameters`.
`ampnn` và `emnn` là các kiến trúc sparse 2D: chúng không yêu cầu hoặc dùng
`pos`, kể cả khi sample cung cấp tọa độ.
`potentialnet` được chọn tường minh qua `model.name` cho cả `csv_smiles` và
`pdbbind_complex`. Nó vẫn nằm ngoài benchmark mặc định để benchmark không tự trộn
ngữ nghĩa ligand-2D và complex-3D; chạy benchmark có model này cần khai báo danh sách
`models` rõ ràng.
`mpnn_3d_distance_bins` yêu cầu mỗi sample có `pos` float32 `[N, 3]` theo Å. Helper
của nó tạo complete graph có hướng không self-loop, gồm bốn bond type và mười
distance bin; vì vậy chi phí là O(N²). Model dùng được với `pdbbind_complex` hoặc
custom featurizer cung cấp đủ tọa độ, và cũng nằm ngoài benchmark mặc định. Readout
của MPNN này dùng toàn bộ atom trong graph, không áp dụng `ligand_mask`.

`dimenet` là model coordinate-backed với contract riêng: `atomic_number`,
`pos`, directed radius graph 5 Å và non-backtracking triplet index theo edge.
Nó không suy atomic number từ canonical `x`, không tạo conformer từ SMILES và
nằm ngoài benchmark mặc định. Structural dataset source native là milestone
riêng; core hiện dùng được với sample chuẩn bị tường minh hoặc custom
featurizer cung cấp tọa độ.

`gpspp` là core hybrid 2D gồm local MPNN, biased global self-attention và
FFN. Helper của nó tạo all-pairs atom view cùng shortest-path distance cho
attention bias; nó không tự suy diễn hoặc dùng tọa độ `pos`. Do global
attention có chi phí O(N²), model này nằm ngoài benchmark mặc định và cần được
chọn tường minh trong `models` khi benchmark.

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
| `ampnn` | `x`, `edge_index`, `ampnn_edge_type`, `batch` | `ampnn_edge_types` thêm nhãn bond type 2D |
| `dmpnn`, `emnn` | `x`, `edge_index`, `edge_attr`, `reverse_edge_index`, `batch` | `directed_edges` thêm reverse-edge map |
| `mpnn` | `x`, `edge_index`, `mpnn_edge_type`, `batch` | `mpnn_edge_types` thêm nhãn bond-type 2D |
| `mpnn_3d_distance_bins` | `x`, `mpnn_3d_edge_index`, `mpnn_3d_edge_type`, `batch` | `mpnn_3d_distance_bins` cần `pos` float32 `[N, 3]`, tạo all-pairs edge view với 4 bond type và 10 distance bin |
| `dimenet` | `atomic_number`, `pos`, `dimenet_edge_index`, `dimenet_triplet_edge_index`, `batch` | `dimenet_inputs` cần explicit `atomic_number` và `pos`; tạo radius edge 5 Å và non-backtracking triplet edge-ID view |
| `gpspp` | `x`, `edge_index`, `edge_attr`, `gpspp_pair_index`, `gpspp_spd`, `batch` | `gpspp_inputs` tạo tất cả ordered atom pairs (gồm self-pair) và shortest-path distance topological cho attention bias |
| `weave` | `x`, `weave_pair_index`, `weave_pair_attr`, `batch` | `weave_inputs` tạo sparse ordered atom-pair view hai chiều, gồm self-pair và các pair cách tối đa hai liên kết |
| `potentialnet` | Bắt buộc: `x`, `potentialnet_bond_edge_index`, `potentialnet_bond_edge_type`, `ligand_mask`, `batch`. Tùy chọn (đi cùng nhau): `potentialnet_stage2_edge_index`, `potentialnet_stage2_edge_type`, `potentialnet_use_spatial` | `potentialnet_inputs` luôn tạo typed covalent graph; có `pos` thì tạo thêm spatial graph, không có `pos` thì dùng nhánh 2D bond-only |
| `hignn` | `x`, `edge_index`, `edge_attr`, `brics_edge_index`, `brics_edge_attr`, `atom_to_fragment`, `batch` | `brics_fragments` thêm BRICS fragment view |
| `himnet` | `himnet_x`, `himnet_edge_index`, `himnet_edge_attr`, `himnet_reverse_edge_index`, `himnet_node_batch`, `himnet_node_type`, `himnet_fp` | `himnet_inputs` thêm unified hierarchy và fingerprint views |
| `molecular_graph_embedding` | `mge_x`, `edge_index`, `mge_edge_attr`, `batch` | `coley_2017_features` thêm feature tensors cho MGE |

Điểm chung là `edge_index` và `batch` luôn mô tả graph batch thực tế; tên và
số chiều feature phụ thuộc từng core architecture. Khi dùng custom featurizer,
hãy xem output của `describe-model` thay vì giả định canonical feature schema
của project là bắt buộc.

- Với AMPNN, `ampnn_edge_type` là nhãn `0..3` cho liên kết single, double,
  triple hoặc aromatic. Helper bundled lấy các nhãn này từ canonical bond
  profile; custom featurizer có thể cung cấp trực tiếp tensor tương thích.
- Với D-MPNN và EMNN, `reverse_edge_index` phải map mỗi directed edge sang đúng cạnh
  đảo chiều và phải là involution: `reverse_edge_index[reverse_edge_index]`
  trả về từng edge ban đầu.
- Với HiGNN, `brics_edge_index`/`brics_edge_attr` là graph giữ lại sau khi cắt
  BRICS; `atom_to_fragment` gán mỗi atom vào connected-component fragment của
  chính molecule đó.
- Với MGE, helper bundled tạo `mge_x`/`mge_edge_attr` theo default 32/8. Custom
  feature schema vẫn hợp lệ nếu đồng thời đặt `input_atom_dim`/`input_bond_dim`
  của model khớp với tensor cung cấp.
- Với `mpnn_3d_distance_bins`, các cạnh covalent giữ nhãn single/double/triple/
  aromatic; mọi pair atom còn lại nhận một trong mười distance bin. Helper nhận
  canonical 14-wide bond features hoặc profile 5-wide của `pdbbind_complex`.
- Với DimeNet, `dimenet_triplet_edge_index` index các DimeNet radius edge (không
  phải atom): hàng đầu là `k -> j`, hàng hai là `j -> i`. Transform chạy trước
  PyG batching để offset edge-ID đúng; khoảng cách và góc vẫn được tính trong
  model để giữ coordinate gradient.
- Với PotentialNet, Stage 1 chỉ nhận covalent typed edges. Nếu có spatial fields,
  Stage 2 nhận cả spatial distance-bin và covalent typed edges; nếu không có, Stage 2
  bị bỏ qua thay vì nhận một graph covalent giả. Readout chỉ sum atom có
  `ligand_mask=True` (CSV-SMILES mặc định là toàn bộ atom). Built-in transform dùng
  profile spatial tương thích DGL-LifeSci (cutoff 4.5 Å, bốn bins và tối đa bốn
  incoming neighbours), nhưng giữ các relation đồng thời thành cạnh song song thay vì
  ghi đè chúng.

## Contract tọa độ của FragNet

`fragnet` yêu cầu sample chưa được batch có `smiles`, `x` và `pos` float32 hữu hạn
`[N, 3]` cùng mô tả một tập atom theo đúng thứ tự. Helper `fragnet_inputs` tạo các
view BRICS, fragment và cosine bond-angle từ các tọa độ được cung cấp. Nó không tự
embed hay tối ưu conformer. `pdbbind_complex` hiện chưa tương thích trực tiếp vì
SMILES của source đó chỉ mô tả ligand, còn `x`/`pos` bao gồm cả ligand–pocket; hãy
dùng source hoặc custom sample có SMILES, graph và tọa độ đồng nhất. FragNet nằm
ngoài benchmark mặc định.

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

`--featurizer` hiện có ABI SMILES-only nên chủ động không hỗ trợ
`data.source: pdbbind_complex`; source này đã nạp ligand, pocket và tọa độ trực tiếp.
`--training-strategy` vẫn dùng được cho cả hai source.

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

Canonical CSV runner biểu diễn mỗi liên kết phân tử bằng hai cạnh ngược chiều và
không đưa self-loop vào input. PDBBind source cũng tạo một complex graph cho mỗi
sample, giữ `pos` và `ligand_mask`; transform riêng của PotentialNet hoặc
`mpnn_3d_distance_bins` có thể dùng các tọa độ đó để tạo spatial edges. Mọi model chặn cạnh nối giữa các sample trong
cùng batch; các model có phương trình nguồn tự cộng trạng thái node cũng chặn
self-loop, còn `dmpnn` kiểm tra thêm reverse-edge map. Nếu dùng featurizer riêng,
hãy giữ các quy ước này khi muốn dùng đúng input contract; `gcn_baseline` vẫn giữ
hành vi PyG hợp lệ với đồ thị directed/self-loop.
