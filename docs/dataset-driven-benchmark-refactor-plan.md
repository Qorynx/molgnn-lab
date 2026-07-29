# Dataset-driven Multi-model Benchmark Refactor Plan

Trạng thái: **Đang triển khai - Phase 1 và Phase 2 đã hoàn tất**

Ngày chốt kế hoạch: **2026-07-24**

Phạm vi: config, registry, data preparation, runner, artifacts, CLI, tests và migration
tài liệu. Truyền hyperparameter trực tiếp qua command line chưa nằm trong phạm vi này.

## 1. Kết quả cần đạt

Sau refactor, một YAML đại diện cho một dataset/benchmark thay vì một cặp
model-dataset. Người dùng có ba mức sử dụng:

1. Không khai báo `models`: chạy tuần tự toàn bộ model benchmark-enabled.
2. Khai báo `models`: chỉ chạy các model trong danh sách và giữ nguyên thứ tự.
3. Khai báo `model_overrides`: thay đổi architecture/training hyperparameter của từng
   model mà không tạo thêm YAML riêng.

Mỗi model vẫn độc lập với config và orchestration. Runner không được thêm nhánh
`if model_name == ...`; mọi khác biệt phải đi qua registry metadata và graph transform.

## 2. Ngoài phạm vi

- Không hỗ trợ `--set` hoặc truyền hyperparameter qua CLI trong refactor này.
- Không triển khai hyperparameter search, Optuna hoặc grid/random sweep.
- Không thay đổi loss, metric hoặc trainer theo tên model.
- Không tạo plugin API công khai cho model bên ngoài package.
- Không thay đổi fidelity contract của các architecture đã triển khai.

## 3. Trạng thái hiện tại và vấn đề

Luồng hiện tại là:

```text
YAML chứa model.name/model.parameters
  -> load_config() tạo một ResolvedConfig có đúng một ModelConfig
  -> run_experiments() lặp qua seed
  -> _run_experiment() load dataset, split, transform, build một model
  -> ghi runs/<experiment>/seed_<seed>/
```

Thiết kế này tạo ra bốn vấn đề khi cần benchmark toàn bộ model:

| Vấn đề | Hậu quả |
|---|---|
| Một config chỉ chứa một model | Số YAML tăng theo `dataset x model` |
| Runner chỉ lặp seed | Không có orchestration model-level |
| Dataset/transform nằm trong seed run | Tiền xử lý bị lặp không cần thiết |
| Artifact path không có model | Các model có thể ghi đè kết quả của nhau |
| Hyperparameter gắn với YAML đơn-model | Khó dùng default chung nhưng vẫn override chọn lọc |

Không giải quyết bằng cách bọc `_run_experiment()` trong một vòng lặp model, vì cách đó
vẫn lặp dataset preparation, không giải quyết artifact collision và không tạo benchmark
summary cấp model.

## 4. Quyết định kiến trúc

### 4.1. Dataset config là public input

Config bắt buộc mô tả dataset, task và experiment. Model selection và overrides là tùy
chọn. Model constructor defaults cùng registry metadata là baseline khi không override.

### 4.2. Registry là nguồn danh sách model

Model built-in chỉ được benchmark tự động khi `benchmark_enabled=True`. Registry cung cấp
thứ tự mặc định ổn định; khi người dùng khai báo `models`, thứ tự YAML được giữ nguyên.

### 4.3. Một split chung cho toàn benchmark

Dataset được load một lần, split được tạo một lần bằng seed đầu tiên trong
`experiment.seeds`, sau đó được tái sử dụng cho mọi model và training seed. Điều này giữ
so sánh công bằng và đáp ứng contract reproducibility hiện hành.

### 4.4. Model outer loop, seed inner loop

Runner chạy xong mọi seed của model hiện tại trước khi chuyển model tiếp theo. Graph
transform được chuẩn bị một lần cho model và tái sử dụng giữa các seed. Dữ liệu đã
transform được giải phóng trước khi chuẩn bị model tiếp theo để giới hạn bộ nhớ.

### 4.5. Failures được cô lập

Một `(model, seed)` lỗi không chặn các model phía sau. Runner ghi status và summary đầy
đủ, tiếp tục benchmark, sau đó trả kết quả tổng hợp để CLI quyết định exit code.

## 5. Config contract mới

### 5.1. Chạy toàn bộ model bằng default

```yaml
extends: ../benchmark_defaults.yaml

experiment:
  name: esol
  seeds: [42, 43, 44]
  output_dir: ../../runs

data:
  path: ../../data/esol.csv
  smiles_column: smiles
  target_columns: [target]
  id_column: null
  split: scaffold
  split_ratios: [0.8, 0.1, 0.1]
  split_column: null
  invalid_smiles: error

task:
  type: regression
  loss: mse
  metrics: [rmse, mae, r2]
  target_scaling: true

training:
  epochs: 200
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.000001
  patience: 30
  monitor: val_rmse
  monitor_mode: min
  device: auto
  num_workers: 0
```

Không có `models` nghĩa là chạy toàn bộ model có `benchmark_enabled=True`.

### 5.2. Chạy subset model

```yaml
models:
  - gcn_baseline
  - dmpnn
  - attentivefp
```

Runner chạy đúng thứ tự trên. Không sort lại danh sách explicit của người dùng.

### 5.3. Override hyperparameter theo model

```yaml
models:
  - gcn_baseline
  - attentivefp

model_overrides:
  gcn_baseline:
    parameters:
      hidden_dim: 128
      num_layers: 4
      dropout: 0.1

  attentivefp:
    parameters:
      hidden_dim: 256
      num_atom_layers: 3
      num_molecule_layers: 2
      dropout: 0.2
    training:
      batch_size: 32
      learning_rate: 0.0005
```

`training` top-level là chính sách chung. `model_overrides.<name>.training` chỉ ghi đè
model tương ứng.

### 5.4. Validation rules

| Input | Kết quả |
|---|---|
| Bỏ `models` hoặc `models: null` | Chạy toàn bộ benchmark-enabled models |
| `models: []` | `ConfigError`, không cho phép benchmark rỗng |
| Tên model rỗng hoặc trùng | `ConfigError` có field path |
| Tên model không tồn tại | Lỗi khi resolve registry, kèm available models |
| Override model không được chọn | Lỗi nếu `models` là subset; hợp lệ nếu chạy all |
| Unknown override/training/parameter key | Lỗi trước khi bắt đầu train |

Các context-derived parameters `atom_dim`, `bond_dim`, `num_targets` và
`feature_schema_version` không được phép xuất hiện trong `model_overrides`; runner/registry
tiếp tục inject chúng từ dataset context.

### 5.5. Inheritance và merge

`extends` vẫn chỉ hỗ trợ một cấp. Merge semantics mới:

- Các section mapping hiện tại tiếp tục shallow-merge như implementation hiện hành.
- `models` ở config con thay thế toàn bộ `models` của base, không concatenate.
- `model_overrides` merge theo model name; `parameters` và `training` merge theo key.
- `model:` dạng singular cũ bị từ chối với migration error rõ ràng.
- Resolved config luôn immutable và serializable.

## 6. Typed config design

`ModelConfig` không còn là field bắt buộc của public `ResolvedConfig`. Thiết kế đề xuất:

```python
@dataclass(frozen=True)
class ModelOverrideConfig:
    parameters: Mapping[str, object]
    training: Mapping[str, object]


@dataclass(frozen=True)
class ResolvedConfig:
    experiment: ExperimentConfig
    data: DataConfig
    training: TrainingConfig
    task: TaskConfig
    models: tuple[str, ...] | None
    model_overrides: Mapping[str, ModelOverrideConfig]
```

Config parsing chỉ validate cấu trúc độc lập với model implementation. Registry resolution
validate tên model và constructor parameters. Tách hai bước này giúp `config.py` không cần
import toàn bộ model modules.

Mỗi run tạo một internal resolved model-run config:

```python
@dataclass(frozen=True)
class ResolvedModelRun:
    model_name: str
    parameters: Mapping[str, object]
    training: TrainingConfig
    seed: int
```

Object này được serialize cùng dataset/task metadata vào checkpoint và seed-level
`config.yaml`.

## 7. Hyperparameter resolution

### 7.1. Architecture parameters

Thứ tự ưu tiên:

```text
constructor defaults
  -> ModelSpec.default_parameters
  -> model_overrides.<name>.parameters
  -> dataset-derived BuildContext injection
```

`resolve_model_parameters()` phải materialize mọi default thực sự được dùng để artifact
không phụ thuộc ngầm vào version source code. Unknown keys tiếp tục được kiểm tra bằng
factory signature.

### 7.2. Training parameters

Thứ tự ưu tiên:

```text
framework training defaults
  -> top-level training
  -> model_overrides.<name>.training
```

Không có CLI override trong phase này. Không thêm training defaults riêng vào `ModelSpec`
cho đến khi có model thực tế bắt buộc cần chúng; per-model training override đã đủ cho
thí nghiệm hiện tại.

### 7.3. Reproducibility

Seed-level config phải lưu effective architecture parameters và effective training config,
không chỉ các key người dùng đã override. Benchmark root phải lưu cả requested và resolved
model lists.

## 8. Registry contract

Mở rộng `ModelSpec`:

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[..., nn.Module]
    default_parameters: Mapping[str, object]
    required_batch_fields: tuple[str, ...]
    graph_transform_name: str | None
    prediction_reducer_name: str
    benchmark_enabled: bool = True
    benchmark_order: int = 0
```

API mới:

```python
def benchmark_models() -> tuple[ModelSpec, ...]: ...
def resolve_benchmark_models(names: Sequence[str] | None) -> tuple[ModelSpec, ...]: ...
def resolve_model_parameters(
    spec: ModelSpec,
    overrides: Mapping[str, object],
    context: BuildContext,
) -> Mapping[str, object]: ...
```

`benchmark_models()` dùng `(benchmark_order, name)` để tạo thứ tự ổn định.
`resolve_benchmark_models(explicit_names)` giữ thứ tự explicit và reject duplicate/unknown.

### 8.1. Thêm model mới sau refactor

Thêm model mới chỉ cần:

1. Implement architecture và public batch contract.
2. Implement/register graph transform nếu canonical graph chưa đủ.
3. Thêm đúng một `ModelSpec` trong built-in registration.
4. Thêm model, registry và integration tests.
5. Không sửa dataset YAML khi benchmark mặc định chạy all.

Model mới tự xuất hiện trong config không có `models` nếu `benchmark_enabled=True`. Config
có subset explicit không tự động thêm model mới, bảo đảm experiment cũ không đổi phạm vi.

## 9. Data preparation contract

Tách dataset canonical khỏi transformed model view:

```python
def prepare_model_samples(
    dataset: MolecularDataset,
    graph_transform: GraphTransform | None,
) -> PreparedDataset: ...


def build_dataloaders(
    prepared: PreparedDataset,
    splits: SplitIndices,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> DataLoaders: ...
```

Lifecycle:

1. Load/validate CSV và canonical features một lần.
2. Tạo/persist split một lần; fit target scaler một lần.
3. Chuẩn bị transformed samples một lần cho model hiện tại.
4. Tạo seeded dataloaders cho từng seed từ cùng prepared samples.
5. Giải phóng prepared view trước model kế tiếp.

Transform phải tiếp tục deterministic, không mutate canonical sample và không tạo
cross-molecule index leakage.

## 10. Runner design

Public orchestration API:

```python
@dataclass(frozen=True)
class BenchmarkResult:
    completed: tuple[Path, ...]
    failed: tuple[RunFailure, ...]
    summary_path: Path
    leaderboard_path: Path


def run_benchmark(config: ResolvedConfig) -> BenchmarkResult: ...
```

Internal boundary:

```python
def _run_model_seed(
    config: ResolvedConfig,
    model_spec: ModelSpec,
    resolved_run: ResolvedModelRun,
    prepared: PreparedDataset,
    splits: SplitIndices,
    scaler: TargetScalerState | None,
    paths: RunPaths,
) -> Path: ...
```

`run_experiment()`/`run_experiments()` có thể được giữ như compatibility wrappers nội bộ
trong một release, nhưng CLI chỉ gọi `run_benchmark()`. Không giữ hai orchestration paths
độc lập.

### 10.1. Preflight

Trước run đầu tiên, benchmark thực hiện preflight cho mọi selected model:

- Resolve model spec, transform và effective parameters.
- Chuẩn bị một representative batch đúng required fields.
- Build model trên CPU và kiểm tra output shape `[batch, num_targets]`.
- Validate effective training monitor/task compatibility.
- Ghi model-level failure và tiếp tục model khác nếu preflight lỗi.

### 10.2. Resource cleanup

Sau mỗi seed, giải phóng model, optimizer và dataloader references. Sau mỗi model, giải
phóng transformed samples, gọi garbage collection và chỉ gọi CUDA cache cleanup khi CUDA
được sử dụng.

## 11. Artifact layout

```text
runs/<benchmark_name>/
├── benchmark_config.yaml
├── split.csv
├── summary.csv
├── leaderboard.csv
├── attentivefp/
│   ├── resolved_model.yaml
│   ├── aggregate_metrics.json
│   └── seed_042/
│       ├── config.yaml
│       ├── status.json
│       ├── loss_history.csv
│       ├── metrics_history.csv
│       ├── test_history.csv
│       ├── best.ckpt
│       ├── last.ckpt
│       └── test_predictions.csv
└── dmpnn/
    └── ...
```

Root artifacts:

| Artifact | Nội dung |
|---|---|
| `benchmark_config.yaml` | Dataset/task, requested/resolved models và shared policy |
| `split.csv` | Split duy nhất dùng cho toàn benchmark |
| `summary.csv` | Một dòng cho mỗi `(model, seed)` kể cả failed |
| `leaderboard.csv` | Mean/std/valid count của từng model |
| `<model>/aggregate_metrics.json` | Metric distribution qua seed của model |

`summary.csv` ưu tiên columns:

```text
model,seed,status,best_epoch,test_loss,<test_metrics>,run_dir,error_type,error_message
```

Path components cho benchmark/model phải reject empty string, separators và traversal.
Khởi tạo run chỉ được xóa artifact trong đúng seed directory đã resolve.

## 12. Failure và CLI behavior

`validate-config` thực hiện structural validation rồi registry validation. Output cần nêu
selected models, seeds và tổng số run:

```text
Valid config: configs/datasets/esol.yaml
models=[gcn_baseline, dmpnn, attentivefp]
seeds=[42, 43]
total_runs=6
```

`train` không nhận hyperparameter override qua CLI:

```bash
molgnn train --config configs/datasets/esol.yaml
```

Exit code contract:

| Code | Ý nghĩa |
|---|---|
| `0` | Tất cả selected runs hoàn thành |
| `1` | Benchmark chạy xong nhưng có model/seed failed hoặc lỗi runtime bất ngờ |
| `2` | Config/input/preflight toàn cục không hợp lệ |

Log mỗi run phải có prefix `[model=<name>][seed=<seed>]`. CLI in đường dẫn summary và số
completed/failed ở cuối, không làm mất kết quả partial success.

## 13. File-level implementation plan

| File | Thay đổi chính |
|---|---|
| `src/molgnn/config.py` | `models`, `model_overrides`, merge/validation và migration error |
| `src/molgnn/registry.py`, `models/registration.py` | Benchmark metadata, selection và default resolution |
| `src/molgnn/dataset.py`, `runner.py` | Prepared dataset, model/seed loops, preflight và cleanup |
| `src/molgnn/artifacts.py`, `cli.py` | Model-aware paths, summary/leaderboard và exit behavior |
| `configs/`, `tests/`, `README.md`, `docs/` | Migration, coverage và user documentation |

Không cần sửa trainer/evaluator/task theo tên model. Nếu implementation yêu cầu thay đổi
như vậy, phải dừng và sửa registry/transform contract thay vì thêm model branch.

## 14. Implementation phases

### Phase 1 - Config và registry contract (4-5 giờ)

Trạng thái: **Hoàn tất ngày 2026-07-24**.

- Migration bridge: trong Phase 1, legacy `model:` tạm thời được normalize thành
  `models`/`model_overrides` để runner hiện hành và regression suite tiếp tục hoạt động.
  Chỉ reject legacy schema sau khi Phase 3 chuyển CLI sang `run_benchmark()`.
- Viết failing tests cho omitted/subset/invalid models và overrides.
- Refactor typed config, defaults, merge và serialization.
- Mở rộng `ModelSpec` và registration metadata.
- Implement selection/default resolution helpers.
- Gate: config/registry unit tests pass, chưa đổi runner.

Kết quả gate: 33 config/registry tests và 258 tests toàn suite pass; Ruff check/format và
BasedPyright error-level đều sạch. CLI/runner vẫn dùng single-model compatibility view cho
đến Phase 2.

### Phase 2 - Data preparation và runner (6-8 giờ)

Trạng thái: **Hoàn tất ngày 2026-07-24**.

- Tách `prepare_model_samples()` khỏi dataloader creation.
- Implement `BenchmarkResult`, preflight và model-outer/seed-inner loops.
- Tái sử dụng dataset, split, scaler và per-model prepared view.
- Implement failure isolation và resource cleanup.
- Gate: fake-model orchestration tests pass, không cần train sáu model thật.

Kết quả gate: `PreparedDataset` được tái sử dụng giữa các seed; dataset, split và target
scaler chỉ được chuẩn bị một lần cho toàn benchmark; preflight chạy cho mọi model trước
training; lỗi model/seed được trả về qua `BenchmarkResult` mà không chặn model sau. Hai
contract tests Phase 2 và toàn bộ **260 tests** đều pass; Ruff cho các file thay đổi và
BasedPyright error-level cho `config.py`/`runner.py` đều sạch.

Boundary chuyển tiếp: `run_benchmark()` đã ghi seed runs theo
`<output>/<benchmark>/<model>/seed_*`, nhưng `summary_path` và `leaderboard_path` vẫn là
`None`. Benchmark-aware path objects, root summary/leaderboard và việc chuyển CLI sang API
mới thuộc Phase 3. `run_experiment()`/`run_experiments()` hiện là compatibility wrappers
dùng chung lifecycle mới và vẫn giữ layout cũ cho CLI hiện hành.

### Phase 3 - Artifacts và CLI (4-5 giờ)

- Thêm benchmark/model-aware path objects.
- Implement root summary, per-model aggregate và leaderboard.
- Chuyển CLI sang `run_benchmark()` và cập nhật validation output.
- Kiểm tra path safety và partial-failure exit codes.
- Gate: artifact/CLI tests pass trên temporary directories.

### Phase 4 - Migration và end-to-end tests (6-7 giờ)

- Loại `model` khỏi base config và gom smoke configs theo dataset.
- Cập nhật integration tests sang `benchmark/model/seed` hierarchy.
- Chạy one-epoch all-model smoke trên tiny regression fixture.
- Chạy subset + per-model override smoke.
- Gate: full test suite, Ruff và static checks pass.

### Phase 5 - Documentation và final QA (2-3 giờ)

- Cập nhật README quickstart/config/output sau khi code đã merge.
- Đánh dấu sections lịch sử bị superseded nhưng không xóa fidelity plans.
- Kiểm tra resolved artifacts thủ công cho all/subset/failed cases.
- Chạy validation/train commands từ clean checkout hoặc installed wheel.
- Gate: Definition of Done bên dưới được xác nhận đầy đủ.

Tổng effort dự kiến: **22-28 giờ làm việc**, khoảng **3-4 ngày** nếu không phát sinh
incompatibility trong model transforms hoặc checkpoint migration.

## 15. Test matrix

### 15.1. Config

- Omitted/null `models` resolve thành all; explicit subset giữ thứ tự.
- Empty, duplicate, blank và unknown names báo lỗi rõ ràng.
- `model_overrides` merge đúng và reject inactive/unknown models.
- Unknown parameter/context-derived parameter bị reject.
- Serialization chứa built-in values và không mutate được.

### 15.2. Registry/model integration

- Sáu built-in model có unique order/name và benchmark-enabled metadata.
- Resolved defaults build được với `BuildContext` hợp lệ.
- Model mới giả tự xuất hiện trong all-mode nhưng không chen vào explicit subset.
- Graph transform và required fields được lấy từ metadata/class contract.
- Runner/trainer/evaluator không có name-based branch.

### 15.3. Runner

- Số run bằng `selected_models x seeds` và đúng model/seed order.
- Dataset/split/scaler chỉ được tạo một lần.
- Transform chỉ chuẩn bị một lần cho mỗi model.
- Failure một run không chặn các run còn lại.
- Effective parameters/training khác nhau đúng theo overrides.

### 15.4. Artifacts/CLI

- Model directories không collision; split chỉ có một bản ở benchmark root.
- Summary chứa completed/failed rows và run paths hợp lệ.
- Leaderboard aggregate đúng qua finite seed metrics.
- `validate-config` in selected models và total run count.
- Exit codes phân biệt success, partial/runtime failure và config error.

### 15.5. End-to-end

- Tiny regression dataset chạy một epoch trên toàn bộ model built-in.
- Subset config chỉ tạo directory cho selected models.
- Per-model overrides xuất hiện trong resolved config/checkpoint.
- Multi-seed dùng cùng split nhưng seeded training order khác nhau.
- Full regression suite bảo toàn standalone model/fidelity invariants.

## 16. Config migration

Project đang ở phiên bản `0.1.0`, vì vậy dùng controlled breaking migration:

| Cấu hình cũ | Cấu hình mới |
|---|---|
| `model.name` | `models: [name]` nếu muốn chạy riêng model đó |
| `model.parameters` | `model_overrides.<name>.parameters` |
| Training riêng trong YAML model | `model_overrides.<name>.training` |
| Không có model section | Chạy toàn bộ model |
| `runs/<experiment>/seed_*` | `runs/<benchmark>/<model>/seed_*` |

Migration work:

- Xóa model defaults khỏi `configs/base.yaml`; có thể đổi tên thành
  `benchmark_defaults.yaml`.
- Gom các smoke YAML theo model thành một dataset smoke config; tests cần subset thì tạo
  temporary config hoặc dùng `models: [name]`.
- Chuyển ESOL/BACE configs sang dataset-oriented naming.
- Giữ old artifact directories read-only; không tự động move/delete kết quả cũ.
- Lỗi legacy `model:` phải chỉ rõ schema thay thế.

## 17. Rủi ro và kiểm soát

| Rủi ro | Kiểm soát |
|---|---|
| Constructor defaults không materialize | Resolve signature và lưu effective params |
| Model mới cần input riêng | Graph transform + `required_batch_fields`, không runner branch |
| Dataset lớn làm tăng RAM | Chỉ giữ prepared view của một model tại một thời điểm |
| Một model fail sau nhiều giờ | Preflight toàn bộ model và failure isolation |
| Explicit subset đổi khi thêm model mới | Giữ chính xác danh sách người dùng |
| Default-all đổi khi registry có model mới | `benchmark_enabled` explicit và resolved model manifest |
| Override vô tình không được dùng | Reject override của inactive model và lưu effective config |
| Split không công bằng | Một root `split.csv` tái sử dụng cho toàn benchmark |

## 18. Definition of Done

- Một dataset YAML không có `models` chạy toàn bộ built-in benchmark models.
- Một YAML có `models` chỉ chạy đúng subset và thứ tự đã khai báo.
- `model_overrides` thay đổi architecture/training config đúng model.
- Dataset, split và scaler không bị tạo lại theo model/seed.
- Mỗi model/seed có artifact riêng; root summary/leaderboard audit được toàn benchmark.

Quality gate cuối:

- Lỗi một run không làm mất kết quả của các run khác.
- Model mới chỉ cần architecture, optional transform, registration và tests.
- Không có CLI hyperparameter override trong public interface.
- Không có model-name branch trong runner/trainer/evaluator/task.
- Full unit/integration/static checks pass và README phản ánh implementation thực tế.

## 19. Documentation policy trong thời gian refactor

Cho đến khi implementation hoàn tất, README và các coding plan lịch sử tiếp tục mô tả
behavior hiện hành có `model.name`. Tài liệu này là nguồn chuẩn cho behavior đích và phải
được ghi rõ trạng thái **Planned**.

Sau khi Phase 4 pass:

1. Cập nhật README quickstart sang dataset-driven config.
2. Thêm migration example từ `model` sang `models`/`model_overrides`.
3. Cập nhật output hierarchy và CLI validation examples.
4. Đánh dấu config/artifact sections cũ trong MVP plan là superseded.
5. Không sửa lại các fidelity/coding-plan lịch sử như thể thiết kế mới đã tồn tại từ đầu.
