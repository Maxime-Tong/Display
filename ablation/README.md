# screen_adaptor — Ablation & Sensitivity Study

独立、只读的实验脚手架：对 `screen_adaptor` 方法做消融实验与 sensitivity study。
**只调用**项目的公开函数（`pipeline` / `model` / `scenes` /
`power` / `perception`），**不修改**任何源码；所有新增代码都在本目录。

## 实验设计 → 实现映射

| 实验 | 变体 / 扫描值 | 对应 CLI |
|---|---|---|
| **消融 A：训练配方** | `pretrain`（仅 base 全局 LUT，single 模式）；`finetune`（逐簇从零训练，cluster 模式）；`pretrain+finetune`（全流程，cluster 模式） | `--ablation-recipe pretrain,finetune,pretrain_finetune` |
| **消融 B：运行时表示** | `w/ LUT`（LUT blockwise, bs=4）；`w/o LUT`（直接跑网络逐像素，无 LUT 量化）；`resolution`（LUT 逐像素, bs=1） | `--ablation-runtime` |
| **Sensitivity 1：blockwise 降采样因子** | `block_size ∈ {1,4,16,32}`（`LUTColorTransformer` 的块分辨率） | `--sens-block 1,4,16,32` |
| **Sensitivity 2：簇数 × LUT 分辨率** | `n_clusters ∈ {1,2,4,8}` × `lut_resolution ∈ {8,16,32}`（外维需重训、内维仅重导出 LUT） | `--sens-lutres --clusters 1,2,4,8 --lut-resolutions 8,16,32` |

消融 B 的可解释分解（在同一 pretrain+finetune 参考模型上）：

- **blockwise 效应**：`w/LUT (bs4)` vs `resolution (bs1)`
- **LUT 量化效应**：`w/o LUT` vs `resolution (bs1)`
- **部署 vs 理想**：`w/LUT (bs4)` vs `w/o LUT`

## 文件

- `_common.py` — 共享引擎：指标（saving/PSNR/SSIM/MetaM）、模型资产加载、
  LUT/网络两种表示的推理、单配置评估与 `report` 聚合。
- `run_recipes.py` — 调用 `pipeline.pretrain / cluster / finetune` 训练三种配方，
  产出 `run_eval.py` 可读的 model-dir 布局。
- `run_eval.py` — 单配置评估器 + `--report` 聚合（CSV/Markdown）。
- `run_ablation.sh` — 顶层驱动，用 flag 选择要跑哪些实验。
- `configs/from_scratch.json` — "finetune-only（从零）" 专用配置（架构与
  finetune 一致、更多 step / 更高 lr，保证配方可比）。

## 指标（与 `benchmark_screen_adaptor.py` 一致）

- **saving** = `1 − power(opt)/power(orig)`（RGB 加权线性域 OLED 内容功耗，
  `power.power_saving`）
- **PSNR / SSIM**（skimage，`data_range=1.0`）
- **MetaM** = foveated `perception.MetamericLoss`（默认缩放到最长边 ≤768 计算）

## 快速开始

```bash
cd screen_adaptor/ablation

# 仅运行时表示消融（用一个已训练好的参考模型）
./run_ablation.sh -d ../../datasets/DIV2K -o runs/demo \
    -m /path/to/trained/model_dir \
    --ablation-runtime --max-images 5

# 训练配方消融（会实际训练，需要 GPU；先训练再评估）
./run_ablation.sh -d ../../datasets/DIV2K -o runs/recipes \
    --ablation-recipe pretrain,finetune,pretrain_finetune \
    --clusters 8 --device cuda

# sensitivity：block 降采样 + 簇数×LUT 分辨率矩阵
./run_ablation.sh -d ../../datasets/DIV2K -o runs/sens \
    --sens-block 1,4,16,32 \
    --sens-lutres --clusters 1,2,4,8 --lut-resolutions 8,16,32
```

也可以直接调用底层脚本：

```bash
# 单配置评估
python run_eval.py --model-dir DIR --data-dir DATASETS --out OUT \
    --mode cluster --repr lut --block-size 4 --max-images 5

# 聚合某组实验的所有 *.json
python run_eval.py --report --results-dir OUT
```

## 输出布局

```
<out-root>/
  recipes/pretrain/                     # pretrain-only
  recipes/finetune/                     # finetune-only（逐簇从零）
  recipes/pretrain_finetune/            # pretrain+finetune
  lutres/k<K>/                           # 簇数 sensitivity 训练资产
  eval/<recipe|runtime|block|lutres>/   # 每配置一个 <key>.json + summary.csv/.md
```

## 约定与注意事项

- **数据**：`--data-dir` 的每个直接子目录视为一个场景（训练与评估同源，与现有
  benchmark 一致；如需留出验证集可另备一份 data-dir 作评估）。
- **LUT 分辨率扫描**：从已保存的 `cluster_*.pt` 网络用 `generate_lut` 重导出，
  **不重新训练**；只有簇数维度需要重新 derive + finetune。
- **表示**：`repr=net` 逐像素直接跑 `ColorModel`，`block_size` 对其无意义；
  `repr=lut` 使用 `apply_lut`，并在本目录实现 blockwise 细节保持。
- **设备**：评估默认自动选择 `cuda`（无则 `cpu`，MetaM 在 CPU 上很慢，建议小
  `--max-images`）；训练默认 `cuda`。
- **可复现**：固定 `--seed`（默认 0），贯穿 pretrain/derive/kmeans/finetune。
- **幂等**：`run_recipes.py` 会跳过已训练完成的配方，可安全续跑。
