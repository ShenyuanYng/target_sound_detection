# 实验三：AudioSet Strong + Dual FSD + Wrong Condition

独立配置：`exp_defs/tse_strong_dual_fsd_wrong.yaml`。独立入口：`tools/run_tse_dual_wrong.py`。旧 SED、单 FSD TSE 配置及损失保持原样。

## 数据和混音

- 沿用 AudioSet 按视频划分的 Train/CV，并筛选有 Strong 标注的背景；FSD 使用官方 dev train/val，沿用已有重复内容排除项和音频审计。检查 Train/CV 的 ID、路径、文件内容以及跨语料内容泄漏。
- 优先选择两个**原始标签均为单标签**的 FSD。若没有满足标签约束的单标签组合，才考虑多标签来源。多标签判定使用完整原始 MID，包含不在 527 类模型映射中的标签。
- A/B 的 MID 不同。两份 FSD 的完整标签彼此之间、与背景的 Strong + 原始 Weak 标签之间，均排除相同 MID 和 Ontology 祖先/后代关系。未知 Ontology 标签的来源被排除并计数。仅共享上层祖先的兄弟类别不自动互斥。
- Wrong MID 从官方 FSD train 的已标注类别词表中采样，须在模型映射和 Ontology 内，排除背景、A、B 的全部原始标签及其祖先/后代；不使用 Silence 作为 Condition。
- 每个背景有两个样本位置：一个不重叠，一个部分重叠。因此完整 Train epoch 和 CV 均严格为 50% / 50%，每轮混音条数是有效背景数的两倍。Train 每次读取重新采样来源、裁剪、SNR 和位置；shuffle 不改变完整 epoch 的比例。
- 部分重叠按**插入区间交集长度 / 较短裁剪长度**计算，范围为 25%–50%，且至少有非零的同时活跃区间。该比例不是 Active RMS 掩码的交集比例。不重叠组的区间允许首尾相接。
- 保留已有 FSD 重采样、去严格零点、短音频重复及跨中点裁剪逻辑。两份裁剪无法按指定关系放入 10 秒背景时，重新采样裁剪，必要时重新选来源，不为满足插入关系额外截短。裁剪/来源各最多尝试 128 次；耗尽后明确报错，不悄悄换组或跳过样本。
- 沿用 Active RMS、背景 zero-mean RMS、原增益公式和共同 SNR（默认 −5～5 dB）。A/B 分别按同一背景计算增益。最后对混音、背景、A/B 共同除以 `max(1, peak(mixture))`。GT 是实际插入、缩放并共同归一化后的干净来源波形；Wrong GT 为同长度全零波形。

这些规则根据标注推断类别缺席，不能证明未标注声音在听觉上不存在。多标签回退时，GT 是完整 FSD 来源，包含共现声音，并非其中单一类别的分离真值。Dataset 报告记录这些限制、排除项和 CV 单标签来源比例。

## CV 可复现性

首次建立 CV 时冻结全部混音组合，保存来源 ID/哈希、三个 MID、裁剪种子、位置、重叠长度、SNR、增益、归一化系数和波形哈希。固定种子为 `mix_seed: 7`。

重用 manifest 时核验配置、输入列表、音频审计、Ontology、class map、混音源码和数值库版本；每次读取再核验完整混音元数据及波形。变化会报错，需为新实验选择新的 manifest 路径，不覆盖原实验记录。源数据目录与缓存、manifest、训练输出目录必须分离。

## 模型、Loss 与评估

同一共享 TSEDPRNNTF 对一条混音进行 A、B、Wrong 三次条件前向。新实验增加 `target_ri` 输出，保留 `target_waveform`；旧配置不启用 RI 输出。

`loss_functions/tse_ri_loss.py` 适配用户提供的 `riinf_enh_loss()` 指定分支：

```text
loss_type="ratiol1", rix2_flag=1, mag_flag=1
permutation_invariant=0, mc_flag=0
mag_sf=1, rix2_weight=1, stft_istft_proj_flag=0
```

使用模型的原始 RI，GT 和 mixture 采用与模型一致的 sqrt-Hann STFT（默认 window=256、hop=128）。每个样本分别对实部、虚部、幅度计算 L1 总和，再各自除以该混音的幅度谱总和。幅度保留原实现的 `sqrt(R² + I² + 1e-5)`；三个分量相加后对 batch 求均值。损失运算使用 FP32。Wrong Condition 同样使用此公式，GT 为零，不需要 SI-SNR 的非零参考波形。

```text
Loss = (Loss_A + Loss_B + Loss_wrong) / 3
```

SI-SNR 仅用于指标，不参与训练。每个 epoch 写入 `train_tse_dual_metrics.json` / `val_tse_dual_metrics.json`，并记录 TensorBoard 标量；JSON 保留最近 epoch，TensorBoard 保留历史。报告包含 `all`、`nonoverlap`、`overlap` 三组：

| Condition | 指标 |
|---|---|
| A、B | 各自 RI + Magnitude loss、SI-SNR、相对同一混音的 SI-SNRi、有效参考数 |
| Wrong | loss、输出 RMS、输出/混音能量比 |
| 三者 | 等权平均总 loss |

SI-SNR 沿用项目的 zero-mean SI-SDR 实现。Wrong 输出 RMS 为 `sqrt(mean(output²))`，能量比为 `mean(output²) / max(mean(mixture²), 1e-12)`，对样本作算术平均，不做去均值；全零输出两项均为零。无样本组报告 `null`。

## 运行

在项目根目录和具有 PyTorch / Lightning 的 Python 环境执行。默认 YAML 是已核对过的 ailab01 语料路径；在其他机器直接使用该入口时，需要用独立 YAML 或命令行覆盖数据路径。下方 Slurm 入口则会自动冻结对应节点的路径。

```bash
# 数据审计与固定 CV 重放检查；不训练
python tools/run_tse_dual_wrong.py --mode check \
  --output outputs/tse_dual_wrong/check_01

# 有界 CPU 检查：完整模型、1 次 optimizer step、2 个 CV batch
python tools/run_tse_dual_wrong.py --mode smoke \
  --train-batches 1 --val-batches 2 \
  --output outputs/tse_dual_wrong/smoke_01

# 完整固定 CV 评估；仅需缩小评估范围时显式给 --val-batches
python tools/run_tse_dual_wrong.py --mode validate \
  --checkpoint /absolute/path/to/experiment3.ckpt \
  --output outputs/tse_dual_wrong/validate_01

# 正式训练入口（此次未执行完整训练）
python tools/run_tse_dual_wrong.py --mode train

# 显式续训：目录内须有 mylast.ckpt，并保持原配置和 CV manifest
python tools/run_tse_dual_wrong.py --mode train \
  --resume-run-dir /absolute/path/to/previous/run
```

通过 `--exp-def /absolute/path/to/config.yaml` 指定独立配置；其余 YAML 参数使用连字符形式覆盖，例如 `--batch-size 1`。源码中的相对产物路径从项目根目录解析，语料相对路径从 `data_path` 解析。`check` 是默认模式；`smoke` 强制 CPU / FP32 / 0 workers，不启动长训练。正式训练默认单 GPU、batch=2、三条件梯度等权平均。

输出目录必须未存在，防止覆盖检查结果。`--samples` 仅限制逐样本抽查数量；底层源数据审计、泄漏检查和首次 CV manifest 创建仍针对整个配置的数据范围，完整语料上的首次运行可能较慢。

## ailab01 / ailab02 的三作业正式训练

两节点分别独立训练，不组成分布式训练。使用新增独立入口，旧 `run.sh` 的 dataset 路由不适用于实验三。在项目根目录执行：

```bash
/home/syy12632604/anaconda3/envs/audioset/bin/python -B -u \
  tools/submit_tse_dual_wrong.py --node ailab01 --num-jobs 3 --time 2-00:00:00 --submit

/home/syy12632604/anaconda3/envs/audioset/bin/python -B -u \
  tools/submit_tse_dual_wrong.py --node ailab02 --num-jobs 3 --time 2-00:00:00 --submit
```

省略 `--submit` 只预览冻结后的配置和资源，不创建文件或提交作业。配置默认 batch=2、50 epochs、FP32、gradient checkpointing；每个作业申请 1 GPU / 8 CPU / 32G 内存。ailab01 使用 L40 分区和 `gpu:1`，允许 Slurm 分配该节点上的 L40 或 RTX 5880 Ada；ailab02 使用 RTX5880Ada 分区和 `gpu:rtx5880ada:1`，不手动指定 `CUDA_VISIBLE_DEVICES`。首次提交及自动串行续训使用同一 GPU 请求；修改默认值不会自动更新已排队作业，需要另行更新其资源请求。

`--num-jobs 3` 包括首次作业和 2 个串行续训作业，单节点最多获得 6 天运行时间。第一作业启动后，复用现有 `slurm/submit_continuation.py` 排入两个 `afterany` 接续作业；后续使用 `resume_training.py` 恢复同一 run 的 `mylast.ckpt`、优化器、学习率和 epoch 进度。目标始终为累计 50 epochs，已达到目标时后续作业直接退出，不追加 50 epochs。超时从最后完成 epoch 的 checkpoint 恢复；普通 Python 错误会写入失败标记，使后续作业跳过训练。若 6 天仍不足 50 epochs，需要再附加续训作业。

提交时冻结 YAML。节点对应的 AudioSet 路径分别为 `/hdd2/syy12632604/AudioSet` 与 `/data/syy12632604/AudioSetMix`；音频缓存独立保存在各节点的 `/data/syy12632604/tse_dual_wrong/<node>/fsd_cache_16k/`，不放在项目所在的 `/home`。CV manifest 和背景 split 等元数据仍分别保存在项目的 `outputs/tse_dual_wrong/data/ailab01/` 与 `outputs/tse_dual_wrong/data/ailab02/`，首次各自建立 CV manifest，避免两节点来源路径混用。直接运行 YAML 时，默认缓存路径对应 ailab01；提交入口会按所选节点覆盖缓存路径。已有运行的冻结配置和 checkpoint 不随默认值修改；迁移过的旧缓存路径通过软链接继续兼容。

每个节点的 run 路径为 `experiments/tse_dual_wrong/<node>/run_job<首次JOBID>/`。首次训练日志为 `logs/training.log`，续训日志为 `logs/training_job<续训JOBID>.log`；作业依赖记录在 `continuation_jobs.json`。每个 epoch 的 Train 和 CV 都输出以下形式的一行，数值是该 epoch 的样本加权平均：

```text
TSE_EPOCH epoch=1/50 phase=train loss=... loss_a=... loss_b=... loss_wrong=... samples=...
TSE_EPOCH epoch=1/50 phase=val loss=... loss_a=... loss_b=... loss_wrong=... samples=...
```

可以用 `grep 'TSE_EPOCH' experiments/tse_dual_wrong/<node>/run_job<JOBID>/logs/*.log` 查看全部 epoch，或 `tail -f` 查看当前作业日志。原 TensorBoard、分组 JSON 指标与最佳 checkpoint 机制继续保留。

终端 epoch 日志不再输出 `overlap_loss` / `nonoverlap_loss`。这两项只是已 detach 的分组统计，
不是额外训练目标；移除终端字段不改变反向传播。分组 JSON / TensorBoard 仍保留用于分析。

## 文件改动

| 文件 | 作用 |
|---|---|
| `datasets/tse_dual_dataset.py` | 独立 Dataset、来源选择、三条件 GT、动态 Train / 固定 CV |
| `datasets/tse_dual_mix.py` | 双来源受控插入、重采样、原缩放逻辑复用 |
| `datasets/tse_ontology.py`、`metadata/audioset_ontology.json` | Ontology 关系过滤和固定本地快照 |
| `loss_functions/tse_ri_loss.py` | 指定 RI L1 + Magnitude L1 的波形/RI 接口适配 |
| `tse_dual_separator.py`、`utils/tse_dual_metrics.py` | 三条件优化与分组评估 |
| `exp_defs/tse_strong_dual_fsd_wrong.yaml`、`tools/run_tse_dual_wrong.py` | 独立配置与长期运行入口 |
| `tools/submit_tse_dual_wrong.py`、`slurm/tse_dual_wrong.sbatch` | 01/02 独立提交、节点配置冻结、复用已有串行续训机制 |
| `creators/dataset.py`、`creators/loss.py`、`train.py` | 新实验路由 |
| `nets/tse_dprnntf.py` | 可选原始 RI 输出，默认关闭 |
| `utils/tse_config.py`、`utils/training_config.py` | 新分支参数约束 |
| `tse_separator.py` | 背景数量日志使用实际背景列表长度 |

## 本次检查记录（2026-10-23）

- 指定 Loss 分支与上传 `ri_loss.py` 的数值、梯度对照通过；零 GT、零混音损失及梯度有限，Wrong 残留输出有抑制梯度。
- 200 组插入几何检查通过，涵盖不重叠、25%–50% 部分重叠及不可行情况。
- 动态 Train、CV 逐元素一致重放、源码/来源变更拒绝、GT 与混音分量求和、三条件语义约束及分割泄漏检查通过。
- 从服务器只读复制 6 条 AudioSet 背景和 12 条 FSD 到本地临时目录，Train/CV 各抽查 6 条混音。完整 5,402,679 参数模型（emb=128、hidden=200、4 层）在 CPU 上完成 1 次参数更新与 2 个 CV batch，三个分支损失及分组指标均为有限值，覆盖重叠和非重叠。
- 缩小模型完成原 `train.main` 入口的单步训练、验证、最佳/续训 checkpoint 写入和严格权重重载；旧 TSE 的默认输出接口及原损失兼容检查通过。
- 本地语法和 Git 空白检查通过。未开展全量语料训练、GPU 显存测试或模型效果评估；单步检查只能证明运行和数值链路正常。
- 后续三作业入口检查：两节点配置隔离、只读预览、不变配置快照、首次作业与两个 `afterany` 依赖链的模拟提交均通过；Train/CV 每 epoch 的总 Loss 和分支 Loss 日志断言通过。未在检查中提交真实 Slurm 作业。

按用户要求，完成后删除本次一次性测试脚本、合成夹具、真实音频临时副本、测试缓存、checkpoint 与报告；仅在本文保留检查结论。长期运行入口与 Ontology 快照属于实验实现，保留在项目中。

## 2026-10-24：采样训练与关闭梯度重计算

原配置继续使用三个 Condition、合并反向和 gradient checkpointing。新增独立配置
`exp_defs/tse_strong_dual_fsd_wrong_fast.yaml`：

```yaml
tse_gradient_checkpointing: false
tse_train_condition_mode: sampled_correct_wrong
tse_condition_backward: sequential
```

Train 对 batch 中每条混音独立以 1/2 概率选择 A 或 B，正确 GT 同步选择；另算 Wrong。
实际训练损失为 `2/3 * Loss_selected_correct + 1/3 * Loss_wrong`，其期望及期望梯度
与原三项平均一致。抽样增加梯度方差，同样 epoch 数不保证相同收敛效果。CV 始终完整计算三个
Condition，并保持原三项平均、固定混音与分组指标。

`sequential` 对每个 Condition 的加权损失立即反向并释放计算图，两个分支梯度累积后仅进行
一次梯度裁剪、一次 Adam 更新。学习率调度器仍根据完整 CV loss 每轮更新一次。
该模式要求 `accumulate_grad_batches: 1`；不改变模型结构或 RI Loss 公式。

Train epoch 总 loss 按每条混音的实际加权估计量累计，不再使用三个条件各自均值的简单平均。
A/B 指标仅统计本轮实际抽中的样本，日志增加 `count_a`、`count_b`、`count_wrong`；
`samples` 仍是混音总数。每轮首个完整更新后输出 `TSE_TRAIN_READY`，包含开关、计算方式、
global_step 和 CUDA 峰值显存，便于核验续训是否真正生效。

### 已有实验续训

```bash
# 仅写入该 run 下的续训设置；不会修改 checkpoint 或停止正在运行的进程。
python -B resume_training.py --run-dir /absolute/path/to/run \
  --configure-only --gradient-checkpointing off \
  --condition-mode sampled_correct_wrong --condition-backward sequential

# 正常续训入口；现有 Slurm 接续作业也会自动读取这些设置。
python -B resume_training.py --run-dir /absolute/path/to/run

# 新实验使用独立配置，原配置保留。
python -B tools/submit_tse_dual_wrong.py --node ailab02 --num-jobs 3 \
  --exp-def exp_defs/tse_strong_dual_fsd_wrong_fast.yaml --submit
```

设置保存在该 run 的 `tse_runtime_overrides.json`，包含变更历史。仅允许覆盖上述三个运行字段，
数据、混音、模型结构和损失定义的续训一致性检查继续有效。自动/手动优化切换时迁移 Lightning
的 optimizer-step 计数，恢复原 Adam、学习率调度器、epoch 和 global_step。原 checkpoint
不会被改写；未保存的 epoch 仍从最近 checkpoint 重跑。配置文件不支持运行中热更新。

### 短测记录

02 的 RTX 5880 Ada，空闲 GPU，batch=2、10 秒、FP32、完整 5.4M 模型；固定合成输入，
包含 RI Loss、分组指标、完整反向、梯度检查/裁剪和 Adam。每组 1 步预热 + 3 步计时取中位数：

| 方式 | 秒/步 | 峰值 allocated / reserved (GiB) |
|---|---:|---:|
| 三条件，开启重计算，合并反向 | 2.664 | 16.89 / 23.31 |
| 三条件，关闭重计算，合并反向 | OOM | 已 allocated 44.24，仍需分配 |
| 两条件，关闭重计算，合并反向 | OOM | 已 allocated 44.24，仍需分配 |
| 三条件，关闭重计算，分次反向 | 2.020 | 41.34 / 44.13 |
| 两条件，关闭重计算，分次反向 | 1.356 | 41.35 / 44.13 |

保留三条件时耗时减少约 24%，未达到用户指定的“每步耗时减半才保留三项”的阈值，因此选择
采样方案。最终方案的计算短测耗时减少约 49%，吞吐约为原来的 1.96 倍；这不包含全量数据加载、
数据审计和完整 CV，不能作为整轮耗时或收敛速度承诺。关闭重计算后需要近乎独占的 48 GB 卡。

本地小模型检查通过：两组互补 A/B 抽样的平均梯度等于完整目标；分次与合并反向梯度一致；
真实 Lightning 自动优化 checkpoint 转手动优化续训后，epoch/global_step、Adam step、scheduler
正确衔接；CV 三条件完整、Train 采样计数和加权指标正确，数据配置变化仍会拒绝续训。
一次性测试使用临时目录，结束自动删除，不保留本地测试文件。

服务器真实音频检查也已通过：重放 02 固定 CV manifest 中一条非重叠、一条重叠混音，四路波形
哈希与原 manifest 一致；从原 `run_job39255/mylast.ckpt` 恢复完整模型和 Adam，执行 2 次更新
与 1 个完整三条件 CV batch，global_step 从 5750 到 5752，损失与梯度有限，峰值 allocated
41.395 GiB、reserved 44.668 GiB。检查只在临时目录写报告，未改写正式 checkpoint。

## 01 从头训练：每条混音只抽取一个 Condition

独立配置：`exp_defs/tse_strong_dual_fsd_wrong_single.yaml`。数据、混音及 RI Loss 公式沿用原实验，
模型与优化器从头初始化，输出目录由新 Job ID 隔离，旧训练产物保留。

```yaml
batch_size: 2
tse_window: 256
tse_hop: 128
tse_gradient_checkpointing: true
tse_train_condition_mode: sampled_one
tse_condition_backward: combined
log_every_n_steps: 1
tse_log_batch_loss: true
tse_record_overlap_loss: false
```

Train 对每条混音独立抽取 A、B、Wrong，概率各为 1/3，按抽中的类别同步选择 Condition 和 GT。
整个 batch 只调用模型一次，训练损失直接是选中条件的 RI+Magnitude Loss，不再额外乘 1/3
或 2/3。其期望等于原三项平均；batch=2 时两条混音可以抽中不同类别。
CV 保持完整三条件评估和原三项平均。02 的 `sampled_correct_wrong` 双前向方案不变。

01 共享 GPU 显存，因此新配置继续启用梯度重计算。每轮每个训练 batch 都在控制台输出：

```text
TSE_BATCH epoch=1/50 phase=train batch=1/5750 global_step=1 loss=... sR=... sI=... sM=... loss_a=... loss_b=... loss_wrong=... count_a=... count_b=... count_wrong=... condition_mode=sampled_one
```

`loss = sR + sI + sM`（允许浮点舍入差异）。未抽中的类别显示 `n/a`、count=0，不为填充日志
额外执行前向。训练 epoch 的 A/B/Wrong count 之和等于混音数；CV 三项 count 各等于混音数。
训练总 loss 按实际抽样样本累计，不能直接拿三个条件各自的均值做平均替代。

`tse_record_overlap_loss: false` 同时关闭 JSON 与 TensorBoard 中重叠/非重叠组的总 Loss 和
各条件 Loss；保留分组 SI-SNR、SI-SNRi、Wrong RMS/能量比及计数。终端 epoch 日志原本已去掉
两个分组 Loss 字段。该开关不改变混音的 50% 重叠比例或任何梯度。

在项目根目录提交新实验（3 个串行作业，每个最长 2 天，累计目标 50 epochs）：

```bash
/home/syy12632604/anaconda3/envs/audioset/bin/python -B -u \
  tools/submit_tse_dual_wrong.py --node ailab01 --num-jobs 3 --time 2-00:00:00 \
  --exp-def exp_defs/tse_strong_dual_fsd_wrong_single.yaml --submit
```

检查覆盖：三组循环抽样的平均 Loss/梯度与完整目标一致；Condition/GT 对齐；Train 单次前向、
CV 三次前向；原双条件方案梯度兼容；真实 Lightning 更新与 checkpoint 重载；首个及每个 batch
日志；JSON/TensorBoard 不包含分组 Loss。一次性检查使用内存和临时目录，结束后删除。
