# TSE：条件目标声音分离

本项目从 `new_example_code` 独立拆出，使用类别 MID 条件提取目标声音波形。
SED 留在原项目。公共工具在本项目内保留独立副本，不导入另一个项目的代码。
网络、损失、条件采样和混音规则延续拆分前的实现；新实验从随机权重开始。

## 位置与数据

- 本地代码：`/Users/shenyuanyang/Desktop/TSE`
- 服务器共享代码：`/home/syy12632604/projects/TSE`
- 新训练结果：`/home/syy12632604/projects/TSE/experiments/`
- 原始 AudioSet、FSD50K 和可复用音频缓存：继续使用计算节点上的现有目录。
- 新 CV 清单、背景筛选清单和条件词表：`experiments/manifests/<variant>/<node>/`。
  原有父划分和 FSD 官方划分仍只读使用，不重新划分训练/验证成员。

在 Slurm 分配的节点上执行共享代码；不要在实验目录或 `/data` 创建代码副本。
每次训练使用独立 run，保存有效配置、Git revision、全部相关源码 SHA-256、
TensorBoard、指标、checkpoint 和调度日志。历史 TSE 实验留在原项目，不迁移、不改写。

## 保留的实验

| 配置 | 任务 |
|---|---|
| `tse_strong_fsd50k.yaml` | 单个 FSD 目标，波形与多分辨率 STFT 损失 |
| `tse_strong_dual_fsd_wrong.yaml` | A/B/Wrong 三条件完整训练 |
| `tse_strong_dual_fsd_wrong_fast.yaml` | 随机正确条件 + Wrong，分次反向 |
| `tse_strong_dual_fsd_wrong_single.yaml` | A/B/Wrong 等概率单条件训练 |
| `tse_strong_dual_fsd_hierarchy_v2.yaml` | 层级标签归约后的 188 类版本 |
| `tse_hierarchy_v2_pilot_10pct_bf16.yaml` | 保留现有 10%/BF16 pilot 设置与诊断 |
| `tse_hierarchy_v2_smallfit.yaml` | 八类固定 256/128 条混音，8000 step 分离能力小训 |
| `tse_hierarchy_v2_smallfit_correct.yaml` | 重放 job39909 的同一批混音，8000 step，仅 A/B 训练 |

配置都在 `exp_defs/`。依赖版本见 `requirements.txt`；可继续使用现有 `audioset` 环境。

## 提交新训练

常用入口是 `run.sh`，结构参考 `example_code`：顶部集中配置实验和资源，
`spksep_run.sh` 负责阶段和运行方式，`tools/submit_training.py` 负责校验参数、
冻结配置、命名输出和提交作业。网络、loss、混音与条件采样实现不由启动脚本改写。

在 `run.sh` 顶部修改 `node`、`exp_def`、`num_jobs`、`cpus`、`mem`、`time_limit`
以及实验 `tag`。`batch_size`、`learning_rate`、`epochs` 等训练参数留空时继承
所选 YAML，需要覆盖时再赋值。这样切换 pilot YAML 会保留其 5 epoch、10% 和
BF16 设置。命令行参数优先于脚本顶部的设置，支持连字符或下划线两种写法。

从服务器共享项目执行，默认 `stage=0` 仅打印计划，不创建模型、目录或作业：

```bash
cd /home/syy12632604/projects/TSE
bash run.sh
bash run.sh --node ailab02 --batch-size 4 --learning-rate 0.0001 --tag baseline
# 确认预览后提交；也可直接在 run.sh 顶部设置 stage=1。
bash run.sh --stage 1 --node ailab02 --batch-size 4 --learning-rate 0.0001 --tag baseline
```

`--submit` 等价于 `--stage 1 --use-slurm 1`。`--node ailab02` 使用 ailab02
的数据路径；`--num-jobs 3` 安排总共三个串行作业，后续作业恢复同一新 run。
pilot 导出配置使用 `--num-jobs 1`。资源可通过 `--cpus`、`--mem`、`--time` 指定。
服务器默认选择 `/home/syy12632604/anaconda3/envs/audioset/bin/python`；
可通过 `PYTHON_BIN` 或 `--python-bin` 指定环境。本地预览需选择已安装项目依赖的 Python。

`--manifest-tag cleanup_v1` 将派生清单放到
`experiments/manifests/<variant>/<node>/cleanup_v1/`，用于源码或混音设置更新后的新实验。
它不改变原始 Train/CV 成员或音频缓存；同一命名空间仍严格校验清单，不能覆盖旧内容。

初始调度日志和提交时的配置位于 `experiments/.submissions/<submission>/`；
训练目录为 `experiments/<variant>/<node>/run_b<batch>_lr<lr>_e<epochs>_<precision>[_<tag>]_job<ID>/`。
例如 `run_b2_lr0.0001_e50_32-true_baseline_job12345`。续训调度日志位于该 run 的
`continuation/scheduler_logs/`。有效配置和源码哈希完整保存；旧 `run_job<ID>` 目录仍可恢复。

已经获 Slurm 分配时，使用 `bash run.sh --stage 1 --use-slurm 0 --node ailab01`，
仍由同一流程冻结配置、创建独立 run 并保存日志。脚本验证实际分配节点与 `--node`
一致，沿用 Slurm 的 GPU 可见性。默认预览也可在该模式下执行。

`python tools/submit_training.py --node ailab01 --exp-def <config>` 的预览/提交接口
继续可用。底层 `train.py --exp-def <config> --save-name <unique-name>` 保留，需在
分配的计算节点执行并自行提供正确数据路径；常规训练推荐使用 `run.sh`。

## 恢复拆分后的新训练

```bash
python resume_training.py --run-dir /home/syy12632604/projects/TSE/experiments/<variant>/<node>/<run>
```

恢复读取该 run 的冻结设置和 checkpoint，保留优化器、学习率调度器与训练进度。
首次训练记录 `.tse_run_created`；迁移前的 checkpoint 不用于本项目续训。
不要移动已开始的新 run。pilot 专用诊断保留单次运行限制。

## 修改 loss 后的固定数据小训

入口仍是 `run.sh → spksep_run.sh → tools/submit_training.py → train.py`。
可以在 `run.sh` 顶部选择 `exp_defs/tse_hierarchy_v2_smallfit.yaml`，或直接传参。
具体类别、混音数量、最大 step 和评估间隔集中在这个 YAML 中；不修改正式训练配置。

```bash
cd /home/syy12632604/projects/TSE
# 预览：不读取音频、不创建 run、不提交作业。
bash run.sh --exp-def exp_defs/tse_hierarchy_v2_smallfit.yaml \
  --node ailab01 --num-jobs 1 --tag smallfit8k --manifest-tag smallfit8k_v1
# 提交：数据构建、基线检查、训练和评估均在分配的 GPU 节点执行。
bash run.sh --stage 1 --exp-def exp_defs/tse_hierarchy_v2_smallfit.yaml \
  --node ailab01 --num-jobs 1 --tag smallfit8k --manifest-tag smallfit8k_v1 \
  --cpus 8 --mem 32G --time 1-00:00:00
```

服务器 `audioset` 环境已有 matplotlib；其他环境需要安装 `requirements.txt` 中的绘图库。
本次配置从随机权重训练到 **8000 次参数更新**，保留 batch=2、LR=1e-4、BF16、
`sampled_one`、原模型和 RI+Mag loss。训练/验证比例都是浮点 `1.0`。
63 epoch 只是覆盖 8000 step 的上界；256 条训练数据每轮 128 个 batch，
实际在第 63 轮中途达到 step 上限停止，不会等待完整的第 63 轮。

八类为 Bark、Meow、Cough、Computer keyboard、Walk/footsteps、Piano、Clapping、Thunder。
保持完整的 188 类训练词表和 527 维条件输入，只限制本次正、负条件的抽样范围。
按原始完整标签排除冲突，不能合法共存的类别不会硬配对（例如带 Animal 父标签的 Bark/Meow）。
每个 split 的 A/B 类别计数分别均衡；每个背景各生成一条非重叠和一条重叠混音。
八类范围内没有合法配对的候选背景按配置的 `exclude` 策略排除，并记录在数据报告中。
沿用官方音源划分和原背景划分，检查 ID、原视频、路径及内容哈希泄漏。
新清单记录源文件、裁剪、位置、SNR、种子、增益和最终波形哈希，生成后预载到内存，
每轮只打乱顺序，不重新抽取声音或重新混音。原始音频和可复用缓存保留在计算节点。

step 0 做完整 Train/CV 基线和 oracle RI、零输出、原混音对照，检查 STFT/iSTFT 重建。
之后每 **500 step** 在 eval 模式下完整评估 256 条训练混音和 128 条验证混音，
每条都计算 A/B/wrong。1000、2000 step 在汇总中标为重点检查点，**不自动早停**。
ReduceLROnPlateau 只在这些完整评估结束时读取新的 val_loss，patience=5 按评估次数计；
不会在中间短 epoch 上重复使用旧的验证值。训练日志仍保留 sampled_one 的 batch/epoch loss；
对比分离能力请用 `train_eval` 与 `val`。

输出位于 `experiments/tse_hierarchy_v2_smallfit/ailab01/<run>/`：

- `effective_config.yaml`、`source_provenance.json`：有效配置、Git revision、源码 SHA-256。
- `fixed_data/`：Train/CV 固定清单、类别/重叠分布和来源隔离检查。
- `step_metrics.jsonl`、`evaluations/step_XXXXXXXX/`：逐步汇总及逐条件 CSV；
  A/B 的 SI-SDRi 均值、中位数、改善比例和输出/GT 能量比，wrong 输出/混音能量比，
  重叠分组指标与 A/B 输出相关系数。`sR/sI/sM` 的 CSV 值是所在 batch 的分量均值，
  汇总按实际样本数加权；总 loss 和 SI-SDRi 是逐样本值。
- `listening/train_eval/`、`listening/val/`：各八条固定样本，覆盖八类 A 条件和两种重叠情况。
  混音、GT、背景只存一次；每个 step 保存三条件 FLOAT WAV 和六联频谱图。
  音频不独立归一化，频谱统一使用 -100～0 dB 色标。
- `checkpoints/step_XXXXXXXX.ckpt`：每 500 step 的模型；`version_<job>/` 保留最佳 val_loss
  checkpoint 和 TensorBoard，`mylast.ckpt` 保留最新状态，`status.json` 记录运行状态。
- `logs/training.log`：训练日志；初始 Slurm 日志仍在提交器打印的 `.submissions/` 路径。

此入口是单次分配的诊断训练，禁止续训或串行续作业；中断后使用新 run 从头启动。
固定小训并非全类别效果评测；通过后再扩大类别/数据，并使用原完整验证集。

### 去掉 wrong 训练的对照

`tse_hierarchy_v2_smallfit_correct.yaml` 保留 batch=2、LR=1e-4、BF16、8000 step、
原模型及 RI+Mag loss。`sampled_correct` 每条混音等概率选择 A 或 B，只有一次前向；
所选条件的 loss 权重为 1，不沿用三条件实验的 2/3 权重。训练日志的 `count_wrong` 为 0。

配置中的 `tse_smallfit_reference_run` 指向已经完成的 job39909，只读取它的固定样本清单，
不加载权重或优化器。新 run 校验数据设置、源音频与混音代码哈希，并逐条重放校验裁剪、
位置、增益及最终波形哈希。数据或参考清单不匹配时直接报错，不重新抽样替代。
`fixed_data/summary.json` 记录参考清单路径、SHA-256 和相同的 `rows_sha256`。

完整评估仍计算 A/B/wrong，并导出三种预测。主 `loss`、`val_loss`、最佳 checkpoint 和
学习率调度只使用 A/B 平均；`loss_conditions` 明确记录 `[a, b]`。
`correct_loss` 用于与旧实验的 `(loss_a + loss_b) / 2` 比较；
`all_conditions_loss` 和 `conditions.wrong` 只供诊断，不参与选模或调度。
step 0 的主 `zero_loss` 也按 A/B 平均，避免把三条件零输出基线混入比较。
同样的 8000 更新下，正确目标监督次数预期增加 50%；该对照不能单独区分这一因素与移除静音负例的影响。

更新服务器代码后，在登录节点、未进入其他 Slurm 作业时执行。实际训练分配到 **ailab01**：

```bash
cd /home/syy12632604/projects/TSE
bash run.sh --stage 1 --use-slurm 1 \
  --exp-def exp_defs/tse_hierarchy_v2_smallfit_correct.yaml \
  --node ailab01 --num-jobs 1 --tag correct8k --manifest-tag correct8k_v1 \
  --cpus 8 --mem 32G --time 1-00:00:00
```

去掉 `--stage 1` 可先只预览。新结果写入
`experiments/tse_hierarchy_v2_smallfit/ailab01/run_b2_lr0.0001_e63_bf16-mixed_smallfit_s8000_correct8k_job<ID>/`，
每 500 step 的评估、试听和 checkpoint 布局沿用前述小训。

## 检查与工具

所有配置预览和作业提交统一使用 `run.sh` / `tools/submit_training.py`。
pilot 使用同一提交器，传入对应配置以及 `--num-jobs 1`。
`tools/evaluate_tse.py` 提供单目标数据审计和固定 CV 评估；
`tools/run_tse_dual_wrong.py` 保留双目标的 check、smoke、validate、train 入口。
单目标与双目标各保留一个固定 CV 试听导出工具，见 [工具清单](tools/README.md)。
`tools/check_local.sh` 检查 Python/shell 语法与 Git 空白错误。

## 数据代码与清理范围

`creators/dataset.py` 只选择三类正式数据集：`tse_dataset.py`（单目标）、
`tse_dual_dataset.py`（双目标 + Wrong）、`tse_hierarchy_dataset.py`（层级版）。
`strong_mix.py` 只负责背景筛选和已有划分校验；`paths/` 读取语料索引，
`fsd50k_cache.py` 处理 FSD 读取与重采样，`mix_utils.py` / `active_rms.py` /
`tse_dual_mix.py` 负责裁剪、SNR 和混音。两个 `mix_*_audit.py` 是训练所需的
源音频检查，保留以排除无效音频。标签层级与冲突检查在 `tse_ontology.py` 和
`tse_hierarchy_labels.py`。

RMS 按零均值波形计算；前景使用活动帧 RMS，背景使用完整 10 秒 RMS。
`mix_snr_db: null` 时按 `mix_snr_db_range` 在 dB 域均匀采样，指定数值则使用
固定 SNR。Train 动态采样，CV 保存并严格重放同一配对、seed、裁剪和增益。

2026-10-30 清理了旧 SED 的 log-mel / 帧标签数据集、旧双 FSD 裁剪实现、
一次性 smallfit / singlefit / 来源和错误条件推理实验，以及重复提交脚本、
相同内容的历史配置、过期工具说明和网络工具末尾的独立测试示例。仍在使用的 CV 清单、音频读取和 pilot
指标函数已移入正式模块。删除内容可从 Git 历史查阅。

此次清理改变源码哈希；新实验须使用新的 CV manifest 路径。已有实验继续配套
原代码和原清单，不覆盖清单、不修改其哈希，也不放宽重放校验。

拆分前文档原文保存在 `docs/history/`，其中旧路径和旧作业号仅作历史记录。
新运行以本 README 和当前配置为准。来源和验证见 [拆分记录](docs/PROJECT_SPLIT.md)。
