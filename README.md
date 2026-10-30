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
