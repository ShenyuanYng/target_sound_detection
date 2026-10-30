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

配置都在 `exp_defs/`。历史命名配置 `tse_strong_fsd50k_b4_lr1e4.yaml` 也保留，
新提交优先使用主配置。依赖版本见 `requirements.txt`；可继续使用现有 `audioset` 环境。

## 提交新训练

从服务器共享项目提交，默认仅打印计划，不创建模型或提交作业：

```bash
cd /home/syy12632604/projects/TSE
/home/syy12632604/anaconda3/envs/audioset/bin/python tools/submit_training.py \
  --node ailab01 --exp-def exp_defs/tse_strong_dual_fsd_hierarchy_v2.yaml
```

检查计划中的配置和路径后，添加 `--submit` 提交。`--node ailab02` 使用 ailab02
的数据路径；`--num-jobs 3` 可安排最多三个串行作业，后续作业恢复同一新 run。
pilot 导出配置使用 `--num-jobs 1`。资源可通过 `--cpus`、`--mem`、`--time` 指定。
`run.sh` 是上述提交器的简短入口，使用 `PYTHON_BIN` 选择 Python。

初始调度日志和提交时的配置位于 `experiments/.submissions/<submission>/`；
训练目录为 `experiments/<variant>/<node>/run_job<ID>/`，续训调度日志位于该 run 的
`continuation/scheduler_logs/`。保存的提交记录可以关联两者。

直接训练入口为 `train.py --exp-def <config> --save-name <unique-name>`，
需在分配的计算节点执行。配置默认数据路径对应 ailab01；跨节点优先用提交器。
`spksep_run.sh` 是已获分配节点上的执行包装，不再包含 SED 分支。

## 恢复拆分后的新训练

```bash
python resume_training.py --run-dir /home/syy12632604/projects/TSE/experiments/<variant>/<node>/<run>
```

恢复读取该 run 的冻结设置和 checkpoint，保留优化器、学习率调度器与训练进度。
首次训练记录 `.tse_run_created`；迁移前的 checkpoint 不用于本项目续训。
不要移动已开始的新 run。pilot 专用诊断保留单次运行限制。

## 检查与工具

`tools/run_tse_hierarchy_v2.py --mode plan` 只检查层级配置。
`tools/evaluate_tse.py` 提供单目标数据审计和固定 CV 评估；
`tools/run_tse_dual_wrong.py` 保留双目标的 check、smoke、validate、train 入口。
小样本拟合、单样本拟合和试听导出工具见 [工具清单](tools/README.md)。
`tools/check_local.sh` 检查 Python/shell 语法与 Git 空白错误。

拆分前文档原文保存在 `docs/history/`，其中旧路径和旧作业号仅作历史记录。
新运行以本 README 和当前配置为准。来源和验证见 [拆分记录](docs/PROJECT_SPLIT.md)。
