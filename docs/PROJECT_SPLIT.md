# SED / TSE 项目拆分记录

2026-10-30 按任务边界拆分。`new_example_code` 保留 SED；同级 `TSE` 为独立代码项目。
来源 Git revision：`69a747735822584eee167a41ef901ac2aa6ad339`。拆分包含当时尚未提交的源码修改，原始文件 SHA-256 见
`project_split_manifest.json`，不能仅靠 Git revision 代表迁移输入。

## 目录与边界

- SED：本地 `/Users/shenyuanyang/Desktop/new_example_code`；服务器 `/home/syy12632604/projects/new_example_code`。
- TSE：本地 `/Users/shenyuanyang/Desktop/TSE`；服务器 `/home/syy12632604/projects/TSE`。
- 新 TSE 训练结果：`/home/syy12632604/projects/TSE/experiments/`。
- 旧实验、checkpoint、冻结配置和原始数据不移动、不改写。
- 公共源码各保留一份。SED 的 `datasets/tse_hierarchy_labels.py` 仍用于层级标签，故保留。
- SED 的网络、数据实现、损失和 5 份实验配置保持原文件内容。
- TSE 保留数据处理与网络实现，拆开工厂/训练入口，训练基础类仅包含本任务实际继承的方法。
- TSE 启动器使用新的代码与结果路径；新背景/CV 清单在 `experiments/manifests/`，音频缓存留在计算节点。
- TSE 新 run 记录 `.tse_run_created`、有效配置和源码来源，后续恢复只针对本项目的新 run。
- 历史 TSE 文档原文位于新项目 `docs/history/`；其中旧路径是历史记录。
- TSE 试听工具读取所选 run 的实际最佳 checkpoint，不再固定第 47 轮。

## 验证

- 两套源码 Python、shell 和 shell 内嵌 Python 语法检查；本地模块依赖完整。
- 所有 5 份 SED、7 份 TSE 配置可校验；TSE 两个节点共 14 组输出路径校验通过。
- 5 套 SED、6 套 TSE：固定输入下输出、损失、梯度和参数与拆分前逐项相等。
- TSE 自动/手动优化两种模式：训练一步保存，再恢复到第二步；检查模型、Adam、调度器保存状态、最佳模型路径、TensorBoard、指标和冻结配置。
- SED 已有 6 项推理测试通过；TSE 3 组试听/混音自检通过。
- 四种 TSE 提交生命周期通过模拟调度验证，包括首次创建 pilot 预览目录；未提交真实作业。
- 训练恢复与数值检查使用合成输入；TSE 缩小通道和层数以控制 CPU 验证成本，不修改正式配置。
- 本次不提交正式训练，不代表完成全量语料审计或 GPU 性能验证。

## 回退

原项目 `.project-split-backups/20261030/` 保存迁移前的源码副本、Git 状态与 diff。
它不包含训练结果副本。不要用覆盖整个项目的方式回退，应从该备份恢复需要的源码文件。
新 TSE 有独立 Git 基线；原 SED 仓库保留未提交变更供检查，没有自动提交或推送。
