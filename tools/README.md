# TSE 工具

所有工具在本项目内解析依赖。服务器共享代码为 `/home/syy12632604/projects/TSE`。

| 工具 | 用途 |
|---|---|
| `submit_training.py` | 三类 TSE 的节点配置、预览、提交及新 run 续训链 |
| `submit_tse_dual_wrong.py` | 使用双 FSD 默认配置的提交快捷入口 |
| `run_tse_hierarchy_v2.py` | 层级版本配置计划和训练 |
| `run_tse_dual_wrong.py` | 双 FSD 数据检查、有界训练、固定 CV 评估 |
| `evaluate_tse.py` | 单目标数据审计与固定 CV 评估 |
| `run_tse_smallfit.py` | 固定小样本准备、审计与拟合 |
| `run_tse_smallfit_direct_cpu.py` | 已准备的小样本 CPU 拟合恢复 |
| `run_tse_singlefit.py` | 已审计单样本 GPU 拟合 |
| `export_tse_audio_compare.py` | 从显式 run 导出单目标 CV 试听；自动读取实际最佳 epoch |
| `export_tse_dual_preview.py` | 双目标试听页及音频导出 |
| `infer_tse_source_tests.py` | 来源条件推理 |
| `infer_tse_wrong_condition.py` | 错误条件推理 |
| `infer_tse_dual_fsd.py` | 双 FSD 条件推理 |
| `audit_common.py`、`tse_smallfit_checks.py` | 数据重放、泄漏和数值检查基础函数 |

训练诊断输出放到 `experiments/diagnostics/<unique-name>`，不写入 `/data`。
小样本工具的 `--background-split` 指向新项目已生成的背景清单；先运行数据检查。
音频缓存继续留在计算节点。推理工具的默认输出位于新项目 `experiments/inference/`，
重复执行时使用新的 `--output`，不覆盖历史导出。

源码哈希用于固定 CV 的重放检查。更改源码/数据规则后为新实验使用新清单，
不要修改已有 manifest 或关闭校验。拆分前的实验操作文档保存在 `docs/history/`。
