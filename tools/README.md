# TSE 工具

所有工具在本项目内解析依赖。服务器共享代码为 `/home/syy12632604/projects/TSE`。

| 工具 | 用途 |
|---|---|
| `submit_training.py` | 三类 TSE 的节点配置、预览、提交及新 run 续训链 |
| `run_tse_dual_wrong.py` | 双 FSD 数据检查、有界训练、固定 CV 评估 |
| `evaluate_tse.py` | 单目标数据审计与固定 CV 评估 |
| `export_tse_audio_compare.py` | 从显式 run 导出单目标 CV 试听；自动读取实际最佳 epoch |
| `export_tse_dual_preview.py` | 双目标试听页及音频导出 |
| `audit_common.py` | 评估工具共用的数据重放、泄漏检查函数 |
| `check_local.sh` | Python/shell 语法与 Git 空白检查 |

训练诊断输出放到 `experiments/diagnostics/<unique-name>`，不写入 `/data`。
音频缓存继续留在计算节点。试听导出使用新的 `--output`，不覆盖历史导出。

层级版和 pilot 也使用统一提交器，例如：

```bash
python tools/submit_training.py --node ailab01 \
  --exp-def exp_defs/tse_hierarchy_v2_pilot_10pct_bf16.yaml --num-jobs 1
```

默认仅预览；添加 `--submit` 才提交。直接执行训练与恢复分别使用项目根目录的
`train.py` 和 `resume_training.py`。旧的一次性拟合、特殊条件推理和重复提交入口
已删除；正式导出所需的只读 FSD 解码在 `datasets/fsd50k_cache.py`，pilot 的指标
函数在 `utils/tse_dual_metrics.py`，训练不再依赖已删除的测试脚本。

源码哈希用于固定 CV 的重放检查。更改源码/数据规则后为新实验使用新清单，
不要修改已有 manifest 或关闭校验。拆分前的实验操作文档保存在 `docs/history/`。
