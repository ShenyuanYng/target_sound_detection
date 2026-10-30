# FSD 层级去重版 TSE v2

独立 dataset：`audioset_strong_dual_fsd_hierarchy_tse_v2`。配置为
`exp_defs/tse_strong_dual_fsd_hierarchy_v2.yaml`，入口为
`tools/run_tse_hierarchy_v2.py`。当前只准备代码，不提交或启动训练。

## 标签和混音规则

1. 从完整的 `original_mids` 出发，沿 AudioSet ontology 的全部父节点递归寻找祖先。
   仅当某个标签的后代也出现在同一条音频的原始标注中，才去掉该祖先标签。
   例如 `Animal + Dog + Bark` 变成 `Bark`；`Animal + Dog` 变成 `Dog`，不会推断为 `Bark`。
2. 去重后恰好一个标签、在 527 类映射中且不是 Silence，才可作为正目标。
   `Bark + Speech` 等仍有多个具体标签的音频不参与正目标采样，也没有多标签回退。
   先去祖先、后检查映射，避免把未映射的最具体标签错误替换成父类。
3. `mids`、`original_mids` 均保留；新增 `most_specific_mids` 和 `positive_mid`。
   音源 A/B、背景及 Wrong 的冲突检查仍使用完整原始标签的精确、祖先、后代关系。
   例如保留了 Animal 原始标签的 Bark 音源，仍会保守地排除其他动物类 Wrong。
4. Wrong 词表只由官方训练集的有效正目标类别生成：先应用既有跨分区重复排除、
   音频审计排除，再进行层级去重。当前配置要求恰好 188 类，否则报错。
   Train/CV 共用该词表，单次 Wrong 再从其中排除与背景、A、B 冲突的类别。
5. 一条样本仍为背景 + 两段 FSD 音源；A/B 目标仍为各自经过裁剪、放置、增益和归一化后的
   完整音源，Wrong 目标为零。每个背景仍各有一个非重叠和部分重叠槽位；训练随机采样、CV 固定。
   网络保持 527 维条件输入。这里的 188 是可采样条件类别数，不改变网络输入维度。

新代码已用现有标签和审计排除项核对：训练池为 **32,726 条 / 188 类**，
验证池为 **3,308 条 / 163 类**。每类条数见
[标签覆盖清单](tse_hierarchy_v2_label_inventory.json)。这是标注覆盖，不表示录音中一定只有一种可听声源。

## 独立路径和启动

所有新输出都在 `/data/syy12632604/tse_hierarchy_v2/`：

| 用途 | 子路径 |
| --- | --- |
| 独立服务器代码快照 | `code_20261027/` |
| 音频缓存 | `fsd_cache_16k/` |
| 正目标/Wrong 词表 | `manifests/condition_vocabulary.json` |
| 背景划分 | `manifests/background_split.json` |
| v2 固定验证集 | `manifests/cv_seed7.json` |
| 模型、日志和训练报告 | `experiments/tse_hierarchy_v2_single/` |

原始 FSD、AudioSet 数据仍从原路径读取，不复制数据集。配置拒绝将上述输出写入 `/home`
或旧版缓存、清单目录。CV 使用 schema 2，记录新 dataset、标签策略、代码指纹、原始与去重标签。
词表和清单生成后不可静默覆盖；输入、策略变化需要新的输出路径。

服务器代码必须部署到独立目录。不要把本次共享入口的 dispatch 改动同步覆盖正在运行的旧项目。
旧任务继续使用 `/home/syy12632604/projects/new_example_code`。

默认入口仅显示配置规划，不导入训练模块、不读取音频、不实例化模型、不生成验证集：

```bash
python tools/run_tse_hierarchy_v2.py
```

将来明确开始新训练时才使用 `--mode train`；当前不执行。默认沿用单条件采样
`sampled_one`、batch size 2、梯度检查点，验证仍完整评估 A/B/Wrong。
首次实际训练时会审计源音频，并生成独立词表、缓存和固定 CV，这些步骤本次不运行。
不使用 v1 checkpoint 续训；dataset 和标签配置不匹配时会拒绝恢复。
若之后恢复本版本，用同一 v2 配置和 `--resume-run-dir` 明确指定其输出目录。
旧 `submit_tse_dual_wrong.py` 仍只服务 v1，不用于提交这个版本。

## 本次验证范围

14 项标签/路径检查通过，覆盖多父节点、传递祖先、多标签保留、未知标签、未映射后代、
原始标签不变、训练词表排除和路径隔离。服务器环境中的默认规划入口、配置校验、
工厂路由、100 次微型标签组合抽样、完整原始标签冲突检查、旧 checkpoint 拒绝也已通过。
这些检查禁止音频读取，临时元数据夹具已清理。用既有 CSV/JSON 元数据核对了类别覆盖。
不执行模型前向、音频效果测试、训练或数据预处理。真实音频混合和完整训练尚未验证。
