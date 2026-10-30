# 当前 SNR 范围采样

```yaml
mix_snr_db: null
mix_snr_db_range: [-5.0, 5.0]
```

`mix_snr_db` 为 null 时在 dB 域均匀采样；指定有限数字时优先采用固定值。公共采样与缩放均位于 `datasets/mix_utils.py`：

```python
sample_snr_db(snr_db_range=(-5., 5.), *, sample_id=None, seed=0, rng=None)
scale_foreground_to_snr(foreground, *, snr_db, background_rms,
                        reference_mask=None, reference_rms=None)
```

给定 `sample_id + seed` 时，使用 SHA256 派生的独立 RNG；不影响文件选择、裁剪和位置随机流。无 manifest 的固定评估调用必须提供稳定的 ID 和 seed。

| 主线 Dataset | Train | CV |
|---|---|---|
| `AudioSetFSD50KMixDataset` | 每次读取的动态混音 seed 派生一个新 SNR | 每条混音确定 SNR 并写入 manifest，读取时直接复用 |
| `StrongMixtureCore` / `StrongFSD50KMixDataset` | 继承单 FSD 动态混音，条件展开方式保留 | 条件共享底层固定 SNR，bundle 共享同一次混音 |
| `DualFSDMixDataset` | 每条混音采一个目标 SNR，A/B 分别相对背景缩放并共用该目标值 | 行级及 event 级保存 SNR，固定重放 |

`get_with_metadata()` 中 `snr_db` 表示请求值，保留 `snr_db_requested`、实测 SNR、误差和 `snr_target_met`。静音回退不承诺达到目标；实测值可能低于采样范围。

三条混音主线均使用 active RMS，背景均使用完整 10 秒。增益前按 20 ms 非重叠帧计算 zero-mean RMS，选择超过 −40 dBFS 的帧；对这些帧的样本并集再计算一次 zero-mean RMS。mask 在缩放前确定，缩放后不重新判定。

采用功率比法（用户提供的第一种方式），在项目中保持背景、缩放前景：

```python
foreground_gain = sqrt(10 ** (snr_db / 10)) * background_rms / foreground_active_rms
```

这等价于原公式的噪声增益 `sqrt(P_foreground_active / P_background / 10**(snr_db/10))` 的倒数。功率由公共 zero-mean RMS 平方得到。这里在 dB 域均匀采样，不采用在线性幅度区间均匀采样的第二种方式。共同峰值归一化、GT 算法、特征 std 归一化和采样布局保留。

无活跃帧时原公式返回的噪声增益为 0，不能取倒数；当前各主线保留有界回退 `max(crop RMS, 10**(threshold_dbfs/20))`，记录 `rms_fallback` 和实际 SNR，不声称达到目标。全静音/纯 DC 前景池条目仍由数据审计排除。

`snr_db_exported` 是 active 前景 / 完整背景的实测值（回退时为 crop / 背景）；旧字段 `snr_db_active_region_exported` 仅作插入区间的局部 crop SNR 诊断，不是缩放依据。

四条主线的 config 中保留三套混音配置，范围可通过 YAML 或 `--mix-snr-db-range '[-10.0, 10.0]'` 覆盖，active 检测参数为 `mix_active_threshold_dbfs` / `mix_active_frame_ms`。范围或 active 参数变化须新 manifest 并重新审计。当前版本使用 `*_active_snr_v2/cv_{node}.json`；旧 checkpoint/manifest 须配套旧代码评估，不可直接用新版混音重放。原始 AudioSet 强标签主线不做人工混音。
