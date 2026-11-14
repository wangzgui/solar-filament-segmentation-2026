# F4 0.40 全流程错误审计

同一张历史照片的预测用于所有标注者视图，计分按官方 IoU > 0.5 与池化口径。
Mul2 有上游训练或历史实验曝光，S0 每张验证图来自对应的两个折外模型，测试集使用十模型。
本报告支持历史错误归因，公开榜收益仍需独立验证。

主视角上游回放验证通过 35 张。增强视角记录为 Refiner 或 coarse 回退后、检测分数与面积过滤后的 donor。
原始 detector 指模型已输出的实例 Mask，覆盖率按逐 GT 最佳 IoU 统计。候选池允许重叠，正式实例集合单独计算 PQ。

| 范围 | 输出 | PQ | SQ | DQ | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| selection | mul2_vote_both | 0.45091 | 0.68639 | 0.65693 | 135 | 77 | 64 |
| selection | f4 | 0.46313 | 0.68966 | 0.67153 | 138 | 74 | 61 |
| selection | s0 | 0.46502 | 0.69248 | 0.67153 | 138 | 74 | 61 |
| audit | mul2_vote_both | 0.46171 | 0.68402 | 0.67500 | 162 | 72 | 84 |
| audit | f4 | 0.46343 | 0.68236 | 0.67917 | 163 | 71 | 83 |
| audit | s0 | 0.47651 | 0.68115 | 0.69957 | 163 | 57 | 83 |
| paired_all | mul2_vote_both | 0.45673 | 0.68510 | 0.66667 | 297 | 149 | 148 |
| paired_all | f4 | 0.46329 | 0.68571 | 0.67565 | 301 | 145 | 144 |
| paired_all | s0 | 0.47113 | 0.68634 | 0.68643 | 301 | 131 | 144 |

445 个 GT 标注视图实例。Mul2 → F4 救回 6，失去 2，其中从 IoU 0.30–0.50 救回 6。
F4 与 S0 共同漏检 109，S0 独有命中 35，F4 独有命中 35。
共同 TP 有 295 个，F4 相对 Mul2 的平均最佳 IoU 变化为 +0.00277。

**F4 gt 错误频数**

- `weak_boundary_tp` 111
- `complete_miss_fn` 71
- `overmerge_fn` 24
- `near_miss_fn` 20
- `fragmentation_fn` 17
- `weak_overlap_fn` 11

**F4 prediction 错误频数**

- `unmatched_fp` 67
- `fragment_fp` 34
- `near_gt_fp` 32
- `overmerge_fp` 10
- `duplicate_fp` 2

**F4 fn_stage 错误频数**

- `last_observed_loss_admission_to_quality_pass` 42
- `last_observed_loss_quality_pass_to_a1_selected` 32
- `no_hit_in_audited_banks` 29
- `last_observed_loss_g3_to_admission` 26
- `cached_augmented_view_hit_without_final_match` 7
- `last_observed_loss_refiner_to_g3` 5

**F4 的 1 − PQ 分解**

- `fp` 占 PQ 分母 16.27%。
- `fn` 占 PQ 分母 16.16%。
- `matched_iou_shortfall` 占 PQ 分母 21.24%。

**合法 Oracle**

每个 Oracle 固定允许的操作，共用官方多标注评估。轮廓变体限 Mul2、F4 Conservative、F4 Moderate 的同一实例槽。
0 张照片的 Moderate 未通过回放核对，仅这部分 Moderate 候选不进入 Oracle。逐图预期值、实测值和缓存路径见 replay_controls.csv。
救回只考虑能匹配当前至少一个 FN 的整块缓存候选。每张照片跨所有标注者共用同一解，像素互斥、唯一匹配，遇到冠军本身标注匹配歧义的图保留原结果。
G3 恢复采用回放中未进入 A1 最终集合的掩码。质量筛选与冲突的归因由逐阶段覆盖变化单独给出。
数值是给定候选与约束下的理想增量，各项不可相加。日期重采样基于固定的历史 Oracle 解。

| Oracle | 状态 | 照片数 | ΔPQ | ΔTP | ΔFP | 正收益日期数 |
|---|---|---:|---:|---:|---:|---:|
| drop_only | solved | 35 | +0.05284 | -14 | -112 | 28 |
| contour_only | solved | 35 | +0.01052 | +4 | -4 | 34 |
| cached_view_rescue | solved | 35 | +0.02707 | +36 | +10 | 16 |
| gq_rejected_rescue | solved | 35 | +0.02772 | +36 | +11 | 16 |
| s0_rescue | solved | 35 | +0.00752 | +11 | +5 | 7 |
| reselect_cached | solved | 35 | +0.13683 | +75 | -101 | 35 |
| joint_with_s0 | solved | 35 | +0.13702 | +75 | -101 | 35 |

下一步判断使用 `stage_transitions.csv` 与 `oracle_summary.csv`。若 G3 已覆盖而质量或选择阶段大量丢失，优先定位该阶段。
若原始候选与全部可用补充候选都很少命中共同 FN，优先研究候选生成。
若 drop-only 理论空间显著且 FP 主要是重复或碎片，优先审查实例关系。
