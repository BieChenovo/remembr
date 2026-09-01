# Published NaVQA reports

GitHub Pages 报告与 `artifacts/eval_reports` 使用同一目录规范：

```text
reports/
├── b0/{v1,v4}/
├── b1/{v1,v2}/
├── b2/{v1,v2}/
├── b3/{v1,v2,v3,v4}/
└── comparison/{v1,v2,v3,v4}/
```

- `v1` 表示修复前的 legacy `per_call + native` 实现。
- `v2` 表示修复后的 question-state 实现。
- `b3/v3` 表示在 v2 基础上进一步实现 controller ↔ retrieval
  闭环、每轮 Q-RAG top-1 的 v3 实验；旧 v1/v2 结果保持不变。
- `b3/v4` 将 text/time/position 统一为 top-1，共享题级
  evidence ledger、全局 ID mask 和 5 条唯一证据预算，并支持 duplicate replan。
- `b0/v4` 保留 mxbai dense scorer，但按修复后的 controller 闭环重跑 210 题；
  text 每次返回 Top-5，time/position 沿用 Top-4，并记录完整 retrieval trace。
- `b0/v1` 仍是修复前历史基线；`comparison/v2` 继续复用它，不能与 B0-v4
  当作只改变 Top-K 的严格配对实验。
- `sequences/`、`analysis/`、`sequence_0_audit/` 分别保存逐题页面、聚合分析和
  sequence 0 深度审计。
- 所有审计页共享 `b0/v1/sequence_0_media/` 中的视频和 contact sheets。
