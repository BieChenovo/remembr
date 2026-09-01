# Published NaVQA reports

GitHub Pages 报告与 `artifacts/eval_reports` 使用同一目录规范：

```text
reports/
├── b0/v1/
├── b1/{v1,v2}/
├── b2/{v1,v2}/
├── b3/{v1,v2}/
│   ├── v2/v3_interleaved/
│   └── v2/v4_unified_top1/
└── comparison/{v1,v2,v3_interleaved,v4_unified_top1}/
```

- `v1` 表示修复前的 legacy `per_call + native` 实现。
- `v2` 表示修复后的 question-state 实现。
- `b3/v2/v3_interleaved` 表示在 v2 基础上进一步实现 controller ↔ retrieval
  闭环、每轮 Q-RAG top-1 的 v3 实验；旧 v1/v2 结果保持不变。
- `b3/v2/v4_unified_top1` 将 text/time/position 统一为 top-1，共享题级
  evidence ledger、全局 ID mask 和 5 条唯一证据预算，并支持 duplicate replan。
- B0 不涉及本次 Q-RAG 状态修复，因此只有 `b0/v1`，v2 对比复用该结果。
- `sequences/`、`analysis/`、`sequence_0_audit/` 分别保存逐题页面、聚合分析和
  sequence 0 深度审计。
- 所有审计页共享 `b0/v1/sequence_0_media/` 中的视频和 contact sheets。
