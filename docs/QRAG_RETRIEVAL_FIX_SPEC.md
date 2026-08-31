# Q-RAG 检索控制器修正规格

- 状态：Implemented，单元测试通过；`question_state_v3_interleaved` 全量实验待运行
- 新实验标签：`question_state_v3_interleaved`
- 适用范围：ReMEmbR controller、工具调用封装、Q-RAG text retrieval、retrieval trace
- 历史结果：保留 `question_state_v2`，不得将其改名或解释为 controller-interleaved Q-RAG

## 1. 已确认的问题

以 sequence 0 第 1 题为例，两个 `retrieve_from_time("07:55:32")` 返回完全相同。主要原因不是 Q-RAG scorer 或 reader：

1. Q-RAG 目前只覆盖 `retrieve_from_text`；time/position 仍使用原始数值检索，所以该题的 B3 与 B0 可以完全相同。
2. 当前 prompt 既禁止重复调用，又要求“至少调用两个不同工具”，并允许一次输出多个并行 tool calls。
3. 工具封装接受同一批次中的所有调用，未对 `(tool, normalized_query)` 去重。报告再将调用扁平编号为 Call 1/2，容易误判为第二次调用已经看过第一次结果。
4. 即使跨 controller 回合，当前实现也会把 `ToolMessage` 转成普通 `AIMessage`、丢失 tool provenance；记录历史参数的正则还会把字典内容删为空字符串。
5. reader 只在 controller 停止检索后生成最终答案，不负责发起第二次 retrieval。

因此，重复 query 是 controller 直接生成、现有 orchestration 未阻止的结果；高概率属于同一批并行调用，第二个调用并未看到第一个结果。

## 2. 当前 v2 语义

| 项目 | 当前行为 |
|---|---|
| B0/B1 text | 每次调用返回 static top-5 |
| B2 Q-RAG static | 编码一次 state，返回 Q 值 top-5 |
| B3 Q-RAG sequential | 一次 text tool call 内部执行 5 步；每步取当前 Q 值 top-1、加入内部 state、重新打分，最后一次性返回 5 条 |
| Q-RAG 题级预算 | 5；B3 第一次 text call 通常全部耗尽 |
| time/position | 非 Q-RAG；每次返回 numeric top-4 |
| controller | 最多 3 个可调用工具的决策回合；单回合可产生多个并行 calls |

这里的“5”是 evidence budget，不应被理解为“controller 看过一次结果后再调用五次”。v2 的 state update 发生在 memory backend 内部，controller 在五步完成前看不到中间证据。

## 3. v3 目标行为

```text
controller turn N
  -> 恰好 0 或 1 个 retrieval call
  -> memory 返回结果
  -> 带 tool 名、query、call id 的结果进入 controller context
  -> controller turn N+1：回答，或生成一个新的 retrieval call
```

必须满足：

- 每个 controller 回合最多执行一个 retrieval tool。
- B3 每次 `retrieve_from_text` 仅选择当前 Q 值最高的 1 条。
- 下一次 B3 text call 的 Q-RAG state 包含：原问题、当前 tool query、此前已选择的 captions。
- 每题最多保留 5 条唯一 text evidence；已选择 ID 全局 mask。
- 每题最多 5 个 retrieval 回合；到达上限后强制进入 reader。
- 相同 `(tool, normalized_query)` 在同一 answer attempt 内不得再次执行；无效输出应让 controller 重试或结束。
- time/position 保持 numeric top-4，不伪装为 Q-RAG，但同样受单调用、重复 query 和 controller 回合上限约束。
- reader 仅在 controller 明确停止或达到上限后执行。

## 4. v3 实验配置

| 参数 | B0 | B1 | B2 | B3 v3 |
|---|---:|---:|---:|---:|
| text 方法 | mxbai static | GTE static | Q-RAG static | Q-RAG interleaved |
| 每次 text 返回数 | 5 | 5 | 5 | 1 |
| 题级唯一 text evidence | 5 | 5 | 5 | 5 |
| Q-RAG steps/call | N/A | N/A | 5 | 1 |
| numeric top-k | 4 | 4 | 4 | 4 |
| 并行 tool calls | 禁止 | 禁止 | 禁止 | 禁止 |
| 最大 retrieval 回合 | 5 | 5 | 5 | 5 |

该设置控制每组最多看到 5 条 text evidence；B0/B1/B2 是一次 top-5，B3 则最多进行五次 top-1 自适应检索。报告必须分别统计 text Q-RAG、time 和 position，不能用 numeric 问题验证 Q-RAG 是否生效。

## 5. 必要代码修改

1. `remembr/prompts/agent_system_prompt.txt`
   - 删除“至少调用两个工具”和并行调用说明。
   - 改为“每回合最多一个工具；收到结果后再决定下一步；不得重复相同 tool/query”。

2. `remembr/tools/functions_wrapper.py`
   - 校验 controller 输出最多包含一个 retrieval call。
   - 对 tool 名和规范化参数生成稳定 signature；重复时不执行。
   - 保留原始参数 JSON，不使用正则删除字典内容。

3. `remembr/agents/remembr_agent.py`
   - 将 controller 指令放入 `system` role。
   - 保留 `ToolMessage` 的 `tool_call_id`；若模型接口不支持，显式序列化 `tool/name/query/result/call_id`，不可伪装成无来源的 assistant 文本。
   - 将硬编码 3 回合替换为可配置的 5 个 retrieval 回合，并在上限后进入 reader。
   - 在 answer attempt 范围维护已执行 signature 集合。

4. `remembr/memory/qrag_local_memory.py`
   - v3 设置 `evidence_budget=1`、`question_evidence_budget=5`、`episode_mode=question`。
   - 保留跨 call 的 selected-ID mask 和 caption state；每次只提交一个 selected item。

5. `remembr/scripts/eval.py` 与审计脚本
   - 新增并持久化 controller/batch 级 trace，审计范围覆盖 text、time、position。
   - 使用新 run tag，禁止覆盖 v2 结果。

## 6. Trace 最低字段

每次调用至少记录：

```text
answer_attempt_id, controller_turn_id, tool_batch_id, tool_batch_size,
tool_call_id, tool, raw_query, normalized_query, duplicate_blocked,
prior_result_ids_visible_to_controller, selected_ids, qrag_state_components
```

只有 `controller_turn_id` 增加且 `prior_result_ids_visible_to_controller` 包含上次返回结果时，才能声称发生了“基于上次结果的下一次 retrieval”。

## 7. 验收标准

- 单元测试：同批多个 calls 被拒绝；重复 signature 不执行。
- 单元测试：B3 第一次 text call 返回 1 条；第二次 state 包含第一条 caption，且不会再次选择同一 ID；累计最多 5 条。
- 集成测试：sequence 0 第 1 题不得执行两次相同的 `retrieve_from_time("07:55:32")`。
- 集成测试：若存在第二次 retrieval，trace 能证明 controller 已看到第一次结果。
- 报告测试：Call 编号同时显示 controller turn/batch；numeric retrieval 明确标记为 `non_qrag`。
- 回归测试：现有 memory trace 与 Q-RAG 单元测试继续通过；v2 报告保持不变。

## 8. 实现顺序

先修 controller 的单调用、消息角色、参数记录与去重，再将 B3 改为 `steps/call=1`，最后补 trace/audit 并运行 v3。否则新实验仍无法证明第二次 query 是否真正使用了第一次 retrieval context。

## 9. 实现结果

- controller 每轮只接受一个选择，`FunctionsWrapper` 在执行前拒绝并行或混合 batch。
- 本地 Ollama 不支持原生 `ToolMessage` 时，工具请求与结果以带 provenance 的 system record 序列化，不再转换成 `AIMessage`。
- answer attempt 维护稳定的 `(tool, normalized_query)` signature、五轮执行预算和此前对 controller 可见的结果 ID；重复调用不会进入 memory backend。
- B3 v3 默认 `steps/call=1`、`episode_mode=question`、题级证据预算 5；Q-RAG 在第二次调用的 state 中携带第一次 caption，并持续 mask 已选 ID。
- trace 已覆盖 controller turn、batch、call ID、去重、可见结果、selected IDs、Q-RAG state 和 numeric `non_qrag` 标识；审计页面同时显示这些字段。
- 历史 `question_state_v2` 结果和报告保持不变；新脚本使用 `question_state_v3_interleaved` run tag。
