# Q-RAG 检索控制器修正规格

- 状态：v3 和 v4 均已实现并完成 210 题；v4 的实现审计为 0 错误
- 已完成实验：`question_state_v3_interleaved`、`question_state_v4_unified_top1`
- 最新实验标签：`question_state_v4_unified_top1`
- 适用范围：ReMEmbR controller、工具调用封装、Q-RAG text retrieval、retrieval trace
- 历史结果：保留 `question_state_v2` 和 v3，v4 不得覆盖或改写已有产物

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

## 3. v3 已实现行为

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

## 4. v3 已运行配置

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

v3 的 210 题审计记录了 222 个 answer attempts、146 次有效 text calls 和 11 个
multi-text attempts；跨回合状态与预算错误为 0。但实验同时记录了 81 次被拦截的
重复调用，且单个 attempt 最多只得到 3 条唯一 text evidence。

## 10. v3 运行后确认的剩余问题

1. B3 text 每次返回 1 条，但 time/position 每次仍返回 4 条，单轮读取预算不一致。
2. 题级 ID mask 只覆盖 text；time/position 可能再次返回 controller 已见过的 memory。
3. 重复 `(tool, normalized_query)` 虽然不会执行，但当前实现立即设置
   `force_reader=True`。controller 没有机会根据纠错消息重新规划。
4. time/position 结果对下一轮 controller 可见，但没有进入后续 text Q-RAG 的
   evidence state；当前 Q-RAG state 只包含此前 text retrieval 的 captions。

因此，v3 已经证明了 controller-interleaved 消息链路，但还不是统一单证据预算、
跨模态共享状态的最终实验。

## 11. v4 项目决策

### 11.1 所有工具统一 top-1

```text
text_k = 1
numeric_k = 1
max_executed_retrieval_rounds = 5
question_unique_evidence_budget = 5
```

- text、time、position 每次只返回 1 条 memory。
- 三类工具共享一个 answer-attempt 级 selected-ID ledger 和全局 mask。
- 每个有效 retrieval 最多新增 1 条唯一 evidence；总计最多 5 条。
- time/position 继续按数值距离排序并标记为 `non_qrag`，不能将其描述为 Q-RAG。

为了公平比较，v4 的单条返回、全局预算、单调用 controller 和跨模态 mask 应用于
B0–B3。B0/B1/B2 保持各自 static scorer；只有 B3 的 Q-RAG scorer 使用此前证据
更新 state，避免把 B2 static 消融变成 sequential 方法。

### 11.2 重复 query 不执行，但允许重新规划

prompt 和程序 guard 必须同时工作：

1. prompt 以结构化 ledger 列出所有已执行的 `(tool, normalized_query)`，要求
   controller 回答、切换工具或生成语义上不同的新 query。
2. 重复 query 不进入 memory backend，不消耗 retrieval round，也不消耗 evidence
   budget。
3. 系统返回 `invalid_retrieval_request` 纠错消息，然后重新进入 controller，而不是
   立即进入 reader。
4. 不自动替模型改写 query；也不允许通过轻微修改空格、大小写或把时间平移 1 秒
   规避去重。
5. 默认最多允许 2 次连续 duplicate replans；成功生成新调用后计数清零。达到上限
   才强制进入 reader，防止 temperature 0 的本地模型形成死循环。

纠错消息至少包含：

```json
{
  "type": "invalid_retrieval_request",
  "reason": "duplicate_query",
  "tool": "retrieve_from_time",
  "query": "07:55:32",
  "executed_retrieval_rounds": 1,
  "instruction": "Use the visible result to answer, switch modality, or formulate a semantically different query."
}
```

### 11.3 time/position 结果进入 Q-RAG state

共享 evidence ledger 按实际返回顺序保存所有工具得到的唯一 memory。下一次 B3
text retrieval 的 Q-RAG state 为：

```text
[
  original_question,
  current_text_tool_query,
  caption_from_prior_text_retrieval,
  caption_from_prior_time_retrieval,
  caption_from_prior_position_retrieval,
  ...
]
```

只把此前 memory 的 caption 文本送入 Q-RAG encoder；tool、query、timestamp、
position、memory ID 等 provenance 单独保存在 trace 中，避免改变 checkpoint 的文本
输入分布。当前调用返回的 memory 在调用完成后写入 ledger，只影响下一次 retrieval。

## 12. v4 必要代码修改

1. `remembr/memory/local_vector_memory.py`
   - 将 text-only episode ledger 抽象为跨模态共享 ledger。
   - `_select()` 对三类工具应用相同的全局 ID mask、top-1 和题级预算。

2. `remembr/memory/qrag_local_memory.py`
   - Q-RAG state 从共享 ledger 读取所有此前工具返回的 captions。
   - 保持 B3 `steps/call=1`，并避免维护与 base memory 冲突的第二套 ID ledger。

3. `remembr/tools/retrieval_control.py` 与 `remembr/agents/remembr_agent.py`
   - 将“有效 retrieval rounds”和“无效 duplicate replans”分开计数。
   - duplicate 返回纠错消息并重新进入 controller；只有达到纠错上限才
     `force_reader`。
   - 成功调用后清零连续纠错计数。

4. `remembr/prompts/agent_system_prompt.txt`
   - 明确列出禁用 signatures，要求基于可见结果形成新 query 或切换 modality。
   - 明确禁止为了绕过去重而做无依据的时间/坐标微调。

5. `remembr/scripts/eval.py`、运行脚本与审计脚本
   - 增加 `numeric_k=1`、全局 evidence budget 和 duplicate replan limit 配置。
   - 使用 `question_state_v4_unified_top1`，不得覆盖 v3。

## 13. v4 Trace 增量字段

在 v3 trace 基础上增加：

```text
retrieval_executed, duplicate_reprompted, duplicate_replan_count,
duplicate_replan_limit, evidence_state_version,
global_selected_entry_ids_before, global_selected_entry_ids_after,
prior_evidence_sources, forced_stop_reason
```

无效 duplicate event 必须出现在 trace 中，但不得增加
`executed_retrieval_rounds`、`evidence_state_version` 或 selected IDs。

## 14. v4 验收标准

- 每次 text/time/position 调用最多返回 1 条。
- 同一 answer attempt 中，任何两个已执行调用不得返回相同 memory ID。
- 所有工具合计最多返回 5 条唯一 memory，且最多执行 5 个有效 retrieval rounds。
- 第一次重复 query 被拒绝后，controller 仍可执行一个不同的新 query；之前的结果
  继续可见，且重复请求不消耗 round。
- 连续重复达到配置上限后才进入 reader，不得形成无限 graph loop。
- time 或 position 返回 memory 后，下一次 B3 text Q-RAG state 必须包含该 memory 的
  caption 和当前新 query。
- B2 static state 不包含此前 evidence captions，但必须应用全局 ID mask。
- numeric calls 始终标记为 `non_qrag`；v2/v3 结果保持不变。

## 15. v4 实现顺序

先实现共享跨模态 ledger、top-1 和全局 mask；再把 Q-RAG state 接到共享 ledger；
随后把 duplicate 行为改为有限次数的 re-prompt；最后扩展 trace/audit，并先用
sequence 0 做集成验证，通过后再运行 210 题。

## 16. v4 当前实现结果

- `LocalVectorMemory`、GTE 与 Q-RAG memory 已支持显式
  `unified_evidence_ledger`；三类工具共享题级 ID mask、top-1、五条 evidence budget
  和按返回顺序记录的跨模态 provenance。
- B3 sequential state 会读取共享 ledger 中此前 text/time/position 返回的 caption；
  B2 static 只使用 original question + 当前 query，但应用相同的全局 ID mask。
- controller ledger 显式列出已执行的 tool、normalized query 与 signature；duplicate
  返回 `invalid_retrieval_request`，不执行 memory、不增加 round/version/IDs。成功
  re-plan 会清零连续错误计数，连续两次 duplicate 才强制进入 reader。
- evaluator、全量运行脚本、trace audit 与 sequence audit 页面已接入 v4 参数和字段；
  默认新标签为 `question_state_v4_unified_top1`，报告写入独立的
  `v4_unified_top1/` 子目录。
- 30 项单元测试已通过，覆盖跨模态 top-1/mask、numeric caption 进入 B3 state、
  B2 static state 隔离以及 duplicate re-prompt 上限。

## 17. v4 全量实验结果

- 7 个 sequence 共 210 题已全部完成；93 题答对，strict accuracy 为
  44.3%，4 个无效输出按错误计入分母。
- reference-memory hit 为 10/210（4.8%），Grounded accuracy 为
  10/210（4.8%）。相比 v3，24 题改善、24 题退化，答案准确率净变化为 0。
- question-state audit 覆盖 218 个 answer attempts：148 次 duplicate 被拦截，
  95 次触发 replan，top-1、预算、全局 mask、跨模态 state 和终止上限错误均为 0。
- 完整报告发布于 `docs/reports/b3/v2/v4_unified_top1/`，跨版本对比发布于
  `docs/reports/comparison/v4_unified_top1/`。
