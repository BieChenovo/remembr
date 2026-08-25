# Q-RAG × ReMEmbR：NaVQA 长时 EQA 检索实验 Idea 规格

> 文档类型：跨论文研究 Idea、实验规格与 Agent 交接协议
>
> 版本：v0.2，2026-08-25
>
> 目标读者：负责数据、模型和实验实现的 Agent
>
> 当前状态：**可进入 Phase 0 NaVQA / ReMEmbR 数据审计；尚未证明核心假设**

## 0. 一分钟版本

第一阶段只组合两项工作：

```text
ReMEmbR
  3 s video captions + position + timestamp
              ↓
       fixed caption memory
              ↓
Q-RAG-style state-conditioned sequential retrieval
              ↓
     fixed ReMEmbR answerer / evaluator
              ↓
             NaVQA
```

要验证的核心问题是：

> 在完全相同的 ReMEmbR caption memory、问题、回答模型与读取预算下，用 Q-RAG 式 `Q(问题 + 已检索证据, 下一条 caption)` 替换原检索过程，能否提高 NaVQA 正确率，或在保持正确率时减少证据和 LLM 调用成本？

**[当前判断]** 这是比直接做多模态 Q-RAG 更干净的第一阶段：ReMEmbR 已经把视频转成 caption 文本，所以 Q-RAG 的 chunk / action 接口不需要改成视觉 encoder。但“不需要太多改动”只对**输入模态**成立；监督信号、ReMEmbR 的 text / position / time 三类工具如何与 Q-RAG 对齐，仍是第一个必须解决的 Gate。

论文中数据集的正式写法是 **NaVQA**（Navigation Video Question Answering）。

### 状态标签

- **[论文事实]**：原文或库内论文笔记已支持；
- **[当前判断]**：跨论文归纳得到的研究判断；
- **[待验证假设]**：需要实验支持或否证；
- **[项目决策]**：为了可归因而暂定的实现边界；
- **[Gate]**：未满足时不得进入下一阶段。

## 1. 本次范围收缩与论文切口

### 1.1 第一阶段只研究 Retrieve

```text
固定 NaVQA 视频历史
    → 固定 ReMEmbR captions / pose / time
    → 替换 retrieval policy
    → 固定 reader / answer prompt
    → 评估答案质量与检索成本
```

第一阶段不同时处理：

- raw image / video retrieval；
- memory write policy、caption 生成优化、压缩、遗忘或冲突更新；
- 主动探索产生新证据；
- FindingDory 的 goal-frame selection 和低层导航；
- 多楼层 map、planner、local policy 或 manipulation。

### 1.2 当前可证伪的 claim

**[待验证假设 H1]**：相比 one-shot dense top-k，state-conditioned retrieval 能根据已取证据选择互补 caption，从而改善 NaVQA 中需要多次检索的问题。

**[待验证假设 H2]**：在与 ReMEmbR 相同的检索项数和 LLM 调用次数下，Q-RAG 式 value retriever 高于相同 encoder 的 dense retriever、监督式 reranker 和 trajectory SFT。

**[待验证假设 H3]**：learned STOP 能在保持 NaVQA 正确率时降低平均证据数、检索步数或 LLM 调用成本。

可支持的第一阶段结论最多是：

> Learned state-conditioned sequential retrieval over ReMEmbR caption memory improves long-video EQA on NaVQA under an equal retrieval and reader budget.

不能由此宣称：多模态证据检索已被解决、系统具有跨 episode 长期记忆、能在动态环境处理过期事实，或能够完成多楼层导航。

## 2. ReMEmbR 与 NaVQA 已核查事实

### 2.1 Memory 与 retrieval

- **[论文事实]** ReMEmbR 每 3 秒生成一条 VILA1.5-13B caption，caption 由 2 FPS 采样的 6 帧生成。[ReMEmbR Sec. III-A]
- **[论文事实]** memory entry 保存 caption embedding、位置和时间戳；文本 embedding 使用 `mixedbread-ai/mxbai-embed-large-v1`。[ReMEmbR Sec. III-A]
- **[论文事实]** 同一 memory 支持 text、position 和 time 三类检索视图。LLM 根据问题和已取回 memory 发出函数调用，每轮后判断证据是否足够，最多三轮检索。[ReMEmbR Sec. III-B]
- **[论文事实]** 输出为结构化 JSON，类型包括 text、position、time 和 duration。[ReMEmbR Sec. III-B]

### 2.2 NaVQA 构成

- **[论文事实]** NaVQA 由 CODa 的 23 条序列中选取 7 条构建；原序列时长约 15–30 分钟。[ReMEmbR Sec. IV-A]
- **[论文事实]** 每条序列分别选 10 个 short、medium 和 long 片段，对应 `<2 min`、`2–7 min` 和 `>7 min`。[ReMEmbR Sec. IV-A]
- **[论文事实]** 5 名机器人专家每条序列标注 30 个问题与答案，总计 210 个问题。[ReMEmbR Sec. IV-A]
- **[论文事实]** 输出类型分布为：yes/no 32%、point-in-time 14%、duration 4%、spatial position 34%、descriptive text 16%。[ReMEmbR Sec. IV-A]
- **[论文事实]** 问题覆盖空间理解、目标检测、标志读取、动态事件和上下文推理。[ReMEmbR Sec. IV-A]

### 2.3 官方评测与可用参照点

- **[论文事实]** 空间答案在 15 m 内算正确；时刻或持续时间在 2 min 内算正确；描述性答案由 LLM 评判。[ReMEmbR Sec. IV-B]
- **[论文事实]** ReMEmbR 的 GPT-4o 结果在 short / medium / long 上分别为 `.72±.5 / .56±.5 / .61±.5`；“1 call only”消融为 `.67±.5 / .48±.4 / .50±.5`。[ReMEmbR Table II]
- **[论文事实]** 12 s caption 消融为 `.54±.5 / .50±.5 / .38±.5`，说明 caption 时间粒度本身也是重要混淆因子。[ReMEmbR Table II]

Table II 的数值在原文中以 mean ± standard deviation 呈现；本文仅将其用作复现校验参考，不将跨配置的点差解读为显著性。

## 3. 什么保持不变，什么被替换

### 3.1 必须冻结的变量

| 层 | 第一阶段处理 |
|---|---|
| 原始视频 | 使用同一 NaVQA 片段 |
| Caption builder | 固定 ReMEmbR 的 3 s caption 或公开预计算 caption，不联合训练 |
| Memory entries | 同一 caption、position、timestamp、segment ID |
| Question / answer | 同一 NaVQA 问题、参考答案与官方评分 |
| Reader | 同一 LLM、prompt、temperature、JSON schema |
| Candidate pool | 同一完整历史，或对所有方法相同的 deterministic pool |
| Budget | 同一最大检索步数、每步条目数、总证据数与 LLM 调用次数 |
| Seeds / evaluator | 同一 split、seed 组和评估器版本 |

不得在 Q-RAG 组同时换更强 captioner、reader、更多候选或更宽读取预算。

### 3.2 两种替换范围

#### Scope A：只替换 text retrieval ranker（推荐的 Stage 1A）

```text
ReMEmbR LLM controller
  ├─ text_search(query, working_context)
  │    → Q-RAG-style state-conditioned next-caption ranker
  ├─ position_search(...)  → 保留原实现
  └─ time_search(...)      → 保留原实现
```

优点：改动最小，最贴近“替换 retrieval 做 ablation”。

每次 controller 选择 text tool 时，ranker 的 state 使用原问题、当前 tool query、已返回的 memory 与剩余预算，并返回与对照系统相同数量的 captions。Stage 1A 不学习 STOP：工具选择和停止仍由原 ReMEmbR controller 负责，避免在一个消融中同时改变 ranker 与 controller。

限制：只能证明语义 caption ranker 的价值；LLM 选择 text / position / time 工具的能力仍会影响结果。除全部 NaVQA 结果外，应报告事先依问题/答案类型划分的子集；不得根据某个 controller 事后是否调用 text tool 来筛选有利样本。

#### Scope B：替换整个 retrieval loop（Stage 1B）

将每条 memory 序列化为：

```text
[caption] ...
[position] x, y, z
[time] start, end
```

Q-RAG 直接在所有 entry 中多步选择，使用自己的 STOP，并向固定 reader 一次性提供已选证据。

优点：形式上更接近 Q-RAG 的“state→next evidence”。

限制：同时改变了工具路由、metadata 访问和 LLM 调用方式，归因不如 Scope A 干净。

**[项目决策]** 先做 Scope A 的可运行复现和最小替换，再做 Scope B。两者不混成一个结果。

## 4. Q-RAG 在 caption memory 上的形式化

### 4.1 Memory item

```text
memory_item = {
  id,
  caption,
  text_embedding,
  position,
  start_time,
  end_time,
  sequence_id,
  segment_id
}
```

对第 $i$ 条记忆：

\[
m_i=(c_i,e_i,p_i,t_i,seq_i,seg_i).
\]

`sequence_id` 只用于 split 和 provenance，不得作为可学习的答案捷径。

### 4.2 State、action 与 STOP

\[
s_t=(q,u_t,m_{a_0},\ldots,m_{a_{t-1}}, b_t),
\]

其中 $q$ 是 NaVQA 原问题，$u_t$ 是 Scope A 中 controller 当前生成的 text-tool query（Scope B 中可等于 $q$），$b_t$ 是剩余证据/调用预算。

\[
\mathcal A_t=\{m_i\notin s_t\}\cup\{\mathrm{STOP}\}.
\]

\[
Q_\theta(s_t,m_i)=\langle E_s(s_t),E_a(m_i)\rangle.
\]

最小版先仅使用 caption 文本；position / time 是之后的可控消融，不在首个结果中无条件混入。

此处 STOP 属于 Scope B 和 A7。Scope A / Stage 1A 只替换 text ranker，必须继续使用 ReMEmbR controller 的停止决策。

### 4.3 Reward 的前提

Q-RAG 原设定依赖 gold supporting facts。如果对问题 $q$ 可定义一组等价 support sets $\mathcal S^*(q)$，则可用：

\[
r_T=\mathbb 1[\exists E\in\mathcal S^*(q):E\subseteq R_T]-\lambda_c|R_T|.
\]

**[Gate]** ReMEmbR 原文说明 NaVQA 有问题和答案，但没有在论文中说明公开了每个问题的 gold supporting captions 或检索轨迹。因此不得假设原 Q-RAG reward 可以直接使用。

若没有 gold support，备选路线必须分开报告：

1. **Support reconstruction**：仅在答案时间/位置能可靠对齐 caption 的问题上生成候选 support，并人工审计歧义；
2. **Answer-reward variant**：以 NaVQA 最终正确性作为 reward，但这是高方差、reader-dependent 的新变体，不是原 Q-RAG 复现；
3. **Synthetic pretraining**：从 caption / metadata 自动生成带 support 的问题用于训练，只在人工 NaVQA 上测试；
4. **Supervised / inference-only fallback**：如果无法构建可靠 reward，先将研究降级为 sequential reranking，不宣称 RL / value learning 收益。

## 5. Phase 0：数据与监督 Gate

### 5.1 必须审计的项目

- [ ] NaVQA 问题、答案、对应 CODa 序列和片段是否全部可获取；
- [ ] ReMEmbR 官方 code / config / prompts / captions 是否可复现；
- [ ] 是否提供已预计算 captions，或必须重新跑 VILA；
- [ ] 每条 caption 与 position / timestamp / segment 的对齐 schema；
- [ ] 是否有 support timestamp、support captions、relevant interval 或 retrieval traces；
- [ ] 210 个问题是否有官方 train / validation / test 拆分；
- [ ] 原始 ReMEmbR 的确切 `k`、`m`、最大调用数、prompt 与 evaluator 版本；
- [ ] 位置与时间类答案能否反向构建无泄漏的支持区间；
- [ ] 在相同 reader 下，oracle support 是否显著高于 dense top-k；
- [ ] 公开 license、数据体积、checksum 与必要计算资源。

### 5.2 无泄漏 split

NaVQA 仅有 7 条来源序列和 210 个人工问题，数据规模对训练 value retriever 很小。

**[项目决策]**：

- 以 CODa sequence 为分组单位，不按问题随机切分；
- 来自同一 sequence 或重叠视频片段的问题必须在同一 split；
- 如果官方只是 evaluation set，不得在全部 210 问上训练后再报同一集的分数；
- 主结果优先使用 leave-one-sequence-out 或按 sequence 固定 train/val/test，并报告不同拆分方案的方差；
- 置信区间和显著性检验以 sequence / video cluster 而不是把 210 个问题当成完全独立样本。

### 5.3 Gate 结论

- **Gate A：数据可运行**：问题、caption memory、metadata、reader 和评分代码可以在一条序列上端到端复现；
- **Gate B：监督可定义**：可靠 support labels / synthetic protocol / answer-reward 方案至少有一个成立，且研究 claim 与其一致；
- **Gate C：评测有灵敏度**：oracle evidence 明显高于弱检索，证明 reader / benchmark 能暴露 retrieval 改进；
- **Gate D：无泄漏训练**：拆分不共享来源序列/重叠片段，且不使用答案字段作为检索输入。

Gate A/C 失败时停止该 benchmark 实验。Gate B/D 失败时，只能报数据审计或 inference-only baseline，不训练 Q-RAG。

## 6. 必须比较的实验矩阵

| ID | 方法 | 目的 |
|---|---|---|
| A0 | 官方 ReMEmbR：最多 3 轮工具式检索 | 必须超过的原系统主基线 |
| A1 | ReMEmbR `1 call only` | 对齐论文已有消融与复现 |
| A2 | one-shot dense caption top-k | 最简单静态文本检索 |
| A3 | iterative query rewriting + dense top-k | 区分 LLM 改写带来的收益 |
| A4 | 监督式 pairwise/listwise reranker | 检查是否普通有监督学习已足够 |
| A5 | trajectory SFT / sequential imitation | 检查是否必须 value learning |
| A6 | Q-RAG-style fixed-step retriever | 只测 state-conditioned 选择 |
| A7 | Q-RAG-style retriever + STOP / cost | 测试质量—成本折中 |
| A8 | oracle support / relevant interval | 分离 retriever ceiling 与 reader ceiling |

若 support label 不可用，A4–A8 中依赖该标签的组别必须删除或改名，不能用伪标签代替 gold 后仍作原结论。

### 6.1 公平性约束

- 与 dense baseline 使用相同 encoder initialization 和 candidate features；
- Q-RAG 若 fine-tune encoder，必须同时给 reranker / SFT 相同训练数据和参数预算；
- 按“总返回 caption 数”与“LLM 调用数”双重对齐，不只对齐 top-k；
- 所有方法的最终证据用相同顺序/模板交给 reader；
- 报告每个 query 的完整 retrieval trace，不只保存 aggregate score；
- 不把论文 Table II 的数值与新 reader / prompt 下的结果直接比较。

## 7. 评测指标

### 7.1 主指标

- NaVQA overall correctness；
- short / medium / long 分组正确率；
- yes/no、point-in-time、duration、spatial position、descriptive text 分类结果；
- spatial position error（m）、time / duration error（min）；
- 与 A0 / A4 / A5 的配对差值、cluster bootstrap CI。

### 7.2 Retrieval 指标

仅在存在经审计的 gold support 时报告：

- support recall / precision / F1 @ budget；
- first-support rank；
- complete-support success；
- redundancy / near-duplicate rate；
- STOP precision / premature-stop rate；
- 按 history length 和 question type 分组的 retrieval gain。

### 7.3 成本指标

- 平均/九十五分位 retrieved captions；
- retrieval steps 与 LLM calls；
- reader input tokens；
- retrieval / end-to-end latency；
- 训练参数、GPU hours 与推理内存。

## 8. 分阶段执行计划

### Phase 0：NaVQA / ReMEmbR 可运行性审计

交付：

```text
dataset_audit.md
navqa_schema.json
question_manifest.jsonl
caption_manifest.jsonl
split_manifest.json
oracle_or_upper_bound_report.md
```

先回答数据是否可训练 Q-RAG，不训练模型。

### Phase 1：复现 ReMEmbR 与简单检索基线

1. 在同一子集上复现 A0 和 A1；
2. 跑 A2，校验 caption、candidate pool、reader 与 evaluator；
3. 输出按问题类型和长度分层的 trace；
4. 复现偏差过大时，不进入 Q-RAG 比较。

### Phase 2：非 value-learning 强基线

运行 A3–A5。这一阶段先回答：收益是否仅来自更好 encoder、监督式相关性排序或多轮 query rewriting。

### Phase 3：Q-RAG-style retrieval

1. 在相同 encoder / data / budget 下运行 A6；
2. 只在 A6 表明 state 有收益后加 STOP / cost；
3. 先完成 Scope A，再实现 Scope B；
4. 任何 answer-reward 结果单独一列，不与 gold-support 训练混报。

### Phase 4：视觉证据与导航扩展（暂不实现）

只在 Phase 3 达到 go 标准后考虑：

1. 将 caption action 扩展为 caption + raw frame / visual feature；
2. 用 RAVEN / FindingDory 测试原图核验与 goal-frame selection；
3. 加入 position / time / episodic provenance 和过期冲突；
4. 最后再将 target floor / room / pose 交给 ASCENT / TravExplorer 类多楼层 planner。

## 9. 必做消融

### 9.1 检索机制

- question-only vs question + selected evidence state；
- fixed step vs learned STOP；
- one-shot top-k vs sequential selection；
- Q-learning / value objective vs trajectory SFT；
- sparse support reward vs answer reward（如可用）；
- no cost vs caption / call cost。

### 9.2 Memory 字段

- caption only；
- caption + time；
- caption + position；
- caption + position + time；
- 3 s caption vs 12 s caption（对齐 ReMEmbR 已有消融）。

### 9.3 长度与问题类型

- short / medium / long；
- 文本、空间、时间、持续时间、yes/no；
- 估计单证据 vs 多证据问题（仅在人工审计后使用该标签）。

## 10. Go / no-go 标准

### Go

在相同 caption memory、reader、证据数和 LLM call 预算下，满足以下之一：

1. Q-RAG 相比最强非 value-learning 基线的 NaVQA overall correctness 提高至少 3 个百分点，且配对区间不支持可忽略差异；或
2. 正确率处于预先规定的非劣界内，同时平均证据数或 LLM calls 下降至少 20%。

此外，受益应在 medium / long 或可审计的多证据子集上更明显，不能只靠 yes/no 问题的波动。

### No-go

任一情况成立时，不继续扩展多模态或多楼层系统：

- 仅超过 `1 call only`，但不超过完整 ReMEmbR 或强 reranker / SFT；
- 监督式 reranker / trajectory SFT 与 Q-RAG 相当；
- 收益来自额外 reader calls、更多 captions 或更强 encoder；
- NaVQA 不提供可用 split / support，且 synthetic / answer reward 结果不稳定；
- oracle evidence 不提高答案质量，说明当前 reader / benchmark 对 retrieval 不敏感；
- 只在训练序列上提升，换 sequence 后消失。

## 11. 失败归因表

| 现象 | 可能原因 | 必做检查 |
|---|---|---|
| support recall 低 | retriever / candidate pool 问题 | oracle candidate recall、first-support rank |
| support 高但 answer 低 | reader 或 prompt 瓶颈 | oracle evidence + same reader |
| short 有效、long 失效 | 负候选增长或 state 压缩不足 | history-length curve、hard negatives |
| Q-RAG 只赢 one-shot | 多轮本身有效，value 无独立贡献 | A3 / A5 |
| 位置/时间题改善、文本题不改善 | metadata 而非 state value 带来收益 | caption-only 与 metadata 消融 |
| 随机 split 高、sequence split 低 | 视频/场景泄漏 | overlap audit |
| STOP 降低成本但丢失准确率 | 过早停止 / cost 过强 | stop calibration curve |

## 12. 实验数据与日志接口

### 12.1 Query manifest

```json
{
  "question_id": "...",
  "sequence_id": "...",
  "clip_start": 0.0,
  "clip_end": 0.0,
  "length_bin": "short|medium|long",
  "question": "...",
  "answer_type": "binary|time|duration|position|text",
  "reference_answer": {},
  "support_entry_ids": [],
  "support_status": "gold|derived|unknown",
  "split": "train|val|test"
}
```

### 12.2 Retrieval trace

```json
{
  "run_id": "...",
  "question_id": "...",
  "method": "...",
  "scope": "text-ranker|whole-loop",
  "budget": {"max_steps": 3, "max_entries": 0, "max_llm_calls": 0},
  "steps": [
    {
      "step": 0,
      "state_query": "...",
      "tool": "text|position|time|stop",
      "selected_entry_ids": [],
      "scores": [],
      "latency_ms": 0.0
    }
  ],
  "reader_input": [],
  "prediction": {},
  "correct": false,
  "cost": {}
}
```

`reference_answer` 和 `support_entry_ids` 只用于训练 target / evaluator，推理 state 不得读取。

## 13. 第一个最小实验

不要一开始就训练 Q-RAG。第一个可交付实验是：

1. 只选 1 条 CODa sequence，打通 question → captions → retrieval → reader → evaluator；
2. 复现 ReMEmbR 最多 3 calls 与 `1 call only`；
3. 加入相同 caption encoder 的 one-shot dense top-k；
4. 人工审查该序列上的 30 个问题，记录是否可指定 support caption / interval；
5. 为每个问题保存检索 trace、答案和评分；
6. 之后才决定 Q-RAG 使用 gold-support reward、synthetic pretraining 还是 answer reward。

该实验的目标是确认接口和评测灵敏度，不是报告可发表的结果。

## 14. 对实验 Agent 的直接交接指令

> 请先完整阅读 `literature/09_notes/02_QRAG具身长期记忆_实验Idea规格.md`。本轮只执行 Phase 0：审计 ReMEmbR 官方代码与 NaVQA 数据，确认 3 s captions、position、timestamp、question / answer、split、support annotations、reader prompt 和 evaluator 的实际可用性。先在一条 sequence 上打通官方 ReMEmbR、`1 call only` 和 one-shot dense top-k，不训练 Q-RAG，不引入 raw images、FindingDory 或多楼层导航。将数据 schema、实际数量、可复现命令、检索 trace、support 标注状态、无泄漏 split 和 Gate A–D 结论写入新的 task 文件与 `dataset_audit.md`。任何未能从代码、数据或原文确认的字段标为 `unknown`，不得猜测。

## 15. 后续路线图

```text
Stage 1: ReMEmbR captions + Q-RAG on NaVQA
  └─ 证明文本 caption memory 上的 sequential value retrieval

Stage 2: RAVEN-style raw evidence / FindingDory goal frames
  └─ 检验 caption 信息瓶颈与视觉证据核验

Stage 3: episodic maintenance + map-only comparison
  └─ 处理跨 episode、过期、冲突与存储增长

Stage 4: multi-floor navigation interface
  └─ memory 输出 floor / room / pose / confidence
     交给 ASCENT / TravExplorer 类 planner
```

本路线中 Stage 2–4 只是后续定位，不是第一阶段实验承诺。

## 16. 相关库内资料

- ReMEmbR：`literature/01_read/01_global_navigation_memory/01_ReMEmbR/ReMEmbR.md`
- ReMEmbR 原文：`literature/01_read/01_global_navigation_memory/01_ReMEmbR/2025_Anwar_ReMEmbR_LongHorizonSpatioTemporalMemory.pdf`
- Q-RAG：`literature/01_read/04_retrieval_foundations/01_Q-RAG/Q-RAG.md`
- RAVEN：`literature/01_read/01_global_navigation_memory/02_RAVEN/RAVEN.md`
- FindingDory：`literature/01_read/03_memory_benchmarks/01_FindingDory/FindingDory.md`
- 当前研究状态：`literature/09_notes/00_CURRENT_RESEARCH_STATE.md`

## 17. 当前结论

**[当前判断]** “ReMEmbR + Q-RAG + NaVQA，只替换 retrieval”是很合理的第一阶段，因为它把视觉 captioning、memory 写入、reader 和任务数据都冻结，能较干净地测试 retrieval policy 的独立价值。

最需要保持警惕的两点是：

1. ReMEmbR 的 caption 是文本，因此数据 encoder 适配很小；但 ReMEmbR 是三类检索工具 + LLM controller，不能含糊地说“替换了 retrieval”；
2. NaVQA 规模小，而原文未说明存在 gold supporting captions 和可训练 split。数据/监督审计是 Q-RAG 训练前的真正第一步。

如果这两个问题得到解决，第一阶段应先用 Scope A 验证最小替换，再以 Scope B 检验 Q-RAG 替换整个多轮检索 loop 的收益。
