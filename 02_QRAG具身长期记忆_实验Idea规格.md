# Q-RAG × ReMEmbR：NaVQA 长时 EQA 检索实验 Idea 规格

> 文档类型：跨论文研究 Idea、实验规格与 Agent 交接协议
>
> 版本：v0.4，2026-08-30
>
> 目标读者：负责数据、模型和实验实现的 Agent
>
> 当前状态：**B0–B3 的旧版 210 题 zero-shot 实验已保留；question-state v2 修复后的 B1/B2/B3 也已于 2026-08-31 全量重跑。三组共 630 题，跨-call ID 去重、题级预算、query state 和 retry episode 审计均为 0 个错误。**

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
- **[当前实测]**：已由当前服务器上的代码、数据或运行产物验证；
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

### 2.4 当前服务器上的 210 题基线与错误审计

**[当前实测]** 当前本地结果使用 VILA1.5-13B 3 s captions、ReMEmbR controller、
Qwen3-8B reader、`no_thinking`、256-token 输出上限和
`America/Los_Angeles` 时区。按 ReMEmbR 论文阈值重新对全部 210 题计分，
无效结构化输出按错误处理：

| 分组 | 正确 / 总数 | 严格准确率 |
|---|---:|---:|
| Overall | 115 / 210 | 54.8% |
| binary | 41 / 67 | 61.2% |
| descriptive text | 26 / 33 | 78.8% |
| position（15 m） | 26 / 71 | 36.6% |
| point-in-time（2 min） | 20 / 30 | 66.7% |
| duration（2 min） | 2 / 9 | 22.2% |
| short | 49 / 70 | 70.0% |
| medium | 38 / 70 | 54.3% |
| long | 28 / 70 | 40.0% |

95 个错误的可观察分类为：43 个有效位置越界、23 个有效 binary 答错、
9 个有效时间越界、7 个有效 duration 越界、6 个有效 text 语义答错，以及
7 个结构化输出失效。7 个失效必须计错并单独报告 output coverage；不能从相应
题型的分母中删除后只报条件准确率。

**[当前实测] reference context 审计：**

- 210/210 个问题都带有非空 `context`；
- `context` 共包含 230 个 reference caption block；
- 230/230 都能通过 caption 文本和位置精确映射回当前 caption memory；
- 按当前 `load_memory()` 的问题起止窗口，只有 191/210 个问题的全部 reference
  entries 位于候选池内；19 个问题至少有一个 reference entry 在池外；
- 19 个池外 entry 中 14 个在起点之前、5 个在终点之后；其中 10 个只相差约
  3–6.1 s，属于明显的 3 s caption 边界/索引问题，其余存在更大的标注窗口不一致；
- reference 全部入池的 191 题严格准确率为 56.0%，reference 在池外的 19 题为
  42.1%。这是相关性描述，不单独证明因果。

`context` 可作为 **derived reference support candidate**，但在确认其标注来源、
穷尽性和是否允许训练前，不应直接称为完整 gold supporting facts。当前历史运行
没有保存实际 retrieval query、top-k entry ID 或 score，因此除候选池遗漏和输出
schema 失败外，尚不能把每个错误严格分解为 retriever 或 reader 责任。

当前审计产物：

- `artifacts/eval_reports/navqa_210_error_analysis_v1/index.html`
- `artifacts/eval_reports/navqa_210_error_analysis_v1/error_analysis.json`
- `artifacts/eval_reports/navqa_210_error_analysis_v1/question_diagnostics.csv`
- `artifacts/eval_reports/navqa_210_error_analysis_v1/reference_context_manifest.jsonl`

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

ReMEmbR 论文没有说明是否公开了逐题 gold supporting captions 或检索轨迹；但
**[当前实测]** 当前 NaVQA 问题文件中的 `context` 可全部映射为 230 个 caption
entry IDs。它为 Q-RAG 监督提供了此前未确认的派生 support 候选。

**[Gate]** 在将这些 entry IDs 用作 reward 前必须确认：

1. `context` 是人工标注者使用的答案证据，而非评测时自动拼接或答案泄漏；
2. 一个问题的 `context` 是否穷尽全部等价 support，还是只给出一个可行证据；
3. 多 block context 应按“全部命中”还是“任一等价集合命中”计 reward；
4. 19 个 reference context 在当前候选池外的问题如何修正，且所有 baseline 使用
   完全相同的候选池；
5. 训练只使用训练序列的 derived support，不能用测试序列 reference context 调参。

在完成上述 Gate 前，公开 checkpoint 只能作为 inference-only / zero-shot baseline，
不能把利用全部 210 个 reference contexts 微调后的同集结果称为泛化性能。

若 derived reference support 通过审计，可按以下优先级推进：

1. **Derived-support reward**：以 `reference_context_manifest.jsonl` 的 entry IDs
   构造 support reward，并按 sequence split；这是当前最直接路线；
2. **Support reconstruction**：为位置/时间可对齐问题补充等价 support sets，并人工
   审计多解；
3. **Answer-reward variant**：以 NaVQA 最终正确性作为 reward，但这是高方差、
   reader-dependent 的新变体，不是原 Q-RAG 复现；
4. **Synthetic pretraining**：从 caption / metadata 自动生成带 support 的问题训练，
   只在人工 NaVQA 上测试；
5. **Supervised / inference-only fallback**：若 support provenance 不通过审计，降级为
   sequential reranking，不宣称 RL / value learning 收益。

原先“无 support 时”的备选路线仍必须分开报告：

- derived support、synthetic support 和 answer reward 不得混在同一结果列；
- pseudo label 不得在命名中伪装成官方 gold；
- support F1 与最终 answer correctness 必须同时报告。

## 5. Phase 0：数据与监督 Gate

### 5.1 必须审计的项目

- [x] 当前实验所需的 7 序列、210 个 NaVQA 问题/答案和 CODa 预处理片段可用；
- [x] 当前 fork 的 ReMEmbR code / prompts / VILA captions 可端到端运行；尚未完成
  与 GPT-4o 官方数值的严格复现；
- [x] 7 个序列的 VILA1.5-13B 3 s captions 已生成；
- [x] caption、position、timestamp 和 3 s segment 对齐 schema 已核对；
- [x] 210 题均有可映射到 caption IDs 的 reference `context`；没有历史 retrieval
  traces，且 support provenance / completeness 仍待确认；
- [ ] 210 个问题是否有官方 train / validation / test 拆分；
- [ ] 原始 ReMEmbR 的确切 `k`、`m`、最大调用数、prompt 与 evaluator 版本；
- [x] 已生成 derived reference-context manifest；仍需按 sequence split 并审计等价
  support sets；
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

0. 先修复并冻结 evaluator：无效输出计错、按官方阈值计算 overall、保存 output
   coverage；审计 19 题 reference context 被候选窗口排除的问题；
   候选池修正必须由与答案无关的统一时间规则产生，不得为单题注入
   reference IDs；
1. 为 A0 增加逐步 retrieval trace（tool query、candidate IDs、scores、selected IDs、
   reader evidence）；历史 210 题结果没有该 trace，不能用于精确 retriever/reader 归因；
2. 在同一子集上复现 A0 和 A1；
3. 跑 A2，校验 caption、candidate pool、reader 与 evaluator；
4. 输出按问题类型和长度分层的 trace；
5. 复现偏差过大时，不进入 Q-RAG 比较。

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

当前 7 序列、210 题 baseline 已经跑通，但发现 output contract、候选窗口和 trace
三个前置问题。因此更新后的最小实验不是再次盲跑 210 题，而是：

1. 用序列 0 的 30 题验证修正后的 candidate-pool policy 与 output schema；
2. 保存 A0 每次 tool call 的完整 retrieval trace；
3. 用 `context` 映射出的 entry IDs 计算 derived-support Recall@K；
4. 跑相同 GTE encoder 的 dense top-k 与 Q-RAG zero-shot fixed-step；
5. 固定同一 reader、候选池、证据数、tool-call 数与输出 schema；
6. 只有 oracle/derived-support evidence 明显提高 reader ceiling 后，才进入 Q-RAG
   微调或 answer reward。

该实验的目标是确认接口、support 标签和评测灵敏度，不是报告可发表的结果。

## 14. 对实验 Agent 的直接交接指令

### 14.1 本轮交付边界

另一个实验 Agent 本轮只实现 **Scope A / Stage 1A 的 inference-only
adapter**：仅替换 `search_by_text()` 的 caption ranker，position/time 检索、
ReMEmbR controller、Qwen reader、prompt、候选池、证据数和 evaluator 都保持不变。

本轮不做：

- 不使用 210 题的 reference context 训练或调参；
- 不实现 learned STOP，停止决策继续交给 ReMEmbR controller；
- 不改 position/time tool，不引入 raw frame；
- 不为了提升 support recall 按问题注入 reference IDs 或改变时间窗口；
- 不把 zero-shot 适配结果称为 Q-RAG 在 NaVQA 上的完整复现。

### 14.2 已就绪资产与未就绪依赖

| 项目 | 当前状态 |
|---|---|
| Q-RAG 代码 | `third_party/Q-RAG`，commit `42358d78ac491843763b90677f07237471c97086` |
| 官方 checkpoint | `third_party/qrag-models/qrag-ft-gte-on-hotpotqa_musique/model_best.pt` |
| checkpoint SHA-256 | `ff2ab1db095fe05f0f854672a224e9dff6ff0a9a8ecda5cf35cdf88a94d37c56` |
| 原始配置 | 同目录 `config_orig.yaml`，HotpotQA + Musique，`max_steps=6`，`positions_processor=none` |
| 当前 3-step 副本 | 同目录 `config.yaml`；仅是本地运行副本，不代表 checkpoint 重训练 |
| encoder | `Alibaba-NLP/gte-multilingual-base`；当前 checkpoint 的实际 inference forward 输出为 768 维 |
| GTE base/tokenizer | 已下载到本地 `third_party/`；模型文件不纳入 Git |
| NaVQA captions | 7 序列共 3260 条，已就绪 |
| derived support manifest | `artifacts/eval_reports/navqa_210_error_analysis_v1/reference_context_manifest.jsonl` |

当前 mxbai caption embeddings 是 1024 维，与 Q-RAG 的 768 维 action encoder
不兼容，不得直接复用。必须用 checkpoint 对应的 GTE action encoder 将
3260 条 caption 重新编码并缓存。

Q-RAG 模型代码中虽定义了 768→384 的线性 head，但官方 `BertPredictor.forward`
没有调用该 head；当前适配器按实际 forward 输出与正式 trace 校验，使用 768 维
state/action embeddings。

### 14.3 建议新增的代码边界

Q-RAG 第三方目录保持原样；适配代码放在 ReMEmbR 内：

```text
remembr/memory/qrag_text_retriever.py
    加载 slim inference weights、编码 state、对 caption actions 打分、去重选择

remembr/memory/qrag_local_memory.py
    实现 Memory 接口；text 调 Q-RAG，position/time 完全沿用 LocalVectorMemory

remembr/scripts/export_qrag_inference_checkpoint.py
    一次性从 11 GB checkpoint 导出 policy state encoder 和 critic action encoder

remembr/scripts/precompute_qrag_caption_embeddings.py
    按 sequence 生成 action embedding cache 和 manifest
```

`eval.py` 只新增显式参数，不更改 dense 默认行为：

```text
--text_retriever dense|gte_dense|qrag_static|qrag
--qrag_checkpoint PATH
--qrag_inference_checkpoint PATH
--embedding_cache_dir PATH
--qrag_evidence_budget 1|3|5
--qrag_state_format native|controller
--qrag_episode_mode per_call|question
--qrag_question_evidence_budget N
--text_episode_mode per_call|question
--question_text_evidence_budget N
```

retrieval trace 随每题结果写入评测 JSON/JSONL，不另设
`--retrieval-trace-dir` 参数。

`dense` 必须回归到当前行为；`gte_dense` 用未经 NaVQA 训练的同款
GTE base 作普通 question/caption dense baseline；`qrag_static` 使用与
Q-RAG 相同的 checkpoint state/action encoders，但不在取回证据后更新
state，而是一次取 question-only score 最高的 top-B；`qrag` 才逐步
追加已选 evidence 并重新打分。`qrag_static` 是区分“学到的 encoder/
scorer”和“state-conditioned sequential update”的必要对照。

### 14.4 zero-shot 检索算法

checkpoint 的 native 训练状态是“question + 已选 evidence”，action 是文本
chunk，且该 checkpoint 不包含可直接迁移的 learned STOP action。旧版 v1 曾以
`native + per_call` 为 zero-shot 主配置：

```text
native:
  original_question [SEP] selected_caption_1 [SEP] ...

controller:
  original_question [SEP] current_tool_query [SEP] selected_caption_1 [SEP] ...
```

2026-08-31 的 trace 审计确认：`native + per_call` 会让 controller 的 tool query
完全不参与打分，并在每次 text call 开头清空 selected-ID mask。只要原问题和候选池
不变，多次 call 就会确定性地返回同一条链。因此 v2 的 ReMEmbR 集成默认改为：

```text
state_format = controller
episode_mode = question
state = original_question [SEP] current_tool_query
        [SEP] question_episode_selected_caption_1 [SEP] ...
```

每一步：

1. 将 state 编码为 768 维；
2. 读取当前候选池的 caption action embeddings；
3. 计算 state/action dot-product，屏蔽本 call 及同一 answer attempt 已选 IDs；
4. greedy 选最高分 action，追加到 state；
5. 重复到 per-call 上限或题级唯一 evidence budget，将 caption 按同一模板返回
   ReMEmbR controller。

同一 answer attempt 的多次 text-tool call 属于同一个 question episode；达到题级
唯一 evidence budget 后仍保留固定的 tool schema，但 retriever 明确返回空结果及
`budget_exhausted=true`，不重复注入旧证据。这里保留 schema 是必要的：实测在回答
过程中动态移除 text tool 会让 Qwen 继续生成该函数调用，并被 strict function
parser 判为失败；固定 schema 加后端预算约束可避免这种非检索因素干扰准确率。
结构化输出失败后的 evaluator retry 会开启新 episode，但旧 trace 继续保留用于审计。
`per_call + native` 只保留用于复现旧结果。本轮仍不让 Q-RAG 自己决定是否继续发起
tool call。

### 14.5 缓存与资源约束

action cache manifest 至少保存：

```json
{
  "sequence_id": "0",
  "caption_file_sha256": "...",
  "checkpoint_sha256": "...",
  "encoder": "Alibaba-NLP/gte-multilingual-base",
  "entry_ids": ["..."],
  "embedding_shape": [0, 768],
  "dtype": "float16|float32"
}
```

任一 hash、entry 顺序、encoder 或维度不一致时必须拒绝使用旧缓存。全部
3260 条 caption 很小，应预编码而不是每题重复跑 action encoder。不要让
每个评测 worker 都加载完整 11 GB 训练 checkpoint；先导出 slim inference
weights，再每张 GPU 一个共享 retriever 进程，或先单进程完成可复现性验证。

### 14.6 候选池与无泄漏规则

在跑 Q-RAG 前，必须把 candidate-pool policy 记录为版本化配置。可考虑
“与 `[start_time,end_time]` 区间相交的 caption + 固定的 3 s 边界容差”，
但不得为了命中 reference 而按题扩窗。19 题中较大时间差的样本必须
人工审计数据含义；如不能修正原始 metadata，则同时报告：

- 全部 210 题的主结果；
- 191 题 reference-complete-in-pool 子集的诊断结果；
- 19 题 pool-mismatch 子集，不将它们从 overall 分母中删除。

### 14.7 对照实验顺序

| 阶段 | 方法 | 目的 |
|---|---|---|
| B0 | 当前 mxbai dense | 只作现有系统参考，不用于归因 Q-RAG encoder |
| B1 | GTE base dense，budget = 1/3/5 | 分离更换 base encoder 本身的影响 |
| B2 | Q-RAG checkpoint static top-B，budget = 1/3/5 | 相同 learned encoders/scorer，不更新 state |
| B3 | Q-RAG checkpoint sequential zero-shot，fixed budget = 1/3/5 | 相对 B2 测 state-conditioned update 的独立收益 |
| B4 | oracle / derived-support evidence | 测 reader ceiling，不是可部署系统 |
| B5 | 按 sequence split 的 derived-support fine-tuning | 只在 support Gate 通过后由后续阶段执行 |

B1/B2/B3 必须共享同一 candidate IDs、caption 文本、evidence budget、reader
input 模板、outer controller 最大 tool calls、Qwen 参数和 evaluator。先在序列 0
上跑 30 题，接口与 trace 通过后再跑 7 序列。

### 14.8 必须落盘的 retrieval trace

每个 text-tool call 至少保存：

```json
{
  "run_id": "...",
  "question_id": "...",
  "sequence_id": "0",
  "method": "gte_dense|qrag_static|qrag_sequential_zero_shot",
  "candidate_pool_policy": "...",
  "candidate_entry_ids": ["..."],
  "original_question": "...",
  "tool_query": "...",
  "state_format": "native|controller",
  "steps": [
    {
      "step": 0,
      "state_text": "...",
      "selected_entry_id": "...",
      "selected_score": 0.0,
      "top_candidate_ids": ["..."],
      "top_candidate_scores": [0.0],
      "latency_ms": 0.0
    }
  ],
  "reader_evidence_entry_ids": ["..."],
  "prediction": {},
  "evaluation": {}
}
```

评测后再用 derived support manifest 计算 Recall@B / complete-support success；检索推理
本身不得读取 `reference_answer` 或 `support_entry_ids`。

### 14.9 验收标准

- checkpoint SHA 校验通过，slim 模型与完整 checkpoint 在小样本上的
  state/action scores 和 greedy IDs 一致；
- `--text-retriever dense` 的回归测试通过，不改当前基线行为；
- position/time tool 在 dense 和 Q-RAG 组返回完全一致的 IDs；
- Q-RAG 选择是 deterministic 的，不重复选同一 ID，所有 ID 均属于冻结候选池；
- cache hash / 维度 / ID 顺序错误时 fail closed，不静默重算或混用；
- 序列 0 的 30 题生成完整 trace，无 adapter 异常、无 NaN score；
- 输出 B1/B2/B3 的 strict accuracy、output coverage、support recall、latency 和
  按题型/长度分组对比。

### 14.10 2026-08-28 全量 210 题结果

以下均为 NaVQA 210 题 zero-shot 评测，未使用 `reference_answer` 或
`support_entry_ids` 训练或参与推理：

| 组别 | 检索方法 | 正确 / 总数 | 严格准确率 | 结构化失败 | 平均延迟 / 题 |
|---|---|---:|---:|---:|---:|
| B0 | mxbai dense | 115 / 210 | 54.8% | 7 | 37.70 s |
| B1 | GTE dense | 102 / 210 | 48.6% | 5 | 40.44 s |
| B2 | Q-RAG static top-B | 112 / 210 | 53.3% | 11 | 41.11 s |
| B3 | Q-RAG sequential zero-shot | 111 / 210 | 52.9% | 6 | 44.39 s |

**[当前实测]** B2 相比同 encoder 的 B1 提高 10 题，说明 checkpoint scorer
有正向信号；B3 相比 B2 净少 1 题，其中 21 题改善、22 题退化，因此当前结果
不支持“sequential state update 带来净收益”。

为排除“没有检索到标注记忆，但 reader 猜对或落入数值容差”对答案准确率的干扰，
另按最终有效 attempt 的 retrieval trace 计算两项指标：成功时只取最后成功 attempt，
整题失败时只取最后一次失败 attempt，不合并旧重试。

- `RetrievalHit = RetrievedIDs ∩ DerivedReferenceIDs ≠ ∅`；
- `GroundedCorrect = AnswerCorrect ∧ RetrievalHit`。

`RetrievedIDs` 联合 text、time、position 三类工具。分母固定为全部 210 题：

| 组别 | 检索命中（any-hit） | 完整 reference 检索 | Grounded 回答正确 | 答对但未命中 |
|---|---:|---:|---:|---:|
| B0 | 不可验证 | 不可验证 | 不可验证 | 不可验证 |
| B1 | 72 / 210（34.3%） | 68 / 210（32.4%） | 45 / 210（21.4%） | 57 |
| B2 | 62 / 210（29.5%） | 58 / 210（27.6%） | 43 / 210（20.5%） | 69 |
| B3 | 62 / 210（29.5%） | 60 / 210（28.6%） | 48 / 210（22.9%） | 63 |

B0 原 210 题结果未保存 retrieval trace，不能把缺失轨迹伪装成检索未命中。后来单独
重跑且保存完整 trace 的 B0 sequence 0（30 题）只能作为补充诊断：检索命中
9 / 30（30.0%），Grounded 回答正确 7 / 30（23.3%）。其中 S0·Q0 的答案按
NaVQA 容差判为正确，但 derived reference 为 `#183`，最终两次 time retrieval 只返回
`#115–#118`，所以新指标明确判为未命中，不计 Grounded 正确。

以上 reference 是由 NaVQA `context` 到 caption entry 的确定性映射，不是穷尽式
gold support；19 题至少一条 reference 位于冻结候选池之外，因此这是一项偏保守的
证据命中指标。早期 comparison v2 中按 text-tool call 条件化的 hit rate 不应解读为
210 题检索准确率；本节的新口径取代该解读。

outer controller 实际产生的 text-tool calls 不完全相同（B1 476、B2 517、
B3 483），所以本轮尚不是严格等总 tool-call budget 的因果隔离实验。完整对比报告
发布于 `docs/reports/navqa_210_retriever_comparison_v2/index.html`；新的检索命中与
Grounded 回答报告发布于
`docs/reports/navqa_210_grounded_accuracy_v1/index.html`。

#### 14.10.1 B2/B3 多次 call 状态审计与 v2 修复

旧 B3 的最终有效 attempts 中，168 / 210 题调用过 text tool，143 题调用至少两次；
这 143 题的多次 call 全部返回完全相同的 entry-ID 链。其中 61 题虽然 controller
生成了不同 query，链仍完全相同。旧 B2 同样有 142 / 142 个多 text-call 问题重复
同一批 IDs。根因不是可视化，而是：

1. 每个 call 都从空 `selected_indices` 和全可用 mask 开始；
2. `native` state 只读取 original question，不读取 tool query；
3. outer controller 允许三次调用，但 retriever 没有跨-call episode state 或去重。

因此 14.10 的 B2/B3 只能解释为 **legacy per-call native**：B3 的 sequential update
只发生在单次 call 内部，不能解释为完整 ReMEmbR retrieval process 的跨-call
statefulness。v2 已实施以下修复：

- B2/B3 默认 `state_format=controller`、`episode_mode=question`；
- B3 state 跨 call 继承已选 captions，并全局 mask 已返回 IDs；
- B2 保持 static scorer，但同样使用 query、全局 ID mask 和题级预算；
- dense/GTE 对照也支持相同的题级唯一 evidence budget；
- text tool schema 在一次 answer attempt 内保持固定；budget 用尽后由 retriever
  返回显式空结果和 `budget_exhausted=true`；
- 每次 evaluator retry 重置 episode budget/mask，但不清除历史 trace；
- trace 新增 episode ID、call index、有效请求数、预算前后余额、全局 IDs 和
  `budget_exhausted`。

默认全量脚本使用新的 `*_210_question_state_v2` tag，防止 `--resume` 误读旧结果。
旧结果可用 `--qrag_state_format native --qrag_episode_mode per_call` 精确复现。

**question-state v2 全量实测（2026-08-31）：**

| 组别 | Strict answer accuracy | Retrieval hit | Grounded correct | 生成失败 | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| B1 · GTE dense | 113 / 210（53.8%） | 82 / 210（39.0%） | 55 / 210（26.2%） | 11 | 38.04 s |
| B2 · Q-RAG static | 87 / 210（41.4%） | 53 / 210（25.2%） | 28 / 210（13.3%） | 17 | 39.29 s |
| B3 · Q-RAG sequential | 104 / 210（49.5%） | 53 / 210（25.2%） | 37 / 210（17.6%） | 12 | 38.09 s |

这里的 Retrieval hit 表示最终用于回答的 attempt 检索到至少一个 reference-memory
ID；Grounded correct 要求答案按 NaVQA 阈值判对且同时 retrieval hit。B3 相比 B2
严格答案净增 17 题（31 题改善、14 题退化），Grounded correct 增加 9 题；但两者
retrieval hit 都是 53 题，说明 sequential update 改变了证据组合及 reader 可用性，
尚未提高 support-ID 的总体覆盖。B1 仍是 v2 三组中 retrieval hit 与 Grounded
correct 最好的方法。

全量 trace 审计覆盖 B1/B2/B3 的 515 / 537 / 504 次 text calls；分别有
171 / 181 / 167 个多 text-call attempts。三组的跨-call 重复 ID、预算记账错误、
retry episode 复用、query 缺失和 B3 step/selection 顺序错误均为 0，且每个 attempt
最多注入 5 条唯一 text evidence。统一可视化与逐题数据见
`docs/reports/navqa_210_question_state_v2_comparison/index.html`。

### 14.11 可直接复制给另一对话的任务

> 请在项目根目录完整阅读 `02_QRAG具身长期记忆_实验Idea规格.md`，重点执行第 14.10.1 节。使用 question-state v2 重跑 B1/B2/B3：Q-RAG 使用 `state_format=controller`、`episode_mode=question`，同一 answer attempt 跨 text calls 继承 evidence state、全局 mask 已返回的 text IDs，并按题级唯一 evidence budget 截止；dense/GTE 使用同一题级预算。evaluator retry 必须重置 episode 但保留 trace，旧 `per_call + native` 结果只作历史对照。先跑 sequence 0 的 30 题并验证没有跨-call 重复 IDs、不同 query 进入 state、budget exhausted 后后端返回显式空结果且不重复旧证据，再扩展到 210 题。不得读取答案或 support IDs 参与推理。完成后生成严格准确率、retrieval-hit、Grounded accuracy 和逐题 trace 可视化。

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
2. NaVQA 规模小；当前数据中存在可精确映射的 derived reference
   contexts，但原文未说明其 provenance、穷尽性或可训练 split。因此
   zero-shot 适配可先进行，训练仍必须等待 supervision/split Gate。

如果这两个问题得到解决，第一阶段应先用 Scope A 验证最小替换，再以 Scope B 检验 Q-RAG 替换整个多轮检索 loop 的收益。
