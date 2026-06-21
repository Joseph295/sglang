# 配套导读：跟着一条请求走一遍（Guided Walkthrough）

> 这是一份**"老师带读"风格**的配套导读，和 `docs/sglang-internals/` 下的参考手册并存。
>
> - 参考手册（`00`–`91`）：结构化、带 `file:line`、当字典查。
> - 本导读（`99`）：口语化、强调"为什么这么设计"、把跨章的联系串起来，适合从头读一遍建立心智模型。
>
> 阅读顺序就是 SGLang 主链路的执行顺序：**01 全景 → 12 调度 → 13 KV → 14 前向 → 17 采样 → 18 解码回流 → 15 并行基础**。每节末尾给出对应参考章节的链接。
>
> 本文随讲随补，已覆盖上面 7 章；后续章节（16 计算层、20 投机解码、21 PD 分离 …）会陆续追加。

---

## 0. 一张图先记住：多进程 + ZMQ 流水线

SGLang runtime（SRT）不是单进程服务，而是**多进程 + ZMQ 流水线**，构成一个环：

```
text "中国的首都是"
  → GenerateReqInput            (HTTP 层)
  → TokenizedGenerateReqInput   ──ZMQ──►  跨进程边界①
  → Req                         (Scheduler 收下，入队)
  → ScheduleBatch               (调度器组批 + 绑显存)
  → ModelWorkerBatch → ForwardBatch  (喂给模型)
  → logits → next_token_ids     (前向 + 采样, 还在 GPU)
  → BatchTokenIDOut             ──ZMQ──►  跨进程边界②
  → BatchStrOut "北"            ──ZMQ──►  回到主进程
  → SSE chunk                   (流回客户端)
```

```
HTTP/OpenAI ─► [主进程: FastAPI → TokenizerManager(asyncio)]
                      │ ZMQ PUSH (TokenizedGenerateReqInput)
                      ▼
              [Scheduler 进程: 调度 ↔ RadixCache/KV Pool → ModelRunner → Sampling]
                      │ ZMQ PUSH (BatchTokenIDOut)
                      ▼
              [DetokenizerManager 进程: token→文本] ─ZMQ PUSH─► 回 TokenizerManager ─► 客户端
```

> ★ **为什么拆进程？** 绕开 Python GIL，让 CPU（分词/解码）和 GPU（前向）解耦。记住这个环，后面任何一章都能立刻定位"我现在在哪个进程里"。

参考：[00-overview](00-overview.md)、[01-request-lifecycle](01-request-lifecycle.md)

---

## 1. 请求全生命周期（8 阶段）

用一个具体请求 `POST /generate`，prompt = `"中国的首都是"` 走一遍。

### ① Entry 接入 — `http_server.py`
FastAPI 路由 `generate_request`（`http_server.py:240`）只做一个分流：看 `obj.stream` 决定返回 `StreamingResponse`（SSE）还是 `await ....__anext__()`（取最后聚合结果）。

> ★ **汇流点**：`/generate` 和 `/v1/chat/completions` 最终都进 `tokenizer_manager.generate_request`。后 7 个阶段对两种 API 通用，差异只在入口参数转换。

### ② Tokenization & 准入 — `tokenizer_manager.py`（进程边界①前）
1. 分配 `rid = uuid4().hex`（`io_struct.py:205`）——这条请求在整条链路里的唯一身份证。
2. 分词 `tokenizer.encode(...)`。
3. **第一次准入校验** `_validate_token_len`（:478）：输入过长直接拒。
4. 打包成 `TokenizedGenerateReqInput`，只剩 token ids + 采样参数 + rid。
5. `send_to_scheduler.send_pyobj`（ZMQ PUSH）发出；登记 `rid_to_state[rid]`（含 `asyncio.Event`）；`_wait_one_response` 挂起 `await event.wait()`。

> ★ **并发模型的精妙点**：发请求的协程（`_wait_one_response`）和收结果的协程（`handle_loop`）靠 `asyncio.Event` 解耦——发的人 wait，收的人 set。这是 TokenizerManager 能编排海量并发的关键。

### ③ Scheduling 调度 — `scheduler.py`（已进 Scheduler 进程）
Scheduler 是**同步 `while True`**（不是 asyncio，要贴着 GPU 跑）。主循环：
```
recv_requests → process_input_requests → get_next_batch_to_run → run_batch → process_batch_result
```
`handle_generate_request`（:897）把 token ids 构造成 `Req` 入 `waiting_queue`。核心决策 `get_next_batch_to_run`（:1286）体现 **prefill 优先、decode 兜底**。新请求在 `get_new_batch_prefill` 里被 `PrefillAdder` 按显存预算试装（**第二次准入**）。

### ④ KV 缓存 & 内存分配 — `schedule_batch.py`
`prepare_for_extend`（:1102）分配 req slot 和 KV token slot（`out_cache_loc`）。RadixAttention 的前缀复用让相同前缀的请求共享 KV，prefill 只算新增部分。

### ⑤ Forward 前向 — `tp_worker.py` → `model_runner.py`
`ScheduleBatch → ModelWorkerBatch → ForwardBatch`，按 forward_mode 派发到 CUDA graph replay / forward_extend / forward_decode。出 `LogitsProcessorOutput`。

### ⑥ Sampling 采样 — `model_runner.py`
`_preprocess_logits`（grammar mask + bias）→ `sampler` 采样，出 `next_token_ids`。

### ⑦ Detokenize & stream — output processor → DetokenizerManager（进程边界②）
`process_batch_result_decode` 追加 token、判完成、`stream_output` 只发增量 → `BatchTokenIDOut` → Detokenizer 增量 detokenize（surr 前缀处理多字节字符）→ `BatchStrOut`。

### ⑧ 响应返回 — 回到 `tokenizer_manager.py`
`handle_loop` 收 `BatchStrOut` → `_handle_batch_output` 累积文本、`state.event.set()` 唤醒挂起的 `_wait_one_response` → `yield` 冒泡回 HTTP → SSE。

### 串成 continuous batching
```
t1: prefill 整个 prompt → "北"
t2: merge 进 running_batch → decode 1 步 → "京"
...
tN: decode → check_finished()=True → 释放 KV → [DONE]
```

> ★ **prefill 只发生一次，decode 每个 token 都要重新走一遍调度循环**。正因为"每个 token 都重新组 batch"，新请求不必等当前请求生成完就能插进来——这就是 continuous batching 的高吞吐+低排队延迟。

**四个枢纽函数（修 bug 必下断点）**：`get_next_batch_to_run`（`scheduler.py:1286`）、`run_batch`（:1533）、`process_batch_result_decode`（`mixin:185`）、`_handle_batch_output`（`tokenizer_manager.py:1119`）。

参考：[01-request-lifecycle](01-request-lifecycle.md)

---

## 2. 调度器：为什么快（continuous batching + overlap）

### Scheduler 是同步死循环
不是 asyncio——要贴着 GPU 跑，避免事件循环抖动。它**本身不做矩阵乘法**，只做决策 + 簿记。

### 问题一：怎么把不同进度的请求拼成一批？
`get_next_batch_to_run`（`scheduler.py:1286`）：
```
① last_batch 是 prefill → filter_batch(踢完成) + merge_batch(并入 running)
② get_new_batch_prefill() 有 → prefill 优先；无 → update_running_batch 做 decode
```
`merge_batch` 直接 `torch.cat` 拼张量。

> ★ continuous batching 的本质：批成员**逐 token 动态变化**（完成的踢出、新来的并入）。设计取舍是 **prefill 优先于 decode**——尽快让新请求进入 decode 流提高并发，代价是 decode 请求被插队（chunked prefill 缓解）。

### 问题二：一批塞多少？PrefillAdder 双账本
```
add_one_req: ① total_tokens >= rem_total_tokens → NO_TOKEN(停)
             ② prefix 命中的部分不算 extend(RadixAttention 省钱)
             ③ input_tokens > rem_chunk_tokens → chunked prefill 截断
```
- `rem_total_tokens` = KV 可用 + 可回收 − 已预留，**含"未来 decode 要生成的 token"估算**。
- `new_token_ratio` 是保守度旋钮：刚 OOM 过就调大，随时间衰减。

> ★ **为什么预留未来的 token？** 请求 prefill 后还要 decode 几百步。只按当前步算预算会塞太满、decode 到一半 OOM 被迫 retract（很贵）。宁可保守预留。

### 问题三：decode 时显存爆了 → retract 抢占式回退
`update_running_batch`（:1496）：`check_decode_mem` 不够 → `retract_decode`（按"输出最长、输入最短"排序从尾部回退）→ 退回 waiting_queue。

> ★ SGLang"宁可回退也不崩"。被回退的请求 KV 可能已进 radix cache，重排后能命中 prefix，不是完全白算。频繁 retract（看 `"Decode out of memory happened"` 日志）= 预算估太激进。

### 重头戏：overlap scheduler = zero-overhead
normal 模式 GPU 算时 CPU 闲、CPU 调度时 GPU 闲。overlap 把前向丢进单独线程让两者重叠。三个关键技巧：
1. **`run_batch` 不阻塞**——丢进 `input_queue` 立即返回，前向在 `forward_thread_func_` 后台线程跑。
2. **future token ids**：返回**负数占位符**让 CPU 不等 GPU 就继续调度，真正前向时 `resolve_future_token_ids` 在 GPU 上 `torch.where` 替换成真值。
3. **结果延迟一拍处理**，启动时造 `DUMMY_FIRST` 批灌满流水线。

> ★ **future token ids 是精髓**："先开支票、后兑现"——CPU 不等 GPU 出结果就用占位符往下排，等真要用那个值时它已在 GPU 备好。整个替换在 GPU 上做，CPU 全程不阻塞。

### overlap 的两个"重构杀手"
- **in-place 字段被下一轮覆盖**（`scheduler.py:1580`）：`result_queue` 存 `batch.copy()`，`sampling_info` 用 `dataclasses.replace` 复制。
- **多生成的"那一拍 token"的 KV 泄漏**（`mixin:218`）：finish 那一拍多算的 token 要 free 掉 `out_cache_loc`，漏掉=KV 缓慢泄漏。

> ★ **黄金调试法**：怀疑 overlap bug，先 `--disable-overlap-schedule` 切回 normal。bug 消失=重叠逻辑问题，否则是更底层的调度/显存问题。

### Req 的隐式状态机
靠字段隐式编码：waiting → running-extend → running-decode → finished/retracted。**abort 不能直接设 `finished_reason`**（会被 filter 立刻摘掉、客户端收不到结束），要设 `to_abort`。

参考：[12-scheduler](12-scheduler.md)

---

## 3. KV 缓存与 RadixAttention：省多少

### 三层映射（理解一切的前提）
```
Req (req_pool_idx)
   │ ReqToTokenPool: [size, max_context_len] int32 表
   ▼ 槽位号 (token index)
   │ TokenToKVPoolAllocator: 管槽位号 free list
   ▼ KVCache.k_buffer[layer][槽位号] / v_buffer  ← 真正显存
```

> ★ **为什么三层而不是一层？** 因为复用。前缀复用 = 让两个请求的某些 token **指向同一个槽位号**——有了中间这层，复用就是 `req_to_token` 表填同样的号,物理显存一份不动。MLA 模型用 `MLATokenToKVPool` 把 KV 压成单 buffer 省显存。

> ★ **slot 0 是 padding 专用**，free list 从 1 开始。CUDA graph 的 padding 假请求就把 `out_cache_loc` 写到 slot 0（回扣第 4 节）。

### paged 分配的"续写三段式"
`alloc_extend_kernel`（triton）一次算出：填旧的半页 → 分新整页 → 开新半页。decode 每步只加 1 token 走更轻的 `alloc_decode_kernel`。开 `SGLANG_DEBUG_MEMORY_POOL=1` 打开页对齐断言。

### radix tree：match / split / evict
每个 `TreeNode`：`key`(一段 token ids，边可多 token) / `value`(KV indices) / `lock_ref`(引用计数) / `last_access_time`(LRU)。
- `match_prefix`：逐边走，匹配到一半 `_split_node` 劈开。
- `_split_node`：把节点在 split_len 处切成父段/子段——**保证公共前缀只存一份**。
- `evict`：只驱逐叶子，按 last_access_time 用最小堆，`lock_ref>0` 跳过。

> ★ **为什么用 radix tree 而非 hash？** hash 只能整条精确命中；radix tree 支持最长公共前缀匹配 + 节点分裂让公共前缀只存一份。多轮对话/few-shot/共享 system prompt 全靠它。prefill 注意力 O(n²)，命中长前缀吞吐能翻数倍——这就是 RadixAttention 名字的由来。

### 最反直觉点：cache_finished_req 的"重复 free"
请求来时为整段（含新 token）分配了槽位；结束 insert 回 tree 时发现前缀别人已插过 → **以 tree 已有的为准，把自己重复分配的那段 free 掉**。

> ★ 看到 `cache_finished_req` 里的 free 别慌，它 free 的是"我多占的那份"，不是公共那份。对账公式：**`available_size()` + `evictable_size()` + `protected_size()` 守恒**。

### 引用计数：保护正在用的 KV
`inc_lock_ref` 从节点**一路 +1 到 root**，evict 遇 `lock_ref>0` 跳过。

> ★ 难点是"锁沿路径到 root"——复用一条前缀整条祖先链都得锁，否则祖先被驱逐子节点 value 悬空。脏数据/崩溃八成是 inc/dec 不配对或忘了锁到 root。

### 三种 cache
`ChunkCache`（不做前缀复用，调试用）/ `HiRadixCache`（GPU+CPU 分层，被驱逐的 KV 写到 CPU 内存）/ `RadixCache`（默认）。

> ★ HiRadixCache 最易出 bug——write-back/load-back 是后台线程**异步**做的，主线程每 batch 要 `writing_check`/`loading_check` 收割并 dec_lock_ref；TP 多卡还要 `all_reduce(MIN)` 对齐进度。

**独门工具**：`RadixCache.pretty_print()` 打印整棵树；`python -m sglang.srt.mem_cache.radix_cache` 单独玩 insert/match，不必起 server。

参考：[13-memcache-radixattention](13-memcache-radixattention.md)

---

## 4. 模型执行：ModelRunner 与 CUDA Graph

### ForwardBatch
`ScheduleBatch → ModelWorkerBatch → ForwardBatch`（越来越瘦、越来越靠 GPU）。decode 路径 `positions = seq_lens-1`；extend 路径用 triton kernel 算所有 token 的 position。

> ★ **模型本体不区分 prefill/decode**——`forward_extend` 和 `forward_decode` 最终调同一个 `model.forward`，区别全在 forward_batch 的元数据和 **attention backend** 的实现。

### forward 分派
```
can_run_cuda_graph → replay；is_decode → forward_decode；is_extend → forward_extend；is_idle → forward_idle
return ret, can_run_cuda_graph   ← 第二个值传回上层供 overlap 用
```

### CUDA Graph（本章难点）
> ★ **为什么 prefill 不进 graph？** graph 要求 shape 静态。decode 每请求 1 token，batch token 数=bs，可枚举几十个 bs 各录一张。prefill 的 token 数=各序列 extend_len 之和，是几十到几万的连续动态值，无法枚举。spec decode 的 `TARGET_VERIFY` 能进，是因为每请求验证的 draft token 数固定。

**capture**：算要捕获的 bs（默认 `[1,2,4,8]+range(16,161,8)`）→ 预分配静态 buffer → 逆序遍历 bs，warmup 2 次后 `torch.cuda.graph` 录制，共享 `global_graph_memory_pool`。

**replay = 拷进去 → replay → 切回来**：
```
bisect 选 >= raw_bs 的最小 capture bs
seq_lens.fill_(1); out_cache_loc.zero_()   把假请求填安全值
copy_ 真实输入进静态 buffer
graph.replay(); 输出切回 raw_num_token
```

> ★ **为什么"拷进去/切回来"？** graph 录的是**固定地址、固定 shape** 的 kernel 串。replay 不能换张量，只能 `copy_` 改静态 buffer 内容。
> ★ **padding 安全值陷阱**：假请求填 `seq_len=1`（不读垃圾 KV）+ `out_cache_loc=0`（写到 padding 专用 slot 0），`num_token_non_padded` 记真实数。

### 一个极重要的连锁规则
**给 ForwardBatch 加要进 CUDA graph 的新字段，必须改四处**：dataclass + `init_new` + `CudaGraphRunner.__init__` 预分配 buffer + `capture/replay` copy_。

> ★ 漏任何一步 → graph 用陈旧数据，症状隐蔽（eager 对、graph 错）。黄金调试法：**`--disable-cuda-graph` 二分**。

### 顺带：显存预算 + 权重加载
`max_num_token = (可用显存 - 总显存*(1-mem_fraction_static)) // cell_size`。`--mem-fraction-static` 越大 → KV cache 越多。权重 shape mismatch 报错在 `default_weight_loader` 的 assert（`weight_utils.py:535`），通常是 TP 切分/weight name 映射错。

参考：[14-model-executor](14-model-executor.md)

---

## 5. 采样与结构化输出

### 施加顺序（骨架）
```
custom processor → penalty → grammar mask(必须最后) → temperature/top-k/top-p/min-p → 采样
```
> ★ **grammar mask 必须最后**：否则被置 -inf 的非法 token 被 penalty 改回有限值，破坏合法性约束。

### greedy 不是开关，是 `top_k==1`
`temperature<1e-6` 被改写成 `top_k=1`；`top_k==-1`(关闭)→`1<<30`。判贪心看 `is_all_greedy = all(top_k<=1)`。

### SamplingBatchInfo 向量化
per-request 标量参数拼成 GPU tensor，支持 `filter_batch`/`merge_batch`。
> ★ **merge 顺序坑**：`__len__` 定义在 `temperatures` 上，用到 `len()` 的合并必须在 cat temperatures 之前。新增 tensor 字段忘加进 filter/merge 列表 → batch 缩放 shape 不一致 crash。

### Sampler
top-p 用 `probs_sum - probs_sort`（不含自身的前缀和）保证至少留概率最大的 token。
> ★ flashinfer 快但 `top_p_renorm` 有数值问题，算 logprob 仍回退 torch。调数值 bug 切 `--sampling-backend pytorch`。默认不 all-reduce token id（赌 kernel 确定性），grammar 在场强制同步否则 hang。

### Penalty：lazy 分配
orchestrator 总建，但 penalizer 先 `_is_required()` 才分配 `[B,V]` tensor。penalty 靠 `cumulate_output_tokens`（`schedule_batch.py:1473`）喂历史 token。

### 结构化输出（FSM/Grammar）
把"合法 JSON/正则"编译成状态机，每步产出"允许哪些 token"，非法的置 -inf。
> ★ **grammar 编译异步**：大 schema 编译秒级，丢线程池 + grammar_queue，编译好再入队。缓存模板，复用时 `copy()`（每请求需独立游标）。
> ★ outlines 用 `[B,V]` bool mask；xgrammar 用 int32 bitmask（每 bit 一 token）+ Triton kernel，大 vocab 省一个数量级带宽。

### jump-forward（compressed FSM）
确定性前缀（如 JSON `{"name": `）一次性"跳"过去，少跑很多步 forward。
> ★ **最深的坑是"重新 tokenize"**：跳的是字符/字节，模型吃 token。直接拼接会因 tokenizer 合并规则错位。`jump_and_retokenize` 找公共前缀 → rollback 不一致尾部 → 逐个 accept 对齐。byte-level + 多字节字符（中文）要按 byte 链处理。

参考：[17-sampling-structured-output](17-sampling-structured-output.md)

---

## 6. Detokenizer 与输出回流

### Scheduler 侧（不 detokenize，只组装增量）
`init_incremental_detokenize`（`schedule_batch.py:671`）返回从 `surr_offset` 起的尾巴 + `read_offset`，第一次回看 5 个 token。
> ★ **为什么回看 5 个？** BPE 解码**上下文相关**：单独解一个 token 和它前面跟着几个 token 时结果可能不同（前导空格/字符合并）。多带 surrounding token 做参照才稳定。

### DetokenizerManager：增量核心技巧
```
new_text = read_text[len(surr_text):]
   surr_text = decode(surr_off .. read_off)
   read_text = decode(surr_off .. 末尾)
```
> ★ 增量本质：**用公共前缀(surr)做差，抵消 tokenizer 上下文相关性**。沿用自 vLLM。

**UTF-8 半字符**：decode 到一半吐 `"�"`。处理：末尾是 `"�"` 就**不推进 offset**，下一步连旧 token 一起重解凑成完整字符。
> ★ SGLang 没有显式 byte buffer——靠"不推进 + 重解"天然补全。**offset 推进和文本提交是绑定的**。

**三层 offset**（修"重复/丢字"bug 的地图）：scheduler `send_decode_id_offset` / detokenizer `surr/read/sent_offset` / tokenizer_manager `state.text`。每层只传新算的，层层做差。

### 结束判定为什么在 scheduler 侧自己 decode 一小段？
> ★ stop string 判定必须自己 decode 尾巴，因为 scheduler 此刻**没有 detokenizer 给的文本**——这是两进程职责分离的小冗余，为了让 detokenize 离开 GPU 关键路径。

### 上层解析 + 反向约束
reasoning(`<think>`)/function-call 解析**不在 detokenizer**，在 OpenAI adapter 消费 `state.text` 时做。流式 tool 用 partial-JSON 解析半截 arguments。
> ★ 一条"解析反向影响生成"的回路：`tool_choice="required"` 时 FunctionCallParser 通过 `get_ebnf` 把约束**下发到采样阶段**（回扣第 5 节）。出口的解析需求变成入口的生成约束。

参考：[18-detokenizer-output](18-detokenizer-output.md)

---

## 7. Worker 与并行（多卡基础设施）

### 四种并行坐标系
| 并行 | 切什么 | 通信 |
|---|---|---|
| TP | 每层权重矩阵 | all-reduce |
| PP | 模型的层分段 | 点对点 send/recv |
| DP | 多个完整副本 | 请求级分发 |
| EP | MoE 专家 | all-to-all |

### 一 rank 一进程
> ★ TP/PP 每个 rank 是独立 OS 进程（绕 GIL）。嵌套：`run_scheduler_process → Scheduler → TpModelWorker → ModelRunner → 模型分片`。代价是进程间靠 NCCL/gloo 同步（随机种子用 `broadcast_pyobj`，否则 TP rank 采样发散）。

### process group = 两个底层 ProcessGroup
> ★ 每个 group 同时建 nccl（GPU 通信）+ gloo（CPU 协调 / broadcast_pyobj）。还有 `CustomAllreduce` 给小张量加速。

### TP 口诀
**Column 切完不通信，Row 切完做一次 all-reduce**。一层 attention+MLP 约 2 次 all-reduce。

### PP
点对点传 `hidden_states`(PPProxyTensors)，**只有最后一段有 logits 和采样**。

### EP 复用 TP group
> ★ EP 没单独 group，复用 `_TP`。后果：`ep_size==tp_size`，专家数必须被 TP rank 数整除。通信是 all-to-all(dispatch/combine)。

### DP attention（和普通 DP 完全两回事）
> ★ GQA/MLA 模型 KV head 极少，TP 按 head 切不动只能复制 KV → 显存浪费。DP attention：attention 按 DP 切（每副本独立持有 KV，**不复制**），MLP/MoE 仍全 TP/EP，边界处 gather/scatter。
> ★ **两个 dp_size 含义不同**：`--dp-size` = 完全独立副本（权重各一份，DataParallelController 分发）；`--enable-dp-attention` = 复用同一 TP group，只 attention 层做 DP，权重不复制。

### DataParallelController
仅 `dp_size>1` 时存在，介于 tokenizer 和 scheduler 之间。`ROUND_ROBIN` 默认（`SHORTEST_QUEUE` 还是空壳）。PD 分离下用 `bootstrap_room % len(workers)` 保证同请求落固定副本。

参考：[15-worker-parallelism](15-worker-parallelism.md)

---

## 8. 计算层：Attention 后端、MoE 与算子

模型前向真正发生计算的地方。回扣 14 章"模型本体不区分 prefill/decode，区别全在 attention backend"。

### RadixAttention layer 怎么把 KV pool 接进 kernel
模型每层持有一个 `RadixAttention` 实例（几乎不算东西，只存 `layer_id` + head 配置），`forward` 转发给 `forward_batch.attn_backend.forward(...)`。
> ★ **backend 是全局单例，RadixAttention 是每层一个**。metadata（kv_indptr/kv_indices/cuda graph buffer）整个 batch 共享，每次 forward 只算一次；每层差异只有 layer_id。这样不重复算 metadata，又能让所有层走同一份 cuda graph。

两条线：写线 `set_kv_buffer(layer, out_cache_loc, k, v)`；读线 kernel 用 `get_key_buffer(layer_id)` + `kv_indices` gather。
> ★ 全貌：**layer 提供 layer_id 选行，backend 提供 kv_indices 选 token，kernel 直接在 pool 连续显存上 gather+attention**。不拷贝、不拼接 KV——RadixAttention + paged KV 高效的核心。`TorchNativeAttnBackend` 是最易读的慢速参考/ground truth。

### backend 选择：先定字符串，再实例化
> ★ 同一个 `"flashinfer"` 字符串按 `use_mla_backend` 落到两个不同类（MHA vs MLA KV 布局不同）。"看着一个 backend、实际两套实现"的坑。

### MoE：TP vs EP（回扣 15 章）
| | 专家权重 | 通信 | 类 |
|---|---|---|---|
| TP | 每 rank 全量持有 | all-reduce | FusedMoE |
| EP | 每 rank 部分专家 | all-to-all dispatch/combine | EPMoE/DeepEPMoE |

路由统一走 `select_experts` → `(topk_weights, topk_ids)`。
> ★ DeepSeek grouped topk：先分组打分、选组、组内选专家——限制跨节点通信。`on_select_experts` 记录专家负载 = **EPLB(25章) 的数据来源**；逻辑→物理专家号映射是在线搬专家的钩子。TP vs EP 是显存与通信的权衡。

### LogitsProcessor：核心难点是剪枝
> ★ 大多数情况只需每 seq 最后一个 token 的 logits（vocab 维巨大）。复杂度全在"要不要 input logprob"——三索引逻辑（`logits_processor.py:288-330`）是 logprob bug 高发区。lm_head 词表并行，需 all-gather 拼回完整分布。

### 量化：方法对象模式
> ★ **接新量化 = 写新 `LinearMethodBase` 子类（create_weights + apply），不碰 layer**。同一个 QKVParallelLinear 能在 fp16/fp8/awq 间切换。`prefix`（层全名）支持按层名 mixed-precision。坏处：method↔layer 契约隐式，易踩 shape 不匹配。

### CustomOp 分发 + torch.compile 隐藏约束
RMSNorm/SiluAndMul/RoPE 继承 CustomOp 按平台分发。`fused_add_rmsnorm` 原地改 x+residual。
> ★ **torch.compile 是隐藏约束**：多处为编译让路（RMSNorm 编译期强制 native、`q.reshape` 绕 rotary_emb 在 compile 下的 3D bug）。改算子务必同时测 eager 和 compile。

### 新增 attention backend（最常见需求）
```
1. 写类(继承 AttentionBackend): init_forward_metadata + forward_extend + forward_decode
2. init_attention_backend 加 elif 懒加载  3. server_args choices 加名字  4.(可选)自动选择
```
> ★ **最小可跑路径**：复制 `TorchNativeAttnBackend`，换 kernel，关 cuda graph 跑通，用它的输出做数值对拍。

参考：[16-layers-attention-moe](16-layers-attention-moe.md)

---

## 9. 投机解码（EAGLE）— 高级特性

### 一句话原理
> ★ 普通 decode：1 次 target 前向 = 1 token。投机解码：小而快的 **draft 模型（EAGLE 头）**猜出多个候选组成**候选树**，大而慢的 **target 模型一次前向并行验证整棵树**，接受最长合法前缀 = 1 次 target 前向产出多个 token。**不改变输出分布**（数学等价于直接从 target 采样），只加速不降质。trade-off：draft 猜得准就赚。

替换了正常 decode 的"前向+采样"：`run_batch` 看 `spec_algorithm.is_none()` 分发到 `EAGLEWorker.forward_batch_speculative_generation`。

### decode 三段式
```
spec_info(topk_p,topk_index,hidden_states)  ← 上步留的猜测种子
 draft()    多步 draft 前向逐层展开候选树
 build_tree 压成扁平树(draft_token + tree_mask + retrive 索引)
verify()    target 1 次大前向 → 接受/拒绝 → free 未接受 KV
 draft_extend_after_decode  用接受 token 重喂 draft 算下步 topk_*
```
- **draft**：`select_top_k_tokens` beam-search 式逐层剪枝。树宽=topk，深=num_steps。
  > ★ EAGLE draft 吃 **target hidden state + 下一 token embedding**，输入要错位一位（"左移一位"）。
- **build_tree**：产出 tree_mask（每候选只 attend 祖先链）/ positions / retrive_*。
  > ★ 搞懂树张量直接读单测 `test_build_tree_kernel_efficient`（带硬编码断言）——读复杂 kernel 的通用技巧。
- **verify**：greedy 沿 retrive 链最长匹配，或树版投机采样保证分布一致。
  > ★ **bonus token**：`seq_lens.add_(accept_length+1)` 那个 +1——哪怕全拒绝，target 根位置预测也是一个合法 token，所以每步至少产出 1 个，永不比普通 decode 慢。

### 最难点：投机 KV 的"分配→回滚→部分释放"
```
draft: 乐观给 topk*num_steps 候选全分配 KV + backup_state → 跑完 restore_state
verify: 重新为 num_verify_tokens 分配 → 只保留接受的, free 拒绝的(evict_mask)
```
> ★ `evict_mask` 与 `accept_index` 对应是 KV bug 头号高发区（漏 free→OOM，多 free→乱码）。`page_size>1` 强制 `topk=1`（释放要按页对齐，topk>1 还没实现）。

### 专用 CUDA graph
> ★ draft 多步循环 in-place 改 `out_cache_loc`/`hidden_states`，`EAGLEDraftCudaGraphRunner` 捕获前后必须**备份还原**这两字段（回扣 14 章 graph "固定地址"本质）。draft 用专用 runner，verify(TARGET_VERIFY) 用 target 普通 runner，DRAFT_EXTEND 不走 graph。

### 认知负担
> ★ `forward_batch_speculative_generation` docstring 警告：**batch 很多字段执行中被 in-place 改，最终状态≠入参**。看到值"莫名变了"先怀疑某步 in-place。三参数 num_steps/topk/num_draft_tokens 强耦合（server_args.py:409 自动调整）。

参考：[20-speculative-eagle](20-speculative-eagle.md)

---

## 10. PD 分离（Prefill/Decode Disaggregation）

### 为什么拆
> ★ prefill 是 **compute-bound**（算整段 prompt，算力打满），decode 是 **memory-bound**（每步 1 token，瓶颈在 KV 带宽）。同实例上 prefill 会打断 decode 造成 TBT 抖动。拆开后**两类实例各自选最优并行**（prefill `deepep normal` 吞吐优先，decode `low_latency` 延迟优先）。根本动机是各自最优,不是省资源。

### 架构：客户端连 LB 不连实例
```
客户端 → mini_lb(选一对 P+D, 注入 bootstrap 三元组, 同时 POST 两台)
prefill: 分词→extend→采1个token→KVSender ─RDMA WRITE─► decode 显存
decode:  预分配KV槽→KVReceiver 收→假extend(跳过prefill)→正常decode→响应
```
> ★ bootstrap 三元组(host/port/room)：room 是请求全局 ID 用来**配对** prefill/decode。**prefill 只采 1 token**(max_new_tokens=1)+logprobs 作 aux 单独传，decode 用**"假 extend"**接上当起点。**传输方向是 prefill 主动 WRITE 到 decode**(single-sided)。

### 两类实例复用同一 Scheduler，只换 event loop
- prefill 三段：Bootstrap(等 decode 注册地址)→Waiting(extend forward)→Inflight(send_kv_chunk 传输)。
- decode 四段：Prealloc(握手+预分配)→Transfer(poll+读首token)→Waiting(假extend)→Running(正常decode)。
> ★ **"假 extend"** `process_prebuilt_extend` 只填 metadata 不跑 forward(KV 已传来)，骗过后续 decode 逻辑。**decode 死锁防护** `_allocatable_tokens` 为传输中/等待的请求预留 decode 空间(num_reserved_decode_tokens=512),否则"KV 传来了没地方 decode"。

### 最烧脑：prefill/decode TP size 不一致
> ★ 三种情况(仅 MLA 支持不等)。**dummy 请求**是精髓：decode TP < prefill TP 时对不要数据的 prefill rank 发 dummy 凑数，因为 prefill 要收齐 `required_dst_info_num` 才翻 WaitingForInput，否则**永久卡 Bootstrapping**。状态机用 `max(old,new)` 合并保证单调前进(prefill 可能先收 decode 注册后设自己状态)。

### KV transfer 后端
mooncake(线程池并发逐层)/nixl(一次 xfer 走 GPUDirect)/fake(warmup)。`group_concurrent_contiguous` 合并连续块减少 RDMA 描述符。
> ★ **overlap 下 KV 传输要延迟**：不立即 send_kv_chunk,推迟到 result resolve 后,否则传到尚未算完的 KV（又一处 overlap 时序陷阱）。

### KV events（独立机制）
把"KV block 存入/移除 radix tree"ZMQ PUB 广播,供外部 KV-aware 路由器订阅做 prefix-aware 路由。带 seq + replay buffer 实现 at-least-once。

修 bug：卡 Bootstrapping(dummy/required_dst_info_num)、高并发 hang(手动 reset batch_is_full)、KV 乱码(page 对齐/分组/overlap 延迟)、decode 死锁(预留不足)。调试用 FAKE 后端排除 RDMA。

参考：[21-pd-disaggregation](21-pd-disaggregation.md)

---

## 11. EPLB：专家并行负载均衡

### 解决什么
> ★ MoE router 学出来天然不均衡——热门专家被路由的 token 远多于others，大规模 EP 下持有热门专家的 GPU 成瓶颈。EPLB 运行时**统计专家热度 → 算更优物理放置(含冗余副本) → 不重启搬权重**。三块：record → rebalance → update。挂在 forward 之后的**旁路控制环路**。

### 地基：logical vs physical + 冗余副本
> ★ logical = 模型真实专家(router 输出 logical id)；physical = 放在某 GPU 上的一份权重 = num_logical + 冗余。**冗余副本=把热门专家复制多份分散到不同 GPU**。logical→physical 解耦让 router 不变、放置可独立变化。`ExpertLocationMetadata` 4 张表，关键是 `logical_to_rank_dispatch_physical_map`(给当前 rank 预选的副本)。

### ① record 统计
forward 用 `with_forward_pass` 包住，MoE topk 后 `on_select_experts`(回扣16章)喂 gatherer。
> ★ 开销 vs 精度：朴素 gatherer 搬 CPU 计数慢；**DeepEP 路径复用 dispatch 已算好的 count 零开销**(大规模部署配 DeepEP 的原因)。一卡只看自己那段,all-reduce(SUM)拼全局;dump 时 scatter_add_ 把物理计数折回逻辑计数。

### ② rebalance 算法
`make_redundant_experts_chunkwise`：贪心放冗余(给"分摊后负载最高"的专家加副本,打分 tokens/(count+1)) + GPU 间蛇形二次均衡。
> ★ **刻意制造"互不相等分数"**(加 1e-4*arange)：各 rank 独立跑算法必须算出**完全一致**放置,有并列会导致不同 rank 选不同 = 灾难。分布式确定性经典技巧。**prefill/decode 不同算法**(回扣21):prefill 考虑 group/node 局部性先 pack_groups,decode 直接全局 chunkwise,phase 由 disaggregation_mode 推断。

### dispatch 翻译 + ③ update 搬权重
forward 用 `topk_ids_logical_to_physical` 翻译(static 查表本地优先/dynamic 实时随机)。update 先搬权重再 `metadata.update()` 原地覆盖 4 表(dispatch 端立刻生效)。单层搬运 5 种 case(unchanged/same-gpu/free-rider/same-node/cross-node)。
> ★ **free-rider(case3)依赖"按 dst 升序遍历"**(后槽复用前槽已收数据),打乱顺序就破坏——隐式契约易踩。P2P 配对不上会全体死锁。

### 修 bug
精度骤降(dump 用那一刻的旧 map)、各 rank 放置不一致(全局 logical_count + 互不相等分数 trick)、update hang(P2P 配对,debug=True 看 case)、dump 全 0(忘 start_record)。
> ★ **circular buffer 耦合**:断言 `eplb_rebalance_num_iterations <= buffer_size`,否则混入陈旧数据。开 `--enable-expert-distribution-metrics` 看 balancedness(越近 1 越均衡)。

参考：[25-eplb-expert-parallel](25-eplb-expert-parallel.md)

---

## 12. Multi-LoRA 批处理

### 数学简单，难点全在工程
```
y = x W + scaling*(x A^T)B^T    A:(r,in) B:(out,r) r≪in/out
```
> ★ 难点不在数学：怎么让 batch 每行用对的 A/B、怎么省显存(挂100个不占100份)、怎么和 CUDA graph/TP 共存。思想来自 S-LoRA + Punica。寄生在 forward 的 linear 层,scheduler 组 batch 时准入(distinct adapter ≤ max_loras_per_batch)。

### 四阶段
- **A 加载+stack**：把 q/k/v 的 lora_A cat 成 qkv_proj、gate/up 拼成 gate_up_proj。
  > ★ 为什么 stack？base model 本身把 q/k/v 融成 QKVParallelLinear(回扣16)，LoRA 必须对齐。**补零陷阱**:只对 q 做 LoRA 没给 k/v 会 zeros_like 补零,可能静默错误。
- **B monkey-patch**：命中 target 的 linear 原地换成 LoRA 版,forward = base GEMM + `if set_lora: apply_lora`。
- **C 换入(显存池精髓)**：只有 max_loras_per_batch 个 slot,所有 adapter 共享按需换入换出。
  > ★ **S-LoRA 核心**:显存只和 max_loras_per_batch 有关,**与挂载多少 adapter 无关**。切换开销只在冷 adapter(H2D 拷贝),热 adapter 零拷贝复用。uid=None(base-only)也是合法 key,slot 清零走同一 kernel。
- **D segmented GEMM(核心优化)**：x 拍平成 (s,in)，LoRABatchInfo(seg_indptr/weight_indices/lora_ranks/scalings)当导航图,kernel 里 `w_index=weight_indices[batch_id]` 选权重、`lora_ranks` 动态裁剪。
  > ★ **一次 launch 覆盖整 batch,每段自动路由到对的 adapter/rank/scaling**(Punica/S-LoRA)。不逐 adapter 循环是因为 launch 开销+decode 瘦 GEMM 利用率差。**rank 不一**:buffer 按 max_lora_dim 分配,kernel 用 lora_ranks 收缩到真实 rank(flashinfer 不支持,要求同 rank)。

### TP + CUDA graph 共存
> ★ **TP 切分在换入时**做,池里存本 rank 分片 kernel 无需感知 TP;切错维度=TP>1 错乱 TP=1 正常(最隐蔽)。**CUDA graph** 单独一套 cuda_graph_batch_info,原地 in-place 更新;capture 时 lora_path=None 要替换占位。

调试:确认 target_modules 被支持→打印 stack 后 key(应是 qkv_proj)→打印 weight_indices/lora_ranks→数值不对先关 cuda graph 切 triton。

参考：[22-lora](22-lora.md)

---

## 学习建议

1. **动手胜过读**：`python -m sglang.srt.mem_cache.radix_cache` 玩 radix 树；`--disable-overlap-schedule` / `--disable-cuda-graph` 二分定位 bug。
2. **用 rid 串日志**：开 `--log-requests`，一条 rid 贯穿三个进程。
3. **四个枢纽断点**：`get_next_batch_to_run` / `run_batch` / `process_batch_result_decode` / `_handle_batch_output`。

> 下一批追加：16 计算层（attention backend / MoE 算子）、20 投机解码、21 PD 分离 …
