# 调度器：连续批处理与 zero-overhead 调度（核心）

> 本章是整套内部文档里最难、也最核心的一章。Scheduler 是 SGLang runtime 的心脏：它决定「下一步算哪些请求、算多少、怎么把不同进度的请求拼在一起、怎么让 CPU 调度和 GPU 计算不互相等待」。读懂这一章，你才真正读懂 SGLang 为什么快。

---

## 1. 一句话职责

Scheduler 是一个**单进程、单线程的无限事件循环**：不断从 tokenizer 收新请求，按显存预算和 prefix 命中情况决定下一个 batch（prefill 还是 decode、装哪些请求），把 batch 交给 GPU worker 前向，再把前向结果处理成 token / logprob 发回给客户端。它通过 **continuous batching**（连续批处理）保证 GPU 永远满载，通过 **overlap scheduler**（重叠调度）把「调度下一批的 CPU 工作」和「当前批的 GPU 计算」并行，从而做到 *zero-overhead*。

核心循环就这四步（见 `event_loop_normal`，`python/sglang/srt/managers/scheduler.py:636`）：

```
recv_requests → process_input_requests → get_next_batch_to_run → run_batch → process_batch_result
```

---

## 2. 在端到端链路中的位置

```
 客户端 HTTP
    │
    ▼
[Tokenizer Manager]  分词 / 多模态预处理        (见 11-tokenizer-manager.md)
    │  ZMQ: TokenizedGenerateReqInput
    ▼
┌─────────────────────────────────────────────────────────────┐
│                        SCHEDULER  (本章)                       │
│                                                               │
│  recv_requests ──► waiting_queue ──► schedule_policy 排序      │
│                          │                                     │
│            get_next_batch_to_run                              │
│           （prefill 优先 / 否则 decode）                       │
│                          │                                     │
│                  ScheduleBatch  ──────► run_batch             │
│                          │                  │                  │
│                          │            (overlap: 单独前向线程)   │
└──────────────────────────┼──────────────────┼────────────────┘
                           │                  │
                           ▼                  ▼
              [RadixCache / KV Pool]   [TpModelWorker → ModelRunner]
              KV 缓存 / prefix 复用       前向 + 采样              (见 13/14/15 章)
                           │                  │
                           └──────► process_batch_result
                                        │ ZMQ: BatchTokenIDOut
                                        ▼
                                  [Detokenizer] 解码 → 响应     (见 18 章)
```

Scheduler 既是「KV 缓存」和「前向」的**调用方**，又是请求生命周期的**状态机持有者**。它本身不做矩阵乘法，只做「决策 + 簿记」。

---

## 3. 关键文件与类

| 符号 | file:line | 作用 |
|------|-----------|------|
| `Scheduler.event_loop_normal` | `python/sglang/srt/managers/scheduler.py:636` | 普通（非重叠）事件循环 |
| `Scheduler.event_loop_overlap` | `python/sglang/srt/managers/scheduler.py:656` | overlap 事件循环（zero-overhead 的入口） |
| `Scheduler.event_loop_pp` | `python/sglang/srt/managers/scheduler.py:700` | pipeline parallelism 事件循环 |
| `Scheduler.recv_requests` | `python/sglang/srt/managers/scheduler.py:802` | 从 ZMQ 收请求并在 TP 组内广播 |
| `Scheduler.process_input_requests` | `python/sglang/srt/managers/scheduler.py:880` | 把收到的请求 dispatch 到各 handler |
| `Scheduler.get_next_batch_to_run` | `python/sglang/srt/managers/scheduler.py:1286` | **核心调度决策**：合并/过滤、prefill vs decode |
| `Scheduler.get_new_batch_prefill` | `python/sglang/srt/managers/scheduler.py:1342` | 用 `PrefillAdder` 组装一个 prefill batch |
| `Scheduler.update_running_batch` | `python/sglang/srt/managers/scheduler.py:1496` | decode 批的过滤、OOM 检测、retract |
| `Scheduler.run_batch` | `python/sglang/srt/managers/scheduler.py:1533` | 调 worker 做前向，返回 `GenerationBatchResult` |
| `Scheduler.process_batch_result` | `python/sglang/srt/managers/scheduler.py:1613` | 分发到 prefill/decode 结果处理 |
| `run_scheduler_process` | `python/sglang/srt/managers/scheduler.py:2252` | 进程入口：建 Scheduler 并选事件循环 |
| `SchedulePolicy` | `python/sglang/srt/managers/schedule_policy.py:69` | 排序 waiting_queue（FCFS/LPM/DFS/LOF/random） |
| `PrefillAdder` | `python/sglang/srt/managers/schedule_policy.py:268` | 准入决策核心：逐个 req 试装，管理 token 预算 |
| `AddReqResult` | `python/sglang/srt/managers/schedule_policy.py:262` | 准入结果枚举：CONTINUE/NO_TOKEN/OTHER |
| `Req` | `python/sglang/srt/managers/schedule_batch.py:420` | 单个请求的输入/输出/状态机 |
| `ScheduleBatch` | `python/sglang/srt/managers/schedule_batch.py:787` | 一个 batch 的全部信息（reqs + 张量） |
| `Req.check_finished` | `python/sglang/srt/managers/schedule_batch.py:683` | 请求终止条件判定 |
| `ScheduleBatch.retract_decode` | `python/sglang/srt/managers/schedule_batch.py:1337` | decode OOM 时回退请求 |
| `TpModelWorkerClient` | `python/sglang/srt/managers/tp_worker_overlap_thread.py:51` | overlap 模式下的前向线程封装 |
| `TpModelWorkerClient.forward_thread_func_` | `python/sglang/srt/managers/tp_worker_overlap_thread.py:125` | 独立线程里跑前向 + 采样 |
| `resolve_future_token_ids` | `python/sglang/srt/managers/tp_worker_overlap_thread.py:43` | 解析 future token（overlap 的关键技巧） |
| `process_batch_result_prefill` | `python/sglang/srt/managers/scheduler_output_processor_mixin.py:32` | prefill 结果处理 |
| `process_batch_result_decode` | `python/sglang/srt/managers/scheduler_output_processor_mixin.py:185` | decode 结果处理 |

---

## 4. 核心数据流 / 执行流程

### 4.1 请求进来什么样、出去什么样

进来：tokenizer 通过 ZMQ 发来 `TokenizedGenerateReqInput`（已分词的 `input_ids` + 采样参数）。`handle_generate_request`（`python/sglang/srt/managers/scheduler.py:897`）把它包装成一个 `Req` 对象，挂进 `self.waiting_queue`。

出去：每跑完一个 batch，`process_batch_result` 给每个 req 追加一个 `output_id`，判定是否结束，并通过 `stream_output` 把 `BatchTokenIDOut` 发回 detokenizer。

### 4.2 主循环（normal 模式）逐步拆解

```
event_loop_normal()  (scheduler.py:636)
└─ while True:
   ① recv_reqs = recv_requests()           # 非阻塞从 ZMQ 收，TP 内广播
   ② process_input_requests(recv_reqs)      # Req 入 waiting_queue
   ③ batch = get_next_batch_to_run()        # ★ 全部调度智慧在这
   ④ if batch:
        result = run_batch(batch)            # GPU 前向 + 采样（同步）
        process_batch_result(batch, result)  # 追加 token、判完成、发输出
      else:
        check_memory(); 重置 new_token_ratio  # idle 自检
   ⑤ last_batch = batch
```

注意 ③④ 在 normal 模式里是**严格串行**的：CPU 准备好 batch → 等 GPU 算完 → CPU 处理结果 → 再准备下一个 batch。GPU 在「CPU 准备/处理」期间是空闲的。这正是 overlap 模式要消除的开销（见 §4.5）。

### 4.3 get_next_batch_to_run：调度决策的核心（scheduler.py:1286）

这个函数干三件事：

**(1) 合并上一批 prefill 到 running_batch（continuous batching 的拼接点）**

```
if last_batch 是 extend(prefill):
    last_batch.filter_batch()          # 踢掉已完成 / chunked 的 req
    running_batch.merge_batch(last_batch)  # 把新 prefill 的 req 并入正在 decode 的批
```

这就是「连续批处理」的本质：刚做完 prefill 的请求，下一步立刻和老请求一起 decode，**不需要等一个 batch 全部结束**。`merge_batch`（`schedule_batch.py:1571`）直接把两个 batch 的张量 `torch.cat` 在一起。

**(2) prefill 优先**

```
new_batch = get_new_batch_prefill()
if new_batch is not None:
    ret = new_batch          # 有新请求要 prefill，就先 prefill
else:
    ret = update_running_batch(running_batch)  # 否则 decode 一步
```

设计取舍：SGLang **prefill 优先**于 decode。原因是 prefill 能尽快把新请求的 KV 填好、让它进入 decode 流，提高并发；代价是正在 decode 的请求会被新 prefill「插队」（latency 抖动）。chunked prefill（见 §5.3）就是为了缓解这个插队。

**(3) DP attention 的对齐**（`prepare_dp_attn_batch`），让各 DP rank 步调一致，这里不展开。

### 4.4 get_new_batch_prefill + PrefillAdder：准入决策（scheduler.py:1342 / schedule_policy.py:268）

这是「显存预算 + prefix 匹配 + chunked prefill」三件事交汇的地方。

```
get_new_batch_prefill():
  ① 若 running_batch 已满 或 waiting_queue 空 → 不 prefill
  ② num_allocatable = max_micro_batch_size - running_bs  # 槽位预算
  ③ prefix_computed = policy.calc_priority(waiting_queue) # ★ 排序 + prefix 匹配
  ④ adder = PrefillAdder(预算: max_prefill_tokens, chunked_prefill_size, ...)
  ⑤ for req in waiting_queue:
        req.init_next_round_input(tree_cache)  # 算 prefix_indices / extend_input_len
        res = adder.add_one_req(req)           # 试装这个 req
        if res != CONTINUE: break              # 预算耗尽就停
  ⑥ can_run_list = adder.can_run_list
  ⑦ 从 waiting_queue 移除已装入的；建 ScheduleBatch.init_new(); prepare_for_extend()
```

**PrefillAdder 的预算模型**（`schedule_policy.py:309-334`）维护两个量：

- `rem_total_tokens` = KV pool 可用 + tree_cache 可回收（evictable） − 已为本批 prefill/decode 预留的 token。注意它包含「未来 decode 要生成的 token」估算：每个 running req 预留 `max_new_tokens * new_token_ratio` 个 token（`schedule_policy.py:297-307`）。
- `rem_chunk_tokens` = 本次单批最多 prefill 多少 token（chunked prefill 上限）。

`add_one_req`（`schedule_policy.py:445`）的判定顺序：
1. `total_tokens >= rem_total_tokens` → 返回 `NO_TOKEN`（显存不够，停止加请求）。
2. prefix 命中：`prefix_len = len(req.prefix_indices)`，命中的部分**不计入** extend，只算 `extend_input_len`（要真正前向的新 token）。这就是 RadixAttention prefix 复用直接降低 prefill 成本的地方。
3. 若 `input_tokens > rem_chunk_tokens`：触发 **chunked prefill**——把这个超长 prompt 截成 `trunc_len`，剩下的留到下一批，记 `self.chunked_req`（`schedule_policy.py:497-510`）。

**new_token_ratio** 是个动态保守系数：刚发生过 decode OOM 时调大（更保守地预留 decode 空间），随时间逐步衰减回 `init_new_token_ratio`（`scheduler.py:1521-1524`、idle 时 `scheduler.py:651` 重置）。

### 4.5 update_running_batch + retract：decode 的 OOM 防护（scheduler.py:1496）

decode 每步给每个活跃 req 生成 1 个 token，需要新分配 KV slot。如果 KV pool 满了：

```
update_running_batch(batch):
  batch.filter_batch()                          # 先踢掉已完成的
  if not batch.check_decode_mem():               # KV 不够（先 evict 再判）
      retracted, new_ratio = batch.retract_decode()  # ★ 回退一部分请求
      new_token_ratio = new_ratio                # 变保守
      _extend_requests_to_queue(retracted)       # 退回 waiting_queue 重排
  batch.prepare_for_decode()
```

`retract_decode`（`schedule_batch.py:1337`）按「输出最长、输入最短」排序，从尾部回退请求：释放它们的 KV，调用 `reset_for_retract` 把 `prefix_indices` 清空、`is_retracted=True`，再扔回 waiting_queue。被回退的请求之前算的 KV 可能已进 radix cache，重排后能命中 prefix，所以不是完全白算。这是 SGLang「宁可回退也不崩」的抢占式调度。

### 4.6 overlap scheduler：zero-overhead 的来源（scheduler.py:656 + tp_worker_overlap_thread.py）

normal 模式的痛点：GPU 算 batch 时 CPU 闲、CPU 调度时 GPU 闲。overlap 把前向放进**单独线程**，让两者重叠。

```
event_loop_overlap()  (scheduler.py:656)
└─ while True:
   batch = get_next_batch_to_run()       # CPU: 调度第 N 批
   if batch:
       batch.launch_done = Event()
       result = run_batch(batch)          # 只是把 batch 丢进 input_queue 就返回！
       result_queue.append((batch.copy(), result))
       ...
   if last_batch:                          # CPU: 处理第 N-1 批的结果
       tmp_batch, tmp_result = result_queue.popleft()
       process_batch_result(tmp_batch, tmp_result, batch.launch_done)
   last_batch = batch
```

关键点 1：**run_batch 不再阻塞**。`run_batch` 调 `TpModelWorkerClient.forward_batch_generation`（`tp_worker_overlap_thread.py:206`），后者把 `model_worker_batch` 放进 `input_queue` 立即返回，真正的前向在 `forward_thread_func_`（`tp_worker_overlap_thread.py:125`）这个后台线程里跑。

关键点 2：**future token ids**（`tp_worker_overlap_thread.py:226-237`）。问题：调度第 N+1 批时，第 N 批的 token 还没算出来（在 GPU 上），但第 N+1 批可能要用第 N 批的输出当输入（同一个请求连续 decode）。解决：`forward_batch_generation` 返回的不是真 token，而是一批**负数占位符** `future_next_token_ids`（`-(ct+1), -(ct+2), ...`）。调度照常进行，把这些负数填进下一批的 `input_ids`。真正前向时，`resolve_future_token_ids`（`tp_worker_overlap_thread.py:43`）在 GPU 上用 `torch.where` 把负数替换成 `future_token_ids_map` 里已算好的真值。整个解析在 GPU 上做，CPU 不必等。

关键点 3：**结果延迟一拍处理**。第 N 批的结果在循环的下一轮（处理第 N+1 批时）才用 `resolve_last_batch_result`（`tp_worker_overlap_thread.py:182`）取回——它从 `output_queue` 拿，`launch_done.wait()` + `copy_done.synchronize()` 确保 GPU→CPU 拷贝完成。所以 `result_queue` 里总积压一个批；这也是为什么开头要造一个 `DUMMY_FIRST` 批（`scheduler.py:675`）来启动流水线。

时间线对比：

```
normal:  [调度N][前向N........][处理N][调度N+1][前向N+1......][处理N+1]
              CPU    GPU        CPU      CPU       GPU          CPU
         (GPU 在 CPU 工作时空闲)

overlap: [调度N+1][处理N]            CPU 线程
              [前向N..............]   GPU 线程（与 CPU 重叠）
         GPU 几乎不空闲 → zero-overhead
```

代价：实现复杂度、一拍延迟、需要小心 in-place 修改（见 §5.5）。可用 `--disable-overlap-schedule` 关闭（`scheduler.py:202`）。

### 4.7 pipeline parallelism 模式（scheduler.py:700）

`event_loop_pp` 是为 pp_size>1 的多 microbatch 流水线。每轮遍历 `pp_size` 个 microbatch 槽位：非最后 rank 把 hidden states / token 用 `send_tensor_dict` 传给下一 stage，最后 rank 算出 token 再回传。`recv_requests` 在 pp_rank>0 时不是从 ZMQ 收，而是从上一 stage `point_to_point_pyobj` 收（`scheduler.py:826`）。这是 normal 的非重叠变体，理解 normal 后看它即可。

---

## 5. 设计难点与权衡

### 5.1 为什么 prefill 优先而不是公平调度？
prefill 优先能最大化 batch 内的并发请求数（吞吐），但牺牲已运行请求的尾延迟。SGLang 选吞吐优先，并用 chunked prefill 限制单次 prefill 的 token 数（`chunked_prefill_size`）来压住对 decode 的插队伤害。

### 5.2 prefix 匹配作为优先级（schedule_policy.py:90-121）
`SchedulePolicy.calc_priority` 默认 LPM（longest prefix match）：把命中 radix cache prefix 最长的请求排前面，让它们能复用已有 KV、扎堆命中。但有两个非显然细节：
- waiting_queue 超过 128 个请求时，`_determine_active_policy`（`schedule_policy.py:123`）退化成 FCFS——因为 prefix 匹配 + 排序本身有 CPU 开销，队列大时得不偿失。
- **in-batch prefix caching**（`schedule_policy.py:170-192`）：如果很多请求共享同一个尚未进 cache 的新 prefix，只放行一个、其余 deprioritize，等第一个把 prefix 写进 cache 后其余再来命中，避免重复 prefill 同一段。

### 5.3 chunked prefill 的内存安全细节
chunked req 跨 batch 存在，`get_next_batch_to_run` 开头会把它从 last_batch 排除、`cache_unfinished_req` 并 `free(req_pool_idx)`（`scheduler.py:1289-1295`），下一轮重新分配 `req_pool_idx`（保留 rid）。`add_one_req` 截断时特意留「至少一个 page」（`schedule_policy.py:498-499`），避免 page_size>1 时算出 0 长度的 chunk 死循环。

### 5.4 显存预算的「双账本」
`rem_total_tokens` 和 `cur_rem_tokens` 是两本账：前者含「未来 decode 预留」，后者只含「当前这一步要占的」。`budget_state`（`schedule_policy.py:325`）任一为负都停。这种保守预留 + `new_token_ratio` 动态调节，是在「尽量多塞请求」和「避免 decode 中途 OOM 而 retract」之间找平衡。retract 很贵（要重排、可能重算），所以宁可预留多一点。

### 5.5 overlap 模式的 in-place 陷阱
`run_batch` 里有一段注释（`scheduler.py:1580-1591`）：`extend_input_len_per_req` 等值必须在这里**拷贝**，因为 overlap 模式下后续调度会 in-place 改 `req` 的字段。同理 `result_queue` 存的是 `batch.copy()`（`scheduler.py:670`）而非 batch 本身。`sampling_info` 也通过 `dataclasses.replace`（`tp_worker_overlap_thread.py:212`）复制，避免下一批的采样参数覆盖正在前向的这一批。**改 overlap 相关代码时，最容易踩的就是「忘了某个字段会被下一轮改掉」。**

### 5.6 overlap 模式多生成的「一拍 token」
overlap 因为流水线延迟一拍，请求 finish 的那一拍可能已经为它多分配/多算了一个 token 的 KV。`process_batch_result_decode`（`scheduler_output_processor_mixin.py:218-230`）和 prefill 路径（`:78-82`）专门 free 这个「延迟的多余 token」的 `out_cache_loc`。这是个非常容易在重构时漏掉、导致 KV 泄漏的点。

### 5.7 Req 状态机
`Req` 的生命周期靠几个字段隐式编码（`schedule_batch.py:420`）：
```
waiting (在 waiting_queue)
  └─ prefix_indices / extend_input_len 由 init_next_round_input 算出
running-extend  (forward_mode=EXTEND, is_chunked>0 表示还在分块)
running-decode  (并入 running_batch, 每步 output_ids.append)
  └─ check_finished() 设 finished_reason → filter_batch 踢出
retracted (is_retracted=True, 回到 waiting)
finished  (finished_reason != None)
```
注意 `to_abort`/`to_abort_message`（`schedule_batch.py:481`）：中途要 abort 不直接设 `finished_reason`（否则会被立刻 filter 掉、永不响应），而是设 `to_abort`，由 `check_finished`（`:687`）在事件循环里转成 `FINISH_ABORT`，保证客户端能收到结束消息。

---

## 6. 改代码 / 修 bug 指南

### 「想改 X → 看这里」

| 你想做的事 | 入手位置 |
|-----------|---------|
| 改准入策略（一次塞多少请求/多少 token） | `PrefillAdder.add_one_req` `schedule_policy.py:445`；预算 `rem_total_tokens` `:309` |
| 改 waiting_queue 排序 / 新增调度策略 | `SchedulePolicy.calc_priority` `schedule_policy.py:90`，加进 `CacheAwarePolicy`/`CacheAgnosticPolicy` |
| 改 prefill vs decode 的优先级 | `get_next_batch_to_run` `scheduler.py:1318` 的分支 |
| 改 chunked prefill 行为 | `add_one_req` 截断逻辑 `schedule_policy.py:497`；跨批清理 `scheduler.py:1289` |
| 改 OOM 时的回退策略 | `retract_decode` `schedule_batch.py:1337`；触发点 `update_running_batch` `scheduler.py:1506` |
| 改请求终止条件（新 stop 规则） | `Req.check_finished` `schedule_batch.py:683` |
| 改 overlap 前向线程 | `forward_thread_func_` `tp_worker_overlap_thread.py:125`；future token `:43` |
| 改新请求的初始化 | `handle_generate_request` `scheduler.py:897` |
| 改输出/流式发送 | `process_batch_result_*` 与 `stream_output` `scheduler_output_processor_mixin.py:445` |
| 加一种新 RPC/控制请求 | `process_input_requests` + `_request_dispatcher` `scheduler.py:880` |

### 该模块常见 bug 高发区

1. **KV / req_pool 泄漏**：chunked req 的 `free(req_pool_idx)`（`scheduler.py:1295`）、overlap 多余 token 的 free（§5.6）、retract 的 free（`schedule_batch.py:1393`）。任何一处漏 free，长跑后 `available_size()` 会缓慢下降直到无法 prefill。用 `check_memory()`（idle 时调用）打的日志对账。
2. **overlap 的张量被提前释放/被覆盖**：`forward_thread_func_` 里特意用 `batch_lists` 持有引用（`tp_worker_overlap_thread.py:136-139`）防止 PyTorch 释放导致 *CUDA illegal memory access*。改 overlap 时遇到这个错，先怀疑是不是某个 batch/tensor 没被 hold 住。
3. **future token 没解析**：忘了走 `resolve_future_token_ids`，会出现 input_ids 是负数 → 越界/乱码输出。
4. **batch_is_full 缓存不一致**：`batch_is_full` 是为省 prefill 检查开销的缓存标志（`schedule_batch.py:803`），多处会重置它（如 `scheduler.py:1308`、`:1502`、`:1527`）。改调度逻辑后若发现「明明有空间却不收新请求」，多半是某条路径忘了把它置 False。
5. **abort 设错字段**：直接设 `finished_reason` 会让请求静默消失，必须用 `to_abort`（§5.7）。

### 调试入手点

- 想看「每个 batch 到底装了啥」：在 `log_prefill_stats`（`get_new_batch_prefill` 末尾 `scheduler.py:1460`）和 decode 日志加打印，或直接打 `len(can_run_list)`、`adder.rem_total_tokens`。
- 想区分是 normal 还是 overlap 的 bug：用 `--disable-overlap-schedule` 关掉 overlap（`scheduler.py:202`），bug 消失 → 问题在重叠逻辑。
- 怀疑显存预算算错：在 `PrefillAdder.budget_state`（`schedule_policy.py:325`）打印 `rem_total_tokens` / `cur_rem_tokens` / `rem_chunk_tokens`。
- 怀疑 retract 抖动：看日志里的 "Decode out of memory happened"（`scheduler.py:1514`）和 `new_token_ratio` 的变化曲线。
- 进程入口和事件循环选择在 `run_scheduler_process`（`scheduler.py:2252`）；scheduler 崩溃会给父进程发 `SIGQUIT`（`:2326`），看父进程日志能拿到完整 traceback。

---

## 相关章节

- KV 缓存与 RadixAttention prefix 复用：[KV 缓存](13-memcache-radixattention.md)
- 前向与 ModelRunner：[前向计算](14-model-executor.md)
- 采样：[采样](17-sampling-structured-output.md)
- 上游分词：[Tokenizer Manager](11-tokenizer-manager.md)
- 下游解码与响应：[Detokenizer](18-detokenizer-output.md)
- 名词解释（prefill / decode / TP / overlap 等）：[术语表](02-glossary.md)
