# 请求全生命周期：从客户端发出到收到响应

> 本章是整套「SGLang 内部架构文档」的主干。它从「链路视角」把一条请求的一生完整走一遍：你发出 `POST /generate` 的那一刻，到最后一个字节流回你的终端，中间到底经过了哪些**进程**、哪些**ZMQ socket**、哪些**关键函数**，数据形态经历了 `text → token ids → Req → ScheduleBatch → ForwardBatch → logits → output ids → text` 的层层变换。
>
> 读完本章，你应该能在脑子里画出整张图，并且知道「想改 X 该去哪个文件」。后续每一章（[调度器](12-scheduler.md)、[KV 缓存](13-memcache-radixattention.md) 等）都是对本章某一阶段的放大。

---

## 一句话职责

把一条 HTTP 请求，经由 **TokenizerManager（分词进程）→ Scheduler（调度+推理进程）→ DetokenizerManager（解码进程）→ TokenizerManager** 这条三进程流水线，在 **continuous batching（连续批处理）** 框架下逐 token 地生成并流式返回给客户端。

最关键的一句话：**一条请求不是"穿过一次"推理引擎就结束的。** 它每生成一个 token，就要重新回到 Scheduler 的事件循环里，和其它请求一起被重新组 batch、重新前向。理解这一点，就理解了 SGLang 的一半。

---

## 在端到端链路中的位置

本章就是整条链路本身。八个阶段与三个进程的对应关系如下：

```
                          ┌──────────────── TokenizerManager 进程 (asyncio) ────────────────┐
  客户端                  │  ①Entry        ②Tokenization & 准入                              │
   │  HTTP POST /generate │  http_server   tokenizer_manager.generate_request                │
   ▼                      │      │              │  text ──tokenizer.encode──▶ token ids       │
 [HTTP]──────────────────▶│  generate_request──┘  封装成 TokenizedGenerateReqInput           │
   ▲                      │      ▲                          │ send_to_scheduler.send_pyobj    │
   │  SSE / JSON          │      │                          ▼ (ZMQ PUSH)                      │
   │                      └──────┼──────────────────────────┼─────────────────────────────────┘
   │                             │ (ZMQ PULL)                ▼
   │                      ┌──────┴────────────────── Scheduler 进程 (同步 while True) ─────────┐
   │                      │  ⑧ 回流              ③Scheduling   ④KV/Mem    ⑤Forward  ⑥Sampling  │
   │   BatchStrOut        │  handle_loop  ◀──┐   recv_requests get_new_   run_batch  sample()   │
   │      ▲               │                  │   get_next_     batch_     tp_worker  sampler    │
   │      │ (ZMQ PULL)    │  send_to_detok ──┼─▶ batch_to_run  prefill    model_runner          │
   │      │               │  .send_pyobj     │   Req→Schedule  alloc KV   logits     output ids │
   └──────┘               │  (BatchTokenIDOut)│  Batch                                          │
                          └──────────────────┼──────────────────────────────────────────────────┘
                                             │ (ZMQ PUSH detokenizer_ipc)
                          ┌──────────────────▼────── DetokenizerManager 进程 (同步 while True) ──┐
                          │  ⑦ Detokenize & stream                                               │
                          │  event_loop: token ids ──tokenizer.batch_decode──▶ 增量文本          │
                          │  封装 BatchStrOut ──send_to_tokenizer.send_pyobj──▶ (回 ① 的 handle_loop)│
                          └──────────────────────────────────────────────────────────────────────┘
```

三个进程通过 **ZMQ inter-process socket**（默认 IPC，TP 多卡时也可走网络）单向连接，构成一个**环**：
`TokenizerManager → Scheduler → DetokenizerManager → TokenizerManager`。

---

## 关键文件与类

| 符号 | file:line | 作用 |
|---|---|---|
| `generate_request` (HTTP route) | `python/sglang/srt/entrypoints/http_server.py:240` | `/generate` 的 FastAPI handler，分流式/非流式 |
| `v1_chat_completions` | `python/sglang/srt/openai_api/adapter.py:1430` | OpenAI 兼容的 chat 入口，最终也走 `tokenizer_manager.generate_request` |
| `TokenizerManager.generate_request` | `python/sglang/srt/managers/tokenizer_manager.py:398` | 分词进程主入口（async generator） |
| `TokenizerManager._tokenize_one_request` | `python/sglang/srt/managers/tokenizer_manager.py:434` | text → token ids，含多模态预处理与长度校验 |
| `TokenizerManager._send_one_request` | `python/sglang/srt/managers/tokenizer_manager.py:622` | 通过 ZMQ 把 `TokenizedGenerateReqInput` 推给 Scheduler |
| `TokenizerManager._wait_one_response` | `python/sglang/srt/managers/tokenizer_manager.py:632` | 等结果、检测客户端断连、`yield` 流式输出 |
| `TokenizerManager.handle_loop` | `python/sglang/srt/managers/tokenizer_manager.py:1111` | 从 Detokenizer 收回结果并唤醒对应请求 |
| `Scheduler.event_loop_normal` | `python/sglang/srt/managers/scheduler.py:636` | 普通调度主循环 |
| `Scheduler.event_loop_overlap` | `python/sglang/srt/managers/scheduler.py:656` | overlap scheduler：CPU 与 GPU 流水线重叠 |
| `Scheduler.recv_requests` | `python/sglang/srt/managers/scheduler.py:802` | 非阻塞收取新请求，TP rank0 收后广播 |
| `Scheduler.process_input_requests` | `python/sglang/srt/managers/scheduler.py:880` | 按类型分发，`TokenizedGenerateReqInput` → `handle_generate_request` |
| `Scheduler.handle_generate_request` | `python/sglang/srt/managers/scheduler.py:897` | 构造 `Req` 对象，入 `waiting_queue` |
| `Scheduler.get_next_batch_to_run` | `python/sglang/srt/managers/scheduler.py:1286` | **每轮循环的核心决策**：prefill 还是 decode |
| `Scheduler.get_new_batch_prefill` | `python/sglang/srt/managers/scheduler.py:1342` | 从 waiting_queue 攒一个 prefill batch（准入策略） |
| `Scheduler.update_running_batch` | `python/sglang/srt/managers/scheduler.py:1496` | 过滤已完成请求、OOM retract、`prepare_for_decode` |
| `Scheduler.run_batch` | `python/sglang/srt/managers/scheduler.py:1533` | 调 `tp_worker` 跑前向 + 采样，产出 `GenerationBatchResult` |
| `ScheduleBatch.prepare_for_extend` | `python/sglang/srt/managers/schedule_batch.py:1102` | prefill 前的 KV/req slot 分配与 tensor 组装 |
| `ScheduleBatch.alloc_token_slots` | `python/sglang/srt/managers/schedule_batch.py:927` | 向 KV pool allocator 申请 `out_cache_loc` |
| `TpModelWorker.forward_batch_generation` | `python/sglang/srt/managers/tp_worker.py:183` | `ModelWorkerBatch → ForwardBatch`，前向 + 采样 |
| `ModelRunner.forward` / `_forward_raw` | `python/sglang/srt/model_executor/model_runner.py:1159` | 按 forward_mode 派发到 cuda graph / extend / decode |
| `ModelRunner.sample` | `python/sglang/srt/model_executor/model_runner.py:1226` | 应用 logits bias / grammar mask，调 sampler |
| `SchedulerOutputProcessorMixin.process_batch_result_decode` | `python/sglang/srt/managers/scheduler_output_processor_mixin.py:185` | 把 next_token 写回 `Req.output_ids`、判完成 |
| `SchedulerOutputProcessorMixin.stream_output_generation` | `python/sglang/srt/managers/scheduler_output_processor_mixin.py:457` | 攒 `BatchTokenIDOut` 推给 Detokenizer |
| `DetokenizerManager.event_loop` | `python/sglang/srt/managers/detokenizer_manager.py:106` | 解码进程主循环 |
| `DetokenizerManager.handle_batch_token_id_out` | `python/sglang/srt/managers/detokenizer_manager.py:140` | 增量 detokenize：token ids → 文本片段 |
| `TokenizerManager._handle_batch_output` | `python/sglang/srt/managers/tokenizer_manager.py:1119` | 拼装 `meta_info`、累积文本、置 event |

---

## 核心数据流 / 执行流程

下面按八个阶段走。每个阶段标注「**进来**」和「**出去**」的数据形态。

### ① Entry（接入）—— `http_server.py`

**进来**：客户端的 HTTP body（JSON）。FastAPI 自动把 JSON 反序列化为 `GenerateReqInput` dataclass（注释见 `http_server.py:238`）。

`generate_request`（`http_server.py:240`）只做一件事：根据 `obj.stream` 选择两种返回方式：

- 流式（`stream=True`）：返回 `StreamingResponse`，内部 `async for out in tokenizer_manager.generate_request(...)` 把每个 chunk 包成 SSE `data: {...}\n\n`（`http_server.py:244-264`）。
- 非流式：`await ....generate_request(...).__anext__()` 只取**最后一个**聚合结果（`http_server.py:266-270`）。

OpenAI 路径 `/v1/chat/completions`（`http_server.py:609`）走的是 `v1_chat_completions`（`adapter.py:1430`），它把 messages 套上 chat template、解析 OpenAI 参数后，**同样**汇入 `tokenizer_manager.generate_request`。所以下文对 `/generate` 的描述对 chat 完全适用，区别只在入口的参数转换。

**出去**：一个 `GenerateReqInput`，交给 TokenizerManager。

### ② Tokenization 与准入 —— `tokenizer_manager.py`（进程边界 1）

`TokenizerManager.generate_request`（`tokenizer_manager.py:398`）：

1. `auto_create_handle_loop()`（:405）首次调用时懒启动后台 `handle_loop` 协程（见 :1053）。
2. `obj.normalize_batch_and_arguments()`（:413）—— 这里会**给请求分配 `rid`**（`io_struct.py:205`，`uuid.uuid4().hex`）。rid 是这条请求在整个链路里的唯一身份证。
3. 单条请求走 `_tokenize_one_request`（:424）：
   - `text → input_ids = self.tokenizer.encode(input_text)`（:460）；若客户端直接传了 `input_ids` 则跳过分词。
   - 多模态走 `mm_processor.process_mm_data_async`（:464）。
   - `_validate_token_len`（:473）—— **准入校验**：输入长度 ≥ context_len 直接 `raise ValueError`（:485-489），`input + max_new_tokens ≥ context_len` 也拒（:493-505）。这是请求**第一次可能被拒**的地方。
   - `_create_tokenized_object`（:507）构造 `SamplingParams` 并 `.verify()`，打包成 **`TokenizedGenerateReqInput`**（:547），里面只剩 token ids + 采样参数 + rid，不再有原始 text 之外的负担。
4. `_send_one_request`（:622）：`self.send_to_scheduler.send_pyobj(tokenized_obj)`（:628）经 **ZMQ PUSH** 发往 Scheduler；同时在本进程登记 `rid_to_state[rid] = ReqState(...)`（:630），这个 state 持有一个 `asyncio.Event`，后面用来"等结果"。
5. `_wait_one_response`（:632）开始 `await state.event.wait()`。

**进来**：`GenerateReqInput`（含 text）。**出去**：`TokenizedGenerateReqInput`（含 token ids + rid），通过 `send_to_scheduler`（PUSH，socket 建于 `tokenizer_manager.py:186`）跨进程发出。

> ZMQ 拓扑：TokenizerManager 的 `send_to_scheduler` 是 PUSH，对端是 Scheduler 的 `recv_from_tokenizer`（PULL，建于 `scheduler.py:227`）。

### ③ Scheduling（调度）—— `scheduler.py`（Scheduler 进程）

Scheduler 是一个**同步的 `while True` 事件循环**（不是 asyncio），跑在自己的进程里。根据配置选三种循环之一（`scheduler.py:2304-2310`）：`event_loop_pp`（PP）/ `event_loop_overlap`（默认开启，`enable_overlap = not disable_overlap_schedule`，:202）/ `event_loop_normal`。

以 `event_loop_normal`（:636）为骨架，一轮循环做四件事：

```
while True:
    recv_reqs = self.recv_requests()            # 收新请求
    self.process_input_requests(recv_reqs)      # Req 入队
    batch = self.get_next_batch_to_run()        # 决定这一轮跑什么
    if batch:
        result = self.run_batch(batch)          # 前向+采样
        self.process_batch_result(batch, result)# 写回+判完成+回流
```

- `recv_requests`（:802）：用 `recv_pyobj(zmq.NOBLOCK)` 把队列里能拿的全拿出来（:808-813）。TP 多卡时只有 `attn_tp_rank == 0` 真收，再 `broadcast_pyobj` 广播给同 TP group 的其它 rank（:871-877）—— 保证所有 TP rank 看到**同一份**请求流。
- `process_input_requests`（:880）：对每个 recv_req 调 `_request_dispatcher`（`TypeBasedDispatcher`，注册于 :433）。`TokenizedGenerateReqInput` → `handle_generate_request`（:897）。
- `handle_generate_request`（:897）：用 token ids 构造一个 **`Req` 对象**（`schedule_batch.py:420` 的 `class Req`，构造调用在 `scheduler.py:917`），挂上 tokenizer、eos id 等，然后 `_add_request_to_queue` 进 **`waiting_queue`**。此时请求"排队中"。

**核心决策在 `get_next_batch_to_run`（:1286）**：

```
get_next_batch_to_run():
  # 1. 把上一轮的 prefill batch 合并进 running_batch（continuous batching 关键）
  if last_batch was extend:
      last_batch.filter_batch()          # 摘掉已完成的请求
      running_batch.merge_batch(last_batch)   # :1316
  # 2. 优先尝试组一个新的 prefill batch
  new_batch = get_new_batch_prefill()    # :1318
  if new_batch: return new_batch         # 有新请求要 prefill，先 prefill
  # 3. 否则做 decode
  else:
      running_batch = update_running_batch(running_batch)  # :1325
      return running_batch
```

这就是 **prefill 优先、decode 兜底** 的混合调度：只要 waiting_queue 里有能塞进去的新请求就先 prefill；没有就让正在生成的请求集体 decode 一步。

`get_new_batch_prefill`（:1342）是**第二次准入**发生的地方：

- 若 `running_batch.batch_is_full` 或 waiting_queue 空，直接返回 None（:1348-1351）。
- 用 `PrefillAdder`（:1372）按 `max_prefill_tokens` / `chunked_prefill_size` / KV 余量逐个 `add_one_req`（:1419）。塞不下就 `batch_is_full = True` 并 break。
- 通过的请求集合 `can_run_list`（:1435）从 waiting_queue 移除（:1444），用 `ScheduleBatch.init_new`（:1463）建 batch，并 `prepare_for_extend()`（:1474）。

**进来**：`TokenizedGenerateReqInput`。**出去**：一个 `ScheduleBatch`（`schedule_batch.py:787`），forward_mode 为 extend（prefill）或 decode。

> 深入：准入/驱逐/优先级策略见 [调度器](12-scheduler.md)。

### ④ KV 缓存与内存分配 —— `schedule_batch.py`

`ScheduleBatch.prepare_for_extend`（`schedule_batch.py:1102`）在 prefill 前完成两类资源分配：

1. **req slot**：`alloc_req_slots` → `req_to_token_pool.alloc(num_reqs)`（:916-917），拿到每个请求在 req-to-token 表里的 `req_pool_idx`。
2. **KV token slot**：根据 page_size 走 `alloc_token_slots(extend_num_tokens)`（:1213）或 paged 版 `alloc_paged_token_slots_extend`（:1220），最终调 `token_to_kv_pool_allocator.alloc(...)`（:935）拿到 **`out_cache_loc`** —— 即这批 token 的 KV 在显存池里的物理槽位。

decode 时则是 `prepare_for_decode`（由 `update_running_batch` 在 :1530 调用），每个请求只需为**1 个新 token** 申请一个 KV 槽（`alloc_paged_token_slots_decode`，:994）。

若分配失败（`out_cache_loc is None`，:936），decode 路径会在 `update_running_batch` 里触发 **retract**（:1506-1519）：把部分请求踢回 waiting_queue 释放显存，并调高 `new_token_ratio` 让后续更保守。这是 OOM 自愈机制。

RadixAttention 的前缀复用（`init_next_round_input` 算 `prefix_indices`，`schedule_batch.py:624`）让相同前缀的请求共享 KV，从而 prefill 只需计算新增部分。

**进来**：`ScheduleBatch`（逻辑）。**出去**：填好 `input_ids / req_pool_indices / seq_lens / out_cache_loc` 的 `ScheduleBatch`（已绑物理显存）。

> 深入：KV pool、page、RadixAttention 见 [KV 缓存与 RadixAttention](13-memcache-radixattention.md)。

### ⑤ Forward（前向）—— `tp_worker.py` → `model_runner.py`

`run_batch`（`scheduler.py:1533`）：

1. `model_worker_batch = batch.get_model_worker_batch()`（:1553）—— 把 `ScheduleBatch` 压成轻量的 **`ModelWorkerBatch`**（`schedule_batch.py:1705`），只含跑模型必需的张量。
2. `tp_worker.forward_batch_generation(model_worker_batch)`（:1556）。

`TpModelWorker.forward_batch_generation`（`tp_worker.py:183`）：

- `ForwardBatch.init_new(model_worker_batch, model_runner)`（:191）—— 转成 **`ForwardBatch`**（`forward_batch_info.py:138`），这是模型真正看到的输入。
- `model_runner.forward(forward_batch)`（:202）→ `_forward_raw`（`model_runner.py:1180`）按 forward_mode 派发：
  - 能用 cuda graph 就 `cuda_graph_runner.replay`（:1192）；
  - 否则 `forward_decode`（:1111）/ `forward_extend`（:1123）/ `forward_idle`（:1146），各自调 `self.model.forward(input_ids, positions, forward_batch)`。
- 返回 **`LogitsProcessorOutput`**（含 `next_token_logits`）。

**进来**：`ModelWorkerBatch`。**出去**：`LogitsProcessorOutput`（logits 张量，还在 GPU 上）。

> 深入：attention backend、cuda graph 见 [ModelRunner 与前向](14-model-executor.md)。

### ⑥ Sampling（采样）—— `model_runner.py`

仍在 `forward_batch_generation` 内（`tp_worker.py:211`）：`next_token_ids = model_runner.sample(logits_output, model_worker_batch)`。

`ModelRunner.sample`（`model_runner.py:1226`）：

1. `_preprocess_logits`（:1247 → :1212）：应用 grammar/regex vocab mask 与 logit bias。注意 :1216-1223 的分支——overlap 模式下 mask 在**上一轮** `process_batch_result` 里就算好了，这里只 `wait`；normal 模式下当场算（CPU 工作与 GPU 前向重叠）。
2. `self.sampler(...)`（:1250）按温度 / top-p / top-k 采样，返回 **`next_token_ids`**（GPU 张量）。

回到 `run_batch`，结果打包成 **`GenerationBatchResult`**（`scheduler.py:1592`，dataclass 定义在 :160），含 `logits_output / next_token_ids / bid` 等。

**进来**：logits。**出去**：`next_token_ids`（每个请求一个新 token id）。

> 深入：采样实现见 [采样器](17-sampling-structured-output.md)。

### ⑦ Detokenize & stream（解码回流）—— output processor → DetokenizerManager（进程边界 2）

`process_batch_result`（`scheduler.py:1613`）按 forward_mode 分流到 `process_batch_result_decode`（mixin :185）或 `_prefill`。以 decode 为例（`scheduler_output_processor_mixin.py:185`）：

1. overlap 模式下先 `tp_worker.resolve_last_batch_result`（:200）把 GPU 结果同步回 CPU；`next_token_ids.tolist()`（:205）。
2. 逐请求 `req.output_ids.append(next_token_id)`（:234），然后 `req.check_finished()`（:236）。
3. **完成判定** `Req.check_finished`（`schedule_batch.py:683`）：达到 `max_new_tokens`（:693）、命中 eos/stop token（:704-722）、命中 stop string（:724-733）任一即 `finished`。完成的请求 `tree_cache.cache_finished_req(req)`（:238）把它的 KV 写回 radix tree 供复用。
4. `stream_output(batch.reqs, ...)`（:270）→ `stream_output_generation`（mixin :457）：对每个该输出的请求（流式按 `stream_interval`，:524；非流式按 `DEFAULT_FORCE_STREAM_INTERVAL`，:527）收集 `decode_ids / read_offset / output_ids / 各种 logprobs`，**只发增量**（用 `send_token_offset` 等偏移，:532/:552），打包成 **`BatchTokenIDOut`**（:651）并 `send_to_detokenizer.send_pyobj(...)`（:650）经 **ZMQ PUSH** 发往 Detokenizer。

`DetokenizerManager.event_loop`（`detokenizer_manager.py:106`）：

- `recv_from_scheduler.recv_pyobj()`（:109）收 `BatchTokenIDOut`。
- `handle_batch_token_id_out`（:140）做**增量 detokenize**：维护每个 rid 的 `DecodeStatus`（含 `surr_offset/read_offset/sent_offset`），用 `tokenizer.batch_decode` 解 `surr_ids` 和 `read_ids`，`new_text = read_texts[i][len(surr_texts[i]):]`（:194）。`surr`（surrounding）前缀的存在是为了正确处理跨 token 的多字节字符（避免把 UTF-8 半个字符 `�` 当成完整输出，:197）。
- 拼成 **`BatchStrOut`**（:215，含 `output_strs` 增量文本）并 `send_to_tokenizer.send_pyobj`（:111）发回 TokenizerManager。

**进来**：`next_token_ids`（写进 `Req`）。**出去**：`BatchTokenIDOut`（token ids 增量）→ Detokenizer → `BatchStrOut`（文本增量）。

> 深入：增量解码与多字节字符见 [Detokenizer](18-detokenizer-output.md)。

### ⑧ 响应返回 —— 回到 `tokenizer_manager.py`，HTTP 流式返回

`TokenizerManager.handle_loop`（`tokenizer_manager.py:1111`）后台协程 `recv_from_detokenizer.recv_pyobj()`（:1115）收 `BatchStrOut`，交 `_result_dispatcher`（:312）→ `_handle_batch_output`（:1119）：

- 按 rid 找回 `state = rid_to_state[rid]`（:1126）。
- 累积文本 `state.text += recv_obj.output_strs[i]`（:1164），拼 `meta_info`（id/finish_reason/prompt_tokens/completion_tokens，:1134-1158），组成 `out_dict = {"text": state.text, "meta_info": ...}`。
- （在 `_handle_batch_output` 后半部分）把 out_dict 追加到 `state.out_list` 并 `state.event.set()` —— **唤醒**正阻塞在 `_wait_one_response`（:632）的那个协程。

`_wait_one_response`（:632）被唤醒后：取 `state.out_list[-1]`（:653），流式则当场 `yield out`（:680），完成则 `yield out; break`（:674-675）。这个 `yield` 一路冒泡回 `http_server.py:246` 的 `async for`，再包成 SSE 发给客户端。

`_wait_one_response` 还负责**断连检测**：等待超时 4s（:642）后若 `request.is_disconnected()`，调 `abort_request` 并抛错终止整条调用链（:644-650 / :682-688）。

**进来**：`BatchStrOut`。**出去**：HTTP SSE chunk / 最终 JSON。

---

## 连续批处理（continuous batching）：一条请求要进调度循环很多次

这是全章最重要的概念。把上面阶段 ③–⑦ 串起来看一条请求的真实轨迹（假设生成 N 个 token）：

```
轮次 t0:  recv → 入 waiting_queue
轮次 t1:  get_new_batch_prefill 选中 → prepare_for_extend(分KV) → run_batch(prefill, 算完整 prompt)
          → sample 出第 1 个 token → stream_output → Detok → 客户端收到第 1 段文本
轮次 t2:  本请求随 last_batch 被 merge 进 running_batch
          → update_running_batch(prepare_for_decode, 分 1 个 KV) → run_batch(decode) → 第 2 个 token
轮次 t3:  ... 第 3 个 token ...
   ...
轮次 t(N): decode → 第 N 个 token → check_finished()=True
          → cache_finished_req(释放/写回 KV) → stream_output(finished) → 客户端收到结尾 [DONE]
```

也就是说：

- **prefill 只发生一次**（或在 chunked prefill 下分几块），把整段 prompt 一次算进 KV。
- **decode 每个 token 都要重新走一遍 `get_next_batch_to_run → run_batch → process_batch_result`**。每一轮，调度器都会重新评估：有没有新请求该 prefill？谁完成了该踢出 batch？显存够不够、要不要 retract？

正因为「每个 token 都重新组 batch」，新到的请求**不必等当前所有请求生成完**就能插进来一起算——这就是 continuous batching 相对静态 batching 的核心优势：**高吞吐 + 低排队延迟**。`get_next_batch_to_run` 里 prefill 与 decode 的交织（:1318-1328）和 `merge_batch`（:1316）/ `filter_batch`（:1304）就是它的实现机制。

### overlap scheduler 又快在哪？

`event_loop_overlap`（`scheduler.py:656`）让**第 t 轮的 GPU 前向**与**第 t-1 轮结果的 CPU 后处理**重叠：本轮 `run_batch` 后把 `(batch, result)` 压进 `result_queue`（:670），然后处理的是**上一轮**弹出的结果（:684-691）。GPU 不用等 CPU 把 token detokenize、判完成就能开下一轮前向。代价是实现复杂度（多出"延迟一个 token"的边界处理，例如 `process_batch_result_decode` :218-230 对 overlap 下额外延迟 token 的 KV 释放）。

> 深入：overlap 的正确性细节见 [调度器](12-scheduler.md)。

---

## 设计难点与权衡

1. **三进程而非单进程**。分词（纯 CPU、可能慢的 Python tokenizer）、调度+推理（GPU、对延迟极敏感）、解码（CPU）被拆进三个 OS 进程，用 ZMQ 连接。好处：tokenizer/detokenizer 的 Python GIL 与慢操作不会阻塞 GPU 事件循环；Scheduler 的同步 `while True` 可以贴着 GPU 跑、不被 asyncio 调度抖动干扰。代价：跨进程序列化（`send_pyobj` 用 pickle）开销，以及调试时要在三个进程间追踪同一个 rid。

2. **Scheduler 是同步循环，TokenizerManager 是 asyncio**。两种并发模型的边界恰好落在 ZMQ。TokenizerManager 用 `asyncio.Event` 把"HTTP 协程"与"后台 handle_loop 协程"解耦（`_wait_one_response` 等 event，`handle_loop` 置 event）。

3. **增量传输无处不在**。从 Scheduler 的 `send_token_offset`（只发新 token，`mixin:532`）到 Detokenizer 的 `sent_offset`（只发新文本，`detokenizer_manager.py:211`），整条链路都在传"增量"。这避免了长输出时反复传整段，但也意味着任何一处 offset 算错都会导致重复/丢字。

4. **rid 是唯一的全局关联键**。它在 TokenizerManager 生成（`io_struct.py:205`），在三个进程间随 io_struct 流转，TokenizerManager 用 `rid_to_state`、Detokenizer 用 `decode_status`、Scheduler 用 `Req.rid` 各自维护以 rid 为键的状态。Detokenizer 的状态表是 **LRU 容量受限**的（`DETOKENIZER_MAX_STATES`，:53），并发请求过多时老状态会被驱逐，触发 `detokenizer_manager.py:186` 的 `RuntimeError`。

5. **两次准入 + 一次回退**。第一次在 TokenizerManager（长度硬校验，:478），第二次在 Scheduler 的 `PrefillAdder`（按显存/token 预算，:1342），运行中显存不够还会 retract 回退（:1506）。三道关卡共同保证不 OOM。

---

## 改代码 / 修 bug 指南

**想改什么 → 看哪里：**

- 改 **HTTP 路由 / 入参**：`http_server.py:240`（generate）、`adapter.py:1430`（chat）。
- 改 **分词 / 准入长度校验**：`tokenizer_manager.py:434`（tokenize）、:478（`_validate_token_len`）。
- 改 **rid 生成 / batch 拆分**：`io_struct.py:112` 的 `normalize_batch_and_arguments`、:205。
- 改 **调度策略（谁先跑、batch 多大）**：`scheduler.py:1286`（`get_next_batch_to_run`）、:1342（`get_new_batch_prefill`）、`PrefillAdder`。
- 改 **KV 分配 / OOM 行为**：`schedule_batch.py:1102`（extend）、:994（decode）、`scheduler.py:1506`（retract）。
- 改 **采样 / grammar mask**：`model_runner.py:1226`（sample）、:1212（preprocess）。
- 改 **完成判定（stop 条件）**：`schedule_batch.py:683`（`check_finished`）。
- 改 **流式粒度 / 增量内容**：`mixin:457`（`stream_output_generation`，`stream_interval`）、`detokenizer_manager.py:140`。
- 改 **最终响应格式 / meta_info**：`tokenizer_manager.py:1119`（`_handle_batch_output`）。

**常见 bug 高发区：**

- **流式文本重复或缺字**：检查 offset 三件套——`send_token_offset`（`mixin:532`）、`DecodeStatus.surr_offset/read_offset/sent_offset`（`detokenizer_manager.py:194-212`）。多字节字符问题通常是 `surr` 前缀逻辑（:197）。
- **请求卡住不返回**：多半是 rid 关联断了。看 TokenizerManager 是否收到了对应 rid 的输出（`_handle_batch_output:1126` 的 "state was deleted" error），或 `state.event` 没被 set。
- **`Decode status not found` RuntimeError**：并发太高、`DETOKENIZER_MAX_STATES`（:53）被打爆，调大环境变量 `SGLANG_DETOKENIZER_MAX_STATES`。
- **偶发 OOM / 吞吐抖动**：看 `update_running_batch` 的 retract 日志（`scheduler.py:1514`）与 `new_token_ratio` 变化。
- **overlap 模式下的 off-by-one**：涉及"额外延迟 token"，集中在 `process_batch_result_decode:218-230` 和 `event_loop_overlap:672-691`。怀疑时可加 `--disable-overlap-schedule` 切回 `event_loop_normal` 二分定位。

**调试入手点：**

1. 用 rid 串起三个进程的日志：开 `--log-requests` 让 TokenizerManager 打印 `Receive/Finish`（:415, :657）。
2. 想看调度行为：关注 `log_prefill_stats`（`scheduler.py:1460`）和 `log_decode_stats`（`mixin:278`）。
3. 想隔离问题进程：`--disable-overlap-schedule` 简化 Scheduler；`skip_tokenizer_init` 绕开分词；单卡（`tp_size=1`）去掉 broadcast 路径（`scheduler.py:871`）。
4. 断点优先放在四个枢纽：`get_next_batch_to_run`（:1286）、`run_batch`（:1533）、`process_batch_result_decode`（mixin:185）、`_handle_batch_output`（:1119）。

---

## 深入阅读

- 各阶段细节：[Tokenizer Manager](11-tokenizer-manager.md) · [调度器](12-scheduler.md) · [KV 缓存与 RadixAttention](13-memcache-radixattention.md) · [ModelRunner 与前向](14-model-executor.md) · [采样器](17-sampling-structured-output.md) · [Detokenizer](18-detokenizer-output.md)
- 概念与缩写：[术语表](02-glossary.md)（rid / prefill / decode / TP / overlap scheduler / continuous batching / RadixAttention）
- 进程与启动：[进程模型与启动流程](00-overview.md)
