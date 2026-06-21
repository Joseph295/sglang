# TokenizerManager：分词、准入与请求编排

> 本章对应源码：
> - `python/sglang/srt/managers/tokenizer_manager.py`
> - `python/sglang/srt/managers/io_struct.py`
> - `python/sglang/srt/managers/multimodal_processor.py`
> - `python/sglang/srt/conversation.py`
> - `python/sglang/srt/hf_transformers_utils.py`

## 1. 一句话职责

`TokenizerManager` 跑在 HTTP server 进程内、基于 `asyncio` 单线程事件循环，负责把每条用户请求**文本化的输入**（text / messages / image / audio）变成 **token ids + 多模态张量**，分配 `rid`，封装成 `TokenizedGenerateReqInput`，用 ZMQ 单向推给 `Scheduler`；同时为每条在飞请求维护一份异步状态，把 Detokenizer 回流的增量结果通过 `asyncio.Event` 唤醒并 `yield` 回 HTTP handler。它还兼任**准入校验**（context length、模型类型）、**abort / 断连处理**、**health check**、**权重热更新 / profile / 内部状态**等控制面 RPC 的客户端。

一句话：它是「HTTP 异步世界」与「Scheduler 同步批处理世界」之间的**翻译官 + 调度编排器**。

## 2. 在端到端链路中的位置

```
                    HTTP server 进程 (asyncio, 单进程)              Scheduler 进程(可多 TP/DP)
 ┌────────┐       ┌──────────────────────────────────────┐        ┌──────────────────┐
 │ client │──────▶│ FastAPI endpoint (http_server.py)      │        │                  │
 └────────┘  HTTP │   │  generate_chat_conv / 套 template   │        │                  │
     ▲            │   ▼  (openai_api/adapter.py)            │        │                  │
     │            │ ┌─────────────────────────────────────┐│        │                  │
     │            │ │        TokenizerManager             ││        │                  │
     │  SSE/JSON  │ │  ① tokenize + 多模态预处理            ││        │                  │
     │◀───────────┤ │  ② 生成 rid / 校验 context_len       ││  ZMQ   │                  │
     │            │ │  ③ TokenizedGenerateReqInput ───────────PUSH──▶│  调度/前向/采样   │
     │            │ │  ④ rid_to_state[rid] = ReqState     ││        │                  │
     │            │ │  ⑥ await event ← handle_loop        ││        │                  │
     │            │ └───────────────▲─────────────────────┘│        └────────┬─────────┘
     │            └─────────────────│──────────────────────┘                 │
     │                              │ ZMQ PULL                                │
     │                       ┌──────┴────────────┐    BatchTokenIDOut         │
     └───────────────────────│ DetokenizerManager│◀───────────────────────────┘
            BatchStrOut       └───────────────────┘   (token ids → text)
```

本章聚焦 **「分词 → 准入 → 发送 → 回流编排」** 这一段。它的左边是 [HTTP / OpenAI 入口](10-entrypoints-api.md)（套 chat template、构造 `GenerateReqInput`），右边是 [调度器](12-scheduler.md)，回流路径上的解码由 [DetokenizerManager](18-detokenizer-output.md) 完成。术语见 [术语表](02-glossary.md)。

> 注意进程拓扑：`TokenizerManager` 与 `DetokenizerManager`、`Scheduler` 是**三个独立进程**（见 `python/sglang/srt/entrypoints/engine.py:639` 起的 `_launch_subprocesses`）。`TokenizerManager` 与 FastAPI app 同进程，用 `zmq.asyncio` 异步 socket；它**只 PUSH 给 Scheduler、只 PULL 从 Detokenizer**，从不直接和 Scheduler 双向通信（控制面 RPC 的「回包」也是经由 Detokenizer 转发回来的）。

## 3. 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `TokenizerManager` | `python/sglang/srt/managers/tokenizer_manager.py:162` | 核心类：构造 socket / tokenizer / processor，编排请求生命周期 |
| `TokenizerManager.__init__` | `tokenizer_manager.py:165` | 建 ZMQ socket、加载 tokenizer/processor、注册 `_result_dispatcher` |
| `ReqState` | `tokenizer_manager.py:127` | **每条在飞请求的状态**：`out_list` / `finished` / `event` / 累积的 text、output_ids、logprobs |
| `generate_request` | `tokenizer_manager.py:398` | 对外入口（async generator）：单条 / batch 分流 |
| `_tokenize_one_request` | `tokenizer_manager.py:434` | 真正分词 + 多模态预处理 + 校验 |
| `_validate_token_len` | `tokenizer_manager.py:478` | 准入：input + max_new_tokens 不得超 `context_len` |
| `_create_tokenized_object` | `tokenizer_manager.py:507` | 组装 `TokenizedGenerateReqInput` / `TokenizedEmbeddingReqInput` |
| `_send_one_request` | `tokenizer_manager.py:622` | `send_pyobj` 给 Scheduler，并登记 `rid_to_state` |
| `_wait_one_response` | `tokenizer_manager.py:632` | per-request async generator：等 `event`、检测断连、yield 增量 |
| `_handle_batch_request` | `tokenizer_manager.py:690` | batch / parallel sampling 的并发编排 |
| `handle_loop` | `tokenizer_manager.py:1111` | 后台 task：从 Detokenizer 收包并分派 |
| `_handle_batch_output` | `tokenizer_manager.py:1119` | 把回流结果写进 `ReqState`、`event.set()` |
| `_result_dispatcher` | `tokenizer_manager.py:312` | `TypeBasedDispatcher`：按回包类型路由 handler |
| `abort_request` | `tokenizer_manager.py:783` | 给 Scheduler 发 `AbortReq` |
| `auto_create_handle_loop` | `tokenizer_manager.py:1053` | 懒启动 `handle_loop` + 信号处理 + watchdog |
| `_Communicator` | `tokenizer_manager.py:1454` | 控制面 RPC 的「请求-应答」原语（带 fan-out / 串行化） |
| `GenerateReqInput` | `io_struct.py:50` | **原始请求**数据类 + `normalize_batch_and_arguments` |
| `TokenizedGenerateReqInput` | `io_struct.py:424` | 发往 Scheduler 的**已分词请求** |
| `BatchTokenIDOut` / `BatchStrOut` | `io_struct.py:578` / `:631` | 从 Detokenizer 回流的批量输出 |
| `get_tokenizer` / `get_processor` | `hf_transformers_utils.py:192` / `:270` | 加载 HF tokenizer / multimodal processor |
| `get_context_length` | `hf_transformers_utils.py:168` | 从 HF config 推断 `context_len` |
| `get_mm_processor` / `import_processors` | `multimodal_processor.py:55` / `:31` | 按模型架构选多模态 processor |
| `generate_chat_conv` | `conversation.py:515` | 把 OpenAI `messages` 套成单条 prompt（在 entrypoint 层调用） |

## 4. 核心数据流 / 执行流程

### 4.1 一条单请求的完整旅程

```
HTTP POST /generate  (GenerateReqInput, 可能只有 text)
      │
      ▼
generate_request()                                    tokenizer_manager.py:398
  ├─ auto_create_handle_loop()   懒启动后台收包 task
  ├─ obj.normalize_batch_and_arguments()  规整 batch/默认值/rid   io_struct.py:112
  ├─ async with model_update_lock.reader_lock:   (与权重热更新互斥)
  │     is_single? → True
  │     ├─ tokenized_obj = await _tokenize_one_request(obj)
  │     │      ├─ input_ids = tokenizer.encode(text)        # 或直接用 obj.input_ids
  │     │      ├─ if contains_mm_input():
  │     │      │     image_inputs = await mm_processor.process_mm_data_async(...)
  │     │      │     input_ids = image_inputs["input_ids"]  # 插入 image placeholder token
  │     │      ├─ _validate_token_len(obj, input_ids)       # 准入校验
  │     │      └─ _create_tokenized_object(...) → TokenizedGenerateReqInput
  │     ├─ _send_one_request(obj, tokenized_obj, created_time)
  │     │      ├─ send_to_scheduler.send_pyobj(tokenized_obj)   # ZMQ PUSH
  │     │      └─ rid_to_state[rid] = ReqState([], False, Event(), obj, ...)
  │     └─ async for response in _wait_one_response(obj, request):
  │              yield response          # ← 增量/最终结果回给 FastAPI
  ▼
```

输入进来时是 `GenerateReqInput`（可能只有 `text`），出去给 Scheduler 时是 `TokenizedGenerateReqInput`（一定有 `input_ids`、`mm_inputs`、`SamplingParams`），回给 client 的是 `dict`：`{"text": ..., "meta_info": {...}}` 或 `{"output_ids": [...], "meta_info": {...}}`。

### 4.2 回流半边：handle_loop 如何唤醒在飞请求

`handle_loop`（`tokenizer_manager.py:1111`）是一个常驻 async task，死循环地从 Detokenizer 拉包：

```
handle_loop:                                  tokenizer_manager.py:1111
  while True:
    recv_obj = await recv_from_detokenizer.recv_pyobj()
    _result_dispatcher(recv_obj)              # 按类型分派
    last_receive_tstamp = time.time()         # health check 用

_result_dispatcher 命中 BatchStrOut/BatchTokenIDOut →
  _handle_batch_output(recv_obj):             tokenizer_manager.py:1119
    for i, rid in enumerate(recv_obj.rids):
      state = rid_to_state.get(rid)           # 找回对应请求状态
      meta_info = {id, finish_reason, prompt_tokens, ...}
      state.text += recv_obj.output_strs[i]   # 增量累积
      state.finished = recv_obj.finished_reasons[i] is not None
      if state.finished:
          del rid_to_state[rid]               # 终态清理
      state.out_list.append(out_dict)
      state.event.set()                       # ★ 唤醒在 _wait_one_response 里 await 的协程
```

而每条请求的协程在 `_wait_one_response`（`tokenizer_manager.py:632`）里以 `await state.event.wait()` 阻塞，被唤醒后取出 `out_list[-1]`、清空、按 `stream` 决定是否 `yield`：

```
_wait_one_response:                            tokenizer_manager.py:632
  state = rid_to_state[obj.rid]
  while True:
    try:
      await asyncio.wait_for(state.event.wait(), timeout=4)   # 4s 超时探活
    except TimeoutError:
      if request and await request.is_disconnected():
          abort_request(rid); raise ValueError(...)           # 断连 → 主动 abort
      continue
    out = state.out_list[-1]; state.out_list = []
    if state.finished:
        yield out; break
    state.event.clear()
    if obj.stream:
        yield out
    else:
        if request and await request.is_disconnected():
            abort_request(rid); raise ValueError(...)
```

这就是「**生产者（handle_loop 一个 task）— 消费者（每请求一个协程）**」模型：单个后台 task 负责把所有 rid 的回包写进各自的 `ReqState` 并 `event.set()`，N 个请求协程各自 `await` 自己的 `event`。这就是「单进程异步事件循环 + 多请求并发」的核心骨架。

### 4.3 分词的三条输入路径

`_tokenize_one_request`（`tokenizer_manager.py:434`）按优先级处理三种输入：

1. `obj.input_embeds is not None`：直接用 embeds，要求 `--disable-radix-cache`（否则报错，`:443`）。
2. `obj.input_ids is not None`：客户端已自带 token ids，跳过 tokenizer。
3. 否则 `input_ids = self.tokenizer.encode(input_text)`；若 `tokenizer is None`（`--skip-tokenizer-init`）则报错（`:454`）。

多模态：当 `obj.contains_mm_input()`（有 image/audio，`io_struct.py:109`）为真，调 `await self.mm_processor.process_mm_data_async(...)`（`:464`）。注意它返回的 dict 里会带**已经插好 placeholder token 的新 `input_ids`**，覆盖原值（`:470-471`）。非多模态模型用 `DummyMultimodalProcessor`（`multimodal_processor.py:18`），其 `process_mm_data_async` 直接返回 `None`。

### 4.4 chat template 在哪里套？（不是这里！）

容易踩的一个认知坑：**`TokenizerManager` 本身不套 chat template**。它收到的 `GenerateReqInput` 里 `text` 已经是渲染好的 prompt。套 template 发生在更上游的 OpenAI adapter：`generate_chat_conv(request, template_name)`（`conversation.py:515`）把 `messages` 渲染成单个字符串 prompt，再构造 `GenerateReqInput` 交给 `generate_request`。`conversation.py` 里有一大套 `register_conv_template`（llama2/llama3/llama4/chatml/gemma3/deepseek 等，从 `:625` 起），以及多模态 placeholder 拼接逻辑 `_get_full_multimodal_text_prompt`（`:497`）。原生 `/generate` 端点则根本不经过 template（直接 text/input_ids）。

> 修 bug 时分清两层：`messages → prompt` 的渲染问题去看 `conversation.py` / `openai_api/adapter.py`；`prompt → token ids` 的问题才在 `TokenizerManager`。

## 5. 设计难点与权衡

### 5.1 为什么分词单独占一个进程？

把 tokenize / detokenize 从 Scheduler 进程剥离，是为了**不让 CPU 密集的字符串处理阻塞 GPU 前向**。Scheduler 进程要尽量贴着 GPU 跑 prefill/decode；tokenizer 用纯 CPU、还可能因 HF tokenizer 的 Rust 实现释放 GIL。三进程经 ZMQ 解耦后，分词与前向天然 pipeline 并行。代价是请求要序列化两跳（PUSH 给 Scheduler、PULL 自 Detokenizer），以及 `rid` 必须全局唯一以便回流时找回状态。

### 5.2 `rid_to_state` + `asyncio.Event`：单线程下的"伪并发"

整个 `TokenizerManager` 是**单线程 asyncio**（`uvloop`，`tokenizer_manager.py:122`）。成百上千并发请求靠协程切换复用一个线程：每条请求在 `_wait_one_response` 里 `await` 自己 `ReqState.event`，让出控制权；`handle_loop` 这唯一的"生产者" task 收到回包后 `event.set()` 把对应协程唤醒。

关键非显然点：
- `_handle_batch_output` 是**同步函数**（不是 coroutine），它在 `handle_loop` 里被直接调用，遍历整个 batch 的所有 rid 一次性更新完。所以它**绝不能 await / 阻塞**，否则会卡住所有请求的回流。
- 终态请求在 `_handle_batch_output` 里 `del rid_to_state[rid]`（`:1206`）。如果回包到达时 state 已被删（例如已 abort），会打 error log 然后 `continue`（`:1127`），而不是崩溃。
- `state.out_list` 用 list 而非单值：两次回包之间消费者可能还没被调度，需要缓冲；非 stream 模式 `_wait_one_response` 只取 `out_list[-1]`（最终态即可），stream 模式则逐个 yield。

### 5.3 4 秒超时轮询 vs 纯事件等待：为了"断连可中止"

`_wait_one_response` 没有直接 `await event.wait()`，而是套了 `asyncio.wait_for(..., timeout=4)`（`:642`）。原因：HTTP client 断连时，asyncio 不会自动取消挂起的 `event.wait()`。所以用 4s 轮询周期性检查 `request.is_disconnected()`，一旦断连就 `abort_request(rid)` 并抛 `ValueError` 杀掉整条调用栈（`:644-650`、`:682-688`）。

文件末尾的注释表（`tokenizer_manager.py:1492`）系统地列出 abort 的四象限矩阵（streaming × {waiting,running}），是理解中止逻辑的"地图"：

```
| entrypoint | streaming | status        | abort engine    | cancel task      | rid_to_state             |
| http       | yes       | waiting queue | background task | fastapi          | del in _handle_abort_req |
| http       | yes       | running       | background task | fastapi          | del in _handle_batch_output |
| http       | no        | waiting queue | type 1          | type 1 exception | del in _handle_abort_req |
| http       | no        | running       | type 3          | type 3 exception | del in _handle_batch_output |
```

- streaming 请求的中止靠 `create_abort_task`（`:1039`）注册的 FastAPI `BackgroundTasks`：流结束/断开后 sleep 2s 再 `abort_request`。
- 非 streaming 请求靠上面 `_wait_one_response` 里的 "type 1 / type 3" 主动探测断连。

### 5.4 batch 与 parallel sampling 的编排

`_handle_batch_request`（`tokenizer_manager.py:690`）处理非单条请求，分三种情况：
- `parallel_sample_num == 1` + `--enable-tokenizer-batch-encode`：`_batch_tokenize_and_process`（`:578`）把整批一次性丢给 `self.tokenizer(texts)`，比逐条 encode 快；但有约束（不能带 image / input_ids / input_embeds，见 `_validate_batch_tokenization_constraints` `:604`）。
- `parallel_sample_num == 1` 普通路径：逐条 `await _tokenize_one_request` 后 `_send_one_request`。
- `parallel_sample_num > 1`：先发一份 `max_new_tokens=0` 的 prefix-only 请求把公共前缀写进 radix cache（`:736-744`），再为每个样本 `regenerate_rid()` 复制发送（`:747-754`）。这是利用 [RadixAttention](13-memcache-radixattention.md) 前缀复用省 prefill。

stream 模式收集 N 个 generator 的方式很讲究（`:762-778`）：用 `asyncio.wait(..., FIRST_COMPLETED)` 抢占式地谁先来谁先 yield，并给每个结果回填 `index`，保证多路结果能区分。

### 5.5 控制面 RPC：`_Communicator` 的串行化请求-应答

数据面是 fire-and-forget（PUSH 完登记 state，回流异步唤醒）。但 `flush_cache` / `update_weights_*` / `profile` / `get_internal_state` 这类**控制面操作需要"等应答"**。`TokenizerManager` 用 `_Communicator`（`tokenizer_manager.py:1454`）封装：调用 `await communicator(obj)` 会 `send_pyobj` 后挂在一个 `asyncio.Event` 上，直到回包数攒够 `fan_out`（通常 = `dp_size`，每个 DP rank 回一份）才 set。注释明确：**同一时刻只允许 1 个 in-flight**（`:1455`），后来者排进 `_ready_queue` 等前一个完成。回包通过 `_result_dispatcher` 路由到对应 communicator 的 `handle_recv`（`:1486`）。

权重热更新还额外用 `RWLock`（`model_update_lock`）：普通请求拿 reader_lock（`:421`），权重更新拿 writer_lock（`:847`），保证**更新权重时不能有请求在飞**。

### 5.6 准入：context length 校验

`_validate_token_len`（`tokenizer_manager.py:478`）做两层检查：输入长度 `>= context_len` 直接拒；`input + max_new_tokens >= context_len` 也拒。`context_len` 来自 `ModelConfig`，最终源头是 `get_context_length`（`hf_transformers_utils.py:168`），它按 `CONTEXT_LENGTH_KEYS` 优先级从 HF config 取，并考虑 rope_scaling factor（`:172`）。

### 5.7 graceful shutdown 与 health check

- `auto_create_handle_loop`（`:1053`）懒启动：第一次有请求时才创建 `handle_loop` + `sigterm_watchdog` task，并装 SIGTERM/SIGQUIT handler（仅当在主线程，`:1067`）。
- `sigterm_watchdog`（`:1084`）收到 SIGTERM 后置 `gracefully_exit`，然后**排空在飞请求**（`rid_to_state` 为空才退出），除非 health check 已失败则立即退。
- health check（`http_server.py:160` 的 `/health_generate`）发一条 `max_new_tokens=1` 的特殊请求，靠 `last_receive_tstamp` 是否被 `handle_loop` 刷新来判断 Scheduler/Detokenizer 是否还活着。

## 6. 改代码 / 修 bug 指南

### 「想改 X → 看这里」

| 想做的事 | 入手位置 |
| --- | --- |
| 改 tokenize 逻辑 / 加新输入形态 | `_tokenize_one_request` `tokenizer_manager.py:434` |
| 改 / 加多模态预处理 | `mm_processor.process_mm_data_async`；注册见 `multimodal_processor.py:55`（`get_mm_processor` 按 `hf_config.architectures` 选类） |
| 加新 chat template | `conversation.py` 的 `register_conv_template`（`:625` 起），别在 TokenizerManager 找 |
| 改准入 / context 校验 | `_validate_token_len` `tokenizer_manager.py:478`；上下文长度推断 `hf_transformers_utils.py:168` |
| 给发往 Scheduler 的请求加字段 | 同时改 `TokenizedGenerateReqInput`（`io_struct.py:424`）和 `_create_tokenized_object`（`tokenizer_manager.py:507`），并在 Scheduler 侧接收处对齐 |
| 改回流结果结构 / meta_info | `_handle_batch_output` `tokenizer_manager.py:1119`；logprob 转换 `convert_logprob_style` `:1217` |
| 改 batch / parallel sampling 编排 | `_handle_batch_request` `tokenizer_manager.py:690` |
| 改 abort / 断连行为 | `_wait_one_response` `:632`、`create_abort_task` `:1039`、`abort_request` `:783`，对照注释矩阵 `:1492` |
| 加新控制面 RPC | 仿照 `_Communicator`（`:1454`）+ 在 `_result_dispatcher`（`:312`）注册回包类型 + 在 io_struct 定义 Req/Output |
| 改 metrics | `collect_metrics` `:1339`、`TokenizerMetricsCollector` |

### 常见 bug 高发区

1. **回包 rid 找不到 state**：abort 与回包竞态导致 `del rid_to_state[rid]` 后又来一包，触发 `:1127` 的 error log。一般是新加的回流类型没正确处理 `finished`，或 abort 路径与正常 finish 路径双删。
2. **`_handle_batch_output` 里误加 await / 重活**：它是同步函数且在唯一的 `handle_loop` 里串行执行，任何阻塞会卡住**全部**请求的回流（典型表现：吞吐骤降、health check 失败）。
3. **stream 与非 stream 的 `out_list` 语义混淆**：非 stream 只看最终态 `out_list[-1]`，stream 逐包 yield；改 `state.output_ids` 累积逻辑（`:1170-1176`）时要兼顾 `last_output_offset`。
4. **多模态 input_ids 覆盖**：`process_mm_data_async` 返回的 `input_ids` 会覆盖原值（`:470`）。新写 processor 若忘了在 dict 里放 `input_ids`，会导致 placeholder token 缺失、后续 embedding 对不齐。
5. **新增字段只改一端**：`TokenizedGenerateReqInput` 是跨进程 `pickle` 传输的 dataclass，两端版本不一致会反序列化错位。
6. **`--skip-tokenizer-init` / `--disable-radix-cache` 组合**：text 输入在 skip 模式下报错（`:454`）；input_embeds 必须配 disable radix cache（`:443`）。改输入校验时容易漏。
7. **控制面 RPC fan-out 不匹配**：`_Communicator` 要收满 `fan_out`（= dp_size）个回包才返回，DP 拓扑或回包数不对会**永久挂起** `await`。

### 调试入手点

- 打开 `--log-requests --log-requests-level 2` 看进出 obj（`get_log_request_metadata` `:994` 控制脱敏字段）。
- 在 `handle_loop` / `_handle_batch_output` 加日志确认回包是否到达、rid 是否匹配。
- 请求"卡住不返回"：先看 `rid_to_state` 是否还有该 rid（在 `_wait_one_response` await）、`handle_loop` 是否还在转（`last_receive_tstamp`）、Scheduler 是否真收到（看 Scheduler 日志）。
- 整个 `TokenizerManager` 抛异常会被 `print_exception_wrapper`（`:1420`）捕获并 `kill_process_tree` 退出，崩溃时去看它打的 traceback。

## 交叉引用

- [入口与 OpenAI 适配层](10-entrypoints-api.md)：chat template 渲染、`GenerateReqInput` 构造
- [调度器](12-scheduler.md)：`TokenizedGenerateReqInput` 的消费方
- [DetokenizerManager](18-detokenizer-output.md)：回流 `BatchStrOut` 的来源
- [RadixAttention / 前缀缓存](13-memcache-radixattention.md)：parallel sampling 前缀复用
- [术语表](02-glossary.md)：rid / prefill / decode / TP / DP 等术语
