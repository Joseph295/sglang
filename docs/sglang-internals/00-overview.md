# 总览：设计哲学与多进程架构

> 本章是整套「SGLang 内部架构文档」的入口。读完后你应该能在脑海里画出一张正确的全景图：一个请求从 HTTP 进来、经过哪些进程、走哪些 ZMQ 管道、最后怎么变成 token 流回到客户端。后续每一章（分词、调度、KV cache、前向、采样、解码）都会反复回到这张图上对号入座。

---

## 1. 一句话职责

SGLang 是一个面向 LLM / VLM 的**高性能推理服务框架**；它把「HTTP 接入 + 分词」「调度 + 前向 + 采样」「解码」拆成**多个独立进程**，用 **ZMQ** 串成一条流水线，从而绕开 Python GIL、把 CPU 工作和 GPU 工作解耦，实现 **zero-overhead scheduler**（CPU 调度与 GPU 计算 overlap）。

几个名词先约定（详见 [术语表](02-glossary.md)）：

- **SRT = SGLang Runtime**，即后端推理引擎。源码注释里反复出现，见 `python/sglang/srt/entrypoints/engine.py:15`。
- **co-design**：SGLang 的核心理念是「后端 runtime 与前端语言协同设计」（README `README.md:44`）。本文档只覆盖后端 runtime（`srt`）。
- **RadixAttention / prefill / decode / TP / overlap scheduler** 等专有名词全程保留英文。

---

## 2. 在端到端链路中的位置

本章是「总览」，覆盖的是**整条链路**，而不是其中某一段。后续各章会聚焦其中一格：

```
 客户端           分词              调度            KV缓存          前向         采样          解码           响应
   │               │                │               │             │            │             │              │
   ▼               ▼                ▼               ▼             ▼            ▼             ▼              ▼
┌──────┐     ┌───────────┐    ┌───────────┐   ┌──────────┐  ┌────────┐  ┌────────┐   ┌────────────┐  ┌──────┐
│ HTTP │ ──▶ │ Tokenizer │──▶ │ Scheduler │──▶│  Radix / │─▶│ Model  │─▶│ Sample │──▶│ Detokenizer│─▶│ HTTP │
│Server│     │  Manager  │    │ (per TP)  │   │ KV Pool  │  │Forward │  │        │   │  Manager   │  │ resp │
└──────┘     └───────────┘    └───────────┘   └──────────┘  └────────┘  └────────┘   └────────────┘  └──────┘
   ▲                                                                                        │
   └────────────────────────── token 流式回传（ZMQ → TokenizerManager → HTTP）─────────────┘

本章 = 整张图 + 进程/ZMQ 拓扑。
对应章节：[分词](11-tokenizer-manager.md) [调度](12-scheduler.md) [KV缓存](13-memcache-radixattention.md)
          [前向](14-model-executor.md) [采样](17-sampling-structured-output.md) [解码](18-detokenizer-output.md)
```

---

## 3. 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `prepare_server_args` / `launch_server` | `python/sglang/launch_server.py:11`、`:14` | 命令行入口。解析参数后调用 `launch_server`，退出时 `kill_process_tree` 清理子进程树。 |
| `launch_server` | `python/sglang/srt/entrypoints/http_server.py:723` | 启动整个 SRT server：先拉子进程，再起 FastAPI/uvicorn。函数注释（`:728`）是最权威的架构概述。 |
| `_launch_subprocesses` | `python/sglang/srt/entrypoints/engine.py:539` | 真正 fork 出 Scheduler / DetokenizerManager 子进程，并在主进程构造 TokenizerManager。 |
| `Engine` | `python/sglang/srt/entrypoints/engine.py:89` | Python API 入口（不走 HTTP）。类 docstring（`:90`）说明三组件模型。 |
| `_set_envs_and_config` | `python/sglang/srt/entrypoints/engine.py:476` | 设置环境变量、信号处理器（SIGCHLD/SIGQUIT）、`mp.set_start_method("spawn")`。 |
| `TokenizerManager` | `python/sglang/srt/managers/tokenizer_manager.py:165` | 主进程内的对象。分词 + 把请求 PUSH 给 scheduler，并起 asyncio `handle_loop` 收回结果。 |
| `Scheduler` / `run_scheduler_process` | `python/sglang/srt/managers/scheduler.py`、`:2252` | 每个 TP rank 一个独立进程；批调度 + 前向 + 采样的中枢。 |
| `DetokenizerManager` / `run_detokenizer_process` | `python/sglang/srt/managers/detokenizer_manager.py:68`、`:264` | 独立子进程，把 token id 解码成字符串再回传 TokenizerManager。 |
| `PortArgs` | `python/sglang/srt/server_args.py:1499` | 4 条 ZMQ 通道（+ nccl_port）的地址定义。`init_new`（`:1513`）分配端口。 |
| `ServerArgs` | `python/sglang/srt/server_args.py` | 所有启动参数。`@dataclasses.dataclass` 字段按功能分组（见第 5 节）。 |
| `get_zmq_socket` | `python/sglang/srt/utils.py` | 统一创建 ZMQ socket 的工具函数（bind/connect 由最后一个 bool 参数控制）。 |

---

## 4. 核心数据流 / 执行流程

### 4.1 进程与 ZMQ 拓扑（全景图）

```
                              主进程 (main process)
        ┌───────────────────────────────────────────────────────────┐
        │  uvicorn / FastAPI  (http_server.py:app)                    │
        │      │  调用 _global_state.tokenizer_manager.generate_request│
        │      ▼                                                       │
        │  TokenizerManager  (asyncio, 跑在主线程的 event loop)        │
        │      ├─ send_to_scheduler   (zmq.PUSH)  ───────┐            │
        │      └─ recv_from_detokenizer(zmq.PULL) ◀────┐ │            │
        │  Engine.send_to_rpc (zmq.DEALER) ──────────┐ │ │            │
        └────────────────────────────────────────────┼─┼─┼───────────┘
                                                      │ │ │
   scheduler_input_ipc_name  (PUSH→PULL) ────────────┘ │ │
   tokenizer_ipc_name        (PULL←PUSH) ──────────────┼─┘
   rpc_ipc_name              (DEALER→ROUTER) ──────────┘
                                                      │
        ┌─────────────────────────────────────────────┼────────────────────────┐
        │  Scheduler 子进程 (TP rank 0)  ◀── 每个 TP rank 一个进程 ──▶ TP rank 1…N │
        │      ├─ recv_from_tokenizer (zmq.PULL)  ← scheduler_input_ipc_name      │
        │      ├─ recv_from_rpc       (zmq.ROUTER)← rpc_ipc_name                  │
        │      ├─ send_to_detokenizer (zmq.PUSH)  → detokenizer_ipc_name          │
        │      └─ send_to_tokenizer   (zmq.PUSH)  → tokenizer_ipc_name (元信息)    │
        │   (各 rank 之间用 NCCL all-reduce，端口 = nccl_port，非 ZMQ)            │
        └─────────────────────────────────────────────┬──────────────────────────┘
                                                       │ detokenizer_ipc_name (PUSH→PULL)
        ┌──────────────────────────────────────────────▼─────────────────────────┐
        │  DetokenizerManager 子进程                                               │
        │      ├─ recv_from_scheduler (zmq.PULL) ← detokenizer_ipc_name            │
        │      └─ send_to_tokenizer   (zmq.PUSH) → tokenizer_ipc_name              │
        └──────────────────────────────────────────────────────────────────────────┘
```

只有 **rank 0** scheduler 持有真正的 ZMQ socket；非 0 rank 用 `SimpleNamespace(send_pyobj=lambda x: None)` 占位（`python/sglang/srt/managers/scheduler.py:249-252`），因为只有 rank 0 负责与外部通信，其余 rank 通过 NCCL 与 rank 0 保持张量同步。

四条命名管道（`PortArgs`，`python/sglang/srt/server_args.py:1500-1511`）：

| 通道 | 方向 | 用途 |
| --- | --- | --- |
| `scheduler_input_ipc_name` | TokenizerManager → Scheduler(rank0) | 把 tokenize 后的请求 PUSH 进调度器 |
| `detokenizer_ipc_name` | Scheduler → DetokenizerManager | 把采样出的 token id PUSH 给解码器 |
| `tokenizer_ipc_name` | DetokenizerManager / Scheduler → TokenizerManager | 把解码后的文本 / 控制类输出回传 |
| `rpc_ipc_name` | Engine → Scheduler | `collective_rpc`（如保存模型）等控制面命令 |

单机默认用 `ipc://` + 临时文件（`:1527`）；开启 `--enable-dp-attention` 或多节点时改用 `tcp://`（`:1556`），因为 IPC 文件无法跨节点。

### 4.2 一个请求的生命周期（数据进来什么样、出去什么样）

```
HTTP POST /generate  {"text": "你好", "sampling_params": {...}}
   │  FastAPI 把 JSON 反序列化为 GenerateReqInput (http_server.py:240)
   ▼
TokenizerManager.generate_request()
   │  - 调 HF tokenizer：text → input_ids  (CPU 工作)
   │  - send_to_scheduler.send_pyobj(tokenized_obj)   ← scheduler_input_ipc (tokenizer_manager.py:628)
   ▼  (ZMQ, 跨进程，pickle 序列化)
Scheduler.event_loop_(normal|overlap)  [每个 TP rank 一个进程]
   │  - recv_requests() 取出请求，process_input_requests() 入队
   │  - get_next_batch_to_run() 组 batch（prefix 命中走 RadixAttention，见 KV 缓存章）
   │  - run_batch() → 模型前向 + 采样，得到下一个 token id
   │  - send_to_detokenizer.send_pyobj(BatchTokenIDOut)  ← detokenizer_ipc
   ▼  (ZMQ)
DetokenizerManager.event_loop  (detokenizer_manager.py:109-111)
   │  - token id → 文本（处理增量解码、特殊 token）
   │  - send_to_tokenizer.send_pyobj(BatchStrOut)   ← tokenizer_ipc
   ▼  (ZMQ)
TokenizerManager.handle_loop  (tokenizer_manager.py:1111-1115, asyncio 后台 task)
   │  - recv_from_detokenizer.recv_pyobj()
   │  - 唤醒对应 rid 的 asyncio Future / 生成器
   ▼
generate_request() 的 async generator 产出 chunk
   ▼
HTTP 以 SSE "data: {...}\n\n" 流式返回 (http_server.py:249)
```

关键点：**整条链路是异步流水线**。TokenizerManager 在主进程用 asyncio 同时处理「往外发请求」和「往回收 token」两件事——`handle_loop` 是一个常驻 asyncio task，由 `auto_create_handle_loop()` 在首个请求到来时懒启动（`python/sglang/srt/managers/tokenizer_manager.py:1053-1061`）。

### 4.3 进程是怎么被拉起来的（启动流程）

```
launch_server.py:main
   └─ launch_server(server_args)                         http_server.py:723
        └─ _launch_subprocesses(server_args)             engine.py:539
             ├─ _set_envs_and_config()                   engine.py:476
             │     └─ mp.set_start_method("spawn")        engine.py:536
             ├─ for pp_rank, tp_rank:                     engine.py:582-605
             │     mp.Process(run_scheduler_process)  ──▶ 启动 N 个 Scheduler 子进程
             │     (N = tp_size × pp_size，dp_size==1 分支)
             ├─ mp.Process(run_detokenizer_process)      engine.py:639-646
             ├─ TokenizerManager(server_args, port_args) engine.py:649   ← 主进程内对象
             └─ 阻塞等待每个 scheduler 通过 mp.Pipe 回报 {"status":"ready"}
                                                           engine.py:662-677
                                                           （对应 scheduler.py:2295 pipe_writer.send）
        ── 回到 launch_server ──
        set_global_state(_GlobalState(tokenizer_manager, scheduler_info))   http_server.py:744
        起一个 warmup_thread（在 FastAPI lifespan 里 .start()）            http_server.py:762 / :131
        uvicorn.run(app, host, port, loop="uvloop")                        http_server.py:778
```

注意 `mp.Pipe`（`engine.py:584`）只用于**一次性的启动握手**——子进程把 `max_total_num_tokens` / `max_req_input_len` 回传给主进程（`scheduler.py:2295-2301`）后，运行期所有通信都走 ZMQ，不再用 Pipe。

`dp_size > 1` 时走另一条分支：不直接起 scheduler，而是起一个 `run_data_parallel_controller_process`（`engine.py:606-615`），由它在内部再分发到各 DP 组（见 [数据并行](15-worker-parallelism.md)）。

`FastAPI lifespan`（`http_server.py:120-132`）负责：跑 `--warmups` 指定的预热、再 `.start()` 那个发首个 warmup 请求的 `warmup_thread`。`_GlobalState`（`:106`）是个全局单例，所有 HTTP handler 通过 `_global_state.tokenizer_manager` 访问引擎。

---

## 5. 设计难点与权衡

### 5.1 为什么是「多进程 + ZMQ」而不是单进程？

1. **绕开 GIL。** CPython 的 GIL 让纯 Python 代码无法真正并行。分词（CPU）、调度（CPU 重逻辑）、解码（CPU）如果和 GPU 前向挤在同一个进程里，它们会互相抢 GIL，导致 GPU 出现空泡。拆成独立进程后，每个进程有自己的 GIL，CPU 工作可与 GPU 计算真正并行。

2. **CPU / GPU 解耦 + zero-overhead scheduler。** Scheduler 进程里 `event_loop_overlap`（`python/sglang/srt/managers/scheduler.py:656`）刻意把「为下一个 batch 做 CPU 调度准备」和「当前 batch 的 GPU 前向」交叠：当 GPU 在算 batch N 时，CPU 已经在组 batch N+1，并用 `result_queue` + `launch_done` event 把上一个 batch 的结果在下一轮才处理（`:670-691`）。这就是 README 所说的 "zero-overhead CPU scheduler"（`README.md:47`）。关掉它用 `--disable-overlap-schedule`，走 `event_loop_normal`（`:636`），代码更直观，便于调试。

3. **故障隔离与清理。** 子进程崩溃时向主进程发 `SIGQUIT`（`scheduler.py:2323-2326`），主进程的 `sigquit_handler`（`engine.py:527-533`）`kill_process_tree` 把整棵树带走，避免僵尸 GPU 进程。`SIGCHLD` handler（`engine.py:513-522`）记录非正常退出码。

### 5.2 每个 TP rank 一个进程

张量并行（TP）下，模型按列/行切到多张卡。SGLang 的选择是**每个 TP rank 一个独立 OS 进程**（`engine.py:582-605` 的双重循环），rank 间用 NCCL all-reduce 同步（端口 `nccl_port`，与 ZMQ 无关）。只有 rank 0 与外界通过 ZMQ 通信，其余 rank 的 send socket 是 no-op 占位（`scheduler.py:251-252`）。好处是每个 rank 独占一个 GIL，前向各跑各的；代价是 rank 间必须严格保持 batch 一致，调度决策只能由 rank 0 主导后广播。

### 5.3 `spawn` 而非 `fork`

`_set_envs_and_config` 强制 `mp.set_start_method("spawn", force=True)`（`engine.py:536`）。CUDA context 不能被 `fork` 安全继承（fork 后子进程的 CUDA 状态会损坏），因此必须用 spawn 全新启动解释器。代价是子进程要重新 import 一切、`server_args`/`port_args` 必须可 pickle 才能传过去。

### 5.4 IPC vs TCP 的地址选择

`PortArgs.init_new` 默认 `ipc://`（Unix domain socket，单机最快），但一旦 `--enable-dp-attention` 或多节点就切 `tcp://`（`server_args.py:1524-1562`），因为 IPC 文件无法跨主机。这是改部署形态时最容易踩的隐式分叉。

### 5.5 主进程里其实有「两个 event loop 概念」

- uvicorn 跑 FastAPI 的 asyncio loop（处理 HTTP）。
- TokenizerManager 的 `handle_loop` / `sigterm_watchdog` 是**挂在同一个 loop 上的后台 task**（`tokenizer_manager.py:1059-1082`），不是独立线程。
两者共享一个 uvloop event loop，靠 asyncio 协作调度。理解这点能避免「为什么我在 handler 里做阻塞调用会卡死整个 server」这类问题。

---

## 6. 改代码 / 修 bug 指南

### 想改 X → 看这里

| 你想做的事 | 入手位置 |
| --- | --- |
| 加一个新的 HTTP endpoint | `python/sglang/srt/entrypoints/http_server.py`（仿照 `:239` 的 `/generate`），转调 `_global_state.tokenizer_manager` |
| 加一个新的启动参数 | `python/sglang/srt/server_args.py` 的对应分组（`# Model and tokenizer` 等，见 `:44` 起），同时改 `add_cli_args` 解析与 `check_server_args` 校验 |
| 改请求字段 / 协议 | `python/sglang/srt/managers/io_struct.py` 的 `GenerateReqInput` 等 dataclass（三个进程都引用它，注意 pickle 兼容） |
| 改进程拓扑 / 加新子进程 | `_launch_subprocesses`（`engine.py:539`）+ 新增 `PortArgs` 通道（`server_args.py:1499`） |
| 改调度 / overlap 行为 | `Scheduler.event_loop_normal` / `event_loop_overlap`（`scheduler.py:636` / `:656`） |
| 用 Python API 而非 HTTP | `Engine`（`engine.py:89`），`.generate()` / `.async_generate()` |

### 该模块常见 bug 高发区

- **「服务起不来 / 卡在启动」**：多半卡在 `_launch_subprocesses` 等 `{"status":"ready"}` 的握手（`engine.py:662-677`）。如果某个 scheduler 在 `Scheduler.__init__`（加载模型、分配 KV）阶段挂了，主进程会收到 `EOFError` 并打印 "Rank i scheduler is dead"（`:665-671`）。**真正的报错在 scheduler 子进程的日志里**，不在主进程。
- **「请求发出去没有响应」**：检查四条 ZMQ 通道是否对接错（PUSH/PULL 方向、bind/connect 由 `get_zmq_socket` 最后一个 bool 决定）。常见错误是新加通道时 bind 端写反。
- **pickle 失败**：spawn 模式下 `server_args` 或 io_struct 里塞了不可序列化对象（如 lambda、打开的文件句柄），会在 `mp.Process.start()` 处炸。
- **多节点 / DP attention 不通**：忘了它会从 IPC 切到 TCP（`server_args.py:1524`），防火墙/`--dist-init-addr` 没配对。
- **健康检查 503**：`/health_generate`（`http_server.py:160`）实际跑了一次单 token 生成，若 detokenizer 长时间无回包会判失败（`:200-207`）——这通常说明 scheduler 卡死或 GPU OOM。

### 调试入手点

- 先看进程是否齐全：`ps aux | grep sglang::scheduler`（进程名由 `setproctitle` 设定，`scheduler.py:2272`）。应看到 `tp_size × pp_size` 个 scheduler + 1 个 detokenizer。
- 单进程化调试调度逻辑：加 `--disable-overlap-schedule` 走 `event_loop_normal`，逻辑线性、断点好打。
- 缩小到最小复现：用 `Engine`（Python API）而非完整 HTTP server，跳过 uvicorn/FastAPI 这一层。
- 看子进程日志：每个 scheduler 用 `configure_logger(server_args, prefix=" TP{rank}")`（`scheduler.py:2281`）带 rank 前缀，便于区分是哪张卡出问题。

### 本地把服务跑起来

```bash
# 1. 启动一个 HTTP server（单卡）
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 30000

# 2. 张量并行 4 卡
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp-size 4 --port 30000

# 3. 关掉 overlap scheduler 以便调试调度逻辑
python -m sglang.launch_server --model-path <model> \
    --disable-overlap-schedule

# 4. 发一个请求（OpenAI 兼容接口）
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"你好"}]}'

# 5. 原生 /generate 接口
curl http://localhost:30000/generate \
    -H "Content-Type: application/json" \
    -d '{"text":"你好","sampling_params":{"max_new_tokens":32}}'
```

不想起 HTTP server、想在脚本里直接调引擎（对应 `Engine`，`engine.py:89`）：

```python
import sglang as sgl

engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
print(engine.generate("你好", {"max_new_tokens": 32}))
engine.shutdown()
```

---

## 下一步阅读

- 请求长什么样、字段含义：[io_struct 与请求协议](90-data-structures.md)
- 主进程内分词与流式回收：[TokenizerManager / 分词](11-tokenizer-manager.md)
- 调度核心与 overlap：[Scheduler / 调度器](12-scheduler.md)
- 前缀复用与显存：[RadixAttention 与 KV 缓存](13-memcache-radixattention.md)
- 名词速查：[术语表](02-glossary.md)
