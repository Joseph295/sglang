# 入口与 API 层（Entrypoints & OpenAI API）

> 本章对应源码：`python/sglang/srt/entrypoints/`（`http_server.py`、`engine.py`、`EngineBase.py`、`http_server_engine.py`、`verl_engine.py`）与 `python/sglang/srt/openai_api/`（`adapter.py`、`protocol.py`），以及 `python/sglang/srt/server_args.py`。

## 1. 一句话职责

入口层（Entrypoints）是 SGLang「面向外部世界的皮肤」：它把各种形式的请求（HTTP JSON、OpenAI 协议、Python 函数调用、RL 训练框架的张量更新）统一翻译成内部唯一的请求结构 `GenerateReqInput` / `EmbeddingReqInput`，交给 `TokenizerManager`，再把内部的流式输出翻译回各协议要求的响应格式。它**不做任何推理计算**，只负责协议适配、参数解析与进程编排。

SGLang 提供两类入口，但它们共享同一个底层引擎：

- **offline Engine**（`Engine` 类）：在**当前 Python 进程内**直接拿到 `TokenizerManager` 对象，函数调用即推理，无 HTTP，无网络开销。适合离线批处理、benchmark、被 RL 框架（verl）嵌入。
- **online HTTP server**（FastAPI app）：用 `uvicorn` 起一个 HTTP 服务，对外暴露 `/generate`、`/v1/chat/completions` 等路由。它在内部仍然创建同一个 `TokenizerManager`，只是多包了一层 FastAPI 路由 + OpenAI 协议适配。

两者的关系可以一句话概括：**HTTP server = Engine 的子进程组 + FastAPI 路由壳**。两者都通过 `_launch_subprocesses` 拉起 Scheduler / Detokenizer 子进程。

## 2. 在端到端链路中的位置

```
                 ┌─────────────────────────── 本章范围 ───────────────────────────┐
                 │                                                                 │
 客户端           │  ① HTTP 路由 / Python API        ② OpenAI 协议适配             │
 (curl/openai ───►│  http_server.py / engine.py  ──► openai_api/adapter.py ──┐     │
  SDK / verl)     │  (FastAPI / Engine.generate)     (v1_*_request)          │     │
                 │                                                           ▼     │
                 │                                              GenerateReqInput   │
                 │                                              EmbeddingReqInput   │
                 └───────────────────────────────────┬─────────────────────┘     │
                                                      │ (Python 对象, 进程内)        │
                                                      ▼
                              TokenizerManager.generate_request()   ← 见 [TokenizerManager](11-tokenizer-manager.md)
                                                      │ (ZMQ IPC)
                  ┌───────────────────────────────────┼───────────────────────────┐
                  ▼                                    ▼                            ▼
              分词/多模态         →  调度(Scheduler)  →  KV缓存/前向/采样  →  解码(Detokenizer)
              [11]                   [12]               [13][16][17]            回流
                                                      │
                                                      ▼ (流式 dict 回到 TokenizerManager)
                 ┌───────────────────────────────────┴───────────────────────────┐
                 │  ③ 响应适配：把内部 dict 翻译回 OpenAI / native JSON / SSE       │
                 │     v1_*_response / generate_request 的 StreamingResponse        │
                 └─────────────────────────────────────────────────────────────────┘
```

本章只负责图中 ①②③ 三个环节：**进来**的是原始 HTTP body / Python 参数，**出去**给下游的是 `GenerateReqInput` / `EmbeddingReqInput`；下游返回的是 `dict`（含 `text` + `meta_info`），本章再把它包装成响应。中间的分词、调度、前向都在后续章节（[分词](11-tokenizer-manager.md)、[调度器](12-scheduler.md)）。

## 3. 关键文件与类

| 符号 | file:line | 作用 |
|------|-----------|------|
| `launch_server()` | `python/sglang/srt/entrypoints/http_server.py:723` | online 入口总函数：拉子进程 → 装中间件 → `uvicorn.run` |
| `_launch_subprocesses()` | `python/sglang/srt/entrypoints/engine.py:539` | **两种入口共用**：起 Scheduler / DataParallelController / Detokenizer 子进程，主进程创建 `TokenizerManager` |
| `app` (FastAPI) | `python/sglang/srt/entrypoints/http_server.py:136` | 全局 FastAPI 应用，所有 `@app.*` 路由挂在其上 |
| `_GlobalState` / `set_global_state` | `python/sglang/srt/entrypoints/http_server.py:106` | 全局单例，持有 `tokenizer_manager` 与 `scheduler_info`，路由函数通过它访问引擎 |
| `generate_request()` (路由) | `python/sglang/srt/entrypoints/http_server.py:239` | native `/generate` 路由，FastAPI 自动把 body 反序列化成 `GenerateReqInput` |
| `Engine` | `python/sglang/srt/entrypoints/engine.py:89` | offline Python API，`generate/encode/...` 同步包裹异步调用 |
| `EngineBase` | `python/sglang/srt/entrypoints/EngineBase.py:7` | 抽象基类，统一 `Engine` 与 `HttpServerEngineAdapter` 的接口 |
| `HttpServerEngineAdapter` | `python/sglang/srt/entrypoints/http_server_engine.py:51` | 「假装是 Engine，实则走 HTTP」的适配器，给 verl server 后端用 |
| `VerlEngine` | `python/sglang/srt/entrypoints/verl_engine.py:29` | RL 训练（verl）专用：按 TP rank 分工 + `broadcast_pyobj` |
| `v1_chat_completions()` | `python/sglang/srt/openai_api/adapter.py:1430` | `/v1/chat/completions` 入口 |
| `v1_chat_generate_request()` | `python/sglang/srt/openai_api/adapter.py:947` | **协议核心**：OpenAI chat → `GenerateReqInput`（应用 chat template、tools、采样参数映射） |
| `v1_generate_request()` | `python/sglang/srt/openai_api/adapter.py:514` | `/v1/completions` → `GenerateReqInput` |
| `v1_embedding_request()` | `python/sglang/srt/openai_api/adapter.py:1784` | `/v1/embeddings` → `EmbeddingReqInput` |
| `ChatCompletionRequest` | `python/sglang/srt/openai_api/protocol.py:334` | OpenAI chat 请求的 pydantic 模型（含 SRT 扩展字段） |
| `ServerArgs` | `python/sglang/srt/server_args.py:43` | 所有配置的中心 dataclass |
| `prepare_server_args()` | `python/sglang/srt/server_args.py:1477` | `argv → ServerArgs`，命令行入口用 |
| `PortArgs.init_new()` | `python/sglang/srt/server_args.py:1514` | 分配进程间通信（ZMQ IPC / TCP）端口 |

## 4. 核心数据流 / 执行流程

### 4.1 两种入口的「同源」启动流程

无论 online 还是 offline，最终都汇聚到 `_launch_subprocesses`（`engine.py:539`）：

```
 online:  python -m sglang.launch_server --model ...
            └► prepare_server_args (server_args.py:1477)
                 └► launch_server (http_server.py:723)
                      └► _launch_subprocesses ──┐
                                                │
 offline: sgl.Engine(model_path=...)            │
            └► Engine.__init__ (engine.py:103)  │
                 └► _launch_subprocesses ───────┤
                                                ▼
                          ┌──────────── 主进程 ────────────┐
                          │ TokenizerManager (engine.py:649)│
                          └────────────────┬────────────────┘
                          ZMQ IPC 端口由 PortArgs.init_new 分配
                          ┌────────────────┼───────────────────────┐
                          ▼                ▼                        ▼
                  Scheduler 子进程   Detokenizer 子进程    (dp>1 时) DataParallelController
                  run_scheduler_process   run_detokenizer_process
```

要点（`engine.py:539-682`）：

1. `configure_logger` → `check_server_args` → `_set_envs_and_config`（设置 NCCL/CUDA 环境、注册 `SIGCHLD`/`SIGQUIT` handler、`mp.set_start_method("spawn")`）。
2. `dp_size == 1` 时直接按 `tp_rank × pp_rank` fork 多个 `run_scheduler_process`（`engine.py:582-605`）；`dp_size > 1` 时只起一个 `run_data_parallel_controller_process`（`engine.py:606-615`）。
3. 非 0 号 node（`node_rank >= 1`）不创建 tokenizer/detokenizer，只等子进程 ready（`engine.py:617-636`）。这是多机部署的关键分支。
4. 主进程创建 `TokenizerManager`，按需加载 chat / completion 模板（`engine.py:650-658`）。
5. 通过 `mp.Pipe` 阻塞等待每个 scheduler 发回 `{"status": "ready", ...}`，把 `max_req_input_len` 回填给 `TokenizerManager`（`engine.py:660-682`）。

online 入口在此之上额外做（`http_server.py:743-787`）：装 `api_key` 中间件、Prometheus 中间件、起一个 warmup 线程（在 FastAPI `lifespan` 中 start，见 `http_server.py:120-132`），最后 `uvicorn.run(app, ...)`。

### 4.2 一个 `/v1/chat/completions` 请求的完整旅程

```
HTTP POST /v1/chat/completions  {messages, tools, temperature, ...}
   │
   ▼ http_server.py:609  openai_v1_chat_completions(raw_request)
   ▼ adapter.py:1430     v1_chat_completions(tokenizer_manager, raw_request)
   │   ├─ raw_request.json()  → ChatCompletionRequest(**json)   (pydantic 校验, protocol.py:334)
   │   └─ v1_chat_generate_request(...)  (adapter.py:947)  ◄── 协议适配核心
   │        ├─ tools 处理 + FunctionCallParser.get_structure_constraint (adapter.py:979-994)
   │        ├─ apply_chat_template / generate_chat_conv  → prompt_ids 或 prompt 字符串
   │        ├─ 采样参数映射: temperature/top_p/max_tokens→max_new_tokens... (adapter.py:1128-1146)
   │        ├─ response_format → json_schema / structural_tag (adapter.py:1148-1159)
   │        └─ 组装 GenerateReqInput (adapter.py:1213-1229)
   │
   ▼ tokenizer_manager.generate_request(adapted_request, raw_request)  ← 离开本章
   │   (async generator, 逐 chunk yield dict: {"text", "meta_info"})
   │
   ▼ 回到 adapter.py
   ├─ stream=True : generate_stream_resp() 把每个 dict 转成 ChatCompletionStreamResponse,
   │               以 "data: {...}\n\n" SSE 格式 yield, 最后 "data: [DONE]\n\n" (adapter.py:1447+)
   └─ stream=False: v1_chat_generate_response() 一次性组装 ChatCompletionResponse
                    (含 reasoning_parser / FunctionCallParser 解析, adapter.py:1234+)
```

**协议适配的本质**就是 `v1_chat_generate_request`：它读 OpenAI 字段，**全部塞进一个 `GenerateReqInput`**。注意几个非平凡映射：

- OpenAI 的 `max_tokens` / `max_completion_tokens` → SRT 的 `sampling_params["max_new_tokens"]`（`adapter.py:1130`）。
- chat template：若启动时设了 `--chat-template`（`chat_template_name` 非 None），走 SGLang 自己的 `generate_chat_conv`（`adapter.py:1072`）；否则走 HF tokenizer 的 `apply_chat_template`（`adapter.py:1029`）。
- 多模态：`is_multimodal` 时传 `text` 让 processor 处理；否则传 `input_ids`（`adapter.py:1185-1211`）。
- tool calling：`tool_choice != "none"` 时构造 `FunctionCallParser` 并可能注入结构化约束（`structural_tag` / `ebnf`），与 `response_format` 的约束互斥（`adapter.py:1161-1178`）。

`/v1/completions`（`v1_generate_request`，`adapter.py:514`）和 `/v1/embeddings`（`v1_embedding_request`，`adapter.py:1784`）是同样套路的简化版。

### 4.3 native `/generate` 路由的「零适配」捷径

native 路由几乎不做协议转换，因为 FastAPI 会**直接把 JSON body 反序列化成 `GenerateReqInput` dataclass**（`http_server.py:239`，注意函数签名 `obj: GenerateReqInput`，注释见 `http_server.py:238`）。所以 native API 的字段就是 `GenerateReqInput` 的字段。流式分支构造 `StreamingResponse`，并挂上 `create_abort_task` 作为 background task（`http_server.py:260-264`），用于客户端断连时中止请求。

### 4.4 offline Engine 的同步包装

`Engine.generate`（`engine.py:139`）把参数塞进 `GenerateReqInput`，然后用 `loop.run_until_complete(generator.__anext__())` 把异步生成器**同步化**（`engine.py:192-208`）。streaming 时返回一个 `generator_wrapper` 同步生成器。`encode`（`engine.py:268`）走同样路径但构造 `EmbeddingReqInput`。

关键认知：**offline Engine 不经过 HTTP、不经过 openai_api/adapter.py**，它直接调用 `TokenizerManager`。OpenAI 协议适配是 HTTP server 独有的一层。

## 5. 设计难点与权衡

### 5.1 为什么 native 与 OpenAI 两套并存？

native（`/generate`）暴露 SGLang 全部能力（`logprob_start_len`、`token_ids_logprob`、`custom_logit_processor`、PD disaggregation 的 `bootstrap_*` 等），字段即 `GenerateReqInput`，零适配开销。OpenAI（`/v1/*`）牺牲表达力换生态兼容（能直接被 openai SDK、LangChain 调用）。adapter 层就是这两者之间的「翻译官」——把受限的 OpenAI 字段映射到丰富的内部结构，把内部 `meta_info` 折叠回 OpenAI 的 `usage` / `logprobs` / `finish_reason`。

### 5.2 `EngineBase` 抽象与 verl 的两种后端

`EngineBase`（`EngineBase.py:7`）让 `Engine`（进程内）和 `HttpServerEngineAdapter`（跨 HTTP）拥有同一接口。`VerlEngine`（`verl_engine.py:29`）据此提供 `backend="engine"|"server"` 两种选择：

- `backend="engine"`：每个 node 的 first rank 直接 `Engine(**kwargs)`，并设 `SGLANG_BLOCK_NONZERO_RANK_CHILDREN="0"`（`verl_engine.py:52-53`），这正对应 `engine.py:625` 那个「非 0 rank 不阻塞」的分支——RL 框架不希望子 rank 卡死在 `proc.join()`。
- `backend="server"`：只有 tp_rank 0 起 `HttpServerEngineAdapter`，其余为 None（`verl_engine.py:58-62`）。

权重更新是 verl 的核心诉求：`update_weights_from_tensor`（`verl_engine.py:126`）用 `dist.gather_object` 把各 rank 的张量收集到 rank 0，再调引擎。`HttpServerEngineAdapter` 走 HTTP 时**只传张量元数据**（base64 序列化的 handle），真实显存数据通过 CUDA IPC 直接拷贝（见 `http_server_engine.py:80-103` 的注释与 `update_weights_from_tensor` 路由 `http_server.py:433`）。这是「HTTP 传不动几十 GB 权重」问题的标准解法。

### 5.3 进程模型与 IPC 端口分配

`PortArgs.init_new`（`server_args.py:1514`）在普通场景用 **ZMQ IPC（Unix socket 文件）**，在 `enable_dp_attention` 时切到 **TCP**（因为 DP attention 可能跨节点，IPC 文件无法跨机），端口按 `dist_init_addr` 派生（`server_args.py:1533-1562`）。`nccl_port` 用随机探测可用端口避免冲突（`server_args.py:1515-1522`）。理解这套端口约定是排查「子进程连不上」类 bug 的前提。

### 5.4 全局单例与全局变量的坑

- `_global_state`（`http_server.py:112`）是模块级全局，路由函数全靠它访问引擎。这意味着**一个进程只能跑一个 HTTP server**。
- `chat_template_name`（`adapter.py:81`）也是模块级全局，由启动时的 `load_chat_template_for_openai_api` / `guess_chat_template_name_from_model_path` 设置（`engine.py:650-655`）。它影响所有后续 chat 请求走 conv 模板还是 HF 模板。
- batch / file 存储也是进程内全局 dict（`adapter.py:91-95`）——`/v1/files` 与 `/v1/batches` 的状态**不持久化**，重启即丢。

### 5.5 流式响应与中止（abort）

流式用 SSE（`text/event-stream`），每个 chunk 是增量 `delta`（adapter 通过 `stream_buffer` 做差分，见 `adapter.py:836` 的 `delta = text[len(stream_buffer):]`）。`StreamingResponse` 的 `background=create_abort_task(obj)`（`http_server.py:263`）保证客户端断连时后端能中止生成，避免算力浪费。这是流式实现里最容易被忽略但很关键的一环。

### 5.6 `ServerArgs.__post_init__` 的「智能默认值」

`ServerArgs` 不是哑数据类——`__post_init__`（`server_args.py:223`）会根据 GPU 显存与并行度**自动推导** `mem_fraction_static`、`chunked_prefill_size` 等（`server_args.py:244-278`）。`check_server_args`（`server_args.py:1440`）则做硬约束校验（如 pp 与 overlap schedule 互斥、LoRA 与 radix cache 兼容性）。调参时若发现「我没设的值变了」，先看这两处。

## 6. 改代码 / 修 bug 指南

### 想加一个新的 native HTTP 路由

在 `http_server.py` 用 `@app.api_route("/your_endpoint", methods=[...])` 加函数，通过 `_global_state.tokenizer_manager` 访问引擎。若要 FastAPI 自动反序列化 body，把参数类型标成一个 dataclass（参考 `/generate` 的 `obj: GenerateReqInput`，`http_server.py:239`），该 dataclass 通常定义在 `managers/io_struct.py`。

### 想加 / 改 OpenAI 协议字段

1. 在 `protocol.py` 对应的 pydantic 模型加字段（如 `ChatCompletionRequest`，`protocol.py:334`）。SRT 私有扩展字段集中在 `# Extra parameters for SRT backend only` 注释段（`protocol.py:377-401`）。
2. 在 `adapter.py` 的 `v1_*_request` 里把它映射进 `sampling_params` 或 `GenerateReqInput`（chat 见 `adapter.py:1128-1229`）。
3. 若影响响应，改对应 `v1_*_response` 与流式 `generate_stream_resp`。

### 想改采样参数 / response_format / tool calling 行为

直接看 `v1_chat_generate_request`（`adapter.py:947`）。采样映射在 `adapter.py:1128`，`response_format`→约束在 `adapter.py:1148`，tools→`FunctionCallParser` 在 `adapter.py:979`。tool / reasoning 的响应侧解析在 `v1_chat_generate_response`（`adapter.py:1234`，约束转换与 `ReasoningParser` 在 `adapter.py:1306-1323`）。

### 想用 offline Engine 嵌入训练 / benchmark

用 `sglang.Engine(model_path=...)`（`engine.py:89`），调 `generate/encode`。注意它默认 `log_level="error"`（`engine.py:113-115`）。RL 场景看 `VerlEngine`（`verl_engine.py:29`）与权重热更新接口。

### 常见 bug 高发区

| 症状 | 先看这里 |
|------|----------|
| 服务起不来 / 子进程卡死 | `_launch_subprocesses`（`engine.py:539`）的 pipe `recv` 与 `node_rank>=1` 分支（`engine.py:617`） |
| 端口冲突 / 子进程连不上 | `PortArgs.init_new`（`server_args.py:1514`），尤其 `enable_dp_attention` 的 TCP 分支 |
| chat 输出格式不对 / 模板错乱 | `chat_template_name` 全局（`adapter.py:81`）+ `v1_chat_generate_request` 模板分支（`adapter.py:996` vs `1071`） |
| `max_tokens` 不生效 | `adapter.py:1130`（OpenAI→`max_new_tokens` 映射） |
| 流式断连后端不停 | `create_abort_task` 是否挂上（`http_server.py:263`） |
| 权重热更新 OOM / 失败 | `update_weights_from_tensor`（`engine.py:392` / `http_server_engine.py:80`），确认走 CUDA IPC 而非 HTTP 传张量 |
| 某默认参数被「莫名修改」 | `ServerArgs.__post_init__`（`server_args.py:223`） |
| 健康检查 503 | `/health_generate`（`http_server.py:160`）的 `HEALTH_CHECK_TIMEOUT` 与 `last_receive_tstamp` 逻辑 |

### 调试入手点

- **复现协议适配 bug**：直接单元调用 `v1_chat_generate_request([req], tokenizer_manager)`，打印返回的 `GenerateReqInput`，对比预期。这能把「协议层」和「推理层」的 bug 隔离开。
- **看请求实际传了什么**：启动加 `--log-requests`（`server_args.py:96`），或在 `generate_request` 路由打印 `obj`。
- **看内部配置**：调 `/get_server_info`（`http_server.py:221`）会返回 `server_args` + `scheduler_info` + 内部状态，是确认实际生效配置的最快方式。
- **看可用路由全貌**：`/openapi.json`（除非设了 `DISABLE_OPENAPI_DOC`，见 `http_server.py:138`）。

## 路由速查表（native + OpenAI + 运维）

| 类别 | 路由 | 处理函数 (file:line) |
|------|------|----------------------|
| 生成 | `/generate` | `http_server.py:239` |
| 生成(文件/embeds) | `/generate_from_file` | `http_server.py:276` |
| 嵌入 | `/encode` | `http_server.py:301` |
| 奖励/分类 | `/classify` | `http_server.py:313` |
| OpenAI | `/v1/completions` | `http_server.py:604` → `adapter.py:749` |
| OpenAI | `/v1/chat/completions` | `http_server.py:609` → `adapter.py:1430` |
| OpenAI | `/v1/embeddings` | `http_server.py:614` → `adapter.py:1871` |
| OpenAI | `/v1/models` | `http_server.py:620` |
| OpenAI | `/v1/files`、`/v1/batches` | `http_server.py:636-674` → `adapter.py` |
| 健康 | `/health`、`/health_generate`、`/ping` | `http_server.py:154/160/678` |
| 信息 | `/get_model_info`、`/get_server_info` | `http_server.py:210/221` |
| 缓存 | `/flush_cache` | `http_server.py:325` |
| Profiling | `/start_profile`、`/stop_profile` | `http_server.py:336/355` |
| Expert 分布 | `/start|stop|dump_expert_distribution_record` | `http_server.py:365-392` |
| 权重热更新 | `/update_weights_from_disk` | `http_server.py:395` |
| 权重热更新 | `/init_weights_update_group` | `http_server.py:418` |
| 权重热更新 | `/update_weights_from_tensor` | `http_server.py:433` |
| 权重热更新 | `/update_weights_from_distributed` | `http_server.py:453` |
| 显存控制 | `/release_memory_occupation`、`/resume_memory_occupation` | `http_server.py:483/494` |
| 会话 | `/open_session`、`/close_session` | `http_server.py:518/532` |
| 中止 | `/abort_request` | `http_server.py:549` |
| 解析工具 | `/parse_function_call`、`/separate_reasoning` | `http_server.py:559/581` |
| 云平台 | `/invocations`(SageMaker)、`/vertex_generate`(Vertex) | `http_server.py:684/690` |

## 交叉引用

- 下游接收方与异步生成器细节：[TokenizerManager](11-tokenizer-manager.md)
- 请求如何被调度成 batch：[调度器](12-scheduler.md)
- 内部请求结构 `GenerateReqInput` / `EmbeddingReqInput` 字段含义：见 `python/sglang/srt/managers/io_struct.py`，并参考 [术语表](02-glossary.md)
- 多模态输入处理：[多模态](24-multimodal.md)
- 约束解码 / grammar backend（`json_schema`、`ebnf`、`structural_tag`）：[约束解码](17-sampling-structured-output.md)
