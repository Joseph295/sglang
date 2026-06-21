# SGLang 内部架构文档 · Internals Guide

> 这是一套**面向源码阅读者**的 SGLang Runtime（SRT）内部架构文档。它不是「怎么用 SGLang 跑模型」的用户手册，而是回答「SGLang 内部到底是怎么工作的」——一个请求从 HTTP 进来后，经过哪些进程、走哪些 ZMQ 管道、在调度器里如何被拼成连续批（continuous batch）、KV cache 怎么靠 RadixAttention 复用、前向/采样/解码各自发生在哪、以及投机解码 / PD 分离 / LoRA / 量化 / 多模态 / 专家并行这些高级特性是如何挂接到主链路上的。
>
> **给谁看**：想读懂 / 修改 / 调试 SGLang 后端的工程师与贡献者。每一章都尽量把概念落到 `file:line`，方便你边读文档边跳源码。
> **不覆盖**：前端 SGLang 语言（co-design 的前端部分）、部署运维、benchmark 结果——本文档只讲后端 runtime（`python/sglang/srt`）。

---

## 全局架构图（多进程 + ZMQ）

SGLang 把「HTTP 接入 + 分词」「调度 + 前向 + 采样」「解码」拆成**多个独立进程**，用 **ZMQ** 串成一条流水线，从而绕开 Python GIL、把 CPU 工作与 GPU 工作解耦（zero-overhead scheduler）。

```
                                  ┌──────────────────────────────────────────────┐
                                  │  主进程 (main process)                         │
   HTTP / OpenAI API              │  ┌────────────────────────────────────────┐  │
 ───────────────────────────────▶│  │ FastAPI / uvicorn  (http_server.py)     │  │
   POST /v1/chat/completions      │  │            │                            │  │
   POST /generate                 │  │            ▼                            │  │
                                  │  │ TokenizerManager (asyncio)              │  │
   token 流式回传 ◀───────────────│  │   - 分词 / 准入 / 请求编排                │  │
                                  │  │   - handle_loop 收回结果                 │  │
                                  │  └──────┬──────────────────────▲───────────┘  │
                                  └─────────┼──────────────────────┼──────────────┘
                                            │ ZMQ PUSH             │ ZMQ PULL
                                  TokenizedGenerateReqInput   BatchTokenIDOut
                                            │                      │
                                            ▼                      │
        ┌───────────────────────────────────────────────┐         │
        │  Scheduler 进程  (每个 TP rank 一个, scheduler.py)│        │
        │                                                 │         │
        │   recv_requests → schedule (continuous batch)   │         │
        │        │                                        │         │
        │        ▼                                        │         │
        │   ┌──────────────┐   prefix 复用   ┌──────────┐ │         │
        │   │ RadixCache / │◀───────────────▶│ KV Pool  │ │         │
        │   │ tree-match   │                 │ (paged)  │ │         │
        │   └──────┬───────┘                 └────┬─────┘ │         │
        │          ▼                              │       │         │
        │   ┌──────────────────────────────┐     │       │         │
        │   │ ModelRunner / ForwardBatch    │◀────┘       │         │
        │   │  + CUDA Graph + Attention BE  │             │         │
        │   └──────────────┬───────────────┘             │         │
        │                  ▼                              │         │
        │            Sampling (logits → token ids)        │         │
        │                  │                              │         │
        └──────────────────┼──────────────────────────────┘         │
                           │ ZMQ PUSH (BatchTokenIDOut)              │
                           ▼                                         │
        ┌──────────────────────────────────────────────┐           │
        │  DetokenizerManager 进程 (detokenizer_manager) │           │
        │   token ids → 文本 (incremental detokenize)    │───────────┘
        └──────────────────────────────────────────────┘  ZMQ PUSH → TokenizerManager

  分布式扩展: TP / PP / DP / EP 下会有多个 Scheduler 进程, 通过 NCCL 互联;
  PD 分离时 Prefill 与 Decode 是两组独立 server, 通过 KV 传输通道连接 (见 21 章)。
```

详细的进程拓扑、ZMQ 通道定义（`PortArgs`）与启动流程，见 [00-overview.md](00-overview.md)。

---

## 章节目录 · Table of Contents

### 一、总览与链路 (Overview & End-to-End)

| 章节 | 一句话简介 |
| --- | --- |
| [00-overview.md](00-overview.md) | **总览**：设计哲学、多进程架构、ZMQ 拓扑——先读这张全景图。 |
| [01-request-lifecycle.md](01-request-lifecycle.md) | **请求全生命周期**：一个请求从客户端发出到收到响应，逐进程追踪。 |
| [02-glossary.md](02-glossary.md) | **术语表**：prefill / decode / RadixAttention / TP / overlap 等专有名词的速查地图。 |

### 二、核心模块 (Core Modules)

| 章节 | 一句话简介 |
| --- | --- |
| [10-entrypoints-api.md](10-entrypoints-api.md) | **入口与 API 层**：HTTP server、OpenAI 兼容 API、Engine Python API。 |
| [11-tokenizer-manager.md](11-tokenizer-manager.md) | **TokenizerManager**：分词、准入控制与请求编排（主进程内的 asyncio 中枢）。 |
| [12-scheduler.md](12-scheduler.md) | **调度器（核心）**：连续批处理与 zero-overhead 调度——runtime 的心脏。 |
| [13-memcache-radixattention.md](13-memcache-radixattention.md) | **内存与 KV 缓存（核心）**：RadixAttention 前缀复用与 paged KV pool。 |
| [14-model-executor.md](14-model-executor.md) | **模型执行**：ModelRunner、ForwardBatch 与 CUDA Graph。 |
| [15-worker-parallelism.md](15-worker-parallelism.md) | **Worker 与并行**：TP / PP / DP / EP 与分布式互联。 |
| [16-layers-attention-moe.md](16-layers-attention-moe.md) | **计算层**：Attention 后端、MoE 与核心算子。 |
| [17-sampling-structured-output.md](17-sampling-structured-output.md) | **采样与结构化输出**：sampling 参数、FSM / Grammar 约束解码。 |
| [18-detokenizer-output.md](18-detokenizer-output.md) | **DetokenizerManager 与输出回流**：token ids → 文本 → 流式回传客户端。 |

### 三、高级特性 (Advanced Features)

| 章节 | 一句话简介 |
| --- | --- |
| [20-speculative-eagle.md](20-speculative-eagle.md) | **投机解码**：Speculative Decoding / EAGLE 草稿-验证流程。 |
| [21-pd-disaggregation.md](21-pd-disaggregation.md) | **PD 分离**：Prefill / Decode 解耦部署与 KV 传输。 |
| [22-lora.md](22-lora.md) | **Multi-LoRA 批处理**：多 adapter 在同一批里共存。 |
| [23-quantization.md](23-quantization.md) | **量化**：FP8 / INT4 / AWQ / GPTQ 权重与激活量化。 |
| [24-multimodal.md](24-multimodal.md) | **多模态（VLM）**：图像/视频等模态输入的处理与对齐。 |
| [25-eplb-expert-parallel.md](25-eplb-expert-parallel.md) | **EPLB**：专家并行（Expert Parallel）负载均衡。 |

### 四、附录 (Appendix)

| 章节 | 一句话简介 |
| --- | --- |
| [90-data-structures.md](90-data-structures.md) | **核心数据结构速查**：Req / ScheduleBatch / ForwardBatch 等关键类字段速查。 |
| [91-debugging-and-hacking.md](91-debugging-and-hacking.md) | **调试与魔改速查**：常见 bug 定位入口、日志/断点位置、改代码的安全边界。 |
| [99-guided-walkthrough.md](99-guided-walkthrough.md) | **配套导读（老师带读）**：口语化串讲主链路 + ★Insight，强调"为什么这么设计"，适合从头读一遍建立心智模型。 |

---

## 推荐阅读路径 · Reading Paths

### 路线 ① 小白上手（建立全局心智模型）

按顺序读，先建立「整条链路」的图景，再逐段深入：

```
00 总览  →  01 请求生命周期  →  02 术语表
        →  12 调度器(核心)   →  13 KV缓存/RadixAttention(核心)
        →  14 模型执行       →  17 采样与结构化输出  →  18 解码与输出回流
        →（按兴趣）20 投机解码 / 21 PD分离 / 24 多模态 ...
```

要点：00/01/02 给你「地图 + 路标 + 词汇表」；12/13 是 SGLang「为什么快」的核心，务必慢读；14/17/18 把前向、采样、解码三段串成完整一圈。高级特性章节都建立在这条主链路之上，可按需挑读。

### 路线 ② 修 bug（从现象直奔模块）

不要从头读。先翻速查表定位，再跳到对应模块章深入：

```
91 调试与魔改速查（定位现象 / 入口 / 日志）   →   90 核心数据结构速查（看清字段）
                          │
        ┌─────────────────┼───────────────────────────────────────────────┐
        ▼                 ▼                  ▼                  ▼            ▼
  请求卡住/不返回      OOM / KV 不足       结果乱码/重复        慢 / 吞吐低     特性相关 bug
        │                 │                  │                  │            │
        ▼                 ▼                  ▼                  ▼            ▼
   11 TokenizerMgr    13 KV/Radix        18 Detokenizer     12 调度器     20/21/22/
   12 调度器          12 调度器           17 采样            14 模型执行     23/24/25
   01 生命周期        90 数据结构         02 术语表          16 计算层      对应特性章
```

要点：先用 **91** 锁定「在哪个进程、哪一段链路出问题」，再用 **90** 对照数据结构字段，最后跳进对应核心模块章读实现细节。涉及并行/分布式的问题额外看 [15-worker-parallelism.md](15-worker-parallelism.md)。

---

## 约定 · Conventions

- 专有名词（prefill / decode / RadixAttention / TP / overlap scheduler 等）全程保留英文，定义见 [02-glossary.md](02-glossary.md)。
- 代码引用统一用 `file:line` 形式（如 `python/sglang/srt/managers/scheduler.py:2252`），便于直接跳转。
- **SRT = SGLang Runtime**，即本文档讨论的后端推理引擎。
