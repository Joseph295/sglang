# 术语表（Glossary）

> 本章是整套「SGLang 内部架构文档」的速查地图。每个术语给出：一句话定义 + 为什么需要它 + 在代码里第一手出现的位置（`file:line`）。
> 行号以撰写时仓库实际状态为准，源码迭代后可能略有偏移，但符号名稳定，可用 `grep` 复核。
> 阅读建议：先看下面的「端到端链路总览」建立全局直觉，再按需跳到具体术语。后续章节（[调度器](12-scheduler.md)、[KV 缓存](13-memcache-radixattention.md) 等）会展开每个子系统。

---

## 一句话职责

术语表把 SGLang 源码里反复出现、但散落在不同模块的核心概念集中解释清楚，让一个刚入门的工程师能在读到 `ForwardBatch`、`rid`、`RadixAttention`、`overlap scheduler` 这些词时立刻知道「它是什么、归谁管、去哪看代码」。

---

## 在端到端链路中的位置

下面这张图是全文档的「骨架」。每个术语都挂在某个阶段上，方括号里标注的是负责该阶段的进程 / 模块。

```
 客户端 HTTP/OpenAI 请求
        │
        ▼
 ┌──────────────────────────┐
 │ TokenizerManager         │  分词 + 分配 rid（请求唯一 id）
 │ (tokenizer_manager.py)   │  术语: rid, Req(雏形 GenerateReqInput)
 └──────────────┬───────────┘
                │ ZMQ
                ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Scheduler (scheduler.py)  —— 单 TP rank 一个进程          │
 │   continuous batching / chunked prefill / overlap          │
 │   术语: Req, ScheduleBatch, prefill, decode,               │
 │         continuous batching, chunked prefill,              │
 │         overlap scheduler                                  │
 │   ┌────────────────────────────────────────────────┐      │
 │   │ Prefix Cache: RadixCache (radix tree)           │      │
 │   │ 术语: radix tree prefix cache, RadixAttention   │      │
 │   └────────────────────────────────────────────────┘      │
 │   ┌────────────────────────────────────────────────┐      │
 │   │ KV Cache Pool: ReqToTokenPool +                 │      │
 │   │   TokenToKVPoolAllocator + MHA/MLA KVPool       │      │
 │   │ 术语: KV cache pool, paged KV cache,            │      │
 │   │       token attention, MLA                      │      │
 │   └────────────────────────────────────────────────┘      │
 └──────────────┬─────────────────────────────────────────────┘
                │ ModelWorkerBatch
                ▼
 ┌──────────────────────────────────────────────────────────┐
 │ TpModelWorker(Client) (tp_worker*.py)                      │
 │   术语: TP/PP/DP/EP, DP attention, overlap (future ids)    │
 │        │                                                    │
 │        ▼                                                    │
 │   ModelRunner → ForwardBatch (forward_batch_info.py)       │
 │   术语: ForwardBatch, CUDA graph,                          │
 │        speculative decoding / EAGLE, LoRA, quantization    │
 │        │                                                    │
 │        ▼                                                    │
 │   model.forward → LogitsProcessor → Sampler                │
 │   术语: logits processor, sampler,                         │
 │        structured output / FSM / grammar                   │
 └──────────────┬─────────────────────────────────────────────┘
                │ next token ids
                ▼
 ┌──────────────────────────┐
 │ DetokenizerManager       │  增量解码 token → 文本
 │ (detokenizer_manager.py) │
 └──────────────┬───────────┘
                ▼
            响应客户端

 ── 横切（cross-cutting）──
 PD disaggregation: 把 prefill 与 decode 拆到不同实例 (disaggregation/)
```

---

## 关键文件与类（索引表）

| 术语 / 符号 | file:line | 作用 |
| --- | --- | --- |
| `rid` | `python/sglang/srt/managers/io_struct.py:205` | 请求唯一 id（`uuid4().hex`） |
| `Req` | `python/sglang/srt/managers/schedule_batch.py:420` | 一个请求在 Scheduler 内的运行时对象 |
| `ScheduleBatch` | `python/sglang/srt/managers/schedule_batch.py:787` | 一次调度迭代要跑的一批 `Req` |
| `ModelWorkerBatch` | `python/sglang/srt/managers/schedule_batch.py:1705` | 喂给 worker 的轻量化 batch |
| `ForwardMode` | `python/sglang/srt/model_executor/forward_batch_info.py:53` | EXTEND/DECODE/MIXED/IDLE 等模式枚举 |
| `ForwardBatch` | `python/sglang/srt/model_executor/forward_batch_info.py:138` | 模型前向所需的全部张量 |
| `Scheduler` | `python/sglang/srt/managers/scheduler.py:176` | 调度核心（continuous batching） |
| `RadixAttention` | `python/sglang/srt/layers/radix_attention.py:38` | 注意力算子层（接 backend） |
| `RadixCache` / `TreeNode` | `python/sglang/srt/mem_cache/radix_cache.py:98` / `:44` | radix tree 前缀缓存 |
| `ChunkCache` | `python/sglang/srt/mem_cache/chunk_cache.py:22` | 关闭 radix cache 时的退化缓存 |
| `ReqToTokenPool` | `python/sglang/srt/mem_cache/memory_pool.py:54` | req → token KV 槽位映射 |
| `TokenToKVPoolAllocator` | `python/sglang/srt/mem_cache/memory_pool.py:169` | KV 槽位分配器 |
| `MHATokenToKVPool` | `python/sglang/srt/mem_cache/memory_pool.py:238` | 标准 MHA/GQA 的 KV pool |
| `MLATokenToKVPool` | `python/sglang/srt/mem_cache/memory_pool.py:511` | DeepSeek MLA 的 KV pool |
| `PagedTokenToKVPoolAllocator` | `python/sglang/srt/mem_cache/paged_allocator.py` | page_size>1 的分页分配器 |
| `Sampler` | `python/sglang/srt/layers/sampler.py:29` | 从 logits 采样 next token |
| `LogitsProcessor` | `python/sglang/srt/layers/logits_processor.py:208` | hidden → logits |
| `BaseGrammarBackend` | `python/sglang/srt/constrained/base_grammar_backend.py:108` | 结构化输出语法后端抽象 |
| `XGrammarGrammarBackend` | `python/sglang/srt/constrained/xgrammar_backend.py:147` | 默认 grammar 后端 |
| `CudaGraphRunner` | `python/sglang/srt/model_executor/cuda_graph_runner.py:185` | decode 阶段 CUDA graph 回放 |
| `TpModelWorkerClient` | `python/sglang/srt/managers/tp_worker_overlap_thread.py:51` | overlap scheduler 的 worker 端 |
| `EAGLEWorker` | `python/sglang/srt/speculative/eagle_worker.py:54` | 投机解码（EAGLE）worker |
| `initialize_dp_attention` | `python/sglang/srt/layers/dp_attention.py:61` | DP attention 初始化 |
| `ServerArgs` | `python/sglang/srt/server_args.py:48` | 全部启动参数（很多术语的开关在此） |

---

## 核心术语（按主题分组）

### 一、请求与批的生命周期

#### `rid`（request id）
**一句话**：每个请求的全局唯一字符串 id，贯穿 TokenizerManager → Scheduler → DetokenizerManager，用来在多进程之间对齐同一个请求。

**实现**：在 `GenerateReqInput.normalize_batch_and_arguments` 里若未指定就生成 `uuid.uuid4().hex`，见 `python/sglang/srt/managers/io_struct.py:205`。当一个 batch 输入被拆成多个子请求、或重试时会调用 `regenerate_rid()`（`io_struct.py:382`、`:549`）。`Req` 对象把它存为 `self.rid`（`schedule_batch.py:445`）。

**坑**：`rid` 是跨进程关联的唯一钥匙。如果你在 batch 拆分逻辑里忘了 `regenerate_rid`，会出现两个请求共用一个 rid，导致输出串台。调试串台/丢响应类 bug，第一步就是按 rid 串起三个进程的日志。

#### `Req`
**一句话**：一个请求在 Scheduler 进程内的「活体」对象，保存输入 token、已生成 token、采样参数、前缀缓存命中信息、KV 槽位索引等全部运行时状态。

**实现**：`python/sglang/srt/managers/schedule_batch.py:420`。关注字段：`origin_input_ids`、`output_ids`、`fill_ids`、`prefix_indices`（前缀缓存命中部分）、`req_pool_idx`（在 `ReqToTokenPool` 中的行号）。`adjust_max_prefix_ids()`（约 `schedule_batch.py:640`）参与前缀匹配。

#### `ScheduleBatch` / `ModelWorkerBatch`
**一句话**：`ScheduleBatch` 是 Scheduler 一次迭代选出来要一起跑的若干 `Req` 的集合（重对象，含调度元数据）；`ModelWorkerBatch` 是把它「瘦身」成只含张量的版本，发给 worker。

**实现**：`ScheduleBatch` 在 `schedule_batch.py:787`，承载 prefill/decode 两类工作；`ModelWorkerBatch` 在 `schedule_batch.py:1705`。两者的关系是「调度视角」对「执行视角」。详见 [调度器](12-scheduler.md)。

#### `ForwardBatch` / `ForwardMode`
**一句话**：`ForwardBatch` 是模型前向真正消费的对象——把 `ModelWorkerBatch` 转成 GPU 张量（input_ids、positions、attention 元数据、KV pool 句柄等）；`ForwardMode` 标记这一批是 prefill（`EXTEND`）、decode（`DECODE`）还是混合（`MIXED`）。

**实现**：`ForwardMode` 枚举在 `python/sglang/srt/model_executor/forward_batch_info.py:53`（`EXTEND=56`、`DECODE=58`、还有 `DRAFT_EXTEND`、`IDLE` 等）。`ForwardBatch` 在同文件 `:138`。`is_extend()` / `is_decode()` 等谓词（`:78` 起）在 attention backend 里被大量分支判断。详见 [前向执行](14-model-executor.md)。

---

### 二、prefill / decode 与批处理策略

#### prefill / decode
**一句话**：prefill 是「一次性处理整段 prompt、把 KV 写满」的阶段（计算密集、并行度高）；decode 是「逐 token 自回归生成」的阶段（访存密集、每步只算 1 个新 token）。

**为什么分**：两个阶段的硬件特征截然不同，调度、batching、CUDA graph 策略都要区别对待。代码里对应 `ForwardMode.EXTEND` 与 `ForwardMode.DECODE`（`forward_batch_info.py:56/58`）。Scheduler 的 `get_new_batch_prefill()`（`scheduler.py:1342`）专门组 prefill 批，decode 批则在 `get_next_batch_to_run()`（`scheduler.py:1286`）里滚动推进。

#### continuous batching（连续批处理）
**一句话**：不等整批请求都结束就动态地把新请求加入、把已完成请求移出，使 GPU 始终满载——这是现代 LLM serving 的基础调度范式。

**实现**：体现在 Scheduler 的事件循环 `event_loop_normal()`（`scheduler.py:636`）/ `event_loop_overlap()`（`scheduler.py:656`）每次迭代都重新 `get_next_batch_to_run()` 决定下一批跑什么。详见 [调度器](12-scheduler.md)。

#### chunked prefill（分块预填充）
**一句话**：把超长 prompt 的 prefill 切成固定大小的 chunk 分多次跑，避免一个长 prompt 独占一整步、卡住其它请求的 decode。

**实现**：开关是 `ServerArgs.chunked_prefill_size`（`server_args.py:72`），默认值在 `__post_init__` 里根据显存/PD 模式确定（`server_args.py:271-278`：一般 8192，PD 模式 16384，小显存 2048）。Scheduler 把它存为 `self.chunked_prefill_size`（`scheduler.py:367`），并在 `get_new_batch_prefill` 中传给 prefill 调度策略（`scheduler.py:1378`）。注意：开启 DP attention 时它会被除以 `dp_size`（`server_args.py:340`）。`-1` 表示关闭。混合 chunk（`enable_mixed_chunk`）允许 prefill chunk 与 decode 同批，见 `scheduler.py:372`。

---

### 三、注意力与前缀缓存

#### RadixAttention
**一句话**：SGLang 的注意力算子层封装，名字源于它天然配合 radix tree 前缀缓存工作；它本身不实现 kernel，而是把请求路由到选定的 attention backend（FlashInfer / FlashAttention / Triton / MLA 等）。

**实现**：`python/sglang/srt/layers/radix_attention.py:38`，`class RadixAttention(nn.Module)`。模型里的每个 attention 层都持有一个它。真正的 kernel 在 `python/sglang/srt/layers/attention/*`（如 `flashinfer_backend.py`、`triton_backend.py`、`flashinfer_mla_backend.py`）。详见 [注意力后端](16-layers-attention-moe.md)。

#### radix tree prefix cache（基数树前缀缓存）
**一句话**：用一棵基数树（radix tree）按 token 序列做前缀去重——多个请求共享相同前缀的 KV，从而跳过重复 prefill，是 SGLang 的招牌特性。

**实现**：`RadixCache` 在 `python/sglang/srt/mem_cache/radix_cache.py:98`，节点 `TreeNode` 在 `:44`。核心方法 `match_prefix()`（`:138`，返回命中的 KV indices 和匹配节点）和 `insert()`（`:170`）。命中结果写入 `Req.prefix_indices`，prefill 时只算未命中部分。关掉它时退化为 `ChunkCache`（`chunk_cache.py:22`），即不跨请求共享。`HiRadixCache`（`hiradix_cache.py`）是带 CPU/分层卸载的扩展。详见 [前缀缓存](13-memcache-radixattention.md)。

**坑**：前缀匹配粒度受 `page_size` 约束（`server_args.py` 中 `chunked_prefill_size % page_size == 0` 的断言，`server_args.py:278`）。改 radix cache 逻辑时务必同时考虑 `page_size>1` 的分页对齐。

#### paged KV cache / token attention
**一句话**：把 KV cache 切成固定大小的「页」（page）统一管理，按需分配/回收，避免为每个序列预留连续大块显存；当 `page_size==1` 时退化为「token attention」——以单 token 为粒度分配 KV 槽位。

**实现**：`page_size` 是 `ServerArgs` 的参数；分页分配走 `PagedTokenToKVPoolAllocator`（`python/sglang/srt/mem_cache/paged_allocator.py`，Triton kernel 里以 `page_size` 为 `tl.constexpr`，见该文件 `:37` 起的索引计算），非分页（token 粒度）走 `TokenToKVPoolAllocator`（`memory_pool.py:169`）。

#### KV cache pool
**一句话**：管理「物理 KV 显存」与「逻辑槽位」的两级结构：`ReqToTokenPool` 记录每个请求用了哪些 token 槽位，`TokenToKVPool(Allocator)` 真正持有每层每 token 的 K/V 张量。

**实现**：
- `ReqToTokenPool`（`memory_pool.py:54`）：二维表 `[max_reqs, max_context_len]`，行号即 `Req.req_pool_idx`。
- `TokenToKVPoolAllocator`（`memory_pool.py:169`）：管理空闲 token 槽位。
- `MHATokenToKVPool`（`memory_pool.py:238`）：标准 MHA/GQA，每层存独立 K、V。
- `MLATokenToKVPool`（`memory_pool.py:511`）：MLA 只存压缩后的 latent（见下文 MLA），显存占用大幅降低。
- Host 侧（CPU 卸载）：`MHATokenToKVPoolHost`（`:935`）、`MLATokenToKVPoolHost`（`:1020`）。

详见 [KV 缓存内存池](13-memcache-radixattention.md)。

---

### 四、并行策略

#### TP / PP / DP / EP
**一句话**：四种把模型/请求切到多卡的方式——
- **TP（tensor parallelism）**：把单层权重矩阵按维度切到多卡，每步都要 all-reduce。开关 `tp_size`（`server_args.py:80`）。
- **PP（pipeline parallelism）**：把模型按层切成多段流水线。开关 `pp_size`（`server_args.py:81`）。
- **DP（data parallelism）**：复制整模型、不同卡处理不同请求。开关 `dp_size`（`server_args.py:115`），由 `data_parallel_controller.py` 协调。
- **EP（expert parallelism）**：MoE 模型里把 experts 分到不同卡。开关 `ep_size`（`server_args.py:119`）/ `enable_ep_moe`（`server_args.py:170`）；开启时 `ep_size` 会被强制等于 `tp_size`（`server_args.py:225-228`、`:360`）。

**实现**：分布式组初始化在 `python/sglang/srt/distributed/` 与 `model_parallel.py`。`tp_size * pp_size` 构成总并行规模（`server_args.py:248`）。详见 [并行与分布式](15-worker-parallelism.md)。

#### DP attention
**一句话**：一种针对 MLA 模型的特殊并行——attention 部分按 data parallel 切（每个 DP rank 独立维护自己那批请求的 KV，避免 KV 被 TP 复制 N 份），而 MoE/FFN 部分仍按 TP/EP 切。

**为什么**：MLA 的 KV 已经很小，再被 TP 复制就浪费；DP attention 让每个 attn rank 只存自己的 KV，显存效率更高。开关 `enable_dp_attention`（`server_args.py:168`），要求 `dp_size>1` 且 `tp_size % dp_size == 0`（`server_args.py:337-339`）。

**实现**：`python/sglang/srt/layers/dp_attention.py`：`initialize_dp_attention()`（`:61`）建立 attention 子 TP 组；`attn_tp_size = tp_size // dp_size`（`:37`）。开启后 `chunked_prefill_size` 会被除以 `dp_size`（`server_args.py:340`，注释解释是为避免 MoE kernel 问题）。详见 [DP attention](15-worker-parallelism.md)。

#### MLA（Multi-head Latent Attention）
**一句话**：DeepSeek 系列用的注意力变体——把 KV 压缩成一个低秩 latent 向量缓存，极大降低 KV cache 显存，是 DeepSeek-V2/V3 能长上下文的关键。

**实现**：模型侧 `DeepseekV2AttentionMLA`（`python/sglang/srt/models/deepseek_v2.py:457`）；KV pool 侧 `MLATokenToKVPool`（`memory_pool.py:511`）；专用 attention backend `flashinfer_mla_backend.py` / `flashmla_backend.py` / `cutlass_mla_backend.py`。MLA 与 DP attention 常配合使用。

---

### 五、性能优化（执行层）

#### overlap scheduler / zero-overhead scheduler
**一句话**：把 CPU 侧调度开销（组 batch、采样后处理）与 GPU 前向计算重叠起来，让 GPU 几乎不空等——通过「先用占位的 future token id 推进调度，等 GPU 算完再回填真实 token」实现，所以又叫 zero-overhead scheduler。

**实现**：worker 端是 `TpModelWorkerClient`（`python/sglang/srt/managers/tp_worker_overlap_thread.py:51`），它在独立线程 `forward_thread_func`（`:115`）里跑前向；用 `future_token_ids_map`（`:74`）保存占位 id，`resolve_future_token_ids()`（`:43`）在下一步把负数占位换成真实 token id。Scheduler 侧对应 `event_loop_overlap()`（`scheduler.py:656`）。开关：`disable_overlap_schedule`（见 `server_args.py`）。详见 [Overlap 调度](12-scheduler.md)。

**坑**：future token id 用「负数索引」编码占位（`resolve_future_token_ids` 里 `torch.clamp(-input_ids, min=0)`），改采样/约束解码逻辑时要小心别把占位 id 当真实 token。投机解码与 overlap 组合是 bug 高发区。

#### CUDA graph
**一句话**：把 decode 阶段固定形状的前向计算录制成一张 CUDA graph，之后每步直接 replay，省掉逐 kernel launch 的 CPU 开销——对小 batch decode 提速明显。

**实现**：`CudaGraphRunner`（`python/sglang/srt/model_executor/cuda_graph_runner.py:185`）。为有限的若干 batch size 各录一张图（`cuda_graph_bs` / `cuda_graph_max_bs`，`server_args.py:186-187`），运行时把实际 batch padding 到最近的录制尺寸（`disable_cuda_graph_padding`，`server_args.py:161`）。`disable_cuda_graph`（`server_args.py:160`）关闭。`cuda_graph_max_bs` 默认值随 `tp_size` 调整（`server_args.py:298-304`）。详见 [CUDA graph](14-model-executor.md)。

**坑**：CUDA graph 要求输入张量地址/形状固定，凡是动态形状（变长 prefill、LoRA 切换、grammar mask）都不能直接进图。新增需要随 batch 变化的张量时，要么纳入图的静态 buffer，要么走非 graph 路径。

#### speculative decoding / EAGLE
**一句话**：用一个小的 draft model 一次猜多个 token，再让大模型一次性并行 verify，命中就白赚——EAGLE 是 SGLang 主推的投机解码算法（draft 复用大模型的 hidden state）。

**实现**：`EAGLEWorker`（`python/sglang/srt/speculative/eagle_worker.py:54`）继承 `TpModelWorker`，核心方法 `draft()`（`:320`）生成候选 token 树、`verify()`（`:491`）用目标模型验证。投机树构建在 `build_eagle_tree.py`，draft 也有专用 CUDA graph（`eagle_draft_cuda_graph_runner.py`）。开关 `speculative_algorithm`（`server_args.py:141`，取值如 `EAGLE`、`EAGLE3`）。`ForwardMode` 里的 `DRAFT_EXTEND`（`forward_batch_info.py:67`）就是为它服务。详见 [投机解码](20-speculative-eagle.md)。

---

### 六、横切能力

#### PD disaggregation（prefill/decode 分离部署）
**一句话**：把 prefill 和 decode 拆到不同的实例（甚至不同机器）上跑，各自按自身负载特征独立扩缩容，再通过 KV 传输把 prefill 产出的 KV 送给 decode 实例。

**实现**：`python/sglang/srt/disaggregation/`：`prefill.py`、`decode.py`、负载均衡 `mini_lb.py`，KV 传输后端 `mooncake/`、`nixl/`。开关 `disaggregation_mode`（`server_args.py:217`，取值 `null`/`prefill`/`decode`）。该模式下 `chunked_prefill_size` 默认更大（16384，`server_args.py:274-275`），`Req` 里有 `bootstrap_room` 字段用于配对（见 `schedule_batch.py:767`）。详见 [PD 分离](21-pd-disaggregation.md)。

#### LoRA
**一句话**：低秩适配器（Low-Rank Adaptation），在不改原权重的前提下叠加小的 A/B 低秩矩阵实现微调；SGLang 支持同时加载多个 LoRA 并按请求路由。

**实现**：`python/sglang/srt/lora/`：`lora_manager.py`（加载/切换）、`layers.py`（注入到 linear 层）、`mem_pool.py`（LoRA 权重显存池）、`triton_ops/`（融合 kernel）。开关 `lora_paths`（`server_args.py:131`）。详见 [LoRA](22-lora.md)。

**坑**：多 LoRA + CUDA graph + 动态 batch 的组合在 padding/路由上容易出错。

#### quantization（FP8 / INT4 / AWQ / GPTQ）
**一句话**：把权重（有时含激活/ KV）量化到低比特以省显存、提吞吐。FP8 多为运行时/权重量化；AWQ、GPTQ 是 4-bit 权重量化算法（需预量化模型）；INT8/INT4 是位宽。

**实现**：`python/sglang/srt/layers/quantization/`：`fp8.py`、`awq.py`、`gptq.py`、`w8a8_int8.py`、`blockwise_int8.py`、`modelopt_quant.py`、`compressed_tensors/` 等。统一开关 `quantization`（`server_args.py:53`），可选 `quantization_param_path`（`:54`）。KV cache 量化见 `quantization/kv_cache.py`。详见 [量化](23-quantization.md)。

#### structured output / FSM / grammar
**一句话**：约束模型只能生成符合某种格式（JSON schema / regex / EBNF）的输出——做法是把「下一步允许哪些 token」编译成一个状态机（FSM）/ grammar，每步对 logits 做 mask。

**实现**：`python/sglang/srt/constrained/`：抽象基类 `BaseGrammarBackend`（`base_grammar_backend.py:108`），默认后端 `XGrammarGrammarBackend`（`xgrammar_backend.py:147`），另有 `outlines_backend.py`、`llguidance_backend.py`。Outlines 路径用 FSM（`outlines_jump_forward.py:72` 的 `FSMInfo`，并支持 jump-forward 跳过确定段）。开关 `grammar_backend`（`server_args.py:138`，默认 `xgrammar`，见 `server_args.py:324-325`）。grammar 生成的 mask 在 Sampler 之前作用于 logits。详见 [结构化输出](17-sampling-structured-output.md)。

---

### 七、输出层

#### logits processor
**一句话**：模型最后一层 hidden state → 词表 logits 的转换层，并处理「只取最后一个 token 还是全部位置」「是否返回 logprobs」等逻辑。

**实现**：`LogitsProcessor`（`python/sglang/srt/layers/logits_processor.py:208`），输入元数据 `LogitsMetadata`（`:93`），输出 `LogitsProcessorOutput`（`:63`，含 `next_token_logits`、可选的 logprobs 等）。详见 [采样与 logits](17-sampling-structured-output.md)。

> 注意：此处的 "logits processor" 指 SGLang 内部的网络层，不要与 HuggingFace `LogitsProcessor`（采样前钩子）混淆——后者那类 per-token 约束在 SGLang 走 grammar / sampling 参数路径。

#### sampler
**一句话**：拿到 logits 后按采样参数（temperature、top_p、top_k、min_p 等）抽出 next token id 的组件。

**实现**：`Sampler`（`python/sglang/srt/layers/sampler.py:29`），`forward()`（`:38`）。采样参数封装在 `python/sglang/srt/sampling/`（`SamplingBatchInfo` 等）。grammar mask 在采样前应用。详见 [采样与 logits](17-sampling-structured-output.md)。

---

## 设计难点与权衡（贯穿全表）

1. **三套 batch 抽象（ScheduleBatch / ModelWorkerBatch / ForwardBatch）不是冗余**：它们对应「调度视角 → 传输视角 → 计算视角」三层关注点分离。调度只关心 `Req` 优先级与显存，传输只想要张量，前向只想要 GPU buffer。改一处数据流，往往要顺着这三层都改。

2. **prefill 与 decode 的二元性渗透到每一层**：`ForwardMode`、KV pool 分配、CUDA graph（只录 decode）、chunked prefill、PD 分离——几乎所有性能特性都建立在「这两个阶段必须区别对待」之上。读任何 attention backend 代码，先看它怎么分 `is_extend()` / `is_decode()`。

3. **overlap + CUDA graph + 投机解码 三者叠加是复杂度峰值**：future token id 占位、固定形状要求、draft 树验证三种约束互相牵制，是历史 bug 最集中的区域。改这块务必小批量、关掉其它优化逐个对拍。

4. **page_size 是个隐藏的全局不变量**：radix cache 前缀对齐、chunked prefill size、paged allocator 都依赖它。`server_args.py:278` 的断言就是在守护这个不变量。

---

## 改代码 / 修 bug 指南（按术语索引）

| 想改 / 排查 | 入手位置 |
| --- | --- |
| 请求串台、丢响应 | 按 `rid` 串日志；检查 `regenerate_rid`（`io_struct.py:382`） |
| 调度卡顿、长 prompt 饿死短请求 | `chunked_prefill_size`（`server_args.py:72`）、`get_new_batch_prefill`（`scheduler.py:1342`） |
| 前缀缓存没命中 / 命中错 | `RadixCache.match_prefix`（`radix_cache.py:138`），注意 `page_size` 对齐 |
| KV 显存 OOM | KV pool（`memory_pool.py:238/511`）、`page_size`、`mem_fraction_static` |
| 加新 attention backend | 实现 `base_attn_backend.py` 接口，接到 `RadixAttention`（`radix_attention.py:38`） |
| decode 慢 / CPU 瓶颈 | 确认 overlap（`tp_worker_overlap_thread.py`）与 CUDA graph（`cuda_graph_runner.py:185`）是否开 |
| CUDA graph OOM / 形状报错 | `cuda_graph_bs` / `cuda_graph_max_bs`（`server_args.py:186-187`），动态张量是否进图 |
| 投机解码命中率/崩溃 | `EAGLEWorker.draft`（`eagle_worker.py:320`）/`verify`（`:491`） |
| 结构化输出不生效 | `grammar_backend`（`server_args.py:138`）、mask 是否在 Sampler 前作用 |
| 采样结果异常 | `Sampler.forward`（`sampler.py:38`）、`SamplingBatchInfo`（`sampling/`） |
| MoE/DP attention 报错 | `enable_dp_attention`（`server_args.py:168`）、`initialize_dp_attention`（`dp_attention.py:61`） |
| 量化模型加载失败 | `layers/quantization/<方法>.py`、`quantization` 参数（`server_args.py:53`） |

**通用调试入手点**：
- 几乎所有行为开关都在 `python/sglang/srt/server_args.py` 的 `ServerArgs`（`:48` 起）及其 `__post_init__`（约 `:220`）——先 `grep` 参数名定位它影响了哪些子系统。
- 多进程问题先确认是哪个进程：TokenizerManager / Scheduler / DetokenizerManager / Worker thread。
- 性能问题先二分关掉 overlap、CUDA graph、投机解码，定位是哪一层引入的。

---

## 交叉引用

- [调度器](12-scheduler.md) · [Overlap 调度](12-scheduler.md)
- [KV 缓存内存池](13-memcache-radixattention.md) · [前缀缓存](13-memcache-radixattention.md)
- [前向执行](14-model-executor.md) · [注意力后端](16-layers-attention-moe.md) · [CUDA graph](14-model-executor.md) · [采样与 logits](17-sampling-structured-output.md)
- [并行与分布式](15-worker-parallelism.md) · [DP attention](15-worker-parallelism.md)
- [投机解码](20-speculative-eagle.md) · [PD 分离](21-pd-disaggregation.md) · [LoRA](22-lora.md) · [量化](23-quantization.md) · [结构化输出](17-sampling-structured-output.md)
