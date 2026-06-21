# 计算层：Attention 后端、MoE 与核心算子

> 本章对应源码目录 `python/sglang/srt/layers/`。这里是「模型前向真正发生计算」的地方：每个 Transformer block 里的 attention、MoE/FFN、归一化、激活、RoPE、最后的 lm_head 投影，都在这一层落地成 kernel。读懂这一层，你才能改一个 attention backend、修一个 MoE 路由 bug、或者接一个新的量化方法。

## 1. 一句话职责

`layers/` 提供模型前向所需的全部「可替换算子」：用 `RadixAttention` 把 KV pool 接进各种 attention kernel（FlashInfer / Triton / FA3 / Torch native 等），用 `FusedMoE` / `EPMoE` 实现专家路由与计算，用 `linear.py` / `quantization/` 提供（可量化的）线性层，用 `LogitsProcessor` 把最后一层 hidden states 变成 logits，再加上 RoPE、LayerNorm、激活、embedding 等基础算子。

模型定义文件（`python/sglang/srt/models/*.py`）只负责「搭积木」——调用这些 layer；真正的硬件相关计算和 backend 切换全部封装在这一层。

## 2. 在端到端链路中的位置

```
客户端 → 分词(TokenizerManager) → 调度(Scheduler) → KV缓存(RadixCache/KV pool)
                                                              │
                                                              ▼
                                  ┌─────────── 前向 forward ───────────┐
                                  │  ModelRunner.forward(forward_batch) │
                                  │     │                               │
                                  │     ▼  (逐层 nn.Module.forward)      │
                                  │  ┌───────────── 本章 ─────────────┐ │
                                  │  │ Embedding (vocab_parallel_*)   │ │
                                  │  │ for each layer:                │ │
                                  │  │   LayerNorm (layernorm.py)     │ │
                                  │  │   QKV proj  (linear.py)        │ │
                                  │  │   RoPE      (rotary_embedding) │ │
                                  │  │   RadixAttention ──► attn_backend──► KV pool
                                  │  │   O proj    (linear.py)        │ │
                                  │  │   LayerNorm                    │ │
                                  │  │   MLP / FusedMoE / EPMoE       │ │
                                  │  │     └─ select_experts(topk.py) │ │
                                  │  │     └─ activation.py           │ │
                                  │  │ final LayerNorm                │ │
                                  │  │ LogitsProcessor ──► logits     │ │
                                  │  └────────────────────────────────┘ │
                                  └─────────────────│──────────────────┘
                                                    ▼
                          采样(Sampler) → 解码(detokenize) → 响应
```

本章是「前向」框里的全部内容。上游是 [model-executor](14-model-executor.md) 构造的 `ForwardBatch` 与 attention backend；下游是 Sampler（见 [model-executor](14-model-executor.md) 中的采样部分）。KV pool 与 RadixCache 见 [memcache-radixattention](13-memcache-radixattention.md)。

## 3. 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `RadixAttention` | `python/sglang/srt/layers/radix_attention.py:38` | 模型里实例化的 attention layer，持有 head 配置，`forward` 把 q/k/v 委托给 backend |
| `AttentionBackend` | `python/sglang/srt/layers/attention/base_attn_backend.py:14` | 所有 attention backend 的抽象基类，定义 `init_forward_metadata` / `forward_extend` / `forward_decode` / cuda graph 接口 |
| `TorchNativeAttnBackend` | `python/sglang/srt/layers/attention/torch_native_backend.py:17` | 最易读的参考实现，用 `scaled_dot_product_attention` 逐 seq 计算 |
| `TritonAttnBackend` | `python/sglang/srt/layers/attention/triton_backend.py:88` | 默认 fallback backend，自带 Triton extend/decode kernel |
| `FlashInferAttnBackend` | `python/sglang/srt/layers/attention/flashinfer_backend.py` | MHA 模型的高性能 backend（非 Hopper 默认） |
| `FlashAttentionBackend` (fa3) | `python/sglang/srt/layers/attention/flashattention_backend.py` | Hopper 上的默认 backend |
| `ModelRunner.init_attention_backend` | `python/sglang/srt/model_executor/model_runner.py:990` | 根据 `server_args.attention_backend` 字符串实例化 backend |
| `ModelRunner.model_specific_adjustment` | `python/sglang/srt/model_executor/model_runner.py:313` | backend 未指定时自动选择（fa3 / flashinfer / triton / aiter） |
| `FusedMoE` | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:258` | TP 版 MoE：所有专家权重在每个 rank 全量持有，靠 `fused_experts` 算 |
| `EPMoE` | `python/sglang/srt/layers/moe/ep_moe/layer.py:131` | Expert Parallel 版 MoE：每个 rank 只持有部分专家 |
| `DeepEPMoE` | `python/sglang/srt/layers/moe/ep_moe/layer.py:820` | 基于 DeepEP 的低延迟/大规模 EP 实现 |
| `select_experts` | `python/sglang/srt/layers/moe/topk.py:297` | MoE 路由的统一入口：分发到 grouped/biased/fused/native 各种 topk |
| `LogitsProcessor` | `python/sglang/srt/layers/logits_processor.py:208` | 从 hidden states 产出 `next_token_logits` 与（可选）input logprobs |
| `LinearMethodBase` | `python/sglang/srt/layers/linear.py:102` | 线性层量化方法抽象；`UnquantizedLinearMethod` 是默认实现 |
| `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` | `python/sglang/srt/layers/linear.py:299` / `:1130` / `:750` | TP 切分的线性层 |
| `RMSNorm` / `GemmaRMSNorm` | `python/sglang/srt/layers/layernorm.py:42` / `:110` | 归一化算子，`CustomOp` 多后端分发 |
| `SiluAndMul` / `GeluAndMul` | `python/sglang/srt/layers/activation.py:41` / `:54` | 融合激活算子 |
| `get_rope` | `python/sglang/srt/layers/rotary_embedding.py:1064` | RoPE 工厂，按配置缓存返回 `RotaryEmbedding` 子类 |
| `VocabParallelEmbedding` / `ParallelLMHead` | `python/sglang/srt/layers/vocab_parallel_embedding.py:175` / `:510` | 词表并行 embedding 与 lm_head |
| `Pooler` | `python/sglang/srt/layers/pooler.py:23` | embedding 模型的池化（LAST / CLS） |

## 4. 核心数据流 / 执行流程

### 4.1 RadixAttention：layer 如何把 KV pool 接进 kernel

模型里每个注意力层持有一个 `RadixAttention(nn.Module)` 实例。它本身几乎不算东西——只保存 head 数、`head_dim`、`scaling`、`layer_id`、`sliding_window_size` 等配置，关键是构造时拿到的 `layer_id`（`radix_attention.py:67`），这是它在 KV pool 里的「行号」。

`RadixAttention.forward`（`radix_attention.py:83`）做两件事：

1. 把扁平的 k/v reshape 成 `[-1, num_kv_heads, head_dim]`（`radix_attention.py:96-99`）；
2. **不自己算注意力**，而是转发给 `forward_batch.attn_backend.forward(...)`（`radix_attention.py:101`）。

注意：`attn_backend` 来自 `ForwardBatch`，是全局唯一的 backend 实例，而不是每层一个。所有层共享同一个 backend，靠传入的 `layer`（即 `self`）里的 `layer_id` 区分访问哪一层 KV。

`AttentionBackend.forward`（`base_attn_backend.py:57`）根据 `forward_batch.forward_mode` 二分：

```
forward_mode.is_decode()  → forward_decode(...)   # 每个 seq 只有 1 个新 token
else                      → forward_extend(...)   # prefill / chunked prefill / 投机验证
```

KV pool 的接入点在每个 backend 的 `forward_extend` / `forward_decode` 里，模式高度一致（以 Triton 为例，`triton_backend.py:515`）：

```
# 1. 把本层新算出的 k,v 写进 KV pool 的第 layer_id 行
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)

# 2. 调 kernel，kernel 直接读 KV pool 的连续 buffer
self.extend_attention_fwd(
    q, k, v, o,
    forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),   # ← KV pool
    forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
    ...,
    self.forward_metadata.kv_indptr,    # ← 哪些 token 属于哪个 seq
    self.forward_metadata.kv_indices,   # ← 每个 token 在 pool 里的物理位置
    ...,
)
```

这里有两条线：
- **写线**：`set_kv_buffer(layer, out_cache_loc, k, v)`（`memory_pool.py:378`）按 `out_cache_loc`（调度器分配的物理 slot）把 k/v 写进 `k_buffer[layer_id]` / `v_buffer[layer_id]`。
- **读线**：kernel 通过 `get_key_buffer(layer_id)`（`memory_pool.py:359`）拿到整个 layer 的连续 buffer，再用 `kv_indices`（每个 token 的物理偏移）gather 出本 batch 需要的 KV。

`kv_indices` 是 backend 在 `init_forward_metadata` 里一次性构建好的。Triton backend 用 `create_flashinfer_kv_indices_triton` kernel，把 `req_to_token`（req → token 物理位置的映射表）按本 batch 的 `req_pool_indices` 和 `seq_lens` 展平成一维 `kv_indices`（`triton_backend.py:195-210`）。这一步每次 forward 只做一次，所有层共用。

这就是「RadixAttention 把 KV pool 接进 kernel」的全貌：**layer 提供 `layer_id` 选行，backend 提供 `kv_indices` 选 token，kernel 直接在 pool 的连续显存上做 gather + attention。** 不需要拷贝/拼接 KV，这正是 RadixAttention 与 paged KV 高效的核心。

`TorchNativeAttnBackend._run_sdpa_forward_extend`（`torch_native_backend.py:27`）是理解这套机制最好的「慢速参考」：它逐 seq 循环，用 `per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]` 取出该 seq 的物理 token 列表，再 `k_cache[per_req_tokens]` gather 出 KV，喂给 `scaled_dot_product_attention`（`torch_native_backend.py:91-104`）。性能差但逻辑一目了然，调试时可作为 ground truth。

### 4.2 attention backend 如何选择与切换

两步：**先定字符串，再实例化**。

第一步在 `ModelRunner.model_specific_adjustment`（`model_runner.py:313`）。当用户没传 `--attention-backend` 时自动选（`model_runner.py:316-350`）：

```
attention_backend is None?
├─ 非 MLA 模型 (Llama/Qwen 等 MHA):
│   ├─ Hopper + CUDA12.3 + 非投机(或topk=1) + FA3-friendly 架构 → "fa3"
│   ├─ ROCm (_is_hip)                                          → "aiter"
│   └─ 否则 → flashinfer 可用就 "flashinfer"，否则 "triton"
└─ MLA 模型 (DeepSeek 等):
    ├─ Hopper + CUDA12.3 → "fa3"
    └─ 否则             → "triton"
```

之后还有一串「修正规则」：fa3 + `fp8_e5m2` KV → 退回 triton（`model_runner.py:370-378`）；开 double sparsity → 强制 triton 且关 cuda graph（`model_runner.py:380-390`）。MLA 模型若显式指定了不支持的 backend，会直接报错（`model_runner.py:351-366`）。

第二步在 `ModelRunner.init_attention_backend`（`model_runner.py:990`），就是一长串 `if/elif` 把字符串映射到类，并**懒加载**对应模块（避免无关 CUDA 依赖被导入）：

```
"flashinfer" → FlashInferAttnBackend / FlashInferMLAAttnBackend (按 use_mla_backend 分)
"aiter"      → AiterAttnBackend
"triton"     → TritonAttnBackend (double sparsity 时换 DoubleSparseAttnBackend)
"torch_native" → TorchNativeAttnBackend
"flashmla"   → FlashMLABackend
"fa3"        → FlashAttentionBackend (断言 SM 在 80~90)
"cutlass_mla"→ CutlassMLABackend
else         → raise ValueError
```

注意 `flashinfer` 分支里同一个字符串会根据 `use_mla_backend` 落到两个不同的类（`model_runner.py:992-1007`）——MLA（DeepSeek 的 multi-head latent attention）和普通 MHA 的 KV 布局不同，必须用不同 backend。

### 4.3 MoE：路由 → 分发 → 专家计算，以及与 EP 的关系

一个 MoE 层的标准三步是 **route（选专家）→ dispatch（把 token 发给专家）→ expert compute（专家算 FFN）→ combine（按权重合并）**。SGLang 用两种并行策略实现这三步，区别在「专家权重放在哪」：

| | 专家权重布局 | 通信 | 类 |
| --- | --- | --- | --- |
| TP（默认） | 每个 rank **全量**持有所有专家，但每个专家的 FFN 中间维被切 | 输出 all-reduce | `FusedMoE` |
| EP | 每个 rank 只持有 `num_experts / tp_size` 个**完整**专家 | token 在 rank 间 dispatch/combine | `EPMoE` / `DeepEPMoE` |

**路由（route）** 对两者是统一的，都走 `select_experts`（`topk.py:297`）。它根据参数选不同的 topk 实现：

```
select_experts:
├─ use_grouped_topk (DeepSeek V2/V3/R1):
│   ├─ correction_bias is None → grouped_topk        (topk.py:98,  softmax + group mask)
│   └─ 否则                     → biased_grouped_topk  (topk.py:232, sigmoid + bias，可走 moe_fused_gate kernel)
├─ torch_native           → fused_topk_native  (topk.py:42)
├─ 默认 (Qwen3MoE 等)      → fused_topk         (topk.py:63, sgl_kernel topk_softmax)
└─ custom_routing_function → 模型自定义
```

输出统一是 `(topk_weights, topk_ids)`：每个 token 选了哪 `top_k` 个专家、各自权重多少。最后 `select_experts` 会调 `get_global_expert_distribution_recorder().on_select_experts(topk_ids)`（`topk.py:379`）记录专家负载——这是 EPLB（专家负载均衡）的数据来源。

DeepSeek 系列的 grouped topk 值得注意：它先把专家分成 `num_expert_group` 组，按组打分（`grouped_topk` 用每组 max、`biased_grouped_topk` 用每组 top-2 之和，`topk.py:115-122` / `:173-177`），先选 `topk_group` 个组，再在选中的组内选专家。这是 DeepSeek 限制跨节点专家通信的设计。

**TP 路径（`FusedMoE`）**：`FusedMoE.forward`（`fused_moe_triton/layer.py:641`）直接调 `self.quant_method.apply(...)`，最终在 `forward_cuda`（`fused_moe_triton/layer.py:156`）里先 `select_experts`，再 `fused_experts(x, w13_weight, w2_weight, topk_weights, topk_ids, ...)`（`:213`）。因为所有专家权重都在本地，没有 token 的跨 rank 搬运，只在最后按 `reduce_results` 做一次 all-reduce（`:661`）。`w13_weight` 是 gate_proj 和 up_proj 合并的权重（MergedColumnParallel 风格），`w2_weight` 是 down_proj。

**EP 路径（`EPMoE`）**：每个 rank 只算自己 `[start_expert_id, end_expert_id]` 区间的专家（`ep_moe/layer.py:172-173`）。`EPMoE.forward`（`ep_moe/layer.py:219`）流程：

```
select_experts → topk_ids
run_moe_ep_preproess(topk_ids)  → reorder_topk_ids, src2dst, seg_indptr  # 把 token 按目标专家排序
pre_reorder_triton_kernel(...)  → gateup_input                          # gather + 可选 fp8 量化
grouped_gemm (w13) → activation → grouped_gemm (w2)                      # 分段 GEMM 算专家 FFN
post_reorder → 按 topk_weights 合并回原 token 顺序
```

`DeepEPMoE`（`ep_moe/layer.py:820`）在 EP 之上接入 DeepEP 通信库，靠 `token_dispatcher.py` 里的 dispatcher 做真正的跨 rank `dispatch` / `combine`，并支持 `normal`（高吞吐 prefill）与 `low_latency`（低延迟 decode）两种模式（`token_dispatcher.py:38-128`）。EP 与 EPLB、expert-location 的协作见 expert distribution / expert location 模块（`topk.py` 里 `expert_location_dispatch_info`、`topk_ids_logical_to_physical` 把「逻辑专家号」映射到「物理专家号」，`topk.py:92`）。

### 4.4 LogitsProcessor：如何产出 logits

`LogitsProcessor.forward`（`logits_processor.py:242`）的输入是整个 batch 的 hidden states `[num_tokens, hidden]`，输出是 `LogitsProcessorOutput`（`logits_processor.py:63`，核心字段 `next_token_logits`）。它的核心难点是**剪枝**：大多数情况只需要每个 seq 最后一个 token 的 logits（用于采样），不需要对所有 token 算 vocab 投影（vocab 维巨大）。

```
forward:
├─ decode / target_verify  → 所有位置都要采样，pruned_states = hidden_states (logits_processor.py:257)
├─ extend 且不要 input logprob → 只取每个 seq 最后一个 token (用 cumsum(extend_seq_lens)-1, :268)
└─ extend 且要 input logprob → 复杂三索引：pruned_states / sample_indices / input_logprob_indices (:288-330)

pruned_states → _get_logits(pruned_states, lm_head, metadata)  (:333)
```

`_get_logits`（`logits_processor.py:435`）才是真正算 logits 的地方：

```
hidden_states @ lm_head.weight.T          # 主投影 (logits_processor.py:457)
(GGUF 模型走 lm_head.quant_method.apply)
× logit_scale (可选)
all-gather across TP (vocab 被切分时拼回完整 vocab，:467-480)
裁到 vocab_size 并转 float32 (:493)
final_logit_softcapping (Gemma 等，:495-496)
```

注意 lm_head 是 `ParallelLMHead`（词表并行），vocab 维被 TP 切分，所以算完每个 rank 只有一段 logits，需要 all-gather 才能得到完整分布。DP attention 场景下还有额外的 gather/scatter（`logits_processor.py:448-491`）。

`capture_hidden_mode`（`logits_processor.py:346-369`）用于 EAGLE 投机解码：需要把 hidden states 存下来给 draft model。

### 4.5 linear / quant 的接入点

量化在 SGLang 里是**「方法对象」**模式，不是「重写 layer」。每个 `LinearBase`（`linear.py:178`）在构造时做一次决策（`linear.py:208-211`）：

```
quant_config is None → self.quant_method = UnquantizedLinearMethod()
否则                 → self.quant_method = quant_config.get_quant_method(self, prefix)
```

`LinearMethodBase`（`linear.py:102`）定义两个抽象方法：`create_weights`（按量化格式创建权重 Parameter）和 `apply`（实际算 `y = xW + b`）。`UnquantizedLinearMethod.apply`（`linear.py:168`）就是一行 `F.linear`。`forward` 永远调 `self.quant_method.apply(...)`，所以**接一个新量化方法 = 写一个新的 `LinearMethodBase` 子类，不用碰 layer 本身**。

同样的模式贯穿全层：
- `RadixAttention` 也有 `quant_method`（`radix_attention.py:77-80`），用于 KV cache 的 fp8 量化（k_scale/v_scale）。
- `FusedMoE` / `EPMoE` 用 `FusedMoEMethodBase` / `Fp8EPMoEMethod`（`fused_moe_triton/layer.py:334-340`、`ep_moe/layer.py:188-206`）。
- `VocabParallelEmbedding` 用 `UnquantizedEmbeddingMethod`（`vocab_parallel_embedding.py:28`）。

具体量化实现都在 `python/sglang/srt/layers/quantization/`（fp8、awq、gptq、marlin、blockwise_int8、modelopt 等），通过 `quant_config.get_quant_method(layer, prefix)` 按 layer 类型返回合适的 method。`prefix`（layer 在模型里的全名，如 `model.layers.0.self_attn.qkv_proj`）让量化配置可以**按层名做 mixed-precision / 跳过某些层**。

### 4.6 基础算子：CustomOp 多后端分发

`RMSNorm`、`SiluAndMul`、`RotaryEmbedding` 都继承 `CustomOp`，统一的分发逻辑见 `RMSNorm.forward`（`layernorm.py:52`）：

```
torch.compiler.is_compiling() → forward_native   # 编译期走纯 PyTorch，避免 custom op 打断图
_is_cuda                      → forward_cuda      # 调 sgl_kernel 的融合 kernel
_is_hip                       → forward_hip       # 调 aiter/vllm 的 ROCm kernel
else                          → forward_native    # CPU/其它平台纯 PyTorch
```

`RMSNorm.forward_cuda`（`layernorm.py:62`）的一个关键细节：传 `residual` 时调 `fused_add_rmsnorm`，它**原地修改** `x` 和 `residual`（融合了残差加法 + 归一化），返回 `(x, residual)`；不传 residual 才返回新 tensor。模型代码里 `hidden_states, residual = norm(hidden_states, residual)` 这种写法就是依赖这个融合。

RoPE 通过 `get_rope`（`rotary_embedding.py:1064`）工厂获取，按 `(head_size, rotary_dim, max_position, base, rope_scaling, ...)` 做 key 缓存（`_ROPE_DICT`，`rotary_embedding.py:1061,1095`），相同配置的层共享同一个 RoPE 实例。不同的 scaling（linear / dynamic NTK / YaRN / Llama3 / DeepSeek / MRoPE）是不同子类。

## 5. 设计难点与权衡

1. **为什么 backend 是全局单例，而 RadixAttention 是每层一个？**
   attention kernel 的 metadata（`kv_indptr` / `kv_indices` / cuda graph buffer）是**整个 batch 共享**的，每次 forward 只需 `init_forward_metadata` 算一次；而每层的差异只有 `layer_id`、head 配置、scaling，放进轻量的 `RadixAttention` 即可。这样既避免重复算 metadata，又能让所有层走同一份 cuda graph。代价是 backend 必须无状态（除了 metadata），不能把「本层特有」的东西藏在 backend 里。

2. **KV 不拷贝的代价：kernel 必须懂 paged 布局。** RadixAttention 高效的前提是 kernel 直接吃 `(kv_buffer, kv_indices)` 这种「物理 buffer + gather 索引」。这就是为什么不是每个 attention 库都能直接接——Torch native 要自己写 gather 循环，FlashInfer/FA3 要专门的 paged kernel。新接一个 backend 的主要工作量就在「把我们的 paged KV 喂成它要的格式」。

3. **MoE 的 TP vs EP 是显存与通信的权衡。** TP 让每个 rank 全量持有专家权重（显存大、无 token 通信），EP 让每个 rank 只持有部分专家（显存小、但要 all-to-all 搬 token）。专家越多、模型越大（DeepSeek V3 有 256 个专家），EP 越划算，但通信复杂度高、对 DeepEP/网络敏感。

4. **logits 剪枝的复杂度全在「要不要 input logprob」。** 不要 input logprob 时只取最后一个 token（一行 cumsum）；要的时候必须区分「采样位置」和「需要 logprob 的位置」，还要处理 chunked prefill 下「最后一块只采样不算 input logprob」的边界（`logits_processor.py:302-322`）。这段三索引逻辑是 logprob 相关 bug 的高发区。

5. **量化用「method 对象」而非「子类 layer」**，好处是同一个 `QKVParallelLinear` 能在 fp16 / fp8 / awq 间切换而不改模型代码，weight_loader 仍挂在 layer 上（量化权重的切分逻辑由 layer 的 `weight_loader` + Parameter 的 `weight_loader` 协作完成，见 `linear.py:937` 的 QKV weight_loader）。坏处是 method 和 layer 的契约（哪些属性、什么 shape）是隐式的，接新量化方法时容易踩 shape/属性不匹配。

6. **`torch.compile` 兼容性是隐藏约束。** 多处代码专门为编译让路：`RMSNorm.forward` 编译期强制 `forward_native`（`layernorm.py:53`）；`AttentionType` 用字符串而非纯 Enum「to be compatible with torch.compile」（`radix_attention.py:28`）；decode 路径里反复出现 `q.reshape(...)` 注释「there is a bug in rotary_emb that causes 3D tensor under torch.compile」（`triton_backend.py:567`、`torch_native_backend.py:235`）。改这些算子时务必同时验证 eager 和 compile 两条路。

## 6. 改代码 / 修 bug 指南

### 想新增 / 切换一个 attention backend

这是本章最常见的需求，标准步骤：

1. **写 backend 类**：在 `python/sglang/srt/layers/attention/` 下新建 `myxx_backend.py`，继承 `AttentionBackend`（`base_attn_backend.py:14`），至少实现：
   - `init_forward_metadata(forward_batch)`：构建 `kv_indptr` / `kv_indices` 等，存到 `self.forward_metadata`。可参考 `triton_backend.py:188`。
   - `forward_extend(q,k,v,layer,forward_batch,save_kv_cache)`：先 `set_kv_buffer`，再调你的 prefill kernel，KV 从 `get_key_buffer(layer.layer_id)` 取。参考 `triton_backend.py:515`。
   - `forward_decode(...)`：同上，decode kernel。参考 `triton_backend.py:558`。
   - 想支持 cuda graph 还要实现 `init_cuda_graph_state` / `init_forward_metadata_capture_cuda_graph` / `init_forward_metadata_replay_cuda_graph` / `get_cuda_graph_seq_len_fill_value`（基类默认 `raise NotImplementedError`，`base_attn_backend.py:22-55`）。先不实现，用 `--disable-cuda-graph` 跑通再说。
2. **注册字符串**：在 `ModelRunner.init_attention_backend`（`model_runner.py:990`）加一个 `elif self.server_args.attention_backend == "myxx":` 分支，懒加载你的类。
3. **加 server arg 选项**：在 `python/sglang/srt/server_args.py` 的 `attention_backend` choices 里加 `"myxx"`（否则参数校验会拒绝）。
4. （可选）想让它被自动选中，改 `model_specific_adjustment`（`model_runner.py:316`）的自动选择逻辑。

**最小可跑路径**：先复制 `TorchNativeAttnBackend`（`torch_native_backend.py`，最短最易懂），把 sdpa 换成你的 kernel，`--attention-backend torch_native` 风格起服务，先关 cuda graph。用 `TorchNativeAttnBackend` 的输出做数值对拍是验证新 backend 正确性的最佳手段。

### 想改 MoE 路由

- 改「怎么选专家」→ `python/sglang/srt/layers/moe/topk.py`，找到对应函数（grouped_topk / biased_grouped_topk / fused_topk）。注意 CUDA 有融合 kernel 分支（`moe_fused_gate`、`topk_softmax`），改了 Python 实现要确认是否也走了 kernel 分支（kernel 不会被你的改动影响）。
- 改「专家怎么算」→ TP 看 `fused_moe_triton/layer.py:156` 的 `forward_cuda` 和 `fused_experts`；EP 看 `ep_moe/layer.py:219`。
- 专家负载 / EPLB 相关 → `topk.py:379` 的 `on_select_experts` 与 `expert_location_dispatch_info`。

### 常见 bug 高发区

- **logprob 数值不对 / 越界**：`logits_processor.py:288-330` 的三索引逻辑，尤其 chunked prefill 边界（`extend_len == extend_logprob_start_len` 那个 case，`:304`）。
- **TP 下 logits 拼接错位**：`_get_logits` 的 all-gather（`logits_processor.py:467-480`）和 DP attention 的 gather/scatter（`:448-491`）。
- **量化权重 shape mismatch**：`linear.py` 各 `weight_loader`（QKV 在 `:937`，Merged 在 `:523`），以及 method 的 `create_weights`。报错通常是 `assert param_data.shape == loaded_weight.shape`。
- **`torch.compile` 下 attention 输出变 3D**：见 `triton_backend.py:567` / `torch_native_backend.py:235` 的 `q.reshape` 注释；新算子记得同时测 compile。
- **MLA 模型选错 backend**：`model_runner.py:351-366` 会对 MLA + 不支持的 backend 报错；`flashinfer` 字符串在 MLA 下实际是 `FlashInferMLAAttnBackend`（`:1003`）。

### 调试入手点

- **数值对拍**：把 `--attention-backend torch_native`（attention）或 MoE 的 `torch_native`/native topk 当 ground truth，和高性能 backend 比对。
- **看 backend 实际选了谁**：启动日志里 `Attention backend not set. Use xxx backend by default.`（`model_runner.py:348`）。
- **KV 写入/读取定位**：在 `set_kv_buffer`（`memory_pool.py:378`）和 `get_key_buffer`（`memory_pool.py:359`）打点，确认 `layer_id` 和 `loc` / `kv_indices` 是否正确。
- **MoE 专家分布**：通过 expert distribution recorder（`topk.py:379`）观察各专家负载是否均衡。

## 相关章节

- [model-executor](14-model-executor.md)：谁构造 `ForwardBatch`、谁调用本层、cuda graph、采样
- [memcache-radixattention](13-memcache-radixattention.md)：KV pool（`set_kv_buffer` / `get_key_buffer`）与 RadixCache 的实现
- [scheduler](12-scheduler.md)：`out_cache_loc` / `req_pool_indices` / `seq_lens` 等 batch 元数据的来源
- [request-lifecycle](01-request-lifecycle.md)：本层在整条请求链路中的位置
- [术语表](02-glossary.md)：RadixAttention / MLA / TP / EP / prefill / decode 等术语
