# Multi-LoRA 批处理

> 本章对应源码目录：`python/sglang/srt/lora/`（整目录）。
> 设计思想来自论文 *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* 与 *Punica: Multi-Tenant LoRA Serving*（见 `python/sglang/srt/lora/lora_manager.py:15`）。

## 1. 一句话职责

让一个 server 同时挂载多个 LoRA adapter，并且让**同一个 batch 里不同请求各自用不同的 adapter**：把所有 adapter 的低秩权重 (`lora_A` / `lora_B`) 放进一块共享显存池 (`LoRAMemoryPool`)，在前向时通过 monkey-patch 过的 linear 层用「分段 GEMM (segmented gemm)」一次性算完整个 batch 的 LoRA 增量，最后加到 base model 的输出上。

LoRA 的数学形式很简单：对一个原始线性层 `y = x W`，LoRA 增加一个低秩旁路

```
y = x W  +  scaling * (x A^T) B^T
            └── base ──┘ └──── LoRA delta ────┘
```

其中 `A` 形状 `(r, in)`、`B` 形状 `(out, r)`，`r`（rank，通常 8/16/32/64）远小于 `in`/`out`，`scaling = lora_alpha / r`。多 LoRA 批处理的难点不在数学，而在**工程**：怎么让 batch 里每行用对的那份 `A`/`B`、怎么省显存、怎么和 CUDA graph / TP 共存。

## 2. 在端到端链路中的位置

LoRA 不是独立的「阶段」，而是寄生在 **前向 (forward)** 阶段的 linear 层里，由 **调度器 (scheduler)** 在组 batch 时做准入控制，由 `ForwardBatch` 构造时触发权重加载。

```
客户端 → 分词 → 调度(组batch) → KV缓存 → 前向 → 采样 → 解码 → 响应
                   │                       │
                   │                       ├─ ForwardBatch.init_new()
                   │                       │     └─ lora_manager.prepare_lora_batch()  ← 加载/换入 adapter 权重，构造 LoRABatchInfo
                   │                       │
                   │                       └─ model.forward()
                   │                             └─ XxxLinearWithLoRA.forward()         ← base GEMM + segmented LoRA GEMM
                   │
                   └─ Scheduler.get_new_batch_prefill():
                         「batch 内 distinct adapter 数 ≤ max_loras_per_batch」准入约束
```

- **准入控制**：`python/sglang/srt/managers/scheduler.py:1386` —— 一个 batch 内出现的不同 `lora_path` 数量不能超过 `max_loras_per_batch`，否则该请求留在 waiting queue。
- **每个请求带 adapter id**：请求经 `lora_path` 字段一路传到 `ScheduleBatch.lora_paths`（`python/sglang/srt/managers/schedule_batch.py:1664`）、再到 `ForwardBatch.lora_paths`（`python/sglang/srt/model_executor/forward_batch_info.py:216`）。
- **触发加载**：`ForwardBatch.init_new()` 末尾调用 `lora_manager.prepare_lora_batch(ret)`（`python/sglang/srt/model_executor/forward_batch_info.py:364`）。

关于调度准入逻辑见 [调度器](12-scheduler.md)；关于 `ForwardBatch` 的构造见 [前向批次](14-model-executor.md)；术语见 [术语表](02-glossary.md)。

## 3. 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `LoRAManager` | `python/sglang/srt/lora/lora_manager.py:44` | 顶层协调者：加载所有 adapter、monkey-patch 模型层、每个 batch 构造 `LoRABatchInfo` 并把权重张量喂给各 LoRA 层 |
| `LoRAManager.prepare_lora_batch` | `python/sglang/srt/lora/lora_manager.py:150` | 每次前向调用一次：换入活跃 adapter、生成 segment 指针、调用每层的 `set_lora_info` |
| `LoRAAdapter` | `python/sglang/srt/lora/lora.py:47` | 单个 adapter 的 CPU 端表示，负责从 checkpoint 加载权重并 stack q/k/v、gate/up |
| `LoRAConfig` | `python/sglang/srt/lora/lora_config.py:21` | 读 `adapter_config.json`，拿到 `r`、`lora_alpha`、`target_modules` |
| `LoRAMemoryPool` | `python/sglang/srt/lora/mem_pool.py:18` | 共享显存池：`max_loras_per_batch` 个 slot，负责换入/换出 (eviction) 与 TP 切分 |
| `LoRABatchInfo` | `python/sglang/srt/lora/utils.py:11` | 每个 batch 的「分段元数据」：`seg_lens`/`seg_indptr`/`weight_indices`/`lora_ranks`/`scalings` |
| `BaseLayerWithLoRA` 及子类 | `python/sglang/srt/lora/layers.py:22` | monkey-patch 进模型的 LoRA 版 linear 层（`QKVParallelLinearWithLoRA` 等） |
| `BaseLoRABackend` | `python/sglang/srt/lora/backend/base_backend.py:24` | 后端抽象：定义 `run_lora_a_sgemm` / `run_qkv_lora` 等 kernel 接口 |
| `TritonLoRABackend` | `python/sglang/srt/lora/backend/triton_backend.py:13` | 默认后端，调 triton segmented gemm kernel |
| `FlashInferLoRABackend` | `python/sglang/srt/lora/backend/flashinfer_backend.py:13` | 备选后端，用 flashinfer `SegmentGEMMWrapper`（限制多，要求同 rank/scaling） |
| `_sgemm_lora_a_kernel` / `_sgemm_lora_b_kernel` | `python/sglang/srt/lora/triton_ops/sgemm_lora_a.py:8` / `sgemm_lora_b.py:8` | 分段 GEMM triton kernel：按 `batch_id` 选对应 adapter 权重 |
| `_qkv_lora_b_kernel` | `python/sglang/srt/lora/triton_ops/qkv_lora_b.py:8` | 把 q/k/v 三个 lora_b 打进一个 kernel |
| 工具函数 (`get_stacked_name` 等) | `python/sglang/srt/lora/utils.py` | HF 模块名 ↔ SGLang stacked 模块名 的映射 |

## 4. 核心数据流 / 执行流程

整个生命周期分四个阶段：**(A) 启动加载 → (B) monkey-patch → (C) 每 batch 换入 → (D) 前向计算**。

### 4.1 启动加载（一次性，CPU）

`LoRAManager.__init__`（`lora_manager.py:44`）依次做：

1. `init_loras()`（`lora_manager.py:92`）：对每个 `lora_paths` 里的 adapter，
   - 读 `LoRAConfig`（`lora_config.py:21`，从 `adapter_config.json` 拿 `r`/`alpha`/`target_modules`），
   - 收集所有 `target_modules` 到 `hf_target_names`（如 `{q_proj, k_proj, v_proj, o_proj}`），
   - 通过 `get_stacked_name`（`utils.py:109`）映射成 SGLang 的 stacked 名字对，例如 `q_proj → ("qkv_proj","q_proj")`、`gate_proj → ("gate_up_proj","gate_up_proj")`，存进 `lora_weight_names`，
   - 为每个 adapter 建 `LoRAAdapter` 并 `initialize_weights()` 把权重读到 **CPU**。
2. `LoRAAdapter.initialize_weights()`（`lora.py:75`）用 `DefaultModelLoader._get_weights_iterator` 流式读权重，按正则 `layers\.(\d+)\.` 分到对应 layer，然后做两件「stack」：
   - `stack_qkv_proj`（`lora.py:98`）：把分开的 `q/k/v` 的 `lora_A` 在 dim 0 上 `cat` 成 `qkv_proj`，`lora_B` 的 `k/v` `stack` 成 `kv_proj`（q 单独）。**注意**：如果某个 adapter 没给 `k_proj` 配 LoRA，会用 `torch.zeros_like` 补零（`lora.py:122`）。
   - `stack_gate_up_proj`（`lora.py:152`）：同理把 `gate_proj`/`up_proj` 拼成 `gate_up_proj`；缺 `up_proj` 时补零并 **强制要求 triton 后端**（`lora.py:166`）。

   为什么要 stack？因为 SGLang 的 base model 本身就把 q/k/v 融合成一个 `QKVParallelLinear`、gate/up 融成 `MergedColumnParallelLinear`，LoRA 权重必须与之对齐才能融合计算。

3. `init_lora_memory_pool()`（`lora_manager.py:135`）建显存池并 `init_buffers` 预分配 GPU 张量（见 4.3）。

```
adapter_dir/
  adapter_config.json   ──► LoRAConfig (r, alpha, target_modules)
  adapter_model.safetensors ──► LoRAAdapter.layers[i].weights  (CPU)
                                  │ stack q/k/v, gate/up
                                  ▼
                          {"qkv_proj.lora_A":..., "q_proj.lora_B":..., "kv_proj.lora_B":..., ...}
```

### 4.2 monkey-patch 模型层（一次性）

`convert_to_lora_layers()`（`lora_manager.py:255`）把 base model 里命中 target 的 linear 层**原地替换**成 LoRA 版：

1. `get_customized_names_from_hf_names`（`utils.py:50`）把 HF 名字翻译成 SGLang 层名（优先调模型自带的 `base_model.get_module_name`，否则用内置 fallback 映射）。
2. 遍历 `base_model.named_modules()`，凡末段名命中的，用 `set_lora_module` → `get_lora_layer`（`layers.py:358`）按层类型选对应的 `XxxLinearWithLoRA` 包装类，再用 `replace_submodule`（`lora_manager.py:252`）替换。
3. 结果存进 `self.lora_modules: Dict[layer_id, List[(module_name, module)]]`（`lora_manager.py:263`），后面每 batch 遍历它来 `set_lora_info`。

包装类的 `forward`（如 `ColumnParallelLinearWithLoRA.forward`，`layers.py:95`）逻辑是：**先跑原层的 GEMM 拿 `base_output`，再 `if self.set_lora: apply_lora(base_output, x)`**。`apply_lora` 调后端 `run_lora_a_sgemm` + `run_lora_b_sgemm`（`layers.py:81`）。

### 4.3 显存池布局与每 batch 换入

`LoRAMemoryPool`（`mem_pool.py:18`）的核心是两块按 layer 切分的 buffer：

```
A_buffer[weight_name][layer_id]: (max_loras_per_batch, stacked_num * max_lora_dim, input_dim)   # row-major
B_buffer[weight_name][layer_id]: (stacked_num, max_loras_per_batch, output_dim, max_lora_dim)   # 见 get_lora_B_shape
```

- 第一维（或 A 的第一维）就是 **slot**：池子里只有 `max_loras_per_batch` 个 slot，所有 adapter 共享、**按需换入换出**，这正是 S-LoRA 的核心思想——不为每个 adapter 永久占显存。
- `max_lora_dim` 是所有 adapter 里最大的 rank（`lora_manager.py:123`）；rank 比它小的 adapter 只填前 `lora_rank` 行/列，kernel 里用 `lora_ranks` 动态裁剪（见 4.4）。

每个 batch，`prepare_lora_batch`（`mem_pool.py:127`）做换入：

```
get_available_buffer_slot():
   1) 优先找空 slot (buffer_id_to_uid == "")
   2) 否则淘汰一个「本 batch 不需要的」adapter (uid not in cur_uids)
   3) 都没有 → 报错 (活跃 adapter 超过 max_loras_per_batch)
```

只有当 `uid not in uid_to_buffer_id`（即还没在池子里）才会真正 `load_lora_weight_to_buffer`（`mem_pool.py:159`）把 CPU 权重 `copy_` 进 GPU slot。**已在池中的 adapter 零拷贝复用**——这就是「切换开销」的来源：只有冷 adapter 才付 H2D 拷贝代价。

几个非显然点：
- `uid is None`（base model，不带 LoRA 的请求）也是合法 uid，对应把 A_buffer 该 slot 清零（`mem_pool.py:163`），这样 base-only 请求也能走同一套 kernel（delta 为 0）。
- TP > 1 时换入前要先按 `tp_rank` 切权重：`slice_lora_a_weights` / `slice_lora_b_weights`（`mem_pool.py:187`），不同层类型切法不同（见 5.4）。

### 4.4 构造 LoRABatchInfo + 前向

`prepare_lora_batch`（`lora_manager.py:150`）换入后构造 `LoRABatchInfo`（`utils.py:11`），这是 segmented gemm 的「导航图」：

```
batch 有 bs 个请求，token 在 x 里是首尾相接的：
   x: (s, in)   s = sum(seg_lens)

seg_lens     = [len_0, len_1, ...]         每个请求的 token 数(decode 时全 1)
seg_indptr   = [0, len_0, len_0+len_1, ...]  前缀和，kernel 用它定位每段起点
weight_indices[i] = 第 i 个请求用的 adapter 在池中的 slot id
lora_ranks[slot]  = 该 slot adapter 的 rank
scalings[slot]    = 该 slot adapter 的 alpha/r
```

triton kernel `_sgemm_lora_a_kernel`（`sgemm_lora_a.py:8`）的工作方式：
- `program_id(axis=1)` = `batch_id`，即每个请求一组 block；
- `w_index = weight_indices[batch_id]` 选出这一段该用哪个 adapter 的权重（`sgemm_lora_a.py:46`）；
- `rank = lora_ranks[w_index]`，然后 `N = min(N, rank*stack_num)`（`sgemm_lora_a.py:50`）—— **同一 kernel 里不同段用不同 rank**，靠这行动态裁剪；
- lora_b kernel 里还会 `partial_sum *= scaling`（`sgemm_lora_b.py:92`）并按 `fuse_scaling_add` 决定是否直接加到 `base_output` 上（`sgemm_lora_b.py:98`）。

这就是「分段/segmented gemm」：**一次 kernel launch 覆盖整个 batch，每段自动路由到正确的 adapter 权重、rank、scaling**，避免了「按 adapter 拆成多个小 GEMM」的 launch 开销和 padding 浪费。

最后 `prepare_lora_batch` 遍历 `self.lora_modules`，对每个 LoRA 层调 `set_lora_info`，把池子里对应 `(weight_name, layer_id)` 的张量指针交给该层（`lora_manager.py:223`）。qkv 层特殊：要同时传 `qkv_proj` 的 A 和 `q_proj`/`kv_proj` 两份 B（`lora_manager.py:225`）。

完整一条数据流：

```
请求.lora_path
  └─ ForwardBatch.lora_paths
       └─ prepare_lora_batch
            ├─ memory_pool.prepare_lora_batch  → 换入冷 adapter (H2D copy)，得到 slot id
            ├─ 构造 LoRABatchInfo (seg_indptr / weight_indices / lora_ranks / scalings)
            ├─ backend.set_batch_info(batch_info)
            └─ 每个 LoRA 层 set_lora_info(A_buffer, B_buffer)   ← 只传指针，零拷贝
  └─ model.forward
       └─ QKVParallelLinearWithLoRA.forward(x)
            ├─ base_output = base_layer GEMM
            └─ apply_lora:
                 ├─ sgemm_lora_a_fwd(x, A, batch_info)        → (s, stack*r)
                 └─ qkv_lora_b_fwd(.., B, batch_info, ..)     → 累加进 base_output
```

## 5. 设计难点与权衡

### 5.1 为什么用 segmented gemm 而不是逐 adapter 循环？

朴素做法是按 adapter 分组、每组单独 GEMM。问题：(1) 每个 adapter 一次 kernel launch，batch 里 adapter 多时 launch 开销爆炸；(2) decode 时每段只有 1 个 token，GEMM 形状极瘦，GPU 利用率差。segmented gemm 把 batch 拍平成 `(s, in)`，**一次 launch**，kernel 内部用 `seg_indptr` + `weight_indices` 自己路由，是 Punica/S-LoRA 的关键优化。

### 5.2 rank 不一时怎么办（max_lora_dim padding + 动态裁剪）

buffer 按 `max_lora_dim` 分配（`mem_pool.py:35`），小 rank adapter 只填前几行（`mem_pool.py:213` 的 `[:lora_rank*c, :]`）。kernel 里通过 `lora_ranks[w_index]` 把实际计算的 K/N 收缩到真实 rank（`sgemm_lora_a.py:50`、`sgemm_lora_b.py:52`）。**权衡**：显存按最大 rank 浪费，但换来了「不同 rank 的 adapter 能进同一 batch」。flashinfer 后端不支持这点，要求所有 adapter 同 rank 同 scaling（`lora_manager.py:125`）。

### 5.3 显存与切换开销

- **显存**：`O(num_layers * num_target_modules * max_loras_per_batch * max_lora_dim * hidden)`，与「挂载了多少 adapter」**无关**，只与 `max_loras_per_batch` 有关。挂 100 个 adapter 但 `max_loras_per_batch=8`，GPU 上始终只有 8 份。代价是 adapter 的 CPU 全量副本常驻内存。
- **切换开销**：只有 batch 里出现「池中没有的冷 adapter」时才付 H2D 拷贝（`load_lora_weight_to_buffer`）。热 adapter（仍在 slot 里）零成本。淘汰策略很朴素：找第一个本 batch 不需要的 slot（`mem_pool.py:139`），不是 LRU。

### 5.4 TP 切分的层差异

不同层类型的 LoRA 在 TP 下切法不同（`layers.py`）：
- `RowParallelLinearWithLoRA`（o_proj/down_proj，列在 `ROW_PARALLELISM_LINEAR_LORA_NAMES`，`utils.py:158`）：切 `lora_A` 的输入维（`layers.py:347`），`lora_B` 不切。
- `ColumnParallel`/`QKV`/`MergedColumn`：`lora_A` 不切，`lora_B` 按 output 分片切（`layers.py:115`、`layers.py:267`、`layers.py:170`）。
- QKV 的 B 切分还要处理 GQA 的 `num_kv_head_replicas`（`layers.py:279`）。

切分发生在换入时（`mem_pool.py:187`），所以池里存的就是本 rank 的分片，kernel 无需感知 TP。

### 5.5 CUDA graph 兼容

CUDA graph 要求张量地址/形状固定，所以 LoRA 单独准备一套 `cuda_graph_batch_info`（`lora_manager.py:75`），在 `init_cuda_graph_batch_info` 里预分配好最大尺寸张量，capture/replay 时**原地 in-place 更新** `seg_lens`/`weight_indices` 等（`lora_manager.py:166`），而不是新建张量。decode batch 且 `bs <= max_bs_in_cuda_graph` 时走这条路径。注意 `cuda_graph_runner.py:443`：capture 时若 `lora_path` 为 None 要替换成一个非 None 占位，因为 None 走的是不同逻辑。

### 5.6 fuse 开关

后端有两个 fuse flag（`base_backend.py:8`、`base_backend.py:16`）：
- `fuse_output_add`（triton=True）：lora_b kernel 直接把结果加进 `base_output`，省一次 add（`layers.py:91`、`sgemm_lora_b.py:98`）。
- `fuse_stacked_lora_b`（triton=True）：把 q/k/v 的 B 拼成一个大张量 `B_buffer_qkv`（`layers.py:198`），配合 `_qkv_lora_b_kernel` 用一个 kernel 算完 qkv，gate_up 同理。flashinfer 不 fuse，所以走 tuple 分别算。

## 6. 改代码 / 修 bug 指南

### 想改 X → 看这里

| 需求 | 入手点 |
| --- | --- |
| 支持新的 target module（如 LoRA on `embed_tokens`/`lm_head`） | `lora_config.py:31` 现在直接 raise；要支持得在 `layers.py` 加包装类、在 `utils.py` 的 `get_stacked_name`/`get_hidden_dim`/`get_stacked_multiply` 加映射 |
| 让一个新的 base model 支持 LoRA | 在模型类里实现 `get_module_name` 和 `get_hidden_dim`（参考 `llama.py`）；否则走 `utils.py:62`/`utils.py:88` 的 fallback，名字对不上就会出错 |
| 改换入淘汰策略（如改 LRU） | `mem_pool.py:133` 的 `get_available_buffer_slot` |
| 加一个新后端 | 继承 `BaseLoRABackend`（`base_backend.py:24`），实现 4 个 `run_*`，在 `get_backend_from_name`（`base_backend.py:120`）注册 |
| 调 kernel 性能（BLOCK 尺寸） | `sgemm_lora_a.py:122`、`sgemm_lora_b.py:124` 的 `BLOCK_S/K/N` 是写死的常量 |
| 改 batch 内最多几个 adapter | server arg `max_loras_per_batch`（`server_args.py:132`，默认 8）；准入约束在 `scheduler.py:1386` |

### 调试一个新 LoRA 的标准流程

1. 启动：`--lora-paths name=path` 或裸 path，解析在 `server_args.py:1466`。
2. 先确认 `adapter_config.json` 的 `target_modules` 都被支持（不含 `embed_tokens`/`lm_head`）。
3. 在 `LoRAAdapter.initialize_weights`（`lora.py:75`）打印 `self.layers[i].weights` 的 key，确认 stack 后出现 `qkv_proj`/`gate_up_proj` 而不是残留的 `q_proj`/`gate_proj`——名字残留通常意味着 `get_stacked_name` 没覆盖。
4. 在 `prepare_lora_batch`（`lora_manager.py:205`）打印 `weight_indices`/`lora_ranks`/`scalings`，确认每个请求路由到对的 slot、rank、scaling 非零。
5. 数值对不上时，先关 CUDA graph（排除 `cuda_graph_batch_info` 路径）、再切到 triton 后端（flashinfer 限制多），缩小范围。

### 常见 bug 高发区

- **stack 补零陷阱**：adapter 只对 q 做 LoRA、没给 k/v，`stack_qkv_proj`（`lora.py:122`）会补零；若 base model 期望 kv 也有非零 LoRA，结果会静默错误。gate-only 同理且会 assert 仅 triton（`lora.py:166`）。
- **rank 超过 max_lora_dim**：`max_lora_dim` 在启动时按当时所有 adapter 算定（`lora_manager.py:123`），后加的大 rank adapter 会越界写 buffer。
- **`None` uid 处理**：base-only 请求 uid 是 None，是合法 key（注释见 `mem_pool.py:53`）。把 None 当「无 LoRA」直接跳过的改动容易破坏混合 batch；CUDA graph capture 也专门处理 None（`cuda_graph_runner.py:440`）。
- **TP slice 方向搞反**：新增层类型时，`slice_lora_a_weights`/`slice_lora_b_weights`（`layers.py:112` 起）切错维度会导致 TP>1 下输出错乱但 TP=1 正常——这是最隐蔽的一类。
- **flashinfer 后端的同 rank/scaling 限制**：`lora_manager.py:129` 的 assert，混用不同 rank adapter 会直接挂。
- **radix cache 冲突**：`server_args.py:1459` 一带的校验——LoRA 与某些缓存配置不兼容，注意 `disable_radix_cache` 条件。
