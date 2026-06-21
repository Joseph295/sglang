# 模型执行：ModelRunner、ForwardBatch 与 CUDA Graph

> 本章对应代码：`python/sglang/srt/model_executor/`（`model_runner.py`、`forward_batch_info.py`、`cuda_graph_runner.py`、`expert_location_updater.py`）与 `python/sglang/srt/model_loader/`。

## 1. 一句话职责

`ModelRunner` 是「调度层」与「真实模型前向」之间的边界层：它负责**加载模型权重**、**初始化 attention backend 和 KV cache 内存池**、**捕获 CUDA graph**，并在每个 step 把上层传下来的 `ForwardBatch` 喂进 `model.forward(...)`，吐出 logits（再交给 sampler 采样）。一句话：**ModelRunner 把"一批要算的 token"变成"一批 next-token logits"。**

`ForwardBatch` 是前向的唯一输入载体——它把 prefill/decode 所需的所有 GPU 张量（`input_ids` / `positions` / `seq_lens` / `out_cache_loc` / attention 元数据……）打包成一个 dataclass。

`CudaGraphRunner` 是 decode 阶段的加速器：把固定 batch size 的前向"录制"成 CUDA graph，运行时直接 replay，绕开 Python/kernel-launch 开销。

## 2. 在端到端链路中的位置

```
客户端
  │
  ▼
Tokenizer  ─────────────────────────────► (02-glossary / 分词)
  │
  ▼
Scheduler (ScheduleBatch)  ──────────────► [12-scheduler.md]
  │   组 batch、分配 KV slot
  ▼
TpModelWorker (ModelWorkerBatch)  ───────► [15-worker-parallelism.md]
  │   ForwardBatch.init_new(...)
  ▼
┌───────────────────────────── 本章 ─────────────────────────────┐
│ ModelRunner.forward(forward_batch)                             │
│   ├─ can_run_cuda_graph? ──► CudaGraphRunner.replay()  (decode)│
│   ├─ forward_extend()                          (prefill/extend)│
│   ├─ forward_decode()             (decode, graph 关闭时的回退)  │
│   └─ forward_idle()                  (DP attention 的空闲 rank) │
│        │                                                       │
│        ▼  读写 token_to_kv_pool（KV cache, 见 13 章）          │
│   model.forward(input_ids, positions, forward_batch)          │
│        ▼                                                       │
│   LogitsProcessorOutput (next_token_logits)                   │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
ModelRunner.sample()  ──► Sampler  ──────► [17-sampling-structured-output.md]
  │
  ▼
Detokenizer → 响应
```

`ForwardBatch` 在数据结构链路上的位置（见 `forward_batch_info.py:17-28` 的文件头注释）：

```
ScheduleBatch  ──►  ModelWorkerBatch  ──►  ForwardBatch
(scheduler.py)      (tp_worker.py)         (model_runner.py)
 大多在 CPU          CPU→GPU 的子集          低层 tensor，大多在 GPU
```

上游调用点：`python/sglang/srt/managers/tp_worker.py:191` 调 `ForwardBatch.init_new(...)`，`tp_worker.py:202` 调 `self.model_runner.forward(...)`。

## 3. 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `ModelRunner` | `python/sglang/srt/model_executor/model_runner.py:133` | 持有 model / attn_backend / KV pool / cuda_graph_runner，驱动前向 |
| `ModelRunner.initialize` | `model_runner.py:237` | 加载模型→建内存池→建 attn backend→建 cuda graph 的总装配 |
| `ModelRunner.model_specific_adjustment` | `model_runner.py:313` | 按模型/硬件自动选 attention backend、关 cuda graph 等 |
| `ModelRunner.load_model` | `model_runner.py:498` | 调 `get_model(...)` 真正加载权重 |
| `ModelRunner.init_memory_pool` | `model_runner.py:821` | profile 显存 → 算 `max_total_num_tokens` → 建 `ReqToTokenPool` / `*TokenToKVPool` |
| `ModelRunner.init_attention_backend` | `model_runner.py:990` | 按 `attention_backend` 字符串实例化对应 backend |
| `ModelRunner.init_cuda_graphs` | `model_runner.py:1081` | 构造 `CudaGraphRunner`（捕获 graph） |
| `ModelRunner.forward` / `_forward_raw` | `model_runner.py:1159` / `:1180` | 前向总入口；分派 replay / extend / decode / idle |
| `ModelRunner.forward_extend` | `model_runner.py:1123` | prefill / extend / target-verify 路径 |
| `ModelRunner.forward_decode` | `model_runner.py:1111` | decode 路径（graph 未命中时） |
| `ModelRunner.sample` | `model_runner.py:1226` | 把 logits 交给 `Sampler` 采样 |
| `ForwardBatch` | `forward_batch_info.py:137` | 前向输入载体（dataclass） |
| `ForwardBatch.init_new` | `forward_batch_info.py:255` | 从 `ModelWorkerBatch` 构造 ForwardBatch，算 positions |
| `ForwardMode` | `forward_batch_info.py:53` | EXTEND / DECODE / MIXED / IDLE / TARGET_VERIFY / DRAFT_EXTEND / DUMMY_FIRST |
| `CudaGraphRunner` | `cuda_graph_runner.py:185` | 捕获与 replay decode graph |
| `CudaGraphRunner.can_run` | `cuda_graph_runner.py:328` | 判断当前 batch 能否走 graph |
| `CudaGraphRunner.capture` / `capture_one_batch_size` | `cuda_graph_runner.py:354` / `:393` | 逐 bs 捕获 graph |
| `CudaGraphRunner.replay` / `replay_prepare` | `cuda_graph_runner.py:602` / `:536` | 拷输入到静态 buffer → replay → 切片输出 |
| `get_batch_sizes_to_capture` | `cuda_graph_runner.py:127` | 决定捕获哪些 batch size |
| `get_model` | `python/sglang/srt/model_loader/__init__.py:15` | 模型加载统一入口 |
| `get_model_loader` | `python/sglang/srt/model_loader/loader.py:1418` | 按 `load_format` 选 loader |
| `DefaultModelLoader` | `loader.py:181` | 从 safetensors/bin/pt 加载权重 |
| `default_weight_loader` | `python/sglang/srt/model_loader/weight_utils.py:526` | 默认的逐参数 copy_ |
| `update_expert_location` | `expert_location_updater.py:29` | EPLB：在线搬运 MoE expert 权重 |

## 4. 核心数据流 / 执行流程

### 4.1 初始化：从 `__init__` 到能跑前向

`ModelRunner.__init__`（`model_runner.py:136`）按顺序做四件事，最后委托给 `initialize`（`model_runner.py:237`）：

```
__init__
 ├─ 解析 args、填 global_server_args_dict        (model_runner.py:190)
 ├─ model_specific_adjustment()  自动选 backend  (model_runner.py:184/313)
 ├─ init_torch_distributed()    建 TP/PP/DP 进程组(model_runner.py:223/414)
 └─ initialize(min_per_gpu_memory)               (model_runner.py:230)
      ├─ load_model()                   加载权重  (model_runner.py:270/498)
      ├─ init_memory_pool()        profile + 建池 (model_runner.py:296/821)
      ├─ init_attention_backend()                 (model_runner.py:303/990)
      └─ init_cuda_graphs()        capture graph  (model_runner.py:304/1081)
```

注意 `initialize` 里对 device 的分支（`model_runner.py:301-307`）：只有 `cuda` 才会 `init_cublas` + `init_cuda_graphs`，其它 device 把 `self.cuda_graph_runner = None`，前向永远走 eager。

**显存预算是怎么算出来的（关键，OOM 必看）**：`init_memory_pool`（`model_runner.py:821`）调 `profile_max_num_token`（`model_runner.py:784`）：

```
rest_memory   = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)
cell_size     = 每个 token 的 KV cache 字节数（MHA: 2 * num_kv_heads * head_dim * layers * dtype_size；
                 MLA: (kv_lora_rank + qk_rope_head_dim) * layers * dtype_size）
max_num_token = rest_memory(GB→B) // cell_size
```

也就是说 `--mem-fraction-static` 越大，留给权重+activation 的越少、留给 KV cache 的越多。算完后还会按 `page_size` 向下取整（`model_runner.py:890`），`<=0` 直接抛 "Not enough memory. Please try to increase --mem-fraction-static"（`model_runner.py:896`）。

KV pool 的三种实现按架构选择（`model_runner.py:912-955`）：MLA 模型用 `MLATokenToKVPool`，double sparsity 用 `DoubleSparseTokenToKVPool`，其余用 `MHATokenToKVPool`；`page_size==1` 用 `TokenToKVPoolAllocator`，否则用 `PagedTokenToKVPoolAllocator`（`model_runner.py:957-972`）。详见 [KV 缓存](13-memcache-radixattention.md)。

### 4.2 ForwardBatch：进来是什么、怎么算 positions

`ForwardBatch.init_new`（`forward_batch_info.py:255`）从 `ModelWorkerBatch` 拷 GPU 张量，核心字段：

- 永远要有：`forward_mode`、`batch_size`、`input_ids`、`req_pool_indices`、`seq_lens`、`out_cache_loc`、`seq_lens_sum`（`forward_batch_info.py:142-155`）。
- **decode 路径**：`positions = clamp_position(seq_lens)`（`forward_batch_info.py:333-335`），即每条序列的位置就是 `seq_len-1`（用 `torch.compile` 后的 `clamp_position`，`forward_batch_info.py:695`）。每个请求只算 1 个 token。
- **extend 路径**（`forward_batch_info.py:336-358`）：要额外算 `extend_seq_lens` / `extend_prefix_lens` / `extend_start_loc`，并用 triton kernel `compute_position_triton`（`forward_batch_info.py:622`）一次性算出所有 token 的 `positions`（`prefix_len + 0..seq_len`）。`torch_native` backend 走纯 torch 版本 `compute_position_torch`（`forward_batch_info.py:678`）。
- **idle 路径**（`forward_batch_info.py:317-319`）：`positions` 设成空张量直接返回——DP attention 下没有序列分到的 rank 走这条。

`ForwardMode`（`forward_batch_info.py:53`）的几个判定方法决定了下游分派，最关键的两个：

- `is_extend()`（`:76`）：`EXTEND / MIXED / DRAFT_EXTEND / TARGET_VERIFY` 都算 extend。
- `is_cuda_graph()`（`:106`）：只有 `DECODE / TARGET_VERIFY / IDLE` 能走 CUDA graph。**prefill（EXTEND）不走 graph**，这是因为 prefill 的 token 数是动态的（见第 5 节）。

### 4.3 forward：分派逻辑

`forward`（`model_runner.py:1159`）把 `forward_pass_id` 自增后包一层 expert-distribution recorder，真正分派在 `_forward_raw`（`model_runner.py:1180`）：

```python
can_run_cuda_graph = (
    forward_batch.forward_mode.is_cuda_graph()
    and self.cuda_graph_runner
    and self.cuda_graph_runner.can_run(forward_batch)
)
if can_run_cuda_graph:        ret = cuda_graph_runner.replay(...)   # 命中 graph
elif is_decode():             ret = forward_decode(...)            # decode eager 回退
elif is_extend():             ret = forward_extend(...)            # prefill / verify
elif is_idle():               ret = forward_idle(...)             # DP 空闲 rank
return ret, can_run_cuda_graph
```

注意返回的第二个值 `can_run_cuda_graph` 会一路传回 tp_worker / scheduler，用于上层统计与 overlap 调度。

`forward_extend`（`model_runner.py:1123`）与 `forward_decode`（`model_runner.py:1111`）的差异：

- 两者都先 `self.attn_backend.init_forward_metadata(forward_batch)` 建 attention 元数据（除非 `skip_attn_backend_init`，spec decode 时会跳）。
- extend 额外处理 `input_embeds`（`.bfloat16()`，`model_runner.py:1135`）和 embedding 模型的 `get_embedding=True`（`model_runner.py:1137`）。
- 最终都调同一个 `self.model.forward(input_ids, positions, forward_batch, **kwargs)`——**模型本体不区分 prefill/decode，区别全在 `forward_batch` 里的元数据和 attention backend 的实现**。

### 4.4 CUDA graph 的捕获与 replay

**捕获时机**：`init_cuda_graphs`（`model_runner.py:1081`）只在 generation 模型且未 `--disable-cuda-graph` 时构造 `CudaGraphRunner`。构造函数（`cuda_graph_runner.py:188`）一次性：

1. `get_batch_sizes_to_capture`（`cuda_graph_runner.py:127`）算出要捕获哪些 bs（见 4.5）。
2. 申请一组**静态输入 buffer**（`cuda_graph_runner.py:241-251`）：`input_ids` / `req_pool_indices` / `seq_lens` / `out_cache_loc` / `positions` / `mrope_positions` / `num_token_non_padded`，全部按 `max_num_token` / `max_bs` 预分配。
3. `attn_backend.init_cuda_graph_state(...)`（`cuda_graph_runner.py:221-224`）让 attention backend 也分配它自己的静态元数据 buffer。
4. `capture()`（`cuda_graph_runner.py:354`）在 `graph_capture()` 上下文里**逆序**遍历 `capture_bs`（逆序是为了跨 graph 更好地共享内存池，`cuda_graph_runner.py:360`），对每个 bs 调 `capture_one_batch_size`。

`capture_one_batch_size`（`cuda_graph_runner.py:393`）：把静态 buffer 切片成当前 bs 的视图，构造一个 `ForwardBatch`（`cuda_graph_runner.py:447`），让 attention backend 进入 capture 模式（`init_forward_metadata_capture_cuda_graph`，`cuda_graph_runner.py:475`），然后：

```python
for _ in range(2):          # 先 warmup 跑两次（cuda_graph_runner.py:505）
    torch.cuda.synchronize(); tp_group.barrier(); run_once()
with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=stream):
    out = run_once()        # 真正录制（cuda_graph_runner.py:512）
```

所有 graph 共享一个 `global_graph_memory_pool`（`cuda_graph_runner.py:172-182`），省显存。

**replay 时机与流程**：命中后 `_forward_raw` 调 `replay`（`cuda_graph_runner.py:602`），它先调 `replay_prepare`（`cuda_graph_runner.py:536`）：

```
replay_prepare:
  ├─ recapture_if_needed()                 spec decode 改 hidden mode 时重捕获
  ├─ bisect 选出 >= raw_bs 的最小 capture bs (cuda_graph_runner.py:552)
  ├─ 若 bs != raw_bs（要 padding）:
  │    seq_lens.fill_(1); out_cache_loc.zero_()  把"多出来的假请求"填成安全值
  ├─ copy_ 把真实输入拷进静态 buffer 的前 raw_num_token 段
  ├─ num_token_non_padded[...] = 真实 token 数
  └─ attn_backend.init_forward_metadata_replay_cuda_graph(...)  注意 seq_lens_sum 补了 (bs-raw_bs)
replay:
  ├─ self.graphs[self.bs].replay()         (cuda_graph_runner.py:616)
  └─ 把 output 切片回 raw_num_token        (cuda_graph_runner.py:619-630)
```

**为什么 replay 是"拷进去 → replay → 切回来"**：CUDA graph 录制的是**固定地址、固定 shape** 的一串 kernel。replay 时不能换张量，只能原地改静态 buffer 的内容（`copy_`），所以真实输入必须拷进预分配 buffer；输出也在固定 buffer 里，要按真实 `raw_num_token` 切回去丢掉 padding 部分。

### 4.5 batch size padding：为什么、怎么选

`get_batch_sizes_to_capture`（`cuda_graph_runner.py:127`）的默认列表（无 spec decode、允许 padding 时）：`[1,2,4,8] + range(16,161,8)`（`cuda_graph_runner.py:136`），大显存（>96GB）再加 `range(160,257,8)`。`--disable-cuda-graph-padding` 时改成更密的 `range(1,33)+range(40,161,16)`。

运行时 `can_run`（`cuda_graph_runner.py:328`）判定：

- 允许 padding：只要 `raw_bs <= max_bs` 就行（用 `bisect` 找 ≥raw_bs 的最近捕获 bs）。
- `disable_padding`：要求 `raw_bs` 精确命中某个已捕获的 bs（`batch_size in self.graphs`）。

**为什么要 padding**：如果给每一个可能的 bs（1..256）都捕获一张 graph，捕获时间和显存都吃不消。所以只捕获稀疏的几十个 bs，运行时把真实 bs **向上取整**到最近的捕获 bs，多出来的"假请求"用 `seq_lens=1` / `out_cache_loc=0` 填充（`replay_prepare`，`cuda_graph_runner.py:554-556`），算完丢弃。代价是少量算力浪费，换来 graph 数量可控。

### 4.6 model_loader：权重怎么进显存

`ModelRunner.load_model`（`model_runner.py:498`）→ `get_model`（`model_loader/__init__.py:15`）→ `get_model_loader(load_config)`（`loader.py:1418`）按 `load_format` 返回 loader：

| load_format | loader | file:line |
| --- | --- | --- |
| AUTO / SAFETENSORS / PT / NPCACHE / MISTRAL | `DefaultModelLoader` | `loader.py:181` |
| DUMMY | `DummyModelLoader`（随机权重，用于跑性能基准） | `loader.py:475` |
| SHARDED_STATE | `ShardedStateLoader` | `loader.py:521` |
| BITSANDBYTES | `BitsAndBytesModelLoader` | `loader.py:697` |
| GGUF | `GGUFModelLoader` | `loader.py:1173` |
| LAYERED | `LayeredModelLoader`（逐层加载+量化，降峰值显存） | `loader.py:403` |
| REMOTE | `RemoteModelLoader` | `loader.py:1271` |

`DefaultModelLoader.load_model`（`loader.py:367`）的两步：

```
with set_default_torch_dtype(dtype):
  with target_device:
     model = _initialize_model(...)          # 在目标 device 上建空 nn.Module (loader.py:143)
  load_weights_and_postprocess(model,
       self._get_all_weights(...), device)    # 喂权重迭代器 (loader.py:381)
```

`_get_weights_iterator`（`loader.py:323`）按文件类型选迭代器：safetensors 用 `safetensors_weights_iterator`（`weight_utils.py:417`，`load_file(..., device="cpu")` 逐文件读），`.bin/.pt` 用 `pt_weights_iterator`（`weight_utils.py:447`）。`load_weights_and_postprocess`（`loader.py:387`）调 `model.load_weights(weights)`（每个模型类自己实现 name→param 的映射），最后对每个有 `quant_method` 的 module 调 `process_weights_after_loading`（`loader.py:391-400`）做 repack/量化。

最底层的逐参数拷贝是 `default_weight_loader`（`weight_utils.py:526`）：标量用 `fill_`，否则 `assert size 相等` 后 `param.data.copy_(loaded_weight)`——**权重 shape mismatch 的报错就在这里**（`weight_utils.py:535-538`）。

加载完会做一次 `dist.monitored_barrier`（`model_runner.py:583`），超时（默认 300s）会报 "there are other ranks that didn't finish loading"，用于发现某个 rank OOM/慢节点导致的 TP 卡死。

### 4.7 在线更新权重 / EPLB

ModelRunner 还提供运行时换权重的能力（RLHF rollout 等场景）：`update_weights_from_disk`（`model_runner.py:603`）、`update_weights_from_distributed`（`model_runner.py:703`，通过 `_model_update_group` 从训练引擎 broadcast）、`update_weights_from_tensor`（`model_runner.py:736`）。

MoE 的专家位置在线迁移由 `update_expert_location`（`model_runner.py:593` → `expert_location_updater.py:29`）实现，配合 `EPLBManager`（在 `initialize` 里按 `--enable-eplb` 创建，`model_runner.py:262`）；它用 `torch.distributed.P2POp`（`expert_location_updater.py:19`）在 rank 间搬运 expert 权重。

## 5. 设计难点与权衡

### 5.1 为什么 prefill 不进 CUDA graph

CUDA graph 要求**输入 shape 静态**。decode 每个请求恰好 1 个 token，整个 batch 的 token 数 = `batch_size`，只随 bs 变化，可以枚举几十个 bs 捕获。但 prefill 的 token 数 = 各序列 `extend_seq_len` 之和，是连续且范围巨大的动态值（几十到几万），无法枚举捕获。所以 `is_cuda_graph()`（`forward_batch_info.py:106`）只放行 `DECODE / TARGET_VERIFY / IDLE`。spec decode 的 `TARGET_VERIFY` 能进 graph 是因为每个请求验证的 draft token 数固定为 `speculative_num_draft_tokens`（`cuda_graph_runner.py:214`），token 数仍是 `bs * num_tokens_per_bs` 的静态值。

### 5.2 padding 的"安全值"陷阱

padding 出来的假请求必须填成不会越界、不会污染真实结果的值。`replay_prepare` 把 `seq_lens.fill_(1)`、`out_cache_loc.zero_()`（`cuda_graph_runner.py:555-556`）：`seq_len=1` 保证 attention 不读到垃圾 KV，`out_cache_loc=0` 让假 token 写到 slot 0（之后被丢弃）。同时 `num_token_non_padded` 记录真实 token 数（`cuda_graph_runner.py:564`），让模型内部（如某些 kernel 的 early-exit）知道哪些是 padding。`seq_lens_sum` 传给 attention backend 时也补了 `(bs - raw_bs)`（`cuda_graph_runner.py:590`），因为每个假请求贡献 1。改这块时一定要保证假请求"既存在又无害"。

### 5.3 attention backend 的"三段式" capture 协议

CUDA graph 不止录模型本体，还要录 attention kernel，所以每个 backend 必须实现三个钩子：`init_cuda_graph_state`（预分配元数据 buffer，`cuda_graph_runner.py:221`）、`init_forward_metadata_capture_cuda_graph`（capture 时，`cuda_graph_runner.py:475`）、`init_forward_metadata_replay_cuda_graph`（replay 时原地更新，`cuda_graph_runner.py:586`）。新增一个 attention backend 若不实现这三个方法，CUDA graph 会失败。`flashmla` 还有特例：它的 `init_cuda_graph_state` 用 `max_bs` 而非 `max_num_token`（`cuda_graph_runner.py:221-224`）。

### 5.4 什么时候 fall back 到 eager

按优先级：

1. **整体关闭**：`--disable-cuda-graph` 或非 generation 模型 → `cuda_graph_runner=None`（`model_runner.py:1085-1090`）；double sparsity 会强制 `disable_cuda_graph=True`（`model_runner.py:380-385`）；非 cuda device 也没有 graph（`model_runner.py:305`）。
2. **单 batch 回退**：`can_run` 返回 False（bs 超过 `max_bs`，或 disable_padding 下未精确命中，或 encoder-decoder 的 `encoder_lens` 含 0 的 mixed batch，`cuda_graph_runner.py:344-352`）→ 走 `forward_decode` eager。
3. **mode 不支持**：prefill/extend 永远 eager。

捕获失败本身会抛带"四条解决方案"的异常（`cuda_graph_runner.py:308-317`：调小 `mem-fraction-static`、调小 `cuda-graph-max-bs`、关 torch.compile、关 cuda graph），这串提示是 OOM-during-capture 的标准排查路径。

### 5.5 torch.compile 与 CUDA graph 的关系

`patch_model`（`cuda_graph_runner.py:78`）在 `bs in compile_bs` 时对 model.forward 套 `torch.compile(..., mode="max-autotune-no-cudagraphs", dynamic=False)`。注意 `dynamic=False`——因为 graph 已经保证 shape 静态了，再让 dynamo 走 dynamic 反而拖慢。compile 只在 `--enable-torch-compile` 且 `bs <= torch_compile_max_bs` 时启用（`cuda_graph_runner.py:164-168`）。MoE 层有特例：只在 `num_tokens==1` 时用 native compile 路径（`cuda_graph_runner.py:66-70`）。

### 5.6 DP attention 下 bs 的含义变了

启用 DP attention/SP layernorm 时，`can_run` 和 `replay_prepare` 用的是 `sum(global_num_tokens_cpu)`（所有 DP rank 的 token 总和）而不是本地 `batch_size`（`cuda_graph_runner.py:329-336`、`:547-549`）。这也是为什么有 `forward_idle`：某些 rank 本地没序列，但为了让集合通信 shape 对齐，仍要参与一次"空"前向。

## 6. 改代码 / 修 bug 指南

**想改 X → 看这里：**

- **改/加 attention backend**：`init_attention_backend`（`model_runner.py:990`）加分支；务必实现三段式 cuda graph 钩子（见 5.3），否则要在 `model_specific_adjustment`（`model_runner.py:313`）里强制 `disable_cuda_graph`。
- **改自动选 backend 的策略**：`model_specific_adjustment`（`model_runner.py:316-409`），MHA/MLA、Hopper、spec decode 的判定都在这。
- **改显存/KV cache 预算**：`profile_max_num_token`（`model_runner.py:784`）和 `init_memory_pool`（`model_runner.py:821`）。OOM 优先调 `--mem-fraction-static`。
- **改/调 CUDA graph 捕获的 bs 列表**：`get_batch_sizes_to_capture`（`cuda_graph_runner.py:127`），或用 `--cuda-graph-bs` / `--cuda-graph-max-bs` / `--disable-cuda-graph-padding`。
- **加 ForwardBatch 新字段**：在 dataclass（`forward_batch_info.py:137`）加字段 + 在 `init_new`（`:255`）赋值；**如果该字段要进 CUDA graph，还得在 `CudaGraphRunner.__init__` 预分配静态 buffer（`cuda_graph_runner.py:241`）、在 `capture_one_batch_size`（`:393`）切片、在 `replay_prepare`（`:536`）`copy_` 进去**——漏任何一步都会让 graph 用到陈旧/错误数据。
- **改权重加载格式**：`get_model_loader`（`loader.py:1418`）加分支 + 写一个 `BaseModelLoader` 子类。
- **加新模型的权重映射**：模型类自己的 `load_weights`，底层 copy 用 `default_weight_loader`（`weight_utils.py:526`）。

**常见 bug 高发区：**

- **CUDA graph 用到 padding 的脏数据**：新加的张量没在 replay 时填安全值/没 copy_ 进静态 buffer。对照 `replay_prepare`（`cuda_graph_runner.py:554-580`）逐字段检查。
- **shape mismatch 加载报错**：`default_weight_loader`（`weight_utils.py:535`）的 assert，通常是 TP 切分 / 模型类 weight name 映射写错。
- **OOM during capture**：`mem-fraction-static` 太大导致捕获 graph 时没显存（异常在 `cuda_graph_runner.py:308`）。
- **TP rank 卡在加载**：`load_model` 末尾的 `monitored_barrier` 超时（`model_runner.py:583`），通常是某 rank OOM 或慢节点。
- **decode 结果错但 prefill 对**：很可能是 graph 路径与 eager 路径行为不一致——先用 `--disable-cuda-graph` 二分确认是不是 graph 的问题。

**调试入手点：**

1. `--disable-cuda-graph` 跑一遍：若问题消失，定位在 `CudaGraphRunner`（capture/replay 的 padding 与 buffer 同步）。
2. 在 `_forward_raw`（`model_runner.py:1180`）打印 `forward_batch.forward_mode` / `batch_size` / `can_run_cuda_graph`，确认走的是哪条路径。
3. 权重问题在 `default_weight_loader`（`weight_utils.py:526`）下断点——注释（`weight_utils.py:542`）专门保留了 `raise` 方便设断点。
4. 在 `capture_one_batch_size` 的 `run_once`（`cuda_graph_runner.py:486`）里检查捕获用的 `ForwardBatch` 是否和 replay 时一致。

## 相关章节

- [调度器](12-scheduler.md)：上游如何组 `ScheduleBatch` 并调用 `run_batch`。
- [TP Worker](15-worker-parallelism.md)：`ForwardBatch.init_new` 与 `model_runner.forward` 的直接调用方。
- [KV 缓存](13-memcache-radixattention.md)：`token_to_kv_pool` / `req_to_token_pool` 的内部结构。
- [采样](17-sampling-structured-output.md)：`ModelRunner.sample` 之后的 logits 处理。
- [术语表](02-glossary.md)：prefill / decode / TP / overlap scheduler / RadixAttention 等名词。
