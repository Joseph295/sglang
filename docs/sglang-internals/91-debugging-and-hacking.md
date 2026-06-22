# 改代码 / 修 bug 实战指南（Debugging & Hacking）

本章是整本内部文档的「工程师操作手册」。前面的章节按模块拆解了 SGLang 的设计与实现，本章则反过来组织：**从「我要干一件事 / 我遇到一个现象」出发，告诉你去看哪一章、哪个文件、哪个函数。** 它不引入新概念，而是把散落在各章 §6 的「改代码 / 修 bug 指南」汇总成一张可检索的速查网，并补上「新增模型」「日志 / metrics / profiling」「本地最小调试环路」这三块全局性的实战内容。

读法建议：
- 如果你**还没读过任何模块章节**，先把 [总览](00-overview.md) 和 [请求全生命周期](01-request-lifecycle.md) 过一遍，对「客户端 → 分词 → 调度 → KV 缓存 → 前向 → 采样 → 解码 → 响应」这条链路有个整体印象，再回来用本章定位。
- 如果你**已经知道大概在哪个模块**，直接跳到对应速查表，再点进该模块的专章 §6 看细节。

---

## 0. 速记卡（TL;DR）：最高杠杆的几招

> 贴在工位上的版本。原理与展开见后文对应小节，**进阶技巧见 §8**。

1. **二分关优化定位 bug**：`--disable-cuda-graph` →（还在）`--disable-overlap-schedule` →（还在）`--disable-radix-cache`。bug 在哪一步消失，问题就在那条路径里。（§5.6 / §7.6）
2. **用 `rid` 串起三个进程的日志**：开 `--log-requests`，一条请求的 `rid` 贯穿 TokenizerManager / Scheduler / DetokenizerManager —— 这是多进程下追一条请求的唯一线索。（§8.1）
3. **不起 server，单进程跑一个 batch**：`python -m sglang.bench_one_batch ... --load-format dummy`，没有 ZMQ / HTTP，断点 print 直达；加 `--correct` 和 HF 对拍数值。（§7.2 / §7.3）
4. **数值对拍用「慢但可信」的实现做 ground truth**：attention 用 `--attention-backend torch_native`、采样用 `--sampling-backend pytorch`、整模型用 `reference_hf.py`。（§7.5）
5. **找复杂 kernel / 逻辑的单测当「活文档」**：`python -m sglang.srt.mem_cache.radix_cache`、`test_build_tree_kernel_efficient` 等带硬编码断言的入口，比读代码快得多。（§7.4 / §8.4）
6. **卡住（hang）就 `py-spy dump` 看每个进程卡在哪一行**，崩溃看父进程的 `SIGQUIT` traceback（不是子进程那一段）。（§8.2）
7. **内存泄漏靠不变量自检**：idle 时 `check_memory` 会校验 KV / req slot 守恒，改了任何 alloc/free 路径后长跑看它有没有报 `memory leak detected`。（§5.3 / §8.5）

---

## 1. 一句话职责

本章是「需求 / 现象 → 代码位置」的全局索引：把改代码的入手点、各模块 bug 高发区、profiling/metrics 开关、以及最小复现环路，集中成一份可直接动手的操作手册。

---

## 2. 在端到端链路中的位置

本章不对应任何单一模块，而是横跨整条链路。下图标出每个阶段对应的专章，作为全章导航地图：

```
 客户端 HTTP/Engine
        │
        ▼
 ┌──────────────────┐   入口 & OpenAI 协议   → 10-entrypoints-api.md
 │  Entrypoints     │   /v1/chat/completions, /generate, /start_profile ...
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   分词 / 准入 / 编排    → 11-tokenizer-manager.md
 │ TokenizerManager │   多模态预处理在这一层触发 → 24-multimodal.md
 └──────────────────┘
        │ (ZMQ)
        ▼
 ┌──────────────────┐   连续批处理 / overlap  → 12-scheduler.md
 │   Scheduler      │   采样参数/grammar 准入  → 17-sampling-structured-output.md
 │                  │   LoRA 准入             → 22-lora.md
 └──────────────────┘   PD / EPLB            → 21-pd-disaggregation.md / 25-eplb-expert-parallel.md
        │
        ▼
 ┌──────────────────┐   KV / RadixAttention   → 13-memcache-radixattention.md
 │  Memory / KV     │
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   ModelRunner / CUDA graph → 14-model-executor.md
 │  Model Executor  │   TP/PP/DP/EP 分布式        → 15-worker-parallelism.md
 │                  │   attention / MoE / 算子    → 16-layers-attention-moe.md
 │                  │   量化                      → 23-quantization.md
 │                  │   投机解码                  → 20-speculative-eagle.md
 │                  │   模型实现 (models/*.py)    → 本章 §4「新增模型」
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   采样 / 结构化输出     → 17-sampling-structured-output.md
 │    Sampling      │
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   detokenize / 输出回流 → 18-detokenizer-output.md
 │ DetokenizerMgr   │
 └──────────────────┘
        │
        ▼
   客户端收到响应
```

横切关注点（不在主链路某一点，而是贯穿全链）：
- **metrics / 日志 / profiling**：`python/sglang/srt/metrics/` + scheduler 的统计 + `/start_profile` 端点，见本章 §5。
- **数据结构**：`Req` / `ScheduleBatch` / `ForwardBatch` / `SamplingBatchInfo` 等，见 [核心数据结构速查](90-data-structures.md)。
- **术语**：prefill / decode / overlap scheduler / RadixAttention / TP 等，见 [术语表](02-glossary.md)。

---

## 3. 关键文件与类（全局速查表：「想改 X → 看这里」）

下表是本章的核心。每行给出「需求」「主要入手位置」「专章」。`file:line` 以仓库根为基准（如 `python/sglang/srt/managers/scheduler.py:1318`），行号随代码演进会漂移，**请把它当锚点，落地时用符号名 grep 确认**。

### 3.1 入口 / 协议层

| 你想做的事 | 入手位置 | 专章 |
|-----------|---------|------|
| 改 OpenAI 协议（请求/响应字段、流式 chunk 格式） | `python/sglang/srt/entrypoints/` 下的 OpenAI adapter；HTTP 路由在 `entrypoints/http_server.py` | [入口与 API](10-entrypoints-api.md) |
| 加一个新的 HTTP 端点 | `http_server.py` 用 `@app.api_route(...)` 注册（参考 `start_profile_async` `http_server.py:336`） | [入口与 API](10-entrypoints-api.md) |
| 改 tool calling / function calling | 工具解析层（最近提交 `feat(Tool Calling)` 引入 `required`/specific function 模式） | [入口与 API](10-entrypoints-api.md) |
| 改分词 / chat template / 准入限流 | `TokenizerManager`（`managers/tokenizer_manager.py`） | [TokenizerManager](11-tokenizer-manager.md) |

### 3.2 调度层

| 你想做的事 | 入手位置 | 专章 |
|-----------|---------|------|
| 改准入策略（一次塞多少请求/多少 token） | `PrefillAdder.add_one_req`（`managers/schedule_policy.py:445`）；预算 `rem_total_tokens`（`:309`） | [调度器](12-scheduler.md) |
| 改 waiting_queue 排序 / 新调度策略 | `SchedulePolicy.calc_priority`（`schedule_policy.py:90`），挂到 `CacheAwarePolicy`/`CacheAgnosticPolicy` | [调度器](12-scheduler.md) |
| 改 prefill vs decode 优先级 | `get_next_batch_to_run`（`scheduler.py:1318`）的分支 | [调度器](12-scheduler.md) |
| 改 chunked prefill 行为 | `add_one_req` 截断逻辑（`schedule_policy.py:497`）；跨批清理（`scheduler.py:1289`） | [调度器](12-scheduler.md) |
| 改 OOM 回退（retract） | `retract_decode`（`schedule_batch.py:1337`）；触发点 `update_running_batch`（`scheduler.py:1506`） | [调度器](12-scheduler.md) |
| 改请求终止条件（新 stop 规则） | `Req.check_finished`（`schedule_batch.py:683`） | [调度器](12-scheduler.md) |
| 改 overlap 前向线程 | `forward_thread_func_`（`managers/tp_worker_overlap_thread.py:125`） | [调度器](12-scheduler.md) |
| 加一种新 RPC / 控制请求 | `process_input_requests` + `_request_dispatcher`（`scheduler.py:880`） | [调度器](12-scheduler.md) |

### 3.3 内存 / KV 缓存层

| 你想做的事 | 入手位置 | 专章 |
|-----------|---------|------|
| 改前缀匹配 / page 对齐规则 | `RadixCache.match_prefix`（`mem_cache/radix_cache.py:138`）、`_match_prefix_helper`（`:336`） | [内存与 KV 缓存](13-memcache-radixattention.md) |
| 改驱逐策略（不只是 LRU） | `RadixCache.evict`（`radix_cache.py:263`）、`TreeNode.__lt__`（`:73`） | [内存与 KV 缓存](13-memcache-radixattention.md) |
| 改 KV 物理布局 / 加 KV pool 类型 | 继承 `KVCache`（`mem_cache/memory_pool.py:102`），参考 `MHATokenToKVPool` / `MLATokenToKVPool` | [内存与 KV 缓存](13-memcache-radixattention.md) |
| 改 paged 分配 | `PagedTokenToKVPoolAllocator`（`mem_cache/paged_allocator.py:157`）及其 triton kernel | [内存与 KV 缓存](13-memcache-radixattention.md) |
| 改分层缓存 / 写回策略 | `HiRadixCache`（`mem_cache/hiradix_cache.py:23`）+ `HiCacheController`（`managers/cache_controller.py:146`） | [内存与 KV 缓存](13-memcache-radixattention.md) |
| 整体开关前缀复用 | `--disable-radix-cache` / `--enable-hierarchical-cache`；scheduler 选择在 `scheduler.py:498` | [内存与 KV 缓存](13-memcache-radixattention.md) |

### 3.4 模型执行 / 算子 / 并行

| 你想做的事 | 入手位置 | 专章 |
|-----------|---------|------|
| 加 / 切换 attention backend | `ModelRunner.init_attention_backend`（`model_executor/model_runner.py:990`）加分支；backend 类放 `layers/attention/`，继承 `AttentionBackend`（`layers/attention/base_attn_backend.py:14`） | [计算层](16-layers-attention-moe.md)、[模型执行](14-model-executor.md) |
| 改自动选 backend 的策略 | `model_specific_adjustment`（`model_runner.py:316`） | [模型执行](14-model-executor.md) |
| 改显存 / KV 预算 | `profile_max_num_token`（`model_runner.py:784`）、`init_memory_pool`（`model_runner.py:821`）；首选调 `--mem-fraction-static` | [模型执行](14-model-executor.md) |
| 调 CUDA graph 捕获 bs 列表 | `get_batch_sizes_to_capture`（`model_executor/cuda_graph_runner.py:127`）；或 `--cuda-graph-bs`/`--cuda-graph-max-bs` | [模型执行](14-model-executor.md) |
| 给 `ForwardBatch` 加字段 | dataclass（`model_executor/forward_batch_info.py:137`）+ `init_new`（`:255`）；进 CUDA graph 还要改 `cuda_graph_runner.py` 的预分配/切片/`copy_` | [模型执行](14-model-executor.md) |
| 改 MoE 路由 | `layers/moe/topk.py`（grouped_topk / biased_grouped_topk / fused_topk） | [计算层](16-layers-attention-moe.md) |
| 改 MoE 专家计算 | TP: `layers/moe/fused_moe_triton/layer.py:156`；EP: `layers/moe/ep_moe/layer.py:219` | [计算层](16-layers-attention-moe.md) |
| 改 TP / PP / DP / EP 分布式 | Worker 与并行初始化、通信组 | [Worker 与并行](15-worker-parallelism.md) |
| 改专家并行负载均衡（EPLB） | EPLB 重排与统计 | [EPLB](25-eplb-expert-parallel.md) |
| 改量化（FP8/INT4/AWQ/GPTQ） | `layers/quantization/` 各方法的 `create_weights` / `apply` | [量化](23-quantization.md) |
| 改投机解码 / EAGLE | `speculative/eagle_worker.py`、`speculative/eagle_utils.py` | [投机解码](20-speculative-eagle.md) |
| 改 PD 分离 | prefill/decode 两侧的 prealloc/transfer 队列 | [PD 分离](21-pd-disaggregation.md) |

### 3.5 采样 / 结构化输出

| 你想做的事 | 入手位置 | 专章 |
|-----------|---------|------|
| 加新采样参数（如 typical_p） | `SamplingParams.__init__`/`verify`（`sampling/sampling_params.py:30/89`）+ `SamplingBatchInfo`（`sampling/sampling_batch_info.py:205/300`）+ `Sampler.forward`（`sampling/sampler.py:202`） | [采样与结构化输出](17-sampling-structured-output.md) |
| 加新 penalty | `sampling/penaltylib/` 仿照 `frequency_penalty.py` 写 `_BatchedPenalizer` 子类 | [采样与结构化输出](17-sampling-structured-output.md) |
| 加新 grammar 后端 | 仿 `constrained/xgrammar_backend.py`，在 `create_grammar_backend`（`constrained/base_grammar_backend.py:167`）注册 | [采样与结构化输出](17-sampling-structured-output.md) |
| 加 custom logit processor | 继承 `CustomLogitProcessor`（`sampling/custom_logit_processor.py:19`），需 `--enable-custom-logit-processor` | [采样与结构化输出](17-sampling-structured-output.md) |

### 3.6 输出 / 多模态 / LoRA / 新模型

| 你想做的事 | 入手位置 | 专章 |
|-----------|---------|------|
| 改 detokenize / 流式增量 / 输出回流 | `DetokenizerManager`（`managers/detokenizer_manager.py`） | [DetokenizerManager](18-detokenizer-output.md) |
| 加新 VLM（多模态模型） | 模型类 + processor，见 [多模态](24-multimodal.md) §6 与本章 §4 | [多模态](24-multimodal.md) |
| 让一个 base model 支持 LoRA | 模型类实现 `get_module_name`/`get_hidden_dim`（参考 `models/llama.py:494/509`） | [Multi-LoRA](22-lora.md) |
| 加 LoRA 后端 / target module | 继承 `BaseLoRABackend`，在 `get_backend_from_name` 注册；target module 在 `lora/lora_config.py` | [Multi-LoRA](22-lora.md) |
| **加一个新模型** | `python/sglang/srt/models/your_model.py`（单文件），见本章 §4 | 本章 §4 |

---

## 4. 新增一个模型：步骤清单

新增模型是入门 SGLang 最高频的「改代码」任务。官方指南见 `docs/supported_models/support_new_models.md`，本节把它落到具体符号上。**核心约定：一个模型 = `python/sglang/srt/models/` 下的一个文件，文件末尾导出 `EntryClass`。** 不需要改注册表，启动时 `import_model_classes()`（`models/registry.py:69`）会自动扫描整个 `models/` 包，凡是模块有 `EntryClass` 属性就注册进 `ModelRegistry`。

### 4.1 注册机制（先理解它）

```
启动
 │
 ▼
import_model_classes()           models/registry.py:69
 │  pkgutil 遍历 sglang.srt.models 下所有模块
 │  对每个模块: 若 hasattr(module, "EntryClass") → 注册
 │  EntryClass 可以是单个类，也可以是 list（一个文件多个架构）
 ▼
ModelRegistry.models[arch_name] = cls        registry.py:99
 │
 ▼
resolve_model_cls(architectures)  registry.py:54
 │  用 config.json 里的 architectures 字段匹配类名
 ▼
实例化模型类，调 load_weights 加载权重
```

注意 `import_model_classes` 对每个模块的 import 用 `try/except` 包住，import 失败只打一条 warning 就跳过（`registry.py:77-79`）。所以**如果你的新模型文件有 import 错误，表现是「架构不被支持」而不是报错堆栈** —— 启动日志里搜 `Ignore import error when loading` 能看到真正的异常。

类名（`EntryClass` 的 `__name__`）必须与 HuggingFace `config.json` 的 `architectures`（如 `LlamaForCausalLM`）一致，否则 `resolve_model_cls` 匹配不上。

### 4.2 纯文本模型：最小骨架

以 `models/llama.py` 为参考样板（它也是从 vLLM 移植过来的标准范例）。一个 causal LM 类需要：

| 组成部分 | llama.py 锚点 | 说明 |
|---------|--------------|------|
| `__init__(config, quant_config, prefix)` | `models/llama.py:403` | 构建 `self.model`（decoder 堆叠）、`self.lm_head`、`self.logits_processor`；处理 `tie_word_embeddings`（`:416`） |
| decoder layer 里用 `RadixAttention` | `models/llama.py:170` | **必须把 `layer_id` 传给 `RadixAttention`** —— 它是 KV pool 寻址的 key |
| `forward(input_ids, positions, forward_batch, ...)` | `models/llama.py:448` | 跑 backbone 得 `hidden_states`，最后过 `self.logits_processor(...)` 返回 `LogitsProcessorOutput`（`:471`） |
| `load_weights(weights)` | `models/llama.py:532` | 把 HF checkpoint 的权重名映射到本类参数；融合权重（qkv、gate_up）用 `stacked_params_mapping`（`:428`） |
| `EntryClass` | `models/llama.py:717` | 文件末尾导出，可以是 list（llama.py 一次导出 3 个架构） |

从 vLLM 移植时的关键替换（来自 `support_new_models.md`）：vLLM 的 `Attention` → `RadixAttention`（带 `layer_id`）；vLLM `LogitsProcessor` → SGLang `LogitsProcessor`；vLLM 的 `RMSNorm`/`SiluAndMul` 等 → SGLang 同名层；删掉 `Sample`；把 `forward` 改成接收 `forward_batch`；末尾加 `EntryClass`；不能残留任何 vLLM 组件。

权重融合是 `load_weights` 最容易踩坑的地方。llama 把 `q_proj/k_proj/v_proj` 融成 `qkv_proj`、`gate_proj/up_proj` 融成 `gate_up_proj`，靠 `stacked_params_mapping`（`models/llama.py:428`）这张表驱动；底层逐张量 copy 走 `default_weight_loader`（`layers/quantization/.../weight_utils.py` 的 `default_weight_loader:526`），shape 对不上会在那里 assert。

### 4.3 多模态模型（VLM）：额外步骤

VLM 在纯文本骨架之外，还要按 [多模态](24-multimodal.md) §6 做：

1. **模型类**：实现 `get_input_embeddings()` 返回 text embedding 层；实现 `get_image_feature(items)` 跑 vision encoder + projector；实现 `pad_input_ids(input_ids, mm_inputs)` 膨胀占位 token；`forward` 里调 `general_mm_embed_routine(..., image_data_embedding_func=self.get_image_feature)`。参考 `models/qwen2_vl.py` 或 `models/gemma3_mm.py`。
2. **processor**：在 `managers/multimodal_processors/`（或 `multimodal_processor.py` 体系）下继承 `BaseMultimodalProcessor`，设 `models = [YourVLMForConditionalGeneration]`，实现 `process_mm_data_async`。processor 也是自动发现，不用手动注册。
3. **注册为多模态**：按 `support_new_models.md`，在 `configs/model_config.py` 的 `is_multimodal_model` 让你的架构返回 `True`；必要时在 `conversation.py` 注册 chat template。
4. **ViT attention**：把 ViT 的多头 attention 换成 SGLang 的 `VisionAttention`。

### 4.4 正确性验证（务必做）

`support_new_models.md` 给的标准对拍流程：

```bash
# 1) HF 参考输出
python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,mllm}

# 2) SGLang 输出（应给出相同文本 + 非常接近的 prefill logits）
python3 -m sglang.bench_one_batch --correct --model [new model]
```

`bench_one_batch.py:15` 的 `--correct`（即 `--correctness-test`，`bench_one_batch.py:108`）就是为模型对拍设计的。通过后，把模型加进 `test/srt/models/test_generation_models.py` 的 `ALL_OTHER_MODELS`，用：

```bash
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

### 4.5 不改源码注册外部模型

不想往 `models/` 里塞文件时，可在起服务前直接写 `ModelRegistry.models[model_name] = model_class`（见 `support_new_models.md` 末尾示例）。适合私有模型 / 快速实验。

---

## 5. 日志 / metrics / profiling

### 5.1 日志级别

`--log-level`（`server_args.py:94`，默认 `info`）控制各进程的 Python logging 级别；`--log-level-http`（`:786`）单独控制 HTTP server 的访问日志，调试时把它压到 `warning` 可以让 scheduler 的批处理日志更清爽。每个进程（TokenizerManager / Scheduler / DetokenizerManager）都是独立进程独立 logger，定位问题时先确认你看的是哪个进程的日志。

scheduler 的两条核心运行日志是 `log_prefill_stats`（`managers/scheduler.py:1125`）和 `log_decode_stats`（`:1178`）—— 它们打印每轮 batch 的 running req 数、token 用量、cache 命中率、生成吞吐等，是观察「服务此刻在干什么」的第一手信息。

### 5.2 Prometheus metrics

用 `--enable-metrics`（`server_args.py:99`）打开后，scheduler 侧的 `SchedulerMetricsCollector`（`metrics/collector.py:150`）和 tokenizer 侧的 `TokenizerMetricsCollector`（`:300`）会把指标暴露到 `/metrics` 端点（Prometheus 格式，前缀 `sglang:`）。常用指标：

| 指标 | 定义位置 | 含义 |
|------|---------|------|
| `sglang:num_running_reqs` | `collector.py:159` | 当前在跑的请求数 |
| `sglang:token_usage` | `collector.py:173` | KV pool token 占用率 |
| `sglang:gen_throughput` | `collector.py:180` | 生成 token 吞吐 |
| `sglang:num_queue_reqs` | `collector.py:187` | waiting_queue 长度 |
| `sglang:cache_hit_rate` | `collector.py:201` | 前缀缓存命中率（RadixAttention 是否生效的直接信号） |
| `sglang:spec_accept_length` | `collector.py:208` | 投机解码平均接受长度（EAGLE 调参核心指标） |
| `sglang:prompt_tokens_total` / `generation_tokens_total` | `collector.py:315/321` | 累计输入/输出 token |
| PD 相关队列（prealloc/inflight/transfer） | `collector.py:223-258` | PD 分离时各阶段队列深度与失败计数 |

`stats` 由 scheduler 在每轮填充（如 `cache_hit_rate` 在 `scheduler.py:1168`），并在 idle 时也每 30s 收一次（`check_memory` 里 `scheduler.py:1266`）。

### 5.3 内存泄漏自检：`check_memory`

`Scheduler.check_memory`（`scheduler.py:1238`）在 idle 时被调用（`:650`/`:694`/`:799`），做两件守恒检查：
- KV 守恒：`allocator.available_size() + tree_cache.evictable_size()` 必须等于 `max_total_num_tokens`（hierarchical cache 下再减 `protected_size`），否则抛 `token_to_kv_pool_allocator memory leak detected!`（`:1251`）。
- req slot 守恒：`req_to_token_pool.free_slots` 必须等于 `size`，否则抛 `req_to_token_pool memory leak detected!`（`:1260`）。

**这是排查 KV / req slot 泄漏最直接的探针** —— 如果你改了任何 alloc/free 路径，长跑后看到这两条异常，就说明某条路径漏了 free。

### 5.4 PyTorch Profiler（看 kernel 时间线）

参考 `docs/references/benchmark_and_profiling.md`。两端都要设环境变量 `SGLANG_TORCH_PROFILER_DIR`：

```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
# 另一个终端
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 2 --sharegpt-output-len 100 --profile
```

服务端的实现是 `/start_profile` / `/stop_profile` 两个端点（`http_server.py:336/355`）→ `TokenizerManager.start_profile` → scheduler 的 `start_profile`（`scheduler.py:2086`）。`activities` 默认 `["CPU","GPU"]`（`:2104`），还支持 `"MEM"`（走 `torch.cuda.memory._record_memory_history`，`:2132`）和 `"CUDA_PROFILER"`（`:2134`）。trace 用 `export_chrome_trace`（`:2154`）落盘，到 https://ui.perfetto.dev 看。

两个实用 tip：
- trace 太大打不开 → 用 `--num-prompts 2 --sharegpt-output-len 100` 缩小。
- 想在 trace 里从 CUDA kernel 跳回 Python 源码 → 起服务时加 `--disable-cuda-graph`（否则 graph 把调用栈吃掉了）。

也可以脱离 server 直接 profile 一个 batch：`python3 -m sglang.bench_one_batch --model-path ... --batch 32 --input-len 1024 --output-len 10 --profile`（`bench_one_batch.py:13`）。

### 5.5 Nsight（更底层）

`nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch ...`，server 模式用 `--delay/--duration` 控制窗口，或 `nsys stop --session=profile-XXXXX` 手动收尾。细节见 `benchmark_and_profiling.md` 的 Nsight 小节。

### 5.6 调试用的「降级」开关汇总（`server_args.py`）

这些开关的价值是**二分定位**：关掉某个优化后 bug 消失，就说明 bug 在那条优化路径里。

| 开关 | 定义 | 作用 / 何时用 |
|------|------|--------------|
| `--disable-cuda-graph` | `server_args.py:160` | 走 eager forward；decode 错但 prefill 对时首选 |
| `--disable-cuda-graph-padding` | `:161` | 怀疑 graph padding 脏数据 |
| `--disable-overlap-schedule` | `:166` | 关 overlap 线程；区分普通/重叠路径 bug |
| `--disable-radix-cache` | `:159` | 关前缀复用；怀疑 cache 命中导致脏 KV |
| `--enable-nan-detection` | `:189` | 采样出 NaN/Inf 时定位 |
| `--enable-torch-compile` | `:184` | 反过来：默认关，怀疑 compile 引入问题时确认是否复现 |
| `--attention-backend torch_native` | `:317`（自动连带关 cuda graph `:321`） | 数值对拍的 ground truth backend |
| `--sampling-backend pytorch` | — | 用可读 torch 实现复现采样问题（`sampler.py:113`） |
| `--mem-fraction-static` | — | OOM / 捕获 graph OOM 时先调它 |

注意有些开关之间有自动联动（如 `torch_native` 后端会强制 `disable_cuda_graph`，`server_args.py:321`；某些后端会 `disable_overlap_schedule`），别被「我没设它怎么也变了」搞迷糊 —— 看 `server_args.py:285-461` 这一段的 post-init 调整逻辑。

### 5.7 环境变量调试开关（部分）

- `SGLANG_TORCH_PROFILER_DIR`：profiler trace 输出目录（§5.4）。
- `SGLANG_DEBUG_MEMORY_POOL=1`：打开 paged allocator 的对齐断言（`paged_allocator.py:198/225/262`），调 paged / page_size 逻辑时必开。
- `SGLANG_VLM_CACHE_SIZE_MB`：多模态 embedding cache 大小（默认 100MB）。
- `SGLANG_IO_WORKERS` / `SGLANG_CPU_WORKERS`：多模态预处理并行度（`base_processor.py:95-99`）。
- `SYNC_TOKEN_IDS_ACROSS_TP=1`：TP 下 grammar/采样跨 rank 不一致 hang 时用。

---

## 6. 各模块「常见 bug 高发区」汇总

下面把各专章 §6 的 bug 高发区压缩成一张索引表，方便从「现象」反查。详细排查步骤请回到对应专章 §6。

| 现象 / 症状 | 最可能的模块 & 高发点 | 专章 |
|------------|----------------------|------|
| 长跑后逐渐无法 prefill / `available_size` 下降 / `memory leak detected` | KV / req_pool 漏 free：chunked req 的 `free(req_pool_idx)`（`scheduler.py:1295`）、retract 的 free（`schedule_batch.py:1393`）、`cache_finished_req` 的两处 free（`radix_cache.py:205/233`） | [调度器](12-scheduler.md)、[内存与 KV 缓存](13-memcache-radixattention.md) |
| CUDA illegal memory access（overlap 下） | overlap 张量被提前释放：`forward_thread_func_` 用 `batch_lists` 持引用（`tp_worker_overlap_thread.py:136-139`） | [调度器](12-scheduler.md) |
| 输出乱码 / input_ids 出现负数 | future token 没 `resolve_future_token_ids` | [调度器](12-scheduler.md) |
| 「明明有空间却不收新请求」 | `batch_is_full` 缓存没被重置（`schedule_batch.py:803`，多处 reset） | [调度器](12-scheduler.md) |
| 请求静默消失 | abort 直接设了 `finished_reason` 而非 `to_abort` | [调度器](12-scheduler.md) |
| 驱逐了正在用的 KV / 读到脏数据 | `inc_lock_ref`/`dec_lock_ref` 不配对 | [内存与 KV 缓存](13-memcache-radixattention.md) |
| page 不对齐 assert | 用 `SGLANG_DEBUG_MEMORY_POOL=1` 打开对齐断言定位 | [内存与 KV 缓存](13-memcache-radixattention.md) |
| decode 结果错但 prefill 对 | CUDA graph 路径 vs eager 路径不一致；新加 `ForwardBatch` 字段没在 replay `copy_`（`cuda_graph_runner.py:554-580`） | [模型执行](14-model-executor.md) |
| 权重加载 shape mismatch | `default_weight_loader` assert（`weight_utils.py:535`）；TP 切分 / weight name 映射写错 | [模型执行](14-model-executor.md)、本章 §4 |
| 某 TP rank 卡在加载 | `load_model` 末尾 `monitored_barrier` 超时（`model_runner.py:583`），通常是某 rank OOM | [模型执行](14-model-executor.md)、[Worker 与并行](15-worker-parallelism.md) |
| logprob 数值不对 / 越界 | `logits_processor.py:288-330` 三索引逻辑，chunked prefill 边界（`:304`） | [计算层](16-layers-attention-moe.md) |
| MLA 模型选错 backend / 报错 | `model_runner.py:351-366` 的 MLA backend 校验 | [计算层](16-layers-attention-moe.md) |
| batch 变化后采样 shape 不一致 crash | 新增 sampling tensor 字段没加进 `filter_batch`/`merge_batch`（`sampling_batch_info.py:205/300`）；`merge_batch` 顺序坑（`:297`） | [采样与结构化输出](17-sampling-structured-output.md) |
| 结构化输出偶发非法 token | grammar mask 没在最后施加；overlap 下 `set_next_batch_sampling_info_done` 没调（`scheduler.py:1805`） | [采样与结构化输出](17-sampling-structured-output.md) |
| 「No processor registered for architecture」 | processor 模块 import 报错被静默吞掉（`multimodal_processor.py:38`）；看 `Ignore import error` warning | [多模态](24-multimodal.md) |
| 多模态 embedding 数量 mismatch | `get_mm_items_offset` 占位区间 vs encoder 输出 token 数不一致（`mm_utils.py:335`） | [多模态](24-multimodal.md) |
| LoRA 结果静默错误 | `stack_qkv_proj` 补零陷阱（`lora.py:122`）；rank 超 `max_lora_dim`（`lora_manager.py:123`） | [Multi-LoRA](22-lora.md) |
| EAGLE 接受率低 / KV 异常 | 投机 KV 「先分配后回滚 / 部分释放」与 page 对齐（`eagle_utils.py:475-487`）；参数耦合（`server_args.py:409-433`） | [投机解码](20-speculative-eagle.md) |
| 「架构不被支持」但你刚加了模型 | 模型文件 import 出错被 `registry.py:77` 跳过；搜 `Ignore import error when loading` | 本章 §4 |

---

## 7. 本地复现与最小调试环路

调 SGLang 的核心痛点是它是**多进程 + GPU + 大模型**，复现一次很慢。下面是把「调试环路」压到最小的建议，按从轻到重排列。

### 7.1 选最小模型 + 单卡 + eager

```bash
python -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    --mem-fraction-static 0.7 \
    --log-level debug
```

- 小模型（TinyLlama 1.1B 等）：加载快、显存小、单卡能跑。
- `--disable-cuda-graph`：走 eager，能在模型 `forward` 里随便 print / 设断点，且 CUDA 报错的栈是准的。
- `--disable-overlap-schedule`：去掉 overlap 线程，执行变成同步单线程，逻辑好追。
- 大多数**逻辑类** bug（调度、采样、KV 记账、模型 forward）在这套配置下都能复现。

### 7.2 用假权重 / 砍层数，跳过加载与显存

来自 `benchmark_and_profiling.md` 的 tips：

```bash
# 假权重：只需要正确的 config.json
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --batch 32 --input-len 256 --output-len 32 --load-format dummy

# 砍到 1 层 / 1 kv head：极快迭代结构性改动
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --batch 32 --input-len 256 --output-len 32 --load-format dummy \
    --json-model-override-args '{"num_hidden_layers": 1, "num_key_value_heads": 1}'
```

改调度 / KV / forward 结构（不关心数值正确性，只关心代码跑通）时，这是最快的环路。

### 7.3 不起 server，直接跑一个 batch

`python -m sglang.bench_one_batch ...` 在**单进程**里把模型 + 调度跑一遍，没有 ZMQ、没有 HTTP，断点和 print 都直达。带 `--correct`（`bench_one_batch.py:15`）还能和 HF 对拍数值。**调模型实现、attention backend、采样数值** 时优先用它。

### 7.4 脱离整个 server 玩单个模块

部分模块文件带 `__main__`，可单独运行：
- `python -m sglang.srt.mem_cache.radix_cache`（`radix_cache.py:499`）：单独玩 RadixCache 的 insert/match/evict。
- grammar 的 jump-forward：`outlines_jump_forward.py:181` 的 `test_main` 可直接打印某 regex 的 jump-forward 链。

### 7.5 数值对拍方法论

正确性 bug（输出对但数值偏 / 偶发错）最有效的手段是**用慢但可信的实现做 ground truth**：
- attention：`--attention-backend torch_native`（连带关 graph）。
- 采样：`--sampling-backend pytorch`。
- 整模型：`scripts/playground/reference_hf.py` 跑 HF，对比 `bench_one_batch --correct` 的 SGLang prefill logits。
- MoE topk：用 native topk 实现对比融合 kernel。

### 7.6 二分定位多进程 / 优化路径

确认 bug 在「哪条路径」上，逐个关优化（§5.6）：先 `--disable-cuda-graph`（graph 问题？）→ 再 `--disable-overlap-schedule`（overlap 问题？）→ 再 `--disable-radix-cache`（cache 脏 KV 问题？）。每关一个跑一遍，bug 在哪一步消失，问题就在那条路径里。scheduler 崩溃时父进程会收到 `SIGQUIT`（`scheduler.py:2326`），完整 traceback 在父进程日志里 —— 子进程的异常不要只看子进程那一段。

---

## 8. 进阶调试技巧（§5/§7 之外的巧思）

前面 §5（日志/metrics/profiling）和 §7（最小复现环路）已经覆盖了主干。这一节补的是**多进程系统特有**、容易被新人忽略、但一旦掌握就能省大量时间的技巧。

### 8.1 用 `rid` 当「贯穿三进程的探针」

SGLang 最反直觉的调试难点是：**一条请求的处理被切碎在三个进程里**（TokenizerManager / Scheduler / DetokenizerManager），任何一个进程的日志单看都只是「半截故事」。唯一能把它们缝起来的是 `rid`（请求在 `io_struct.py` 生成的 uuid，见 [生命周期](01-request-lifecycle.md) §②）。

- 开 `--log-requests`（配合 `--log-requests-level` 调详细度）让 TokenizerManager 打印每条请求的 `Receive`/`Finish`，带 `rid` 和参数。
- 排查「请求卡住不返回」时：按 `rid` 依次确认它**走到了哪个进程就没动了**——Scheduler 收到了吗？进 `waiting_queue` 了吗？`process_batch_result` 发出 `BatchTokenIDOut` 了吗？DetokenizerManager 收到了吗？TokenizerManager 的 `_handle_batch_output` 按 `rid` 找到 `state` 了吗（还是报 "state was deleted"）？
- 自己加临时日志时，**第一个字段永远打 `rid`**，否则多请求并发时日志无法归并。

### 8.2 进程 hang / 卡死：`py-spy` + 进程名

逻辑 bug 能靠 print 抓，但**死锁 / 卡住**时进程不报错、不退出，print 也不会触发。这时：

- **`py-spy dump --pid <pid>`**（无需重启、无需改代码）直接打印目标进程当前的 Python 调用栈，瞬间知道它卡在哪一行——是卡在 `recv_pyobj`（等不到上游消息）、`all_reduce`（某 rank 没到齐）、还是 `req.wait()`（P2P 配对不上）。多卡 hang 时对每个 rank 的进程都 dump 一次，对比谁在等谁。
- **怎么找到是哪个 pid**：SGLang 用 `setproctitle` 给每个角色起了名字——`sglang::scheduler`、`sglang::detokenizer`、`sglang::data_parallel_controller` 等（见 [Worker 与并行](15-worker-parallelism.md)）。`ps aux | grep sglang::` 一眼定位角色。
- 典型场景：[PD 分离](21-pd-disaggregation.md) 卡在 `Bootstrapping`、[EPLB](25-eplb-expert-parallel.md) 的 P2P 搬权重 hang、TP 下 grammar 不一致 hang（§5.7 的 `SYNC_TOKEN_IDS_ACROSS_TP`）——全靠 `py-spy` 看栈定位是谁没到齐。

### 8.3 崩溃：看**父进程**的 traceback，不是子进程

子进程（Scheduler 等）抛异常时，往往只在自己的日志里留半段栈，然后向父进程发 `SIGQUIT`（`scheduler.py` 的 `run_scheduler_process` 尾部、`tp_worker_overlap_thread.py`、`data_parallel_controller.py` 都有）。**完整的、可读的 traceback 在父进程日志里**。所以「server 突然整个退出」时，别盯着某个子进程的最后几行，往上翻到父进程捕获 `SIGQUIT` 打出的那一坨。

### 8.4 把「单测 / `__main__`」当活文档和试验台

读复杂 kernel / 数据结构时，**与其逐行啃实现，不如跑它的单测**——单测里的硬编码输入输出断言就是最精确的「行为规格」：

- `python -m sglang.srt.mem_cache.radix_cache`（文件尾 `__main__`）：手动 insert 几个有公共前缀的序列，`pretty_print()` 看树长什么样、`match_prefix` 返回什么。理解 RadixAttention 半小时胜过看半天代码。
- `test_build_tree_kernel_efficient`（[投机解码](20-speculative-eagle.md)）：给了 bs/topk/depth 具体值 + `positions`/`retrive_*` 的硬编码断言，是搞懂 EAGLE 候选树张量含义的唯一捷径。
- 改算法时**反过来用**：先改单测的期望值表达你想要的新行为，再改实现让它通过——等于给自己先写好规格。
- 离线复现算法：[EPLB](25-eplb-expert-parallel.md) 的 `rebalance_experts`、[采样](17-sampling-structured-output.md) 的 jump-forward `test_main` 都能脱离 server 单机喂构造输入验证。

### 8.5 改了 alloc/free 路径？主动加不变量断言

KV / req slot 泄漏是最难查的一类 bug，因为它**不立即报错**，而是长跑后慢慢耗尽。SGLang 的对策是 `check_memory`（§5.3）在 idle 时校验「守恒等式」。借鉴这个思路：

- 你改了任何分配/释放路径（retract、chunked free、投机 KV evict、LoRA 换入换出），**先想清楚守恒等式是什么**（如「`available + evictable + protected == max_total`」），临时在你的改动点后面 `assert` 它。错误会在**第一次失衡时**就炸，而不是几千个请求后才 OOM——把「滞后的症状」变成「即时的断言失败」。
- 同理，给 `ForwardBatch` 加字段进 CUDA graph 时（[模型执行](14-model-executor.md)），临时 assert「replay 时静态 buffer 的该字段 == 真实输入」，能立刻抓到忘了 `copy_` 的脏数据。

### 8.6 回归 bug：`git bisect` + 固定最小命令

「上周还好好的，今天输出变了」这类回归，手工读 diff 很慢。固定一条**确定性的最小复现命令**（贪心采样 `--temperature 0` + `bench_one_batch --correct` + 小模型 dummy 权重），然后 `git bisect run` 自动二分到引入问题的 commit。关键是复现命令必须**确定性**（关 overlap、关 cuda graph、greedy），否则 bisect 会被随机性误导。

### 8.7 在热路径安全地打印

Scheduler 主循环每个 token step 都跑，直接 `print` 会刷屏且拖慢到改变时序（掩盖 race）。技巧：

- **按 rank 门控**：只在 `tp_rank == 0` 打，避免 N 卡 ×N 倍日志。
- **按条件 / 抽样**：`if step % 100 == 0` 或只在某个特定 `rid` 命中时打。
- **优先用现成的统计**：`log_prefill_stats` / `log_decode_stats`（§5.1）已经把每轮 batch 的关键量打出来了，先看它们，不够再加。
- 调时序敏感的 bug（overlap、race）时，**用变量累积、最后一次性 dump**，而不是边跑边打——打印的 I/O 本身会改变时序。

### 8.8 善用「降级开关之间的联动」反推

§5.6 提到开关有自动联动（`torch_native` 连带关 cuda graph、某些后端关 overlap）。反过来用：当你「没设某开关它却变了」时，不要困惑——去读 `server_args.py` 的 post-init 调整段（`:285-461` 一带），它集中表达了「这些配置组合下系统实际跑的是什么」。**排查任何「实际行为和我设的参数不符」的问题，第一站永远是这段 + 启动日志里打印的最终 `server_args`。**

---

## 相关章节

- [总览](00-overview.md)：多进程架构与设计哲学，定位问题时先建立全局视图。
- [请求全生命周期](01-request-lifecycle.md)：一条请求从入到出的完整路径，配合本章导航图使用。
- [术语表](02-glossary.md)：prefill / decode / overlap scheduler / RadixAttention / TP 等名词。
- [核心数据结构速查](90-data-structures.md)：`Req` / `ScheduleBatch` / `ForwardBatch` / `SamplingBatchInfo` 字段速查。
- 各模块专章 §6（[调度器](12-scheduler.md)、[内存与 KV 缓存](13-memcache-radixattention.md)、[模型执行](14-model-executor.md)、[计算层](16-layers-attention-moe.md)、[采样与结构化输出](17-sampling-structured-output.md)、[多模态](24-multimodal.md)、[Multi-LoRA](22-lora.md)、[投机解码](20-speculative-eagle.md) 等）：本章是它们的索引，深入排查回到原章。
- 外部参考：`docs/supported_models/support_new_models.md`、`docs/references/benchmark_and_profiling.md`。
