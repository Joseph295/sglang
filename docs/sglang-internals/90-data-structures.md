# 核心数据结构速查

> 本章是配合 [请求生命周期](01-request-lifecycle.md) 的「字段速查手册」。把贯穿全链路的核心数据结构按「它在哪个阶段出现 / 关键字段含义 / 定义位置 file:line」做成一张张速查表，方便你读源码、改代码、修 bug 时随时翻查。术语（prefill / decode / TP / overlap scheduler / RadixAttention 等）见 [术语表](02-glossary.md)。

## 一句话职责

SGLang 一条请求从「客户端文本」到「响应字符串」要经过 5～6 个进程/线程边界，每跨一次边界，数据就被「换装」成一种更贴近当前阶段需求的结构。本章把这些「换装形态」全部列出来：用户层的 `GenerateReqInput` → 分词后的 `TokenizedGenerateReqInput` → 调度器内的状态对象 `Req` → 一个 batch 的 `ScheduleBatch` → 发给 worker 的 `ModelWorkerBatch` → 前向用的 `ForwardBatch` + `SamplingBatchInfo` → 回传的 `BatchTokenIDOut` → 最终 `BatchStrOut`。理解了这套「数据形态流水线」，就理解了 SGLang 的骨架。

## 在端到端链路中的位置

每个箭头都是一次「数据结构换装」，括号里是承载数据的类：

```
 客户端 HTTP
    │  GenerateReqInput            (io_struct.py:50)   用户原始请求，text/sampling_params 可单条可批
    ▼
 TokenizerManager  ── tokenize ──►
    │  TokenizedGenerateReqInput   (io_struct.py:424)  已分词、单条、SamplingParams 已构造
    ▼  (ZMQ 进程间)
 Scheduler.recv_requests
    │  Req                         (schedule_batch.py:420)  调度器内长生命周期状态机对象
    ▼  组 batch
 Scheduler.get_next_batch_to_run
    │  ScheduleBatch               (schedule_batch.py:787)  一个 batch 的全部调度信息 + KV pool 句柄
    ▼  get_model_worker_batch()
 TpModelWorker
    │  ModelWorkerBatch            (schedule_batch.py:1705) 精简版、只含 worker 需要的张量
    ▼  ForwardBatch.init_new()
 ModelRunner.forward
    │  ForwardBatch                (forward_batch_info.py:138)  前向一次 pass 的全部输入（含 attn backend 句柄）
    │  SamplingBatchInfo           (sampling_batch_info.py:20)  批量采样参数 + grammar mask + penalizer
    ▼  采样
 process_batch_result
    │  BatchTokenIDOut             (io_struct.py:578)   token id 级结果，回传给 DetokenizerManager
    ▼  detokenize (ZMQ)
 DetokenizerManager
    │  BatchStrOut                 (io_struct.py:631)   字符串级结果，回传 TokenizerManager → 客户端
    ▼
 客户端 HTTP 响应
```

调度/KV/前向各阶段的细节分别见 [调度器](12-scheduler.md)、[KV 缓存与内存池](13-memcache-radixattention.md)、[模型执行与前向](14-model-executor.md)、[采样](17-sampling-structured-output.md)。

## 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `GenerateReqInput` | python/sglang/srt/managers/io_struct.py:50 | 用户层生成请求，支持单条/批量混合输入 |
| `TokenizedGenerateReqInput` | python/sglang/srt/managers/io_struct.py:424 | 分词后、发给 Scheduler 的单条请求 |
| `EmbeddingReqInput` / `TokenizedEmbeddingReqInput` | python/sglang/srt/managers/io_struct.py:469 / 564 | embedding 任务对应的输入对 |
| `BatchTokenIDOut` | python/sglang/srt/managers/io_struct.py:578 | Scheduler → Detokenizer 的 token id 级结果 |
| `BatchStrOut` | python/sglang/srt/managers/io_struct.py:631 | Detokenizer → Tokenizer 的字符串级结果 |
| `BatchEmbeddingOut` | python/sglang/srt/managers/io_struct.py:681 | embedding 结果 |
| `AbortReq` | python/sglang/srt/managers/io_struct.py:817 | 中途中止某 rid 的控制消息 |
| `Req` | python/sglang/srt/managers/schedule_batch.py:420 | 调度器内单请求状态机（含 logprob/grammar/disagg 等全部状态） |
| `BaseFinishReason` 及子类 | python/sglang/srt/managers/schedule_batch.py:101 | 结束原因：`FINISH_MATCHED_TOKEN/STR`、`FINISH_LENGTH`、`FINISH_ABORT` |
| `MultimodalInputs` / `MultimodalDataItem` | python/sglang/srt/managers/schedule_batch.py:311 / 169 | 多模态输入容器 |
| `ScheduleBatch` | python/sglang/srt/managers/schedule_batch.py:787 | 一个 batch 在 scheduler 侧的全量信息 |
| `ModelWorkerBatch` | python/sglang/srt/managers/schedule_batch.py:1705 | ScheduleBatch 的精简快照，跨 worker 传递 |
| `ForwardMode` | python/sglang/srt/model_executor/forward_batch_info.py:53 | 前向模式枚举：EXTEND/DECODE/MIXED/IDLE/TARGET_VERIFY/DRAFT_EXTEND/DUMMY_FIRST |
| `CaptureHiddenMode` | python/sglang/srt/model_executor/forward_batch_info.py:120 | hidden states 捕获模式 NULL/FULL/LAST |
| `ForwardBatch` | python/sglang/srt/model_executor/forward_batch_info.py:138 | 单次前向 pass 的全部输入 + backend 句柄 |
| `SamplingBatchInfo` | python/sglang/srt/sampling/sampling_batch_info.py:20 | 批量采样参数张量 + grammar mask + penalizer |

## 核心数据流 / 执行流程

### 1. GenerateReqInput（用户层，支持批量与并行采样）

定义在 python/sglang/srt/managers/io_struct.py:50。它的特点是「字段几乎都可单条可列表」（`Union[List[...], ...]`），因为同一个类既要表示单条也要表示一个 batch。关键入口方法 `normalize_batch_and_arguments()`（io_struct.py:112）会做：校验输入（`_validate_inputs`，io_struct.py:136）→ 推断 batch_size（`_determine_batch_size`，io_struct.py:149）→ 处理并行采样 n（`_handle_parallel_sampling`，io_struct.py:177）→ 填默认值并展开。

```
GenerateReqInput(text="hi", sampling_params={"n":2})
   │  normalize_batch_and_arguments()
   ▼
is_single=False, batch_size=1, parallel_sample_num=2
text=["hi"] → 展开成 2 条 → 各分配 rid
```

| 字段 | 含义 / 备注 |
| --- | --- |
| `text` / `input_ids` / `input_embeds` (io_struct.py:52/54/56) | 三选一的输入；校验要求恰好提供一种 |
| `image_data` / `audio_data` (io_struct.py:63/67) | 多模态输入，单图/列表/嵌套列表均可 |
| `sampling_params` (io_struct.py:69) | dict 或 dict 列表；`n` 决定并行采样数 |
| `rid` (io_struct.py:71) | 请求 id，未提供则 `uuid4().hex` 生成 |
| `return_logprob` / `logprob_start_len` / `top_logprobs_num` / `token_ids_logprob` (io_struct.py:73-80) | logprob 控制；`logprob_start_len=-1` 表示只返回 output 的 logprob |
| `stream` (io_struct.py:84) | 是否流式 |
| `lora_path` (io_struct.py:91) | LoRA adapter 路径 |
| `custom_logit_processor` (io_struct.py:99) | 序列化的自定义 logit processor |
| `return_hidden_states` (io_struct.py:102) | 是否返回 hidden states |
| `bootstrap_host/port/room` (io_struct.py:105-107) | disaggregated（PD 分离）推理用 |
| `is_single` / `batch_size` / `parallel_sample_num` | 非声明字段，`normalize_*` 运行时设置 |

### 2. TokenizedGenerateReqInput（分词后，单条）

定义在 io_struct.py:424。这是 TokenizerManager 分词、把 dict 形式的 `sampling_params` 实例化成 `SamplingParams` 对象之后，通过 ZMQ 发给 Scheduler 的形态。注意它**总是单条**（没有 `Union[List, ...]`），批量在 TokenizerManager 里已经被拆开。

| 字段 | 含义 |
| --- | --- |
| `rid` (io_struct.py:426) | 请求 id |
| `input_text` / `input_ids` (io_struct.py:428/430) | 原始文本与分词后 token ids |
| `mm_inputs` (io_struct.py:432) | 处理好的多模态输入 dict |
| `sampling_params` (io_struct.py:434) | 已构造的 `SamplingParams` 对象（非 dict） |
| `return_logprob` / `logprob_start_len` / `top_logprobs_num` / `token_ids_logprob` (io_struct.py:436-442) | logprob 控制（均为标量） |
| `stream` / `lora_path` / `input_embeds` / `session_params` / `custom_logit_processor` / `return_hidden_states` (io_struct.py:444-460) | 同上游语义，单条化 |
| `bootstrap_host/port/room` (io_struct.py:463-465) | disaggregation |

### 3. Req（调度器内的请求状态机）

定义在 python/sglang/srt/managers/schedule_batch.py:420，构造函数 schedule_batch.py:423。这是**整个系统里最重要、字段最多的对象**：一条请求在 Scheduler 的整个生命周期内只有一个 `Req`，从入队、prefill、多轮 decode、到 finish/abort 全程复用。Scheduler 把 `TokenizedGenerateReqInput` 转成 `Req` 入队。

字段按用途分组速查：

**输入/输出基础（schedule_batch.py:444-458）**

| 字段 | 含义 |
| --- | --- |
| `rid` | 请求 id |
| `origin_input_text` / `origin_input_ids` | 原始 prompt 文本与 token ids |
| `origin_input_ids_unpadded` | 图像 padding 之前的原始 ids（增量 detokenize 用） |
| `output_ids` | 已生成的 token id 列表（每步 decode 追加一个） |
| `fill_ids` | `origin_input_ids + output_ids`，chunked 时会被更新（见 `init_next_round_input`，schedule_batch.py:624） |

**结束判定（schedule_batch.py:476-485）**

| 字段 | 含义 |
| --- | --- |
| `finished_reason` | `BaseFinishReason` 子类实例，`None` 表示未结束；`finished()`（schedule_batch.py:620）即判其非 None |
| `to_abort` / `to_abort_message` | 中途中止标记；**注意不要在事件循环中途直接设 `finished_reason`**，要走 `to_abort`，否则请求会被 filter 掉而永不响应（见 schedule_batch.py:479 注释） |
| `eos_token_ids` | EOS 集合 |

结束判定逻辑集中在 `check_finished()`（schedule_batch.py:683）：依次检查 `to_abort` → `max_new_tokens`（`FINISH_LENGTH`）→ grammar 终止 → EOS/stop_token_ids（`FINISH_MATCHED_TOKEN`）→ stop_strs（`FINISH_MATCHED_STR`）。

**增量 detokenize（schedule_batch.py:496-498）**

```
----- | --------- read_ids -------|
----- |   surr_ids  |
xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
----- ^ ----------- ^ ----------- ^
   surr_offset   read_offset   last token
```

| 字段 | 含义 |
| --- | --- |
| `surr_offset` / `read_offset` | 增量 detokenize 的两个游标，配合 `init_incremental_detokenize()`（schedule_batch.py:671）抵消 detokenizer 的清理算法边界问题 |
| `decoded_text` | 已 detokenize 出的文本 |

**前缀与 prefill（schedule_batch.py:505-516）**

| 字段 | 含义 |
| --- | --- |
| `prefix_indices` | RadixAttention 命中的共享前缀对应的 KV cache 索引 |
| `extend_input_len` | 本次 prefill 需要真正跑的 token 数 = `len(fill_ids) - len(prefix_indices)` |
| `last_node` / `last_node_global` | radix tree 上的命中节点（hierarchical cache 用） |
| `is_chunked` | chunked prefill 计数器；分块时 +1，处理一块 -1 |
| `req_pool_idx` (schedule_batch.py:472) | 在 `ReqToTokenPool` 中的行号 |

**logprob（参数 schedule_batch.py:529-535；返回值 schedule_batch.py:537-568）**

`return_logprob`、`logprob_start_len`、`top_logprobs_num`、`token_ids_logprob` 为输入参数；一大批 `input_*_logprobs_*` / `output_*_logprobs_*` 以及 `temp_*` 临时持有字段用于累积与回传。`return_logprob=True` 时才初始化 `output_token_logprobs_val` 等列表（schedule_batch.py:553）。

**其他**

| 字段 | file:line | 含义 |
| --- | --- | --- |
| `multimodal_inputs` | schedule_batch.py:501 | `MultimodalInputs`，由 `extend_image_inputs` 合并 |
| `grammar` / `grammar_wait_ct` | schedule_batch.py:574 | 受限解码（structured output）的 grammar 对象 |
| `cached_tokens` / `already_computed` | schedule_batch.py:578 | 命中缓存/已计算 token 计数 |
| `spec_verify_ct` | schedule_batch.py:583 | speculative decoding 的 verify pass 次数（算平均接受长度） |
| `time_stats` | schedule_batch.py:586 | `TimeStats`，metrics 用 |
| `bootstrap_*` / `disagg_kv_sender` / `start_send_idx` / `tmp_end_idx` / `metadata_buffer_index` | schedule_batch.py:592-608 | disaggregation（PD 分离）KV 传输状态 |
| `is_retracted` | schedule_batch.py:519 | 因 OOM 被回退（retract）；`reset_for_retract()`（schedule_batch.py:735）清空前缀/计算状态等待重排 |

辅助属性 `seqlen`（schedule_batch.py:610）= `len(origin_input_ids)+len(output_ids)`。

### 4. ScheduleBatch（一个 batch 在 scheduler 侧）

定义在 python/sglang/srt/managers/schedule_batch.py:787。它聚合了一批 `Req`、内存池/缓存的句柄，以及把这些 `Req` 拼成张量后的批处理输入。由 `init_new()`（schedule_batch.py:879）构造，然后通过 `prepare_for_extend()`（schedule_batch.py:1102）或 `prepare_for_decode()`（schedule_batch.py:1449）填充张量字段。

```
List[Req] ── init_new ──► ScheduleBatch(reqs, pools, tree_cache, ...)
                              │  prepare_for_extend / prepare_for_decode
                              ▼  填充 input_ids / seq_lens / out_cache_loc / sampling_info ...
                              │  get_model_worker_batch()
                              ▼
                          ModelWorkerBatch
```

**请求与缓存句柄（schedule_batch.py:791-794）**

| 字段 | 含义 |
| --- | --- |
| `reqs` | `List[Req]`，本 batch 的请求 |
| `req_to_token_pool` / `token_to_kv_pool_allocator` / `tree_cache` | KV 内存池与 RadixAttention 缓存句柄（见 [KV 缓存](13-memcache-radixattention.md)） |

**batch 配置（schedule_batch.py:797-809）**

| 字段 | 含义 |
| --- | --- |
| `forward_mode` | 本 batch 的 `ForwardMode` |
| `enable_overlap` | 是否 overlap scheduler |
| `batch_is_full` | 优化标记：满了就跳过新 prefill 检查 |
| `chunked_req` | PP 下 chunked prefill 的当前分块请求 |

**批处理张量（schedule_batch.py:812-828）**——这些就是要喂给模型的东西：

| 字段 | shape / 含义 |
| --- | --- |
| `sampling_info` / `next_batch_sampling_info` | `SamplingBatchInfo`；后者供 overlap 提前准备 |
| `input_ids` | `[total_tokens]` int64 |
| `input_embeds` | `[b, hidden]` float32（走 embeds 输入时） |
| `req_pool_indices` | `[b]` int64，定位 `ReqToTokenPool` 行 |
| `seq_lens` | `[b]` int64，各序列长度 |
| `out_cache_loc` | 本次输出 token 在 KV pool 的写入位置 |
| `output_ids` | `[b]` int64，上一步采样结果（decode 用） |
| `seq_lens_sum` | 所有 seq_len 之和 |

**extend/mixed 专用（schedule_batch.py:845-851）**：`prefix_lens`、`extend_lens`、`extend_num_tokens`、`decoding_reqs`、`extend_logprob_start_lens`、`extend_input_logprob_token_ids`。

**DP attention（schedule_batch.py:831-833）**：`global_num_tokens`、`global_num_tokens_for_logprob`、`can_run_dp_cuda_graph`。

**speculative / 其他**：`spec_algorithm` / `spec_info`（schedule_batch.py:869）、`encoder_*`（encoder-decoder，schedule_batch.py:854-857）、`has_stream` / `has_grammar`（schedule_batch.py:860-863）、`return_hidden_states`（schedule_batch.py:876）。

关键方法：`prepare_for_extend` / `prepare_for_decode` / `retract_decode`（OOM 回退，schedule_batch.py:1337）/ `filter_batch`（schedule_batch.py:1514，剔除已完成请求）/ `merge_batch`（schedule_batch.py:1571，合并 running batch）/ `get_model_worker_batch`（schedule_batch.py:1610）。

### 5. ModelWorkerBatch（跨 worker 的精简快照）

定义在 python/sglang/srt/managers/schedule_batch.py:1705，由 `ScheduleBatch.get_model_worker_batch()`（schedule_batch.py:1610）生成。它**只保留 worker / model runner 需要的字段**，把 `Req` 列表里逐请求的信息提前抽成列表/张量（如 `lora_paths=[req.lora_path for req in reqs]`，schedule_batch.py:1664），避免把整个 `Req` 对象传过去。

| 字段 | file:line | 含义 |
| --- | --- | --- |
| `bid` | schedule_batch.py:1707 | 全局自增 batch id（schedule_batch.py:783 的 `bid`），overlap 下做事件配对 |
| `forward_mode` / `input_ids` / `req_pool_indices` / `seq_lens` / `seq_lens_cpu` / `out_cache_loc` / `seq_lens_sum` | schedule_batch.py:1709-1721 | 核心张量，语义同 ScheduleBatch |
| `return_logprob` / `top_logprobs_nums` / `token_ids_logprobs` | schedule_batch.py:1724-1726 | logprob |
| `global_num_tokens*` / `can_run_dp_cuda_graph` | schedule_batch.py:1729-1731 | DP attention |
| `extend_*` | schedule_batch.py:1734-1738 | extend 信息 |
| `multimodal_inputs` / `encoder_*` / `lora_paths` | schedule_batch.py:1741-1750 | 多模态 / encoder-decoder / LoRA |
| `sampling_info` | schedule_batch.py:1753 | `SamplingBatchInfo` |
| `input_embeds` / `spec_algorithm` / `spec_info` / `capture_hidden_mode` | schedule_batch.py:1756-1762 | embeds / 投机解码 / hidden 捕获模式 |
| `launch_done` | schedule_batch.py:1765 | overlap 事件 |

`seq_lens_cpu` 仅在特定 attention backend（MLA+flashinfer / flashmla / fa3 / cutlass_mla）下才填（schedule_batch.py:1619）。`capture_hidden_mode` 的取值逻辑（schedule_batch.py:1669）：`return_hidden_states` 为真则 FULL，否则取 spec_info 的设定，再否则 NULL。

### 6. ForwardBatch（单次前向 pass 的全量输入）

定义在 python/sglang/srt/model_executor/forward_batch_info.py:138，由 `ForwardBatch.init_new(batch, model_runner)`（forward_batch_info.py:256）从 `ModelWorkerBatch` 构造。相比 `ModelWorkerBatch`，它额外持有 **attention backend / 内存池句柄 / positions / DP 通信 buffer** 等真正跑前向才需要的运行期对象。

```
ModelWorkerBatch ── ForwardBatch.init_new(model_runner) ──► ForwardBatch
                       │  挂上 attn_backend / token_to_kv_pool / positions / mrope_positions ...
                       ▼
                    model.forward(input_ids, positions, forward_batch)
```

| 字段 | file:line | 含义 |
| --- | --- | --- |
| `forward_mode` / `batch_size` / `input_ids` / `req_pool_indices` / `seq_lens` / `out_cache_loc` / `seq_lens_sum` | forward_batch_info.py:142-155 | 核心输入 |
| `return_logprob` / `top_logprobs_nums` / `token_ids_logprobs` | forward_batch_info.py:161-163 | logprob |
| `temp_scaled_logprobs` / `temperature` / `top_p_normalized_logprobs` / `top_p` | forward_batch_info.py:166-169 | logprob 后处理参数 |
| `positions` | forward_batch_info.py:172 | 位置编码 |
| `extend_*`（含 `extend_start_loc`、`*_cpu`、`extend_input_logprob_token_ids_gpu`） | forward_batch_info.py:175-182 | extend 信息（含 GPU 化的 logprob token ids） |
| `prefix_chunk_*` / `attn_attend_prefix_cache` / `num_prefix_chunks` | forward_batch_info.py:186-204 | MLA chunked prefix cache（见 `prepare_chunked_prefix_cache_info`，forward_batch_info.py:533） |
| `mm_inputs` | forward_batch_info.py:207 | 多模态 |
| `encoder_*` / `lora_paths` / `input_embeds` | forward_batch_info.py:210-219 | encoder-decoder / LoRA / embeds |
| `sampling_info` | forward_batch_info.py:222 | 采样信息 |
| `req_to_token_pool` / `token_to_kv_pool` / `attn_backend` | forward_batch_info.py:225-227 | **运行期句柄**：attention backend 和 KV 内存池 |
| `global_num_tokens_*` / `dp_local_*` / `gathered_buffer` / `can_run_dp_cuda_graph` | forward_batch_info.py:230-241 | DP attention 通信 |
| `spec_info` / `spec_algorithm` / `capture_hidden_mode` | forward_batch_info.py:244-246 | 投机解码 |
| `padded_static_len` / `num_token_non_padded` | forward_batch_info.py:249-250 | CUDA graph padding |
| `mrope_positions` | forward_batch_info.py:253 | Qwen2-VL 的 mrope（`_compute_mrope_positions`，forward_batch_info.py:411） |

`ForwardMode`（forward_batch_info.py:53）的判别方法很常用：`is_extend()` 会把 `EXTEND/MIXED/DRAFT_EXTEND/TARGET_VERIFY` 都算作 extend（forward_batch_info.py:76），`is_cuda_graph()` 判定 `DECODE/TARGET_VERIFY/IDLE`（forward_batch_info.py:106）——改投机解码或 cuda graph 逻辑时务必注意这两个的覆盖范围。

### 7. SamplingBatchInfo（批量采样参数）

定义在 python/sglang/srt/sampling/sampling_batch_info.py:20，由 `from_schedule_batch(batch, vocab_size)`（sampling_batch_info.py:59）构造：把每个 `Req` 的 `sampling_params` 抽成**列方向张量**。

| 字段 | file:line | 含义 |
| --- | --- | --- |
| `temperatures` / `top_ps` / `top_ks` / `min_ps` | sampling_batch_info.py:22-25 | 批量采样参数张量（`temperatures` 是 `[b,1]`，其余 `[b]`） |
| `is_all_greedy` | sampling_batch_info.py:28 | 全部贪心则可走快路径 |
| `need_min_p_sampling` | sampling_batch_info.py:31 | 是否有请求需要 min_p |
| `vocab_size` / `grammars` / `vocab_mask` / `apply_mask_func` | sampling_batch_info.py:34-37 | structured output 的 grammar mask；`update_regex_vocab_mask()`（sampling_batch_info.py:149）生成 mask |
| `sampling_info_done` | sampling_batch_info.py:40 | overlap scheduler 用的事件 |
| `penalizer_orchestrator` / `linear_penalty` | sampling_batch_info.py:43-44 | frequency/presence/min-new-tokens 等惩罚 |
| `has_custom_logit_processor` / `custom_params` / `custom_logit_processor` | sampling_batch_info.py:47-53 | 自定义 logit processor（按字符串 hash 合并同类，sampling_batch_info.py:97） |

关键方法：`filter_batch`（sampling_batch_info.py:199）、`merge_batch`（sampling_batch_info.py:274）必须与 `ScheduleBatch.filter_batch/merge_batch` 同步维护；`apply_logits_bias`（sampling_batch_info.py:187）在采样前对 logits 施加 mask/penalty。

### 8. BatchTokenIDOut / BatchStrOut（回传结果）

`BatchTokenIDOut`（io_struct.py:578）由 Scheduler 在 `process_batch_result` 后产生，发给 DetokenizerManager；后者 detokenize 成文本，封装为 `BatchStrOut`（io_struct.py:631）回传 TokenizerManager。两者都是**列表对齐**结构（第 i 个元素对应 `rids[i]`）。

`BatchTokenIDOut` 关键字段：`rids` / `finished_reasons`（io_struct.py:580-582）、增量 decode 三件套 `decoded_texts` / `decode_ids` / `read_offsets`（io_struct.py:584-586）、`output_ids`（仅 `--skip-tokenizer-init` 时，io_struct.py:588）、detokenize 配置 `skip_special_tokens` 等（io_struct.py:590-592）、token 计数 `prompt_tokens` / `completion_tokens` / `cached_tokens` / `spec_verify_ct`（io_struct.py:595-598）、一大批 logprob 字段（io_struct.py:601-612）、`output_hidden_states`（io_struct.py:615）。

`BatchStrOut` 与之类似，把 `decoded_texts/decode_ids/read_offsets` 换成最终 `output_strs`（io_struct.py:637），其余 token 计数与 logprob 字段一一对应。embedding 任务则走 `BatchEmbeddingOut`（io_struct.py:681，含 `embeddings`）。

## 设计难点与权衡

- **为什么要这么多「换装」？** 每一层只携带该层需要的字段，跨进程/线程边界时传输量最小、耦合最低。例如 `Req` 有几十个字段（grammar、disagg、metrics……），但 worker 前向只需要其中一小撮张量，于是 `get_model_worker_batch()` 把它压成 `ModelWorkerBatch`；`Req` 对象本身不跨 worker 边界。
- **单条 vs 批量的二义性。** `GenerateReqInput` 用 `Union[List, scalar]` 同时表达单条和批量，代价是 `normalize_batch_and_arguments()` 里大量分支（io_struct.py:112 起）。`is_single`/`batch_size` 是运行时才存在的非声明属性——直接读未 normalize 的对象会 `AttributeError`。
- **`Req` 是长生命周期可变状态机。** 同一个 `Req` 跨越 prefill→多轮 decode→finish，字段在不同阶段含义会变（如 `fill_ids` 在 chunked 时被改写）。这是 bug 高发区：retract（OOM 回退）后必须用 `reset_for_retract()`（schedule_batch.py:735）把前缀/计算状态清干净，否则重排时状态错乱。
- **中止要走 `to_abort` 而非 `finished_reason`。** 见 schedule_batch.py:479 的注释：事件循环中途若直接设 `finished_reason`，请求会在 `filter_batch` 时被剔除却没机会把结果发回，导致客户端永久挂起。正确做法是设 `to_abort=True`，由 `check_finished()` 在循环末尾统一转成 `FINISH_ABORT`。
- **ForwardMode 的「extend 家族」。** `is_extend()` 把 MIXED / DRAFT_EXTEND / TARGET_VERIFY 都算 extend（forward_batch_info.py:76）。新增模式或改投机解码时，若漏判这一点，attention backend 会走错分支。
- **SamplingBatchInfo 与 ScheduleBatch 的 filter/merge 必须配对。** batch 增删请求时，采样张量也要同步切片/拼接。两边的 `filter_batch`/`merge_batch` 是「孪生」实现，改一边忘改另一边会导致采样参数与请求错位（典型表现：温度/top_p 串到别的请求上）。
- **overlap scheduler 的事件字段。** `launch_done`（ScheduleBatch / ModelWorkerBatch）、`sampling_info_done`（SamplingBatchInfo）、`next_batch_sampling_info` 用来在「调度下一批」与「上一批前向/采样完成」之间做同步。`DUMMY_FIRST` 模式（forward_batch_info.py:71）专为启动这条流水线而存在。

## 改代码 / 修 bug 指南

- **想加一个新的用户可配置参数（如新的采样选项）→** 顺着这条链路逐处加字段：`GenerateReqInput`（io_struct.py:50）→ `normalize_*` 默认值 → `TokenizedGenerateReqInput`（io_struct.py:424）→ TokenizerManager 填充 → `Req.__init__`（schedule_batch.py:423）→ `SamplingParams` / `SamplingBatchInfo`（若是采样参数）→ 直至 `ForwardBatch`。漏掉中间任何一环，参数都会被「吃掉」。
- **想改结束条件 / stop 逻辑 →** 看 `check_finished()`（schedule_batch.py:683）与 `BaseFinishReason` 子类（schedule_batch.py:101-158）。
- **想改增量流式输出 / detokenize 偏移 →** 看 `Req.init_incremental_detokenize`（schedule_batch.py:671）、`surr_offset/read_offset`（schedule_batch.py:496），以及 `BatchTokenIDOut` 的 `decode_ids/read_offsets`（io_struct.py:585-586）。
- **想改采样 / 加 logit 处理 →** `SamplingBatchInfo.from_schedule_batch`（sampling_batch_info.py:59）和 `apply_logits_bias`（sampling_batch_info.py:187）；自定义 processor 走 `custom_logit_processor` 字段链。
- **想改 prefill 张量构造 / KV 分配 →** `ScheduleBatch.prepare_for_extend`（schedule_batch.py:1102）、`prepare_for_decode`（schedule_batch.py:1449）、`alloc_*_token_slots`（schedule_batch.py:927 起）。
- **OOM / retract 相关 bug →** `retract_decode`（schedule_batch.py:1337）+ `Req.reset_for_retract`（schedule_batch.py:735）；检查回退后 `prefix_indices`、`already_computed`、`req_pool_idx` 是否被正确清空。
- **filter/merge 后采样错位 →** 同时检查 `ScheduleBatch.filter_batch/merge_batch`（schedule_batch.py:1514/1571）与 `SamplingBatchInfo.filter_batch/merge_batch`（sampling_batch_info.py:199/274）是否对齐。
- **调试入手点：** `Req.__repr__`（schedule_batch.py:773）打印 rid/输入/输出/grammar/sampling；`ScheduleBatch.__str__`（schedule_batch.py:1697）打印 forward_mode 与请求数；给这两处加日志能快速定位「某请求在哪个 batch、什么模式下出问题」。

## 交叉引用

- [请求生命周期](01-request-lifecycle.md) —— 本章是它的字段速查附录
- [术语表](02-glossary.md)
- [调度器](12-scheduler.md) —— `Req` / `ScheduleBatch` 的使用方
- [KV 缓存与内存池](13-memcache-radixattention.md) —— `ReqToTokenPool` / `TokenToKVPoolAllocator` / RadixAttention
- [模型执行与前向](14-model-executor.md) —— `ForwardBatch` 的消费方
- [采样](17-sampling-structured-output.md) —— `SamplingBatchInfo` 详解
