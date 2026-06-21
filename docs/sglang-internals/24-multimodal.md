# 多模态（VLM）输入处理

## 一句话职责

把用户传入的图像 / 视频 / 音频，从「原始字节 / URL」一路转换成「可以被 LLM 消费的 embedding 向量」，并保证这些 embedding 在 prefill 时能被**精确地 scatter 到 input_ids 中正确的位置**——同时让占位 token 与 RadixAttention 的前缀缓存兼容、让 vision encoder 的结果能被多模态 cache 复用。

这一章覆盖的是「文字 token 序列里夹着图像」这一整套机制：从 `TokenizerManager` 调用 processor 预处理，到 `Scheduler` 把单个 image token 膨胀（pad）成多个占位 token，再到模型 forward 时 vision encoder 算出 embedding、`masked_scatter` 替换占位。

---

## 在端到端链路中的位置

多模态处理横跨了「分词」和「前向」两个阶段，是少数几个**前后端都要改**的子系统。

```
                                    本章覆盖范围
        ┌───────────────────────────────────────────────────────────────┐
客户端 → │ 分词(TokenizerManager)                          前向(ModelRunner) │ → 采样 → 解码 → 响应
请求   │  ├─ get_mm_processor() 选 processor              ├─ pad_input_ids   │
(图片) │  └─ process_mm_data_async()                      │   (Scheduler 调用)│
        │      ├─ load_mm_data() 下载/解码图片            ├─ general_mm_      │
        │      ├─ HF processor 算 pixel_values            │   embed_routine() │
        │      └─ 产出 mm_items + input_ids              │   ├─ vision encoder│
        │                                                 │   ├─ MM cache 查/存│
        │   ─── IPC ───►  Scheduler                       │   └─ masked_scatter│
        │                  └─ pad_input_ids_func()        │       占位→embedding│
        └───────────────────────────────────────────────────────────────┘
```

关键的两次「输入变形」：

1. **TokenizerManager 阶段**：`process_mm_data_async` 把一张图替换成模型期望的若干个特殊 token（比如 Qwen-VL 的 `<|vision_start|><|image_pad|>...<|vision_end|>`），同时算好 `pixel_values`。
2. **Scheduler 阶段**：`pad_input_ids` 把这些 placeholder token 全部替换成该图的 `pad_value`（一个由图像内容 hash 得到的伪 token id），让相同图像在 RadixAttention 里能命中前缀缓存。

参见 [TokenizerManager](11-tokenizer-manager.md)、[调度器](12-scheduler.md)、[KV 缓存与 RadixAttention](13-memcache-radixattention.md)、[Model Executor](14-model-executor.md)。

---

## 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `import_processors` | `python/sglang/srt/managers/multimodal_processor.py:31` | 扫描 `multimodal_processors/` 包，按 `cls.models` 自动注册 architecture → processor |
| `get_mm_processor` | `python/sglang/srt/managers/multimodal_processor.py:55` | 根据 `hf_config.architectures` 选出对应 processor 实例 |
| `PROCESSOR_MAPPING` | `python/sglang/srt/managers/multimodal_processor.py:15` | architecture(类) → processor 类 的全局映射表 |
| `BaseMultimodalProcessor` | `python/sglang/srt/managers/multimodal_processors/base_processor.py:83` | 所有 processor 的基类，含数据加载、并行解码、offset 工具 |
| `BaseMultimodalProcessor.load_mm_data` | `base_processor.py:255` | 把 prompt 按特殊 token 切分、并行下载解码、重组成新 prompt + images 列表 |
| `BaseMultimodalProcessor.process_mm_data` | `base_processor.py:102` | 调用 HF `AutoProcessor` 算 `pixel_values` 等张量 |
| `BaseMultimodalProcessor.get_mm_items_offset` | `base_processor.py:346` | 从 input_ids 找出每个 mm item 占据的 `(start,end)` 区间 |
| `MultimodalSpecialTokens` | `base_processor.py:37` | 描述某模型的 image/video/audio 特殊 token 与正则 |
| `MultimodalDataItem` | `python/sglang/srt/managers/schedule_batch.py:169` | 单个图/视频/音频的数据载体（pixel_values、hash、pad_value 等）|
| `MultimodalDataItem.set_pad_value` | `schedule_batch.py:210` | 对数据做 hash，得到 `pad_value`（占位伪 token id）|
| `MultimodalInputs` | `schedule_batch.py:311` | 一条请求的全部多模态输入（mm_items + 各种 token id）|
| `MultiModalityDataPaddingPatternMultimodalTokens` | `python/sglang/srt/managers/mm_utils.py:122` | 占位策略：连续重复 token（`<image><image>...`）|
| `MultiModalityDataPaddingPatternTokenPairs` | `mm_utils.py:47` | 占位策略：成对 token 包裹（`<img_start>...<img_end>`）|
| `embed_mm_inputs` | `mm_utils.py:360` | 核心：算 vision embedding 并 scatter 进 text embedding |
| `get_embedding_and_mask` | `mm_utils.py:255` | 查 cache / 算 embedding，并生成占位 mask |
| `general_mm_embed_routine` | `mm_utils.py:505` | 模型 forward 的统一入口：embed 多模态 + 调 language model |
| `MultiModalCache` | `python/sglang/srt/mem_cache/multimodal_cache.py:6` | vision encoder 结果的 LRU-ish cache（按字节大小限容）|
| `gpu_tensor_hash` | `python/sglang/srt/layers/multimodal.py:49` | Triton kernel，在 GPU 上对 tensor 做 hash（给 pad_value 用）|
| `process_anyres_image` / `expand2square` | `python/sglang/srt/mm_utils.py` | LLaVA-NeXT 的 anyres / pad 切图逻辑 |

---

## 核心数据流 / 执行流程

### 阶段一：TokenizerManager 里的预处理

入口在 `python/sglang/srt/managers/tokenizer_manager.py:463`：只要请求带多模态输入（`obj.contains_mm_input()`），就调用 `self.mm_processor.process_mm_data_async(...)`，并把返回的 `input_ids` 覆盖掉纯文本分词的结果。

`mm_processor` 本身是在初始化时通过 `import_processors()` + `get_mm_processor()` 选出来的（`tokenizer_manager.py:201` 和 `:213`）。

以 Qwen2.5-VL 为例（`python/sglang/srt/managers/multimodal_processors/qwen_vl.py:45`），`process_mm_data_async` 的内部流程：

```
原始输入: prompt="...<|vision_start|><|image_pad|><|vision_end|>...", image_data=[url]
   │
   ▼  load_mm_data()  (base_processor.py:255)
   │   1. 把 image_token / regex 转成字符串, collect() 成一个总正则
   │   2. re.split() 按特殊 token 把 prompt 切成 text_parts
   │   3. submit_data_loading_tasks(): 用 io_executor 线程池并行下载/解码图片
   │      - 图片 → PIL.Image; 视频 → encode_video() 抽帧; 音频 → load_audio()
   │   4. 重新拼 new_text: 每张图/每帧 → 一个 image_token
   │
   ▼  (Qwen 特有) smart_resize(): 把图缩放到 28 的整数倍, 像素落在 [min,max]
   │
   ▼  process_mm_data()  (base_processor.py:102)
   │   调 HF AutoProcessor: text=[new_text], images=[...]
   │   → 返回 input_ids(已被 HF 膨胀成多个 image_pad), pixel_values, image_grid_thw
   │
   ▼  get_mm_items_offset(input_ids, image_token_id)  → [(start,end), ...]
   │
   ▼  组装 MultimodalDataItem(pixel_values=..., image_grid_thws=..., image_offsets=...)
   │   (Qwen 还顺便算了 mrope_positions)
   │
   ▼ 返回 dict: { input_ids, mm_items, im_start_id, im_end_id, im_token_id, ... }
```

输出的 dict 会经 IPC 送到 Scheduler，在那里被 `MultimodalInputs.from_dict`（`schedule_batch.py:338`）重建成对象，并对每个 item 调 `set_pad_value()`。

### 阶段二：pad_value 与占位膨胀

`set_pad_value`（`schedule_batch.py:210`）对图像数据做 SHA256 / GPU hash，取低位得到一个 0..2^30 的整数：

```python
self.pad_value = self.hash % (1 << 30)   # schedule_batch.py:270
```

这个 `pad_value` 同时承担两个角色：

1. **RadixAttention 的前缀键**：两张内容完全相同的图会 hash 出相同 pad_value，于是被填进 input_ids 的占位 token 序列也完全一致，前缀缓存就能命中。
2. **embedding scatter 的定位标记**：forward 时用 `torch.isin(input_ids, pad_values)` 找出哪些位置该被 vision embedding 替换。

随后 Scheduler 在 `handle_generate_request` 里调用 `pad_input_ids_func`（`scheduler.py:970`），它实际指向模型类的 `pad_input_ids` 方法（`tp_worker.py:166`）。模型用两种占位策略之一：

- **连续重复型**（Qwen-VL、`qwen2_vl.py:485`）：`MultiModalityDataPaddingPatternMultimodalTokens`，把所有 `im_token_id` 连续区段替换成对应 item 的 pad_value。
- **成对包裹型**（Gemma3、`gemma3_mm.py:197`）：`MultiModalityDataPaddingPatternTokenPairs`，把 `<img_start>...<img_end>` 之间的所有 token 替换成 pad_value，并记录 `data_offsets`。

```
膨胀前(单 image_token):   [.. , <vis_start>, IMG, <vis_end>, ..]   (HF 已展开为 N 个 IMG)
                                              ↓ pad_input_ids
膨胀后(pad_value 填充):    [.. , <vis_start>, P, P, ..., P, <vis_end>, ..]
                                              └── N 个相同 pad_value ──┘
```

### 阶段三：模型 forward 时把 embedding 塞回去

模型的 `forward` 不会自己拼 embedding，而是统一委托给 `general_mm_embed_routine`（`mm_utils.py:505`）。Qwen2-VL 的调用见 `qwen2_vl.py:546`，Gemma3 见 `gemma3_mm.py:368`。

```
general_mm_embed_routine(input_ids, forward_batch, language_model, image_data_embedding_func=get_image_feature)
   │
   ├─ 仅在 prefill 且 batch 含 mm_inputs 时进入多模态分支 (mm_utils.py:535)
   │     decode 阶段没有新图, 直接 embed_tokens(input_ids)
   │
   ▼ embed_mm_inputs()  (mm_utils.py:360)
   │   1. 摊平所有 mm_items, 按 is_image()/is_audio() 分类
   │   2. placeholder_tensor = 各 item 的 pad_value 张量
   │   3. get_embedding_and_mask():
   │        - get_embedding_hash() 查 MultiModalCache
   │        - miss → data_embedding_func(items) 真正跑 vision encoder
   │        - get_embedding_chunk() 处理 chunked-prefill 的部分区间
   │        - mask = isin(input_ids, placeholder_tensor)
   │        - 断言: input_ids 里占位 token 数 == embedding 行数 (mm_utils.py:334)
   │   4. input_ids.clamp_(0, vocab_size-1)   ← 关键! pad_value 远超 vocab
   │      inputs_embeds = embed_tokens(input_ids)
   │   5. masked_scatter(mask, embedding)  把 vision embedding 写入占位
   │
   ▼ language_model(input_ids=None, input_embeds=inputs_embeds, ...)
```

注意 `mm_utils.py:490` 那次 `input_ids.clamp_`：因为占位 token 填的是 pad_value（最大可达 2^30，远超 vocab_size），直接喂给 `nn.Embedding` 会越界。所以先 clamp 算出「垃圾」text embedding，再用 `masked_scatter` 把这些位置整体覆盖成 vision embedding——被覆盖的位置算什么无所谓。

`get_image_feature` 是模型自己实现的「vision encoder + projector」。例如 Qwen2-VL（`qwen2_vl.py:488`）把 `pixel_values` 喂给 `self.visual`；Gemma3（`gemma3_mm.py:278`）逐张过 `vision_tower` 再过 `multi_modal_projector`。两者都先检查 `precomputed_features`——如果客户端已经传了算好的 embedding，就直接 concat 跳过 encoder。

---

## 设计难点与权衡

### 1. 为什么用「内容 hash 当 token id」这种诡异做法？

最直接的痛点是 **RadixAttention 前缀缓存**。文本 LLM 里相同前缀天然有相同 token id；但图像是连续张量，没有 token id。SGLang 的取巧方案：对 pixel_values 做 hash，取低 30 位当作占位 token id。这样：

- 相同图片 → 相同 pad_value → 相同占位序列 → 前缀命中、KV 复用；
- 不同图片 → 不同 pad_value → 不会误命中。

代价是占位 token id 超出了真实 vocab，于是 forward 时必须先 `clamp_` 再 `masked_scatter`（`mm_utils.py:490-501`）。这是个非显然但很巧妙的 trick：占位位置的 text embedding 是垃圾，但马上会被 vision embedding 覆盖，所以无所谓。

### 2. 两种占位策略，何时用哪个？

- `MultiModalityDataPaddingPatternMultimodalTokens`（重复 token）适合 HF processor 已经把图展开成 N 个连续 `<image_pad>` 的模型（Qwen-VL、大多数新模型）。它用 `torch.isin` + 找连续区段，向量化、快。
- `MultiModalityDataPaddingPatternTokenPairs`（成对包裹）适合用 `<start>...<end>` 标记边界的模型（MiniCPM、Gemma3）。它能处理 `data_start_token_ids`，并在 `mm_inputs.data_offsets` 里记录每段起点，供 mrope 之类的位置编码用。

一个隐藏坑在 `mm_utils.py:97`：成对策略里如果 `start_indices` 和 `end_indices` 数量不等，会**直接原样返回 input_ids 不做替换**——前缀缓存里被截断的 `im_start` 会触发这个分支，需要靠 `get_multimodal_data_bounds`（`mm_utils.py:577`）里那段「补一个 0 起点」的兜底逻辑（`mm_utils.py:602-613`）。

### 3. 多模态 cache 与 chunked prefill 的纠缠

`MultiModalCache`（`multimodal_cache.py:6`）按字节大小限容（默认 100MB，可用 `SGLANG_VLM_CACHE_SIZE_MB` 调，见 `scheduler.py:2288`）。它缓存的是**整张图的完整 embedding**，键是 `get_embedding_hash`（所有 item hash 的组合）。

难点在 chunked prefill：一张图可能被切成多个 chunk 分批 prefill。`get_embedding_chunk`（`mm_utils.py:210`）根据 `extend_prefix_len`/`extend_seq_len`/`items_offset` 算出当前 chunk 需要 embedding 的哪一段。整张图的 embedding 算一次进 cache，每个 chunk 取自己那一片；当 `end_index` 到达整张图末尾时（`mm_utils.py:320`）才 `free` 掉。

这也是为什么 `general_mm_embed_routine` 里有句注释「considering chunked-prefill is disabled for multimodal models」并把 `forward_batch.mm_inputs = None`（`mm_utils.py:562-564`）——多模态默认不开 chunked prefill，cache 主要服务于**跨请求复用同一张图**和 prefix-cache 命中后只重算被截断的尾部。

### 4. embedding 数量对不齐怎么办？

`get_embedding_and_mask`（`mm_utils.py:334`）有一段防御逻辑：当 input_ids 里占位 token 数 ≠ embedding 行数时打 warning。如果占位 token 少于 embedding 行（常见于 chunked prefill 截断），就从 embedding 尾部截取（`mm_utils.py:348`）；如果占位 token 多于 embedding，则直接 `RuntimeError`，因为那是真正的 bug（offset / processor 算错了）。

### 5. processor 的自动注册

`import_processors`（`multimodal_processor.py:31`）用 `pkgutil.iter_modules` 扫整个 `multimodal_processors/` 包，对每个 `BaseMultimodalProcessor` 子类读它的 `models` 类属性（一个模型类列表），把每个 architecture 注册进 `PROCESSOR_MAPPING`。`get_mm_processor` 再用 `hf_config.architectures` 反查。注意它用 `lru_cache` 保证只扫一次，且导入失败的模块会被 try/except 吞掉只打 warning（`multimodal_processor.py:38`）——所以新 processor 写错 import 不会让整个 server 崩，但也意味着**注册悄悄失败、报「No processor registered」时要去看启动日志**。

`LlavaMultimodalProcessor`（`llava.py:177`）是个特殊的间接层：它读 `vision_config.model_type` 再去 `PROCESSOR_MAPPING` 里找匹配的 vision processor 包装，用于 HF 新式 `LlavaForConditionalGeneration` 那种「vision 部分可插拔」的配置。

---

## 改代码 / 修 bug 指南

### 想接入一个新的 VLM —— 标准步骤

1. **写模型类**（`python/sglang/srt/models/your_vlm.py`）：
   - 实现 `get_input_embeddings()` 返回 text embedding 层；
   - 实现 `get_image_feature(items)`：跑你的 vision encoder + projector，返回 `[num_tokens, hidden]` 的 embedding，并优先处理 `precomputed_features`（照抄 `qwen2_vl.py:488` 或 `gemma3_mm.py:278`）；
   - 实现 `pad_input_ids(input_ids, mm_inputs)`：选一个 padding pattern（`qwen2_vl.py:481` 或 `gemma3_mm.py:188`）；
   - `forward` 里调 `general_mm_embed_routine(..., image_data_embedding_func=self.get_image_feature)`。
2. **写 processor**（`multimodal_processors/your_vlm.py`）：
   - 继承 `BaseMultimodalProcessor`，设 `models = [YourVLMForConditionalGeneration]`；
   - 在 `__init__` 里定义 image_token 字符串 + regex + 各种 token id；
   - 实现 `process_mm_data_async`：调 `load_mm_data` → (可选 resize) → `process_mm_data` → `get_mm_items_offset` → 组装 `MultimodalDataItem`，返回含 `input_ids`/`mm_items`/各 token id 的 dict。
   - 照抄最接近的现有 processor（Qwen 风格 `qwen_vl.py`，Gemma 风格 `gemma3.py`，LLaVA 风格 `llava.py`）。
3. 不用手动注册——`import_processors` 会自动发现 `models` 列表。

### 「想改 X → 看这里」

| 想做的事 | 去这里 |
| --- | --- |
| 改图片下载 / 视频抽帧逻辑 | `base_processor.py:188` `submit_data_loading_tasks` / `:167` `_load_single_item` |
| 改 prompt 切分 / image_token 替换 | `base_processor.py:255` `load_mm_data` |
| 改 anyres / pad 切图 (LLaVA) | `python/sglang/srt/mm_utils.py` `process_anyres_image` / `expand2square` |
| 改占位 token 膨胀规则 | `mm_utils.py:47` / `:122` 两个 padding pattern + 模型的 `pad_input_ids` |
| 改 vision encoder 接法 | 模型类的 `get_image_feature` |
| 改 embedding scatter / clamp | `mm_utils.py:360` `embed_mm_inputs` |
| 改多模态 cache 大小/策略 | `multimodal_cache.py` + 环境变量 `SGLANG_VLM_CACHE_SIZE_MB` |
| 改并行度 | 环境变量 `SGLANG_IO_WORKERS` / `SGLANG_CPU_WORKERS`（`base_processor.py:95-99`）|

### 常见 bug 高发区

- **「No processor registered for architecture」**：八成是 processor 模块 import 报错被 `multimodal_processor.py:38` 静默吞了。先看启动日志里的 `Ignore import error when loading ...` warning。
- **embedding 数量 mismatch warning**（`mm_utils.py:335`）：通常是 `get_mm_items_offset` 算的占位区间和 vision encoder 实际输出 token 数对不上。检查 image_token_id 是否正确、HF processor 展开的 token 数（如 Qwen 的 grid_thw / spatial_merge）是否和 `get_image_feature` 输出一致。
- **越界 / clamp 相关 CUDA 报错**：`pad_value` 进了 embedding 没被 clamp，或者新模型 forward 没走 `general_mm_embed_routine`。
- **前缀缓存对图像不命中**：检查 `set_pad_value` 的 hash 是否稳定（`schedule_batch.py:210`），以及成对策略下 `im_start` 被前缀截断的边界情况（`mm_utils.py:97` 的「数量不等就放弃」分支）。
- **cache 被撑满 warning**（`mm_utils.py:303`）：大图 / 高分辨率把 100MB 默认 cache 占满，调 `SGLANG_VLM_CACHE_SIZE_MB`。

### 调试入手点

1. 在 `tokenizer_manager.py:464` 打断点，看 `process_mm_data_async` 返回的 `input_ids` 和 `mm_items`——这是预处理结果的 ground truth。
2. 在 `embed_mm_inputs`（`mm_utils.py:360`）看 `num_mm_tokens_in_input_ids` vs `num_mm_tokens_in_embedding`——对不齐就是 offset / encoder 输出不匹配。
3. 在模型 `get_image_feature` 里打印输出 shape，和占位区间长度对比。
4. 怀疑 cache 时，直接在 `multimodal_cache.py` 的 `get`/`put` 加日志，看命中率与 `current_size`。

---

## 交叉引用

- [请求生命周期](01-request-lifecycle.md)：多模态请求如何从 entrypoint 流到这里
- [TokenizerManager](11-tokenizer-manager.md)：processor 的调用方与初始化
- [调度器](12-scheduler.md)：`pad_input_ids_func` 的调用、cache 初始化
- [KV 缓存与 RadixAttention](13-memcache-radixattention.md)：pad_value 为什么要参与前缀匹配
- [Model Executor](14-model-executor.md)：`forward_batch` / chunked prefill 的来源
- [术语表](02-glossary.md)：prefill / decode / RadixAttention / mrope 等术语
