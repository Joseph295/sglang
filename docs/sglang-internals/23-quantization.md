# 量化（FP8 / INT4 / AWQ / GPTQ）

> 本章面向想读懂 SGLang 量化源码、能改代码 / 修 bug 的工程师。先讲「为什么这么设计」，再讲「怎么实现」，最后讲「常见坑」。

## 1. 一句话职责

量化模块负责：**识别 checkpoint 用的是哪种量化方案（FP8 / INT4 AWQ / GPTQ / w8a8 / compressed-tensors / torchao 等），把模型里每个 `Linear` / `FusedMoE` 层的「乘法实现」替换成对应的量化 kernel，并在权重加载阶段完成反量化 / 重打包 / scale 处理**，从而用更少的显存、更快的速度跑同一个模型。

它本质是一个「策略工厂」：`QuantizationConfig`（描述「用什么量化」）→ `get_quant_method(layer, prefix)` → `QuantizeMethodBase`（描述「这一层怎么 create_weights / apply」）。普通层和量化层在 `LinearBase` 看来是完全一样的，区别只在 `self.quant_method` 这个对象。

## 2. 在端到端链路中的位置

量化不是一个独立的「运行时阶段」，而是发生在 **模型构建期 + 权重加载期**，运行时只是「forward 时调用被替换过的 kernel」。

```
客户端
  │
分词 (TokenizerManager)
  │
调度 (Scheduler)
  │
KV 缓存 (RadixAttention)
  │
┌─────────────── 前向 (ModelRunner / forward) ───────────────┐
│  每个 Linear/MoE 层: output = self.quant_method.apply(...)  │ ← 量化 kernel 在这里被调用
└────────────────────────────────────────────────────────────┘
  │
采样 → 解码 → 响应


量化逻辑真正介入的时刻（启动期，一次性）：

ServerArgs.quantization ─┐
                         ├─► ModelConfig._verify_quantization()   ← 识别方案 (§4.1)
HF config.json           ─┘        │
                                   ▼
                    get_quant_config()  (weight_utils.py)         ← 构造 QuantizationConfig
                                   │
                                   ▼
                 建模时每个 LinearBase.__init__:
                   self.quant_method = quant_config.get_quant_method(layer, prefix)  ← 替换层 (§4.2)
                                   │
                                   ▼
                   quant_method.create_weights(...)               ← 申请量化权重/scale 占位
                                   │
                          load_weights() 从磁盘读 safetensors
                                   │
                                   ▼
                   quant_method.process_weights_after_loading()   ← 反量化/重打包/转置 (§4.3)
```

参见 [模型执行器](14-model-executor.md) 了解 `ModelRunner` 如何驱动 forward，参见 [入口与 API](10-entrypoints-api.md) 了解 `ServerArgs`。

## 3. 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `QuantizationConfig` | `python/sglang/srt/layers/quantization/base_config.py:38` | 所有量化 config 的抽象基类；定义 `from_config` / `get_quant_method` / `override_quantization_method` |
| `QuantizeMethodBase` | `python/sglang/srt/layers/quantization/base_config.py:11` | 所有量化「方法」的基类；定义 `create_weights` / `apply` / `process_weights_after_loading` |
| `QUANTIZATION_METHODS` | `python/sglang/srt/layers/quantization/__init__.py:105` | 方案名 → config 类 的全局注册表 |
| `get_quantization_config()` | `python/sglang/srt/layers/quantization/__init__.py:108` | 按名字取 config 类，并检查 vllm 依赖 |
| `get_linear_quant_method()` | `python/sglang/srt/layers/quantization/__init__.py:178` | 通用逻辑：根据 `dynamic` 规则决定某一层走量化还是 `UnquantizedLinearMethod` |
| `get_dynamic_override()` | `python/sglang/srt/layers/quantization/__init__.py:157` | GPTQModel 风格的 per-module 正/负则匹配（`+:` / `-:`） |
| `LinearBase` | `python/sglang/srt/layers/linear.py:178` | 所有 linear 层基类；`__init__` 里调用 `quant_config.get_quant_method` |
| `LinearMethodBase` / `UnquantizedLinearMethod` | `python/sglang/srt/layers/linear.py:102` / `:143` | linear 量化方法基类 / 不量化时的默认实现 |
| `WEIGHT_LOADER_V2_SUPPORTED` | `python/sglang/srt/layers/linear.py:37` | 哪些量化方法走 v2 weight loader（按 dim 分片） |
| `Fp8Config` / `Fp8LinearMethod` / `Fp8MoEMethod` | `python/sglang/srt/layers/quantization/fp8.py:97` / `:176` | FP8（block-wise + per-tensor + 在线动态量化） |
| `AWQConfig` / `AWQLinearMethod` | `python/sglang/srt/layers/quantization/awq.py:27` / `:104` | AWQ INT4（weight-only，运行时 `awq_dequantize`） |
| `GPTQConfig` / `GPTQMarlinConfig` / `GPTQMarlinMoEMethod` | `python/sglang/srt/layers/quantization/gptq.py:57` / `:161` / `:451` | GPTQ 及其 Marlin 加速变体 |
| `W8A8Int8Config` / `W8A8Fp8Config` | `python/sglang/srt/layers/quantization/w8a8_int8.py:21` / `w8a8_fp8.py` | per-channel weight + per-token dynamic activation，走 sgl-kernel CUTLASS |
| `BlockInt8Config` | `python/sglang/srt/layers/quantization/blockwise_int8.py` | block-wise INT8 |
| `ModelOptFp8Config` / `ModelOptFp4Config` | `python/sglang/srt/layers/quantization/modelopt_quant.py` | NVIDIA ModelOpt FP8 / NVFP4 |
| `CompressedTensorsConfig` | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` | neuralmagic/llm-compressor 的统一格式 |
| `MoeWNA16Config` / `QoQConfig` | `moe_wna16.py` / `qoq.py` | MoE weight-only N-bit / QoQ (W4A8) |
| `get_quant_config()` | `python/sglang/srt/model_loader/weight_utils.py:131` | 从 HF config 或独立 json 构造 config 实例 |
| `_verify_quantization()` | `python/sglang/srt/configs/model_config.py:327` | 识别 / 校验量化方案，处理 override |
| `apply_torchao_config_to_model()` | `python/sglang/srt/layers/torchao_utils.py:39` | torchao 路径（`--torchao-config`），与上面体系完全独立 |

## 4. 核心数据流 / 执行流程

### 4.1 方案识别：「我到底要用哪种量化」

入口是 `ModelConfig._verify_quantization()`（`model_config.py:327`）。优先级与逻辑：

1. 用户用 `--quantization xxx`（即 `ServerArgs.quantization`）显式指定，或留空。
2. `_parse_quant_hf_config()`（`model_config.py:305`）从 HF `config.json` 读 `quantization_config`；compressed-tensors 用的是 `compression_config`；ModelOpt 没有 HF 字段，靠探测根目录有没有 `hf_quant_config.json` 来识别（`:314-323`）。
3. 遍历所有注册方案，调用 `method.override_quantization_method(quant_cfg, user_quant)`（`:369-376`）。这是「方案可以自我升级」的钩子——典型例子是 `GPTQMarlinConfig.override_quantization_method`（`gptq.py:281`）：如果 checkpoint 是 GPTQ 格式、当前是 CUDA、且 bits/sym/group_size 都满足 Marlin 要求（`is_gptq_marlin_compatible`，`gptq.py:327`），它会把方案从 `gptq` 自动改写成 `gptq_marlin`，从而走更快的 kernel。
4. 一致性校验（`:378-392`）：用户指定的方案和 checkpoint 里写的方案不一致时报错，除非落在 `compatible_quantization_methods` 白名单里（例如 `w8a8_fp8` 兼容 `compressed-tensors`——这正是 docs 里说的「用 `--quantization w8a8_fp8` 覆盖 HF 的 compressed-tensors，改走 sgl-kernel」的实现）。
5. `optimized_quantization_methods`（`:338`）之外的方案会打印「not fully optimized」warning。

> 关键设计点：**offline 量化（GPTQ/AWQ/FP8 checkpoint）不需要 `--quantization`**，方案从 HF config 自动解析；**online 动态量化（如对 BF16 模型现场转 FP8）才需要 `--quantization fp8`**。两者同时给会冲突。详见 `docs/backend/quantization.md`。

确定方案名后，`get_quant_config()`（`weight_utils.py:131`）负责造出 config 实例：取 `quant_cls = get_quantization_config(name)`，把 HF 的 `quantization_config` 字典塞进去（还会注入 `packed_modules_mapping`，`:152`），调 `quant_cls.from_config(hf_quant_config)`。若该方案没有 HF 字段（`get_config_filenames` 返回非空，如 AWQ 的 `quant_config.json`），则去模型目录 glob 这些文件再读。

### 4.2 层替换：「普通 Linear 怎么变成量化 Linear」

这是整个体系最优雅的地方——**没有真正替换类，只替换了 `quant_method` 这个成员**。

```
LinearBase.__init__(quant_config=...)          linear.py:190
        │
        ├─ quant_config is None
        │     └─► self.quant_method = UnquantizedLinearMethod()    linear.py:209
        │
        └─ quant_config 存在
              └─► self.quant_method = quant_config.get_quant_method(self, prefix)   linear.py:211
                        │
                        ▼
              Fp8Config.get_quant_method(layer, prefix)   fp8.py:159
                  if isinstance(layer, LinearBase):
                      if is_layer_skipped(prefix, ignored_layers): return UnquantizedLinearMethod()
                      return Fp8LinearMethod(self)
                  elif isinstance(layer, FusedMoE):
                      return Fp8MoEMethod(self)
```

每个 config 的 `get_quant_method` 都做同一件事：**看这一层是 `LinearBase` 还是 `FusedMoE`，再看 prefix 是否在「跳过列表」里**，分别返回 linear method / moe method / `UnquantizedLinearMethod`。

跳过逻辑有三种风格：

- **FP8 / compressed-tensors**：`is_layer_skipped(prefix, self.ignored_layers)`（fp8.py:165）。
- **AWQ**：`is_layer_skipped_awq(prefix, modules_to_not_convert)`（awq.py:23），子串匹配。
- **GPTQ（GPTQModel `dynamic`）**：`get_linear_quant_method` + `get_dynamic_override`（`__init__.py:178` / `:157`），支持正则的 `+:`（正向匹配并覆盖 bits/group_size）和 `-:`（负向匹配，整层不量化）。例子见 `gptq.py:84-93` 的注释（按层号给不同 bit 数）。

`forward` 时（`linear.py:287`）统一调用 `output = self.quant_method.apply(self, x, bias)`，调用方完全不知道底层是 FP8 还是 INT4。

### 4.3 权重生命周期：create_weights → load → process_weights_after_loading

层构造时，`LinearBase` 子类（如 `ReplicatedLinear`，`linear.py:251`）立刻调用 `quant_method.create_weights(...)`，传入 `input_size_per_partition` / `output_partition_sizes` 等 TP 切分后的尺寸。每个方法在这里 **注册量化布局的占位 Parameter**，而不是普通的 `[out, in]` fp16 权重。

以 **AWQ**（awq.py:114）为例，INT4 打包格式：

```
qweight : int32 [in,  out/pack_factor]   pack_factor = 32/4 = 8   ← 8 个 int4 塞进 1 个 int32
qzeros  : int32 [in/group_size, out/pack_factor]
scales  : fp16  [in/group_size, out]
```

以 **FP8**（fp8.py:212）为例，分支取决于 `is_checkpoint_fp8_serialized` 和 `block_quant`：

- checkpoint 已是 FP8：weight 直接是 `float8_e4m3fn`；block-wise 量化用 `weight_scale_inv`（`BlockQuantScaleParameter`，按 `block_n × block_k` 分块，fp8.py:279），per-tensor 量化用 `weight_scale`（`PerTensorScaleParameter`，每个 fused shard 一个 scale，fp8.py:292）；static activation 还会有 `input_scale`。
- checkpoint 是 BF16/FP16（online 量化）：weight 先按 `params_dtype` 申请，scale 留到加载后再算。

加载阶段，weight loader（`linear.py:276` 的 v1，或 v2 的 `BasevLLMParameter`）把磁盘上的 shard 按 `input_dim`/`output_dim`/`packed_dim` copy 到占位 Parameter 里。**v2 weight loader** 才知道这些参数是「packed/沿某 dim 分片」的——所以你的量化方法名必须加进 `WEIGHT_LOADER_V2_SUPPORTED`（`linear.py:37`），否则 fused QKV/MLP 的分片加载会出错。

加载完成后，`load_weights_and_postprocess`（loader.py:387）遍历所有 module，对有 `quant_method` 的调用 `process_weights_after_loading`（loader.py:400）。这一步是「真正算 kernel 需要的最终布局」：

- **AWQ**（awq.py:181）：只是把 `PackedvLLMParameter` 转成普通 `nn.Parameter`（`requires_grad=False`）。
- **W8A8Int8**（w8a8_int8.py:74）：把 weight **转置**成 `[in, out]` 以适配 `int8_scaled_mm`。
- **FP8**（fp8.py:311）最复杂：
  - block-wise：基本不动，ROCm 上把 e4m3fn 归一到 e4m3fnuz（`normalize_e4m3fn_to_e4m3fnuz`）。
  - online（checkpoint 非 fp8）：现场调 `per_token_group_quant_fp8` 或 `input_to_float8` 把 BF16 权重压成 FP8，存 `weight_scale`（fp8.py:335-349）。
  - per-tensor checkpoint：若硬件支持 CUTLASS，`convert_to_channelwise` 把 per-tensor scale 展开成 per-channel；否则 `requantize_with_max_scale`（fp8.py:382）取所有 fused shard 的最大 scale 反量化再重量化，统一成一个 per-tensor scale。
  - 若 `use_marlin`（fp8.py:396），调 `prepare_fp8_layer_for_marlin` 重排成 Marlin 布局。

### 4.4 运行时 apply：调哪个 kernel

| 方法 | apply 路径 | kernel |
| --- | --- | --- |
| AWQ | `awq_dequantize(qweight, scales, qzeros)` 反量化成 fp16 再 `torch.matmul`（awq.py:199） | sgl-kernel `awq_dequantize`（weight-only，激活不量化） |
| W8A8Int8 | `per_token_quant_int8(x)` 量化激活 → `int8_scaled_mm`（w8a8_int8.py:115） | sgl-kernel CUTLASS int8 |
| FP8 | `apply_fp8_linear` / `apply_w8a8_block_fp8_linear`（fp8_utils.py） | CUTLASS / DeepGEMM / `torch._scaled_mm` |
| GPTQ-Marlin | vllm `GPTQMarlinLinearMethod.apply` | Marlin kernel |

注意 **weight-only（AWQ / GPTQ / int4wo / fp8wo）** 是「反量化权重→高精度矩阵乘」，省的是显存和带宽，算的是 fp16 GEMM；**w8a8 / FP8 GEMM** 是「权重和激活都量化→低精度 Tensor Core」，省的是计算。这决定了它们在不同 batch size 下的收益曲线（小 batch decode 受带宽限制，weight-only 收益大；大 batch prefill 受算力限制，w8a8 收益大）。

### 4.5 torchao：另一条独立路径

`--torchao-config`（如 `int4wo-128` / `int8dq` / `fp8dq-per_row`）走的是完全独立的 `apply_torchao_config_to_model`（torchao_utils.py:39），它在 `LayeredModelLoader` 里 **逐层加载完就地 `quantize_()`**（loader.py:461），不经过 `QuantizationConfig` / `get_quant_method` 那套体系。支持的字符串见 torchao_utils.py 与 `docs/backend/quantization.md:135`。

### 4.6 vllm 复用与 monkey patch

很多 kernel（GPTQ/Marlin/AWQ-Marlin/compressed-tensors MoE）直接复用 vllm 的实现（`__init__.py:11-34`）。难点是 vllm 的 `get_quant_method` 内部用 `isinstance(layer, vllm.LinearBase)` 判断层类型，而 SGLang 的层是自己的类。解决办法是 `monkey_patch_isinstance_for_vllm_base_layer`（`__init__.py:236`）：临时把内建 `isinstance` 换掉，让 vllm 的 `LinearBase`/`FusedMoE`/`VocabParallelEmbedding` 判断重定向到 SGLang 的对应类。`monkey_patch_quant_configs`（`__init__.py:326`）则给 vllm 的 `GPTQConfig` / `GPTQMarlinConfig` 替换上 SGLang 的 `gptq_get_quant_method`，并用 `monkey_patch_moe_apply` 把 SGLang 的 MoE 调用参数翻译成 vllm 签名。

## 5. 设计难点与权衡

1. **策略工厂而非继承爆炸**。如果给每种「层 × 量化方案」组合都建一个类（`Fp8ColumnParallelLinear`…），会爆炸成几十个类。SGLang 选择「层只有一种」+「`quant_method` 可插拔」，新增方案只需写一个 `QuantizationConfig` + 若干 `*Method`，不碰任何模型代码。代价是 `LinearBase` 必须把所有 TP/分片信息透传给 `create_weights`，签名很长（linear.py:106）。

2. **create_weights / process_weights_after_loading 两段式**。磁盘格式（packed int32、per-tensor scale）≠ kernel 需要的格式（转置、per-channel、Marlin 重排）。如果在加载时就转换，会和 weight loader 的分片逻辑打架；所以统一「先按磁盘格式建占位、加载、再后处理」。`process_weights_after_loading` 是「checkpoint 格式」和「kernel 格式」之间的唯一翻译层，几乎所有「加载后精度异常」的 bug 都在这里。

3. **override_quantization_method 的双刃剑**。它能自动把 GPTQ 升级成更快的 GPTQ-Marlin（gptq.py:290），对用户透明。但也意味着「同一个 checkpoint 在不同 GPU / vllm 版本上跑的是不同 kernel」，复现 bug 时必须先确认实际生效的方案（看启动日志里的 `Using gptq_marlin kernel.`）。

4. **per-tensor vs per-channel vs block-wise 的精度/速度权衡**。粒度越细（block-wise > per-channel > per-tensor）精度越好但 kernel 越复杂、scale 存储越多。FP8 的 `requantize_with_max_scale`（fp8.py:382）在不支持 CUTLASS 的硬件上被迫退化到 per-tensor，会损失精度——这是「同一模型在 A100 和老卡上精度不同」的常见根因。

5. **vllm 依赖的脆弱性**。`VLLM_AVAILABLE` 为 False 时，AWQ/GPTQ/Marlin 等整片不可用（`__init__.py:37-49` 用 `DummyConfig` 占位，`get_quantization_config` 在 `:114` 直接报错要求装 vllm）。而 FP8/w8a8/blockwise_int8/compressed-tensors/modelopt 是 SGLang 自带 sgl-kernel 的，不依赖 vllm（`BASE_QUANTIZATION_METHODS`，`__init__.py:75`）。

6. **MoE 量化是另一套接口**。`FusedMoE` 层不走 `LinearMethodBase`，而是 `FusedMoEMethodBase`（如 `GPTQMarlinMoEMethod`，gptq.py:451 / `Fp8MoEMethod`），权重是 3D `[num_experts, …]`，`create_weights` 完全不同，且常需要 `monkey_patch_moe_apply` 适配 vllm 的 expert 路由签名。改 MoE 量化别套用 linear 的经验。

## 6. 改代码 / 修 bug 指南

### 想做 X → 看这里

| 目标 | 入手点 |
| --- | --- |
| 新增一种量化方案 | 写 `QuantizationConfig` 子类（实现 `get_name` / `from_config` / `get_quant_method` / `get_config_filenames`）+ `LinearMethodBase` 子类（`create_weights` / `apply` / `process_weights_after_loading`），注册到 `QUANTIZATION_METHODS`（`__init__.py:105`） |
| 让某个新 quant method 支持 fused QKV/MLP 分片 | 把类名加进 `WEIGHT_LOADER_V2_SUPPORTED`（`linear.py:37`），并在 `create_weights` 里用带 `input_dim`/`output_dim`/`packed_dim` 的 `*vLLMParameter` |
| 让某些层不量化 | 看该方案的 skip 逻辑：FP8 用 `ignored_layers`，AWQ 用 `modules_to_not_convert`，GPTQ 用 `dynamic` 的 `-:` 规则（`get_dynamic_override`，`__init__.py:157`） |
| 改 FP8 是否走 Marlin / CUTLASS | `Fp8LinearMethod.__init__`（fp8.py:200）：`SGLANG_FORCE_FP8_MARLIN` 环境变量 + `cutlass_fp8_supported()` |
| 加 torchao 新模式 | `apply_torchao_config_to_model`（torchao_utils.py:39），与主体系无关 |
| 改方案自动识别/升级 | `override_quantization_method`（如 gptq.py:281）+ `_verify_quantization`（model_config.py:327） |

### 该模块 bug 高发区

- **`process_weights_after_loading`**：转置方向错、scale 形状错（per-tensor vs per-channel）、ROCm 的 e4m3fnuz 归一化漏掉 → 表现为「能加载但输出乱码 / 精度暴跌」。先在这里加 print 看 `layer.weight.shape` / `layer.weight_scale.shape`。
- **TP 切分对齐**：AWQ 的 `create_weights`（awq.py:124、:132）会因为 `input_size_per_partition % group_size != 0` 或 `output % pack_factor != 0` 报错——TP size 调太大就会触发。FP8 block-wise 同理（fp8.py:233-249）。
- **方案识别走错**：用户以为在跑 A 实际跑了 B（override 改写了）。第一步永远是看启动日志确认实际方案，以及 `quant_cfg.get("quant_method")` 的值。
- **vllm 版本/缺失**：报「requires some operators from vllm」就是 `VLLM_AVAILABLE=False` 或版本不对（`__init__.py:114`，需要 `vllm==0.8.4`）；MoE correction_bias 相关报错见 `__init__.py:316`。
- **online 量化和 offline checkpoint 撞车**：对已量化模型又加 `--quantization fp8`，会触发 `_verify_quantization` 的不一致报错（model_config.py:381），或精度异常。
- **fused module 的 N 个 scale**：FP8 per-tensor 时 QKV 三段各有一个 weight_scale，`requantize_with_max_scale`（fp8.py:382）取最大值统一——若某段 scale 异常会污染全部三段。

### 调试入手点

1. 启动日志：`Detected fp8 checkpoint.`（fp8.py:109）、`Using gptq_marlin kernel.`（gptq.py:295）、`not fully optimized` warning（model_config.py:406）确认实际方案。
2. 在 `LinearBase.__init__`（linear.py:211）打印 `type(self.quant_method)`，确认每层被替换成了预期的方法（或意外退化成 `UnquantizedLinearMethod`）。
3. 在 `process_weights_after_loading` 打印权重/scale 的 shape 与 dtype，对照 §4.3。
4. 怀疑精度时，临时把方法换成 `UnquantizedLinearMethod` 跑一版做 A/B（可在对应 config 的 `get_quant_method` 里强制 return 它）。
5. 用 `docs/backend/quantization.md` 给的小模型（如 `Llama-3.2-1B` GPTQ）做最小复现，再用 benchmark 对比量化前后的精度回归（docs 反复强调量化必须 benchmark 验证）。

---

相关章节：[模型执行器](14-model-executor.md)、[Worker 与并行](15-worker-parallelism.md)、[术语表](02-glossary.md)、[请求生命周期](01-request-lifecycle.md)。
