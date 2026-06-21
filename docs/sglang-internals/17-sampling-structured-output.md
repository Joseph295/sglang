# 采样与结构化输出（FSM / Grammar）

> 本章对应源码目录：`python/sglang/srt/sampling/`、`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/constrained/`。
>
> 行号以仓库当前 `main` 为准（写作时基于 commit `ed0c3035`）。如对不上，以你本地 `grep` 到的为准。

## 1. 一句话职责

把一个 batch 的 model forward 产出的 `next_token_logits`（形状 `[batch_size, vocab_size]`）变成每个请求的 **下一个 token id**：先按各请求的 `SamplingParams` 施加 penalty / grammar mask / custom logit processor，再做 temperature / top-k / top-p / min-p / greedy 采样。**结构化输出（structured output）** 是其中一条特殊路径：用 compressed FSM / grammar（outlines / xgrammar / llguidance）在每一步把「不合法」的 token 的 logit 置为 `-inf`，从而保证整段输出是合法的 JSON / 正则 / EBNF。

## 2. 在端到端链路中的位置

```
客户端 → 分词(TokenizerManager) → 调度(Scheduler) → KV缓存(RadixAttention) → 前向(ModelRunner.forward)
                                                                                      │
                                                                          next_token_logits [B, V]
                                                                                      │
                                                              ┌───────────────────────▼───────────────────────┐
                                                              │  ModelRunner.sample()  (本章)                   │
                                                              │   _preprocess_logits():                         │
                                                              │     ① custom logit processor (Sampler 内部)     │
                                                              │     ② penalty (freq/presence/min_new_tokens)    │
                                                              │     ③ grammar vocab_mask (FSM / xgrammar)       │
                                                              │   Sampler.forward():                            │
                                                              │     ④ temperature / top-k / top-p / min-p / argmax│
                                                              └───────────────────────┬───────────────────────┘
                                                                                      │ next_token_ids [B]
                                                          accept_token() 推进 grammar 状态 ▼
                                  解码(Detokenizer) → 响应         (output_processor_mixin)
```

- 调用入口在 `python/sglang/srt/model_executor/model_runner.py:1226` 的 `ModelRunner.sample()`，它先调 `_preprocess_logits()`（model_runner.py:1212）做 bias/mask，再调 `self.sampler(...)`（model_runner.py:1250）。
- `Sampler` 实例在 `ModelRunner.__init__` 处创建（model_runner.py:269）。
- 采样完成后，Scheduler 在结果处理阶段调 `req.grammar.accept_token(next_token_id)`（`scheduler_output_processor_mixin.py:129` 和 `:266`）推进 grammar 状态机，并把当前 token 喂给 penalizer（`schedule_batch.py:1473`）。

上游调度参见 [调度器](12-scheduler.md)，前向与 KV 缓存参见 [Model Executor](14-model-executor.md) 与 [RadixAttention](13-memcache-radixattention.md)。术语见 [术语表](02-glossary.md)。

## 3. 关键文件与类

| 符号 | file:line | 作用 |
|---|---|---|
| `SamplingParams` | `python/sglang/srt/sampling/sampling_params.py:21` | 单个请求的采样参数容器，含 `verify()` / `normalize()` |
| `SamplingBatchInfo` | `python/sglang/srt/sampling/sampling_batch_info.py:20` | 把一个 batch 的采样参数**向量化**成 GPU tensor；持有 grammar mask、penalizer、custom processor |
| `SamplingBatchInfo.from_schedule_batch` | `sampling_batch_info.py:58` | 从 `ScheduleBatch` 构造（建 temperatures/top_ps/... tensor） |
| `SamplingBatchInfo.update_regex_vocab_mask` | `sampling_batch_info.py:149` | 向每个 grammar 取下一步合法 token 集合，填进 `vocab_mask` |
| `SamplingBatchInfo.apply_logits_bias` | `sampling_batch_info.py:187` | 把 penalty + grammar mask 施加到 logits |
| `Sampler` | `python/sglang/srt/layers/sampler.py:29` | `nn.Module`，真正从 logits 采样 |
| `top_k_top_p_min_p_sampling_from_probs_torch` | `sampler.py:202` | PyTorch fallback 采样实现 |
| `BatchedPenalizerOrchestrator` | `python/sglang/srt/sampling/penaltylib/orchestrator.py:12` | 统一管理多个 penalizer 的生命周期 |
| `BatchedFrequencyPenalizer` | `penaltylib/frequency_penalty.py:9` | 频率惩罚 |
| `BatchedPresencePenalizer` | `penaltylib/presence_penalty.py:9` | 存在惩罚 |
| `BatchedMinNewTokensPenalizer` | `penaltylib/min_new_tokens.py:9` | min_new_tokens 之前屏蔽 stop token |
| `CustomLogitProcessor` | `python/sglang/srt/sampling/custom_logit_processor.py:19` | 用户自定义 logit 处理器（dill 序列化） |
| `BaseGrammarObject` / `BaseGrammarBackend` | `python/sglang/srt/constrained/base_grammar_backend.py:29` / `:108` | grammar 抽象接口 + 带缓存的后端基类 |
| `create_grammar_backend` | `base_grammar_backend.py:167` | 工厂：按 `--grammar-backend` 选 outlines/xgrammar/llguidance |
| `OutlinesGrammar` / `OutlinesGrammarBackend` | `python/sglang/srt/constrained/outlines_backend.py:41` / `:114` | outlines FSM 后端（bool mask） |
| `OutlinesJumpForwardMap` | `python/sglang/srt/constrained/outlines_jump_forward.py:142` | compressed FSM / jump-forward 加速 |
| `XGrammarGrammar` / `XGrammarGrammarBackend` | `python/sglang/srt/constrained/xgrammar_backend.py:44` / `:147` | xgrammar 后端（bitmask + Triton kernel） |
| `apply_token_bitmask_inplace_triton` | `python/sglang/srt/constrained/triton_ops/bitmask_ops.py:84` | 把 bitmask 应用到 logits 的 Triton kernel |
| `ReasonerGrammarBackend` | `python/sglang/srt/constrained/reasoner_grammar_backend.py:78` | reasoning 模型：`</think>` 之前不做 grammar 约束 |

## 4. 核心数据流 / 执行流程

### 4.1 SamplingParams：单请求参数

`SamplingParams.__init__`（sampling_params.py:30）的字段大致分四类：

- **长度控制**：`max_new_tokens`、`min_new_tokens`、`stop`（`stop_strs`）、`stop_token_ids`、`ignore_eos`、`no_stop_trim`。
- **概率分布整形**：`temperature`、`top_p`、`top_k`、`min_p`。
- **惩罚**：`frequency_penalty`、`presence_penalty`、`repetition_penalty`。
- **结构化输出 / 杂项**：`json_schema`、`regex`、`ebnf`、`structural_tag`、`n`、`custom_params`、`stream_interval`、`skip_special_tokens` 等。

两个非显然的预处理（sampling_params.py:82-87）：

```python
if 0 <= self.temperature < _SAMPLING_EPS:   # _SAMPLING_EPS = 1e-6
    self.temperature = 1.0
    self.top_k = 1          # temperature≈0 → 等价于 greedy（top_k=1）
if self.top_k == -1:
    self.top_k = 1 << 30    # -1（关闭）→ 用一个超大值表示「整个 vocab」
```

也就是说：**「greedy」在 SGLang 内部不是单独的开关，而是 `top_k == 1`**。后面 `SamplingBatchInfo.is_all_greedy`（sampling_batch_info.py:135）就是 `all(top_k <= 1)`。

`verify()`（sampling_params.py:89）做范围检查，并强制 `json_schema`/`regex`/`ebnf` 三者互斥（sampling_params.py:131-137）。`normalize()`（:139）把 `stop` 编码成 token，算出 `stop_str_max_len`（detokenize 时判断 stop 字符串要回看多少 token）。

### 4.2 SamplingBatchInfo：向量化一个 batch

`from_schedule_batch`（sampling_batch_info.py:58）把 `batch.reqs` 里每个请求的标量参数拼成列向量 tensor 并搬到 device：

```
reqs = [r0, r1, r2]
                        temperatures = [[t0],[t1],[t2]]   shape [B,1]  (注意 view(-1,1))
SamplingParams ───────► top_ps       = [p0, p1, p2]       shape [B]
                        top_ks       = [k0, k1, k2]       int32
                        min_ps       = [m0, m1, m2]
is_all_greedy        = all(top_k <= 1)
need_min_p_sampling  = any(min_p > 0)
```

注意 `temperatures` 是 `[B,1]`（方便 `logits.div_(temperatures)` 广播），而 `top_ps/top_ks/min_ps` 是 `[B]`，在 sampler 里再 `view(-1,1)`。

同时它构造 `BatchedPenalizerOrchestrator`（:120），把三种 penalizer 注册进去。**注意它无条件创建 orchestrator**，但每个 penalizer 会先 `_is_required()` 判断本 batch 是否真的有人用到，没用到就不分配 tensor（见 4.4），所以开销很小（设计注释见 sampling_batch_info.py:113-119）。

`SamplingBatchInfo` 还要支持 batch 动态变化：
- `filter_batch`（:199）：请求结束时按 `keep_indices` 收缩所有 tensor + penalizer + custom processor。
- `merge_batch`（:274）：新请求加入时把两个 batch 的 tensor `cat` 起来。**关键坑**：`__len__` 定义在 `temperatures` 上（:146），所以涉及 `len()` 的 merge（custom processor / params）必须在 cat `temperatures` **之前**做完（代码注释 :297-299）。

### 4.3 Sampler：从 logits 采样

`Sampler.forward`（sampler.py:38）流程：

```
logits = logits_output.next_token_logits        # [B, V]
   │
   ├─ ① has_custom_logit_processor? → _apply_custom_logit_processor()   (sampler.py:61)
   │
   ├─ ② NaN 检测（enable_nan_detection）：NaN → -1e5                      (sampler.py:64)
   │
   ├─ ③ is_all_greedy?
   │      yes → batch_next_token_ids = argmax(logits, -1)               (sampler.py:74)
   │      no  → logits.div_(temperatures); softmax → probs              (sampler.py:79-81)
   │             ├─ sampling_backend == "flashinfer":                   (sampler.py:84)
   │             │     need_min_p? → top_k_renorm → top_p_renorm → min_p_sampling_from_probs
   │             │     else        → top_k_top_p_sampling_from_probs (filter_apply_order="joint")
   │             └─ sampling_backend == "pytorch":                      (sampler.py:113)
   │                   top_k_top_p_min_p_sampling_from_probs_torch(...)
   │
   ├─ ④ return_logprob? → 计算 next_token_logprobs / top_logprobs       (sampler.py:134)
   │
   └─ ⑤ grammar 或 SYNC_TOKEN_IDS_ACROSS_TP? → all_reduce(MIN) 跨 TP 同步 (sampler.py:152)
```

PyTorch fallback `top_k_top_p_min_p_sampling_from_probs_torch`（sampler.py:202）是理解采样语义的最佳教材：

```python
probs_sort, probs_idx = probs.sort(dim=-1, descending=True)   # 降序
probs_sum = torch.cumsum(probs_sort, dim=-1)
# top-k：排名 >= top_k 的位置清零
probs_sort[arange(V) >= top_ks.view(-1,1)] = 0.0
# top-p：累计概率（不含自身）超过 top_p 的清零
probs_sort[(probs_sum - probs_sort) > top_ps.view(-1,1)] = 0.0
# min-p：低于 (最大概率 * min_p) 的清零
if need_min_p_sampling:
    probs_sort[probs_sort < (probs_sort[:,0]*min_ps).view(-1,1)] = 0.0
sampled_index = torch.multinomial(probs_sort, num_samples=1)  # 在剩余分布里抽样
```

`top_p` 用 `probs_sum - probs_sort`（即「不含当前 token 的前缀和」）做阈值，保证至少保留概率最大的那个 token。

**flashinfer vs pytorch**：由 `global_server_args_dict["sampling_backend"]` 选（`--sampling-backend`）。flashinfer 用融合 kernel（`sgl_kernel` 里的 `top_k_top_p_sampling_from_probs` 等，sampler.py:16-21），更快；但代码注释指出 flashinfer 的 `top_p_renorm_prob` 有数值问题（sampler.py:86），所以算 logprob 时仍走 torch 实现 `top_p_normalize_probs_torch`（sampler.py:229）。

**第 ⑤ 步的 TP 同步**很值得注意（sampler.py:152-164）：为省一次 all-reduce，SGLang 默认**不**在 TP rank 间同步最终 token id（依赖各 kernel 的确定性）。但用 grammar 时不确定性更易触发 rank 间不一致，进而 hang，所以 `sampling_info.grammars` 非空时强制 `all_reduce(MIN)` 同步。

### 4.4 Penalty：重复 / 频率 / 存在 / min_new_tokens

四种 penalty 的施加方式不同：

| 类型 | 公式 / 行为 | 累积方式 | file:line |
|---|---|---|---|
| **frequency** | `logits -= count(token) * freq_penalty` | `scatter_add_`（按出现次数累加） | `frequency_penalty.py:42-50` |
| **presence** | 出现过一次就减 `presence_penalty` | `scatter_`（只标记存在，覆盖式） | `presence_penalty.py:42-50` |
| **min_new_tokens** | 未达 `min_new_tokens` 前把所有 stop/eos token 设 `-inf` | 预计算 `stop_token_penalties` | `min_new_tokens.py:50-78` |
| **repetition** | （在 `SamplingParams` 中定义，penaltylib 当前未提供独立 penalizer） | — | `sampling_params.py:67` |

`BatchedPenalizerOrchestrator`（orchestrator.py:12）的核心设计是 **lazy 准备**：

- 构造时对每个 penalizer 调 `prepare_if_required()`（orchestrator.py:26）→ `_is_required()` 检查本 batch 有没有人用到该 penalty（如 `frequency_penalty != 0.0`，frequency_penalty.py:18），用到才 `_prepare()` 分配 `[B, V]` 的累积 tensor。
- `cumulate_output_tokens(output_ids)`（orchestrator.py:33）：每步采样后，Scheduler 喂入刚生成的 token（`schedule_batch.py:1473`），让 freq/presence 更新计数、min_new_tokens 累加长度。
- `apply(logits)`（orchestrator.py:43）：把累积惩罚减到 logits 上（in-place）。
- `filter` / `merge`：batch 变化时同步收缩/拼接每个 penalizer 的 tensor。

`min_new_tokens` 实现尤其巧妙（min_new_tokens.py:32-65）：把 `stop_token_ids ∪ additional_stop_token_ids ∪ {eos}` pad 到统一长度，用 `scatter_add_` 把这些位置填成 `-inf`，得到 `stop_token_penalties` 这张 `[B, V]` 表；`_apply` 时只对「输出长度 < min_new_tokens」的行加上这张表（min_new_tokens.py:76-78），从而在达到最短长度前禁止结束。

#### 两种施加时机：overlap vs non-overlap

`SamplingBatchInfo` 提供了两条施加路径（关注 sampling_batch_info.py:176-197）：

```python
def update_penalties(self):                 # overlap mode 用
    if orchestrator.is_required:
        self.linear_penalty = zeros([B, V])
        orchestrator.apply(self.linear_penalty)   # 先把惩罚物化成一张 tensor

def apply_logits_bias(self, logits):
    if self.linear_penalty is not None:           # overlap：直接 add 预先算好的
        logits.add_(self.linear_penalty)
    if orchestrator and orchestrator.is_required: # non-overlap：现场 apply
        orchestrator.apply(logits)
    if self.vocab_mask is not None:               # grammar mask 永远在最后
        self.apply_mask_func(logits=logits, vocab_mask=self.vocab_mask)
```

overlap scheduler 模式下，`update_penalties()` 在 overlap thread 里先把惩罚算成 `linear_penalty`（`tp_worker_overlap_thread.py:211`），forward 完成后只需一次 `add_`，把 CPU-heavy 的准备和 GPU forward 重叠起来。non-overlap 模式则在 `apply_logits_bias` 里现场 apply。两条路最终都在 grammar mask 之前完成——**grammar mask 必须最后施加**，否则被 mask 成 `-inf` 的 token 又可能被 penalty 改回有限值。

### 4.5 结构化输出：FSM / Grammar 约束 logits

宏观思路：把「合法 JSON / 正则」编译成一个**状态机**，在每个解码步根据当前状态产出「下一步允许哪些 token」的集合，把不允许的 token 的 logit 置 `-inf`，于是采样必然落在合法分支。

#### grammar 的生命周期

```
请求带 json_schema/regex/ebnf/structural_tag
        │  Scheduler.handle_generate_request (scheduler.py:1029)
        ▼
key = ("json"|"regex"|"ebnf"|"structural_tag", spec)        (scheduler.py:1037-1044)
grammar_backend.get_cached_or_future_value(key)             (base_grammar_backend.py:151)
   ├─ cache 命中 → value.copy()，直接进 waiting_queue
   └─ 未命中 → 提交 ThreadPoolExecutor 异步编译，请求进 grammar_queue (scheduler.py:1055)
        │  编译（compile_json_schema / RegexGuide.from_regex ...）较慢，所以异步
        ▼
move_ready_grammar_requests (scheduler.py:1754)：编译好的请求转入 waiting_queue，并 set_cache
        │
        ▼  每个解码步：
update_regex_vocab_mask (sampling_batch_info.py:149)：grammar.fill_vocab_mask 填 mask
apply_logits_bias → apply_vocab_mask：logits[mask] = -inf
sampler 采样 → next_token_id
grammar.accept_token(next_token_id) (scheduler_output_processor_mixin.py:129)：推进状态
```

编译是异步的：`get_cached_or_future_value`（base_grammar_backend.py:151）命中缓存就 `copy()` 一份返回，否则 `executor.submit(self._init_value_dispatch, key)` 返回一个 future，请求暂存到 `grammar_queue`，等编译完再放进 waiting queue（`scheduler.py:1754` 的 `move_ready_grammar_requests`）。这避免编译大 schema 阻塞整个调度循环。`_init_value_dispatch`（base_grammar_backend.py:136）按 `key_type` 分发到 `dispatch_json/regex/ebnf/structural_tag`。

#### mask 的填充与施加

`SamplingBatchInfo.update_regex_vocab_mask`（sampling_batch_info.py:149）：

```python
first_grammar = next(g for g in self.grammars if g)
self.vocab_mask = first_grammar.allocate_vocab_mask(vocab_size, batch_size, device)
self.apply_mask_func = first_grammar.apply_vocab_mask   # 用 static method
for i, grammar in enumerate(self.grammars):
    if grammar and not grammar.finished:
        grammar.fill_vocab_mask(self.vocab_mask, i)      # 逐行填本请求的 mask
self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)
```

不同后端的 mask 表示与施加方式不同，这正是 `BaseGrammarObject`（base_grammar_backend.py:29）抽象出来的接口：

| 后端 | mask 表示 | fill | apply |
|---|---|---|---|
| **outlines** | `[B, V]` bool tensor | `RegexGuide.get_next_instruction(state).tokens` 得到允许集合，先全 1 再把允许位置 0（outlines_backend.py:65-71） | `logits.masked_fill_(mask, -inf)`（outlines_backend.py:74） |
| **xgrammar** | 压缩 bitmask（int32，每 bit 一个 token） | `matcher.fill_next_token_bitmask`（xgrammar_backend.py:88） | CUDA 走 Triton kernel `apply_token_bitmask_inplace_triton`，CPU 走 `apply_vocab_mask_cpu`（xgrammar_backend.py:94-100） |

xgrammar 用 **bitmask**（一个 int32 装 32 个 token 的 0/1，bitmask_ops.py:25）大幅压缩 mask 体积，并用 Triton kernel（bitmask_ops.py:84）原地把被屏蔽 token 设 `-inf`，比 outlines 的 `[B,V]` bool tensor + `masked_fill_` 更省显存和带宽。

#### 状态推进

每生成一个 token，Scheduler 调 `grammar.accept_token(next_token_id)`（scheduler_output_processor_mixin.py:129/266）：
- outlines：`self.state = guide.get_next_state(state, token)`（outlines_backend.py:53）。
- xgrammar：`matcher.accept_token(token)`，失败会抛 `ValueError`（xgrammar_backend.py:62-71，对调试很有用）；并支持 `rollback(k)`（xgrammar_backend.py:75）用于 speculative decoding 回退（`MAX_ROLLBACK_TOKENS = 200`）。

#### reasoning 模型的特殊处理

`ReasonerGrammarObject`（reasoner_grammar_backend.py:23）包装真正的 grammar：在 `</think>`（`think_end_id`）之前 `is_in_reasoning=True`，`fill_vocab_mask` 直接跳过（reasoner_grammar_backend.py:42-44），即**思考阶段不施加 JSON 约束**，只有进入正式回答后才约束。由 `create_grammar_backend`（base_grammar_backend.py:193）在 `--reasoning-parser` 且 tokenizer 有 `think_end_id` 时自动套上。

### 4.6 compressed FSM / jump-forward 加速

参考博客：https://lmsys.org/blog/2024-02-05-compressed-fsm/（见 outlines_jump_forward.py:16）。

**动机**：很多结构化输出有**确定性前缀**。例如 JSON `{"name": ` 这几个字符在 FSM 里是一条**唯一出边的链**——无论模型怎么想，下一个合法 token 只有一个。逐个 token 去 forward + 采样纯属浪费。jump-forward decoding 直接把这条确定链一次性「跳」过去，少跑很多步 forward。

`init_state_to_jump_forward`（outlines_jump_forward.py:62）预计算「哪些状态只有唯一出边」：

```
把 regex 编译成 byte-level deterministic FSM (make_byte_level_fsm + make_deterministic_fsm)
遍历所有 transition：
   统计每个 state 的出边数 outgoings_ct（final 状态预置为 1，因为它能终止）
   若某 state 出边数 == 1 → 记入 state_to_jump_forward[state] = JumpEdge(symbol, next_state)
   若出边数 > 1 → 从表里删除（不是确定链）
```

`OutlinesJumpForwardMap.jump_forward_symbol`（outlines_jump_forward.py:146）沿着唯一出边一路走到底，拼出可以直接「填充」的字符串和落点状态。

grammar 对象暴露三个配合方法（`BaseGrammarObject`，base_grammar_backend.py:73-99）：
- `try_jump_forward(tokenizer)`：返回可跳过的字符串（xgrammar 用 `matcher.find_jump_forward_string()`，xgrammar_backend.py:117）。
- `jump_forward_str_state(helper)`：得到跳过后字符串和新状态。
- `jump_and_retokenize(old_ids, new_ids, next_state)`：跳过后需要**重新 tokenize**（见下文难点），并把 grammar 状态对齐（xgrammar 通过 rollback + 逐个 accept 对齐，xgrammar_backend.py:126-141）。

## 5. 设计难点与权衡

1. **greedy 不是开关，是 `top_k==1`**。`temperature<1e-6` 被改写成 `temperature=1.0, top_k=1`（sampling_params.py:82）。改采样逻辑时别去找「greedy flag」——看 `is_all_greedy`（all top_k<=1）。

2. **penalty 的 lazy 分配**。orchestrator 总是被创建，但只有 `_is_required()` 为真才分配 `[B,V]` tensor（orchestrator.py:26，frequency_penalty.py:18）。这是「实现简单」与「不浪费显存」之间的折中：不预先判断要不要建 orchestrator（避免 `filter_batch`/`merge_batch` 的复杂分支，注释见 sampling_batch_info.py:113-119），但 penalizer 内部 lazy。

3. **施加顺序：custom processor → penalty → grammar mask → 采样**。grammar mask 必须最后，否则被置 `-inf` 的非法 token 又被 penalty 拉回有限值，破坏合法性约束（apply_logits_bias，sampling_batch_info.py:187）。

4. **overlap mode 下的 mask 预计算与同步**。overlap scheduler 把 CPU-heavy 的 `update_regex_vocab_mask` / `update_penalties` 放到上一批结果处理时做（scheduler.py:1808、tp_worker_overlap_thread.py:211），用 `sampling_info_done`（threading.Event，sampling_batch_info.py:40）让 forward 线程等待 mask 就绪（model_runner.py:1216-1220）。这是「不阻塞 GPU」的关键，但也引入了**线程同步的隐蔽 bug 面**。

5. **TP 下不同步 token id 的赌注**。默认不 all-reduce token id（省一次通信，sampler.py:152 注释），赌各 kernel 确定性。grammar 在场时强制同步——因为 grammar 状态机一旦在不同 rank 走岔，会直接 hang。

6. **grammar 编译是异步的**。大 JSON schema 编译可能秒级，放进 `ThreadPoolExecutor` + `grammar_queue`（scheduler.py:1046、:1754），并按 `(key_type, key_string)` 缓存（base_grammar_backend.py:111）。**缓存的是 grammar，复用时 `copy()`**（每个请求要独立的状态机），见 `get_cached_or_future_value`（base_grammar_backend.py:151）。

7. **bitmask vs bool mask**。outlines 用 `[B,V]` bool（实现直观），xgrammar 用 int32 bitmask（每 token 1 bit）+ Triton kernel（bitmask_ops.py），在大 vocab（如 150k+）上显存/带宽优势明显。

8. **jump-forward 的「重新 tokenize」难点**。跳过的是**字符/字节**，但模型吃的是 **token**。直接把跳过的字符串 append 到已生成 token 后再 detokenize 会因 tokenizer 的合并规则（同一段文本可能有不同切分）产生边界 token 不一致。所以需要 `jump_and_retokenize`（xgrammar_backend.py:126）：找到与旧 token 序列的公共前缀 `k`，`rollback` 掉不一致的尾部，再逐个 `accept` 新 token，保证 grammar 状态与实际喂给模型的 token 序列严格对齐。这也是 compressed FSM 最容易出隐蔽 bug 的地方。

9. **xgrammar 的 byte-level 与多字节字符**。compressed FSM 在 byte 级别构建（outlines_jump_forward.py:69 `make_byte_level_fsm`），中文等多字节字符（如「霍格」= `\xe9\x9c\x8d\xe6\xa0\xbc`）必须按 byte 链处理，`try_jump_forward` 里专门处理 UTF-8 continuation bytes（outlines_backend.py:90-101）。

## 6. 改代码 / 修 bug 指南

**想加一个新采样参数（比如 typical_p）**
- 在 `SamplingParams.__init__` 加字段 + `verify()`（sampling_params.py:30/89）。
- 在 `SamplingBatchInfo.from_schedule_batch` 加对应 tensor，并在 `filter_batch`/`merge_batch` 的循环列表里加上（sampling_batch_info.py:205/300，否则 batch 变化时 shape 对不上 → crash）。
- 在 `Sampler.forward` 和 `top_k_top_p_min_p_sampling_from_probs_torch` 里实现裁剪逻辑（sampler.py:202）。

**想加一个新 penalty**
- 在 `penaltylib/` 仿照 `frequency_penalty.py` 写一个 `_BatchedPenalizer` 子类，实现 6 个抽象方法（orchestrator.py:154）。
- 在 `penaltylib/__init__.py` 导出，并加进 `from_schedule_batch` 的 `penalizers={...}`（sampling_batch_info.py:123）。

**想加一个新 grammar 后端**
- 仿照 `xgrammar_backend.py` 实现 `BaseGrammarObject`（mask 表示 + fill/apply/accept_token）和 `BaseGrammarBackend`（dispatch_json/regex/ebnf）。
- 在 `create_grammar_backend`（base_grammar_backend.py:167）注册新分支，并加 `--grammar-backend` 选项。

**想加 custom logit processor**
- 继承 `CustomLogitProcessor`（custom_logit_processor.py:19），实现 `__call__`，参考 `DisallowedTokensLogitsProcessor`（:42）。通过 dill 序列化（`to_str`/`from_str`）随请求传入，需 `--enable-custom-logit-processor`（`enable_custom_logit_processor` 标志，sampling_batch_info.py:82）。

**常见 bug 高发区**
- **`merge_batch` 顺序坑**：涉及 `len(self)` 的合并必须在 cat `temperatures` 之前（sampling_batch_info.py:297）。新增 tensor 字段忘了加进 filter/merge 列表 → batch 缩放后 shape 不一致。
- **grammar mask 没在最后施加** → 结构化输出偶发非法 token。
- **overlap 模式 `sampling_info_done` 没 set/wait 对** → forward 拿到未填好的 mask，或死等 hang（scheduler.py:1805 `set_next_batch_sampling_info_done` 必须被调用）。
- **xgrammar `accept_token` 抛 `Tokens not accepted`**（xgrammar_backend.py:67）：通常是 jump-forward / speculative decoding 的 token 序列与 grammar 状态没对齐，或 detokenize 边界问题；`XGrammarGrammar.__repr__`（:143）会打出 `key_string` 和 `accepted_tokens`，是首要排查点。
- **TP hang**：grammar + 跨 rank 不确定，先确认 sampler.py:152 的 `all_reduce` 分支是否触发；必要时设 `SYNC_TOKEN_IDS_ACROSS_TP=1`。

**调试入手点**
- 采样数值异常：开 `--enable-nan-detection`（sampler.py:64），或切 `--sampling-backend pytorch` 用可读的 torch 实现复现（sampler.py:113）。
- penalty 不生效：检查 `_is_required()`（对应 penalizer），以及 `cumulate_output_tokens` 是否被调（schedule_batch.py:1473）。
- grammar 不约束：确认请求确实进了 grammar 路径（scheduler.py:1029）、`grammar_backend` 非 None（`--grammar-backend` 不是 `none`）、`update_regex_vocab_mask` 是否填了 mask（sampling_batch_info.py:149）。
- jump-forward 调试：`outlines_jump_forward.py:181` 的 `test_main` 可直接打印某个 regex 的 jump-forward 链。

---

相关章节：[请求生命周期](01-request-lifecycle.md)、[调度器](12-scheduler.md)、[Model Executor](14-model-executor.md)、[术语表](02-glossary.md)。
