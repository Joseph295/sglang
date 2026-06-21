# 投机解码（Speculative Decoding / EAGLE）

> 本章讲清楚 SGLang 里 EAGLE 投机解码是怎么把「一次前向产出一个 token」变成「一次前向验证多个 token」的：draft 模型如何猜，target 模型如何一次性验证，候选树怎么构建与展开，专用 CUDA graph 如何加速 draft，接受/拒绝逻辑怎么保证分布正确，以及它和 continuous batching / KV cache 协同时的难点。
>
> 行号以撰写时仓库实际状态为准，源码迭代后可能漂移，但符号名稳定，可用 `grep` 复核。

---

## 1. 一句话职责

投机解码用一个**小而快的 draft 模型**（EAGLE 头）在每个 decode step 先「猜」出多个候选 token，组织成一棵**候选树（tree）**，再让**大而慢的 target 模型**在**一次前向**里并行验证这整棵树；接受最长的合法前缀，从而用「1 次 target 前向」产出「多个 token」，在不改变输出分布的前提下降低端到端 decode 延迟。

核心入口是 `EAGLEWorker.forward_batch_speculative_generation`（`python/sglang/srt/speculative/eagle_worker.py:251`），它替代普通路径里的 `TpModelWorker.forward_batch_generation`。

---

## 2. 在端到端链路中的位置

投机解码并不是独立一段，而是**替换了正常 decode step 的「前向 + 采样」环节**。在整条链路里它的位置是：

```
客户端 → 分词(TokenizerManager) → 调度(Scheduler) → KV缓存(RadixAttention)
                                        │
                                        ▼
                            Scheduler.run_batch  (scheduler.py:1551)
                                        │
              spec_algorithm.is_none()? │
                ├── 是 ──► tp_worker.forward_batch_generation   (普通：1 forward = 1 token)
                └── 否 ──► draft_worker.forward_batch_speculative_generation
                                        │   (eagle_worker.py:251)
                                        ▼
            ┌──────────────────────────────────────────────────────────┐
            │  EXTEND(prefill)阶段:                                       │
            │    forward_target_extend → forward_draft_extend            │
            │  DECODE 阶段(核心循环):                                      │
            │    draft()  ──►  verify()  ──►  forward_draft_extend_after_decode
            │     │             │                  │                      │
            │   draft模型      target模型         draft模型               │
            │  多步前向猜树   一次前向验证树    为下一步预热hidden_states  │
            └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                采样/接受拒绝(verify) → 解码(DetokenizerManager) → 响应
```

要点：

- **prefill（EXTEND）**：先让 target 跑一遍拿到 full hidden states，再用这些 hidden states 给 draft 模型「预热」一次（`forward_draft_extend`），产出第一组 `topk_p / topk_index / hidden_states`，存进 `batch.spec_info`。
- **decode**：每一步是 `draft → verify → draft_extend_after_decode` 三段；前两段是真正产出 token 的地方，第三段是为下一个 decode step 准备 draft 输入。
- 普通 decode 一次只走一次 target 前向；投机 decode 一次 step 里 draft 模型要跑 `speculative_num_steps` 次小前向，target 跑 1 次大前向。trade-off：用多次便宜的 draft 前向换一次昂贵的 target 前向产出多个 token。

调度侧的统计（接受率）见 `scheduler.py:1571` 附近：`spec_num_total_accepted_tokens += num_accepted_tokens + batch.batch_size()`。

---

## 3. 关键文件与类

| 符号 | file:line | 作用 |
|------|-----------|------|
| `SpeculativeAlgorithm` | `python/sglang/srt/speculative/spec_info.py:4` | 枚举 `NONE / EAGLE / EAGLE3`，`is_eagle()` 等判断散落全代码库 |
| `EAGLEWorker` | `python/sglang/srt/speculative/eagle_worker.py:54` | 投机 worker，继承 `TpModelWorker`，持有 draft 模型 runner，并引用 `target_worker` |
| `EAGLEWorker.forward_batch_speculative_generation` | `eagle_worker.py:251` | 顶层分发：按 forward_mode 走 prefill / decode / idle 路径 |
| `EAGLEWorker.draft` | `eagle_worker.py:320` | 分配 draft KV 槽、跑多步 draft 前向、构建 verify 输入 |
| `EAGLEWorker.draft_forward` | `eagle_worker.py:437` | draft 模型的多步循环（被 CUDA graph 捕获的核心） |
| `EAGLEWorker.verify` | `eagle_worker.py:491` | target 一次前向验证整棵树，调用接受/拒绝采样 |
| `EAGLEWorker.forward_draft_extend_after_decode` | `eagle_worker.py:647` | 用接受后的 token 重新喂 draft，产出下一步的 `topk_*` |
| `EAGLEWorker.capture_for_decode` | `eagle_worker.py:681` | 把 draft logits softmax + topk 写回 `EagleDraftInput` |
| `EagleDraftInput` | `eagle_utils.py:48` | draft 阶段在 batch 间传递的状态（topk_p/index、hidden_states、verified_id…） |
| `EagleVerifyInput` | `eagle_utils.py:182` | verify 阶段的输入（draft_token、tree mask、retrive_* 索引） |
| `EagleVerifyOutput` | `eagle_utils.py:167` | verify 结果（accept 长度、verified_id、下一步 draft_input） |
| `EagleVerifyInput.verify` | `eagle_utils.py:308` | **接受/拒绝逻辑的核心**：greedy / 概率两条路径、free 未接受 KV |
| `select_top_k_tokens` | `eagle_utils.py:747` | 每一步 draft 把上一步 topk 展开成新一层候选（树的展开规则） |
| `build_tree_kernel_efficient` | `build_eagle_tree.py:43` | 把 score/token/parents 列表压成扁平候选树 + tree mask + retrive 索引 |
| `EAGLEDraftCudaGraphRunner` | `eagle_draft_cuda_graph_runner.py:30` | 专门捕获 `draft_forward` 多步循环的 CUDA graph runner |
| `assign_draft_cache_locs` | `eagle_utils.py:626` | Triton kernel：把 draft 的 KV 槽位写进 `req_to_token` |
| `verify_tree_greedy` / `tree_speculative_sampling_target_only` | `sgl_kernel`（在 `eagle_utils.py:28` 导入） | 树上 greedy / 概率接受的 CUDA kernel |

入口绑定：`Scheduler` 在 `scheduler.py:287` 检测 `spec_algorithm.is_eagle()` 后实例化 `EAGLEWorker` 作为 `self.draft_worker`。

---

## 4. 核心数据流 / 执行流程

### 4.1 三种 forward_mode 与新增 ForwardMode

投机解码引入了两个新的 forward mode（`forward_batch_info.py:65`）：

- `TARGET_VERIFY`：target 模型验证候选树（被当成一种 extend，见 `forward_batch_info.py:81`）。
- `DRAFT_EXTEND`：draft 模型用接受的 token 做小 extend。

注意 `is_cuda_graph()`（`forward_batch_info.py:106`）把 `DECODE / TARGET_VERIFY / IDLE` 都列为可走 CUDA graph，而 `DRAFT_EXTEND` 不在其中。

### 4.2 prefill 阶段（EXTEND）

`forward_batch_speculative_generation` 的 else 分支（`eagle_worker.py:290`）：

```
forward_target_extend(batch)              # target 跑 prefill, capture_hidden_mode=FULL
  └─ 拿到 logits_output.hidden_states (全量) + next_token_ids
forward_draft_extend(batch, hidden_states, next_token_ids)   # eagle_worker.py:617
  ├─ EagleDraftInput.prepare_for_extend: 把 input_ids 左移一位, 末尾填 verified_id
  ├─ draft 模型 forward (capture_hidden_mode=LAST)
  └─ capture_for_decode: softmax + fast_topk → 写回 topk_p / topk_index / hidden_states
```

`prepare_for_extend`（`eagle_utils.py:71`）这个「左移一位」很关键：EAGLE 的 draft 模型吃的是 **target 的 hidden state + 下一个 token 的 embedding**，所以输入序列要相对原序列错位一位。

### 4.3 decode 阶段：draft（猜树）

`draft()`（`eagle_worker.py:320`）做三件事：

1. **分配 draft KV 槽**。每个 seq 要为 `topk * speculative_num_steps` 个 draft token 分配 KV（`eagle_worker.py:334`）。`page_size==1` 走 `alloc_token_slots`；`page_size>1 && topk==1` 走 paged 分配；`page_size>1 && topk>1` **明确 `NotImplementedError`**（`eagle_worker.py:364`）。分配时 `backup_state=True`，因为 draft 写的 KV 是「投机性的」，后面要 `restore_state` 回滚（见 `eagle_worker.py:422`）。
2. **跑多步 draft 前向**：能走 CUDA graph 就 `cuda_graph_runner.replay`，否则 `draft_forward`。
3. **构建 verify 输入**：`EagleVerifyInput.create`（`eagle_utils.py:195`）→ `build_tree_kernel_efficient`。

`draft_forward`（`eagle_worker.py:437`）是树的「逐层展开」循环：

```
for i in range(speculative_num_steps):
    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(...)   # 展开第 i 层
    score_list.append(tree_info[0]); token_list.append(tree_info[1]); parents_list.append(tree_info[2])
    if i == last: break          # 最后一层不必再前向(下一层不用了)
    # 设置 out_cache_loc 的第 i 段、positions += 1、换第 i 个 attn backend
    logits_output = draft_model.forward(...)
    probs = softmax(logits); topk_p, topk_index = fast_topk(probs, topk)   # 下一层候选
```

`select_top_k_tokens`（`eagle_utils.py:747`，带 `@torch.compile(dynamic=True)`）定义了树的展开规则：

- **第 0 步**：把 prefill 得到的 `topk_index` 直接展平成 `topk` 个根候选；`hidden_states` 按 topk repeat；parents 是 `[-1, 0..topk-1]`。
- **后续步**：把当前 `scores (b,topk)` 与新一层 `topk_p (b,topk,topk)` 相乘得到 `(b,topk,topk)` 的联合分数，再 `fast_topk` 选出 topk 条路径继续展开。`selected_input_index` 用来挑出对应的父 hidden state（`eagle_utils.py:782`）。parents 编号用 `topk_cs_index + (topk**2*(i-1)+topk)` 保证全树编号唯一。

也就是说：树宽固定为 `topk`，深度为 `speculative_num_steps`，每层都做一次 beam-search 式剪枝，只保留 topk 条最有希望的路径。

### 4.4 把列表压成树：build_tree_kernel_efficient

`build_tree_kernel_efficient`（`build_eagle_tree.py:43`）把 `draft_forward` 收集的 `score_list / token_list / parents_list` 转换成 verify 阶段需要的所有张量。

预处理（`build_eagle_tree.py:15`）：

```
score_list  cat → (b, n)            # n = 1 + (num_steps-1)*topk
token_list  cat → (b, ...)
top_scores = topk(score_list, num_verify_tokens-1)     # 全树里挑分数最高的 num_verify_tokens-1 个节点
top_scores_index = sort(...)                            # 排序保证父在子前
draft_tokens = gather(token_list, top_scores_index)
draft_tokens = cat(verified_id, draft_tokens)          # 根是上一轮已验证的 token
```

注意：树**最多** `1 + (num_steps-1)*topk` 个节点，但最终只取 `num_verify_tokens`（= `speculative_num_draft_tokens`）个去验证 —— 这是又一次全局剪枝。

CUDA kernel `sgl_build_tree_kernel_efficient`（`build_eagle_tree.py:92`）产出：

- `tree_mask`：每个 draft token 在 attention 里能看见哪些 token 的布尔扁平 mask（含 prefix + 树内祖先），保证树形 causal —— 每个候选只能 attend 到它在树上的祖先链，不能 attend 到兄弟分支。
- `positions`：每个 draft token 的绝对位置（`build_eagle_tree.py:88` 的注释例子：深度 `[0,1,1,2]` + prompt 长 7 → `[7,8,8,9]`）。
- `retrive_index / retrive_next_token / retrive_next_sibling`：用「first-child / next-sibling」表示法编码树结构，供接受 kernel 在树上 DFS。

`build_eagle_tree.py:115` 的 `test_build_tree_kernel_efficient` 是理解这些张量含义的**最佳活文档**：它给了一个 bs=2、topk=4、depth=4、num_draft_token=8 的完整 example，并对 `positions / retrive_* / draft_tokens` 做了硬编码断言。想搞懂树结构，直接读这个测试用例和断言值。

### 4.5 decode 阶段：verify（一次前向 + 接受/拒绝）

`verify()`（`eagle_worker.py:491`）：

```
spec_info.prepare_for_verify(batch)         # input_ids = draft_token; 分配/写入 verify 的 KV 槽
batch.forward_mode = TARGET_VERIFY
target_worker.forward_batch_generation(mwb, skip_sample=True)   # 一次大前向, 拿全树 logits
# (可选) 语法约束 vocab_mask 在 CPU 上与前向 overlap 生成 (generate_token_bitmask)
res = spec_info.verify(batch, logits_output, allocator, page_size, vocab_mask)  # 接受/拒绝
# 只保留被接受的 logits/hidden_states
batch.forward_mode = DECODE; batch.spec_info = res.draft_input
```

`EagleVerifyInput.verify`（`eagle_utils.py:308`）是接受/拒绝的核心，分两条路径：

- **全 greedy**（`eagle_utils.py:359`）：`target_predict = argmax(logits)`，调用 `verify_tree_greedy` kernel 在树上沿 retrive 链做最长匹配：从根往下，只要某节点的 draft token 等于 target 在其父位置的 argmax 就接受，否则停。
- **带采样**（`eagle_utils.py:373`）：先对 target logits 做 temperature / top_k / top_p 重归一化得 `target_probs`，配 `coins = rand` 调 `tree_speculative_sampling_target_only`。这是 EAGLE 的**树版投机采样**，用 `threshold_single / threshold_acc`（来自 `global_server_args_dict`）控制接受阈值，理论上保证输出分布与直接从 target 采样一致。

接受之后的关键收尾（`eagle_utils.py:431`～）：

1. 逐 req 逐接受 token 调 `req.check_finished()`，命中 finish 就把后面 accept 全置 -1（`eagle_utils.py:451`）；同时更新 grammar 状态（`req.grammar.accept_token`，`eagle_utils.py:458`）。
2. 构造 `evict_mask`：所有 draft 槽中**未被接受**的位置都要 free 掉（`eagle_utils.py:475`）—— 这是投机解码 KV 管理最容易出 bug 的地方。`page_size != 1` 时还要 `align_evict_mask_to_page_size`（`eagle_utils.py:478`）按页对齐，避免释放半页。
3. `token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])`：把被拒绝候选的 KV 还回去。
4. `assign_req_to_token_pool` 把接受 token 的 KV 槽写进 `req_to_token`，`batch.seq_lens.add_(accept_length + 1)`（+1 是 bonus token：哪怕全拒绝，target 在根位置的预测也是一个合法新 token，accept_length 至少为 0 但总产出 1 个 token）。

返回的 `EagleVerifyOutput.draft_input` 已经带上接受后的 `hidden_states[accept_index]` 和 `verified_id`，供下一步 draft 使用。

### 4.6 decode 阶段：forward_draft_extend_after_decode

`forward_draft_extend_after_decode`（`eagle_worker.py:647`）：用刚接受的 token 序列（长度 = accept_length+1）让 draft 模型做一次小 extend（`DRAFT_EXTEND` mode），重新算出下一个 decode step 要用的 `topk_p / topk_index / hidden_states`。它会先备份 `seq_lens / req_pool_indices / accept_length`（因为 `prepare_extend_after_decode` 会 in-place 改 `batch.seq_lens`），跑完再恢复，并把 `forward_mode` 还原成 `DECODE`（`eagle_worker.py:675`）。

整个 decode step 的数据形态流转：

```
spec_info(EagleDraftInput: topk_p,topk_index,hidden_states)
    │ draft()  ─ draft_forward 多步 ─►  score/token/parents 列表
    │ build_tree_kernel_efficient    ─►  EagleVerifyInput(draft_token, tree_mask, retrive_*, positions)
    ▼
verify()  ─ target 1 forward ─►  全树 logits
    │ verify kernel ─► accept_index, predict, accept_length
    │ free 未接受 KV ; 写接受 KV
    ▼
EagleVerifyOutput(verified_id, accept_length, draft_input=新的EagleDraftInput无topk)
    │ forward_draft_extend_after_decode ─► capture_for_decode 补上 topk_p/topk_index
    ▼
下一个 decode step 的 spec_info
```

---

## 5. 专用 CUDA graph runner

普通 decode 用 `CudaGraphRunner`。draft 的多步循环结构特殊（每步换 attn backend、in-place 改 `out_cache_loc / positions`），所以单独写了 `EAGLEDraftCudaGraphRunner`（`eagle_draft_cuda_graph_runner.py:30`）。

设计要点：

- **每个 token 数固定为 topk**：`num_tokens_per_bs = speculative_eagle_topk`（`eagle_draft_cuda_graph_runner.py:46`），因为 draft 第 0 步的输入正好是 `bs * topk` 个根候选。
- **out_cache_loc buffer 大小** = `max_num_token * speculative_num_steps`（`eagle_draft_cuda_graph_runner.py:70`），覆盖所有步。
- **捕获的就是 `draft_forward`**：`run_once` 里调 `self.eagle_worker.draft_forward(forward_batch)`（`eagle_draft_cuda_graph_runner.py:156`）。因为 `draft_forward` 会 in-place 改 `out_cache_loc` 和 `spec_info.hidden_states`，捕获前后要**备份并还原**这两个字段（`eagle_draft_cuda_graph_runner.py:153`、`:158`），否则第二次 replay 用的是被改过的指针。
- **padding 到 capture_bs**：`replay`（`eagle_draft_cuda_graph_runner.py:183`）用 `bisect` 找到 >= raw_bs 的最近捕获 bs，多出来的部分用 `seq_lens.fill_(1)` 填 dummy，replay 完用 `_postprocess_output_to_raw_bs` 把输出裁回 raw_bs（`eagle_draft_cuda_graph_runner.py:176`）。
- **attn backend 的多步状态**：`draft_attn_backend` 是 `*MultiStepDraftBackend`（按 attention_backend 在 `init_attention_backend`，`eagle_worker.py:151` 选择），持有 `speculative_num_steps` 个 sub-backend，`draft_forward` 里 `forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]`（`eagle_worker.py:475`）逐步切换。CUDA graph 捕获/replay 时分别调 `init_forward_metadata_capture_cuda_graph` / `init_forward_metadata_replay_cuda_graph`。

verify 阶段（`TARGET_VERIFY`）用的是 **target worker 自己的** CUDA graph（普通 `CudaGraphRunner`，因为 `is_cuda_graph()` 把 `TARGET_VERIFY` 算作可捕获），不在本 runner 里。`DRAFT_EXTEND` 目前不走 CUDA graph（`init_cuda_graphs` 里 `cuda_graph_runner_for_draft_extend = None`，`eagle_worker.py:226`；`draft_extend_attn_backend` 一旦非空就 `NotImplementedError`，`eagle_worker.py:244`）。

---

## 6. 设计难点与权衡

### 6.1 draft 与 target 共享 KV pool 但各有一份

`EAGLEWorker.__init__` 里 draft 和 target 共享 `req_to_token_pool` 和 `token_to_kv_pool_allocator`（`eagle_worker.py:88`），但**各自拥有自己的 KV cache pool**（注释 `eagle_worker.py:87`）。共享 allocator 是为了让 draft 的投机 KV 和 target 的真实 KV 用同一套槽位管理，方便 `restore_state` 回滚和 `free` 未接受候选。

draft 还**共享 target 的 embedding 和 lm_head**（`eagle_worker.py:121`、`:138`）——EAGLE 的设计前提就是 draft 头复用 target 的词表投影。EAGLE3 例外：不共享 lm_head，且用 `hot_token_id` 把 draft 输出映射回完整词表（`eagle_worker.py:123`、`draft_forward` 里 `topk_index = self.hot_token_id[topk_index]`，`eagle_worker.py:447`）。

### 6.2 投机 KV 的「先分配后回滚 / 部分释放」

这是与 continuous batching + KV cache 协同最难的点：

- draft 阶段乐观地给 `topk * num_steps` 个候选都分配 KV，并 `backup_state`；draft 跑完立刻 `restore_state`（`eagle_worker.py:422`）——因为 draft 写的 KV 只是为了让后续 draft step 能 attend，验证阶段不需要它们长期存在。
- verify 阶段重新为 `num_verify_tokens` 个候选分配 KV、跑 target，然后**只保留被接受的，free 掉被拒绝的**（`eagle_utils.py:475`～`:487`）。
- `page_size > 1` 时释放必须按页对齐（`align_evict_mask_to_page_size`），否则会把还在用的半页释放掉。这也是为什么 `page_size > 1` 时强制 `topk = 1`（`server_args.py:420`）：topk>1 的「最后半页复制给每个 top-k 段」的逻辑还没实现（`eagle_worker.py:343` 的大段注释 + `NotImplementedError`）。

### 6.3 接受长度可变 → batch 内长度不齐

每个 req 接受的 token 数不同（accept_length 各异），所以 `accept_length` 是 `(bs,)` 张量，后续 `seq_lens`、`positions`、`req_to_token` 全要按变长处理（大量 Triton kernel：`assign_req_to_token_pool`、`create_extend_spec_info`）。`forward_draft_extend_after_decode` 里 `batch.extend_lens = [x+1 for x in accept_length_cpu]`（`eagle_utils.py:90`）就是把变长接受结果转成变长 extend。

### 6.4 与 overlap scheduler / continuous batching 协同

- `forward_batch_speculative_generation` 返回 `model_worker_batch.bid`（`eagle_worker.py:279`）供 overlap 调度对齐结果，和普通路径接口对齐。
- batch 的 filter / merge（请求中途加入/退出）通过 `EagleDraftInput.filter_batch`（`eagle_utils.py:144`）和 `merge_batch`（`eagle_utils.py:150`）维护 `topk_p/topk_index/hidden_states/verified_id` 的一致裁剪与拼接。改 continuous batching 相关逻辑时这两个方法必须同步。
- 注意 `forward_batch_speculative_generation` 的 docstring 明确警告（`eagle_worker.py:256`）：**batch 的很多字段会在执行过程中被 in-place 修改**，最终状态不保证等于入参状态。这是读这段代码最大的认知负担来源。

### 6.5 参数耦合

`speculative_num_steps`（树深）、`speculative_eagle_topk`（树宽）、`speculative_num_draft_tokens`（最终验证的节点数）三者强耦合，`server_args.py:409`～`:433` 有一堆自动调整：topk==1 时强制 `num_draft_tokens = num_steps + 1`，page_size>1 时强制 topk=1。改这些参数前先读这段约束。

---

## 7. 改代码 / 修 bug 指南

### 想改 X → 看这里

| 你想做的事 | 入手位置 |
|------------|----------|
| 改树的展开/剪枝策略（每层怎么选候选） | `select_top_k_tokens`（`eagle_utils.py:747`） |
| 改树结构编码 / tree mask / positions | `build_tree_kernel_efficient`（`build_eagle_tree.py:43`）+ `sgl_kernel` 里的 `sgl_build_tree_kernel_efficient` |
| 改接受/拒绝判定（greedy 或采样） | `EagleVerifyInput.verify`（`eagle_utils.py:308`）+ kernel `verify_tree_greedy` / `tree_speculative_sampling_target_only` |
| 改接受阈值 | `global_server_args_dict["speculative_accept_threshold_single"/"_acc"]`（`eagle_utils.py:411`） |
| 改 draft KV 分配 / page 逻辑 | `EAGLEWorker.draft`（`eagle_worker.py:332`）+ `assign_draft_cache_locs`（`eagle_utils.py:626`） |
| 支持新 attention backend | `EAGLEWorker.init_attention_backend`（`eagle_worker.py:151`），需提供对应 `*MultiStepDraftBackend` |
| 改/调 draft CUDA graph | `EAGLEDraftCudaGraphRunner`（`eagle_draft_cuda_graph_runner.py:30`），尤其 `capture_one_batch_size` 的备份还原（`:153`/`:158`） |
| 加 EAGLE3 / hot token map | `eagle_worker.py:92`～`:138`、`draft_forward` 里 `hot_token_id` 映射（`:447`） |
| 改语法约束(structured output)与树的交互 | `traverse_tree` / `generate_token_bitmask`（`eagle_utils.py:827` / `:893`） |

### 常见 bug 高发区

1. **KV 泄漏 / 错放**：`verify` 里 `evict_mask` 与 `accept_index` 的对应（`eagle_utils.py:473`～`:487`）。改接受逻辑时极易漏 free 或多 free，表现为 KV pool 慢慢耗尽 OOM，或读到错误 KV 导致乱码。`page_size>1` 时漏掉 `align_evict_mask_to_page_size` 会破坏分页。
2. **CUDA graph 字段污染**：`draft_forward` in-place 改 `out_cache_loc` / `hidden_states`，捕获器靠备份还原（`eagle_draft_cuda_graph_runner.py:153`/`:158`）。新增 in-place 修改的字段而忘了备份，会让第二次 replay 用脏指针，表现为 batch>1 或多次 replay 后结果错乱。
3. **batch state 被 in-place 改后未恢复**：`forward_draft_extend_after_decode` 备份/恢复 `seq_lens` 等（`eagle_worker.py:648`～`:679`），漏一个字段就会污染下一 step。
4. **变长接受的 finish 处理**：`verify` 里某 req finish 后要把后续 accept 置 -1 并重算 accept_length（`eagle_utils.py:451`、`:470`），否则会多吐 token 或把已 free 的 KV 当作有效。
5. **filter/merge 不同步**：continuous batching 中途加退请求，`EagleDraftInput.filter_batch/merge_batch`（`eagle_utils.py:144`/`:150`）与实际 batch 维度不一致会直接 shape mismatch。

### 调试入手点

- **接受率异常低**：开 `SIMULATE_ACC_LEN` 环境变量（`eagle_utils.py:45`、`_generate_simulated_accept_index`，`eagle_utils.py:796`）可强制模拟固定接受长度，隔离「是 draft 质量差」还是「verify/KV 逻辑 bug」。
- **数值发散（NaN）**：`--enable-nan-detection` 触发 `_detect_nan_if_needed`（`eagle_worker.py:688`），draft 各步前向后都会检查。
- **树结构对不上**：直接跑 `python build_eagle_tree.py`（文件末尾 `eagle_utils.py` 同款的 `test_build_tree_kernel_efficient`，`build_eagle_tree.py:387`），对比断言值理解 retrive_* 含义。
- **统计接受效率**：看 scheduler 打印的 accept length，源头是 `scheduler.py:1571` 累加的 `spec_num_total_accepted_tokens / spec_num_total_forward_ct`。

---

## 相关章节

- [调度器](12-scheduler.md)：`run_batch` 如何分发到 `forward_batch_speculative_generation`。
- [KV 缓存与 RadixAttention](13-memcache-radixattention.md)：投机 KV 的分配/回滚/释放依赖的 allocator 与 `req_to_token` 池。
- [Model Executor](14-model-executor.md)：`ForwardBatch` / `ForwardMode` / 普通 `CudaGraphRunner`，本章的 draft runner 复用了其 `capture` 流程。
- [术语表](02-glossary.md)：draft / target / topk / tree mask 等术语速查。
